"""
AI evaluation agent for StangWatch.

Subscribes to the event bus, evaluates events using a local LLM (Ollama),
and sends Telegram alerts when suspicious activity is detected.

Runs Ollama inference in a background thread so it never blocks the
detection pipeline (inference takes 1-3 seconds).

Usage:
    from agent.evaluator import EvalAgent

    agent = EvalAgent(config, storage, scene_memory, redis_client)
    agent.subscribe(bus)
    agent.start()
    # ... events flow through the pipeline ...
    agent.stop()
"""

import os
import queue
import threading
import time
from datetime import datetime

from agent.ollama_client import OllamaClient
from agent.vision import VisionDescriber, GeminiVisionDescriber, GroqVisionDescriber
from agent.prompts import SYSTEM_PROMPT, build_user_prompt
from agent.decisions import DecisionStorage
from agent.telegram import TelegramSender
from events.tracker import (
    EVENT_APPEARED, EVENT_COMPANION,
    EVENT_OBJECTS_CHANGED, EVENT_RETURNED,
    EVENT_TRACK_SUMMARY,
)


# Feedback learning thresholds
FEEDBACK_MIN_SAMPLES = 10
FEEDBACK_CACHE_TTL = 600

# Events worth evaluating — every summary goes to the AI, it decides everything
EVAL_EVENTS = [
    EVENT_APPEARED,
    EVENT_COMPANION,
    EVENT_OBJECTS_CHANGED,
    EVENT_RETURNED,
    EVENT_TRACK_SUMMARY,
]


class EvalAgent:
    """
    AI agent that evaluates detection events and sends alerts.

    Architecture:
    - Event handlers (on bus) enqueue events (non-blocking, <1ms)
    - Worker thread dequeues and calls Ollama (1-3 seconds per eval)
    - Never blocks the detection pipeline
    """

    def __init__(self, config, storage, scene_memory=None, redis_client=None,
                 escalation=None, camera_id="cam1"):
        """
        Args:
            config: StangWatchConfig
            storage: EventStorage for track history lookups
            scene_memory: SceneMemory for current scene context (or None)
            redis_client: Redis connection for cooldown tracking (or None)
            escalation: EscalationManager instance (or None for simple sends)
            camera_id: identifier for this camera
        """
        self.config = config
        self.storage = storage
        self.scene_memory = scene_memory
        self.redis = redis_client
        self.camera_id = camera_id

        # Ollama text client (reasoning + decisions)
        self._ollama = OllamaClient(
            model=config.agent.model,
            host=config.agent.ollama_host,
            timeout=config.agent.timeout_seconds,
        )

        # Vision describer (snapshot → plain-English description)
        vp = config.agent.vision_provider
        if vp == "groq" and config.secrets.groq_api_key:
            self._vision = GroqVisionDescriber(
                api_key=config.secrets.groq_api_key,
                timeout=config.agent.timeout_seconds,
            )
            print(f"  Vision provider: Groq (Llama 4 Scout)")
        elif vp == "gemini" and config.secrets.gemini_api_key:
            self._vision = GeminiVisionDescriber(
                api_key=config.secrets.gemini_api_key,
                timeout=config.agent.timeout_seconds,
            )
            print(f"  Vision provider: Gemini Flash (cloud)")
        else:
            self._vision = VisionDescriber(
                model=config.agent.vision_model,
                host=config.agent.ollama_host,
                timeout=config.agent.timeout_seconds,
            )
            print(f"  Vision provider: Ollama ({config.agent.vision_model})")

        # Decision storage (same database)
        db_path = str(
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                config.storage.db_path,
            )
        )
        self._decisions = DecisionStorage(db_path)

        # Telegram sender (fallback when escalation is disabled)
        self._telegram = TelegramSender(
            bot_token=config.secrets.telegram_bot_token,
            chat_id=config.secrets.telegram_chat_id,
        )

        # Escalation manager (optional — if configured, routes alerts through chain)
        self._escalation = escalation

        # Work queue (maxsize prevents unbounded memory if Ollama is slow)
        self._queue = queue.Queue(maxsize=50)

        # Worker thread
        self._running = False
        self._thread = None

        # Cooldown tracking
        self._cooldown_seconds = config.agent.cooldown_seconds
        # In-memory fallback if Redis is unavailable
        self._memory_cooldowns = {}

        # Feedback cache (refreshed every FEEDBACK_CACHE_TTL seconds)
        self._feedback_cache = {}
        self._feedback_refreshed_at = 0.0

    def subscribe(self, bus):
        """Register handlers for evaluated event types on the bus."""
        for event_type in EVAL_EVENTS:
            def make_handler(et):
                def handler(event_data):
                    self._enqueue(et, event_data)
                return handler
            bus.on(event_type, make_handler(event_type))

    def start(self):
        """Start the background worker thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._worker,
            daemon=True,
            name=f"eval-agent-{self.camera_id}",
        )
        self._thread.start()
        print(f"[{self.camera_id}] EvalAgent started (model: {self.config.agent.model})")

    def stop(self):
        """Stop the worker thread."""
        self._running = False
        if self._thread is not None:
            # Put sentinel to unblock the worker
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                pass
            self._thread.join(timeout=5)
            print("EvalAgent stopped")

    @property
    def decision_storage(self):
        """Expose decision storage for API endpoints."""
        return self._decisions

    def _enqueue(self, event_type, event_data):
        """Add an event to the evaluation queue (non-blocking)."""
        try:
            self._queue.put_nowait((event_type, event_data))
        except queue.Full:
            # Queue full — Ollama is falling behind, drop oldest
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait((event_type, event_data))
            except queue.Full:
                pass

    def _worker(self):
        """Background thread: dequeue events and evaluate with Ollama."""
        while self._running:
            try:
                item = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            # Sentinel value signals shutdown
            if item is None:
                break

            event_type, event_data = item

            # Check cooldown — skip if we recently evaluated this track
            track_id = event_data.get("track_id", 0)
            if self._is_on_cooldown(track_id):
                continue

            # Check Ollama health
            if not self._ollama.is_healthy():
                print(f"  AGENT: Ollama unavailable, skipping {event_type} for Track #{track_id}")
                continue

            # Step 1: Find video clip and snapshot
            video_path = self._find_video(event_type, event_data)
            snapshot_path = self._find_snapshot(event_type, event_data)

            # Step 2: Vision perceives — describe what the camera shows
            # Prefer multi-frame video analysis over single snapshot
            visual_description = None
            if video_path and self._vision.is_available():
                visual_description = self._vision.describe_video(video_path)
            if visual_description is None and snapshot_path and self._vision.is_available():
                visual_description = self._vision.describe(snapshot_path)

            # Step 3: Build context (now includes visual description + cross-camera)
            scene_summary = None
            cross_camera = None
            if self.scene_memory is not None:
                scene_summary = self.scene_memory.get_scene_summary()
                cross_camera = self.scene_memory.get_all_cameras_summary()

            track_history = self.storage.get_by_track(track_id, limit=10,
                                                       camera_id=self.camera_id)

            feedback_context = self._get_feedback_context(event_type)

            user_prompt = build_user_prompt(
                event_type, event_data, scene_summary, track_history,
                visual_description=visual_description,
                cross_camera_context=cross_camera,
                feedback_context=feedback_context,
            )

            # Step 4: Text model reasons — decides severity
            start_ms = time.time()
            result = self._ollama.evaluate(SYSTEM_PROMPT, user_prompt)
            eval_ms = int((time.time() - start_ms) * 1000)

            if result is None:
                print(f"  AGENT: Evaluation failed for {event_type} Track #{track_id}")
                continue

            alert = result["alert"]
            severity = result["severity"]
            reason = result["reason"]
            recommendation = result.get("recommendation", "")

            # Save decision (always, for auditability)
            decision_id = self._decisions.save_decision(
                event_type=event_type,
                track_id=track_id,
                alert=alert,
                severity=severity,
                reason=reason,
                recommendation=recommendation,
                eval_duration_ms=eval_ms,
                camera_id=self.camera_id,
            )

            if alert and severity in ("medium", "high"):
                # Set cooldown so we don't spam
                self._set_cooldown(track_id)

                # Route through escalation if configured, otherwise simple send
                if self._escalation and self._escalation.is_configured():
                    alert_id = self._escalation.escalate(
                        decision_id=decision_id,
                        event_type=event_type,
                        track_id=track_id,
                        severity=severity,
                        reason=reason,
                        recommendation=recommendation,
                        description=visual_description or "",
                        snapshot_path=snapshot_path,
                        video_path=video_path,
                        camera_id=self.camera_id,
                    )
                    status = f"escalated (#{alert_id})" if alert_id else "escalation failed"
                    print(f"  AGENT: ALERT {severity.upper()} | {event_type} Track #{track_id} | {status} | {eval_ms}ms")
                elif self._telegram.is_configured():
                    sent = self._telegram.send_alert(
                        event_type=event_type,
                        track_id=track_id,
                        severity=severity,
                        reason=reason,
                        recommendation=recommendation,
                        description=visual_description or "",
                        snapshot_path=snapshot_path,
                        video_path=video_path,
                    )
                    status = "sent" if sent else "FAILED"
                    print(f"  AGENT: ALERT {severity.upper()} | {event_type} Track #{track_id} | Telegram: {status} | {eval_ms}ms")
                else:
                    print(f"  AGENT: ALERT {severity.upper()} | {event_type} Track #{track_id} | No alert channel configured | {eval_ms}ms")
            else:
                print(f"  AGENT: {severity} | {event_type} Track #{track_id} | {reason[:60]} | {eval_ms}ms")

    def _get_feedback_context(self, event_type):
        """
        Build feedback context from past alert outcomes for prompt injection.

        Returns dict with guidance strings, or None if not enough data.
        """
        if self._escalation is None:
            return None

        now = time.time()
        if now - self._feedback_refreshed_at > FEEDBACK_CACHE_TTL:
            try:
                self._feedback_cache = self._escalation.storage.get_outcome_stats(
                    days=30, camera_id=self.camera_id,
                )
                self._feedback_refreshed_at = now
            except Exception:
                return None

        stats = self._feedback_cache
        if not stats:
            return None

        resolved_total = stats.get("true_alerts", 0) + stats.get("false_alarms", 0)
        if resolved_total < FEEDBACK_MIN_SAMPLES:
            return None

        overall_fp = stats.get("false_positive_rate", 0.0)
        result = {
            "overall_fp_rate": overall_fp,
            "resolved_total": resolved_total,
            "event_type_guidance": None,
            "severity_guidance": [],
        }

        # Event-type specific guidance (only if FP rate >= 25%)
        by_et = stats.get("by_event_type", {})
        if event_type in by_et:
            et_stats = by_et[event_type]
            et_resolved = et_stats["true"] + et_stats["false"]
            if et_resolved >= 5 and et_stats["fp_rate"] >= 0.25:
                result["event_type_guidance"] = (
                    f"{event_type} events have a {int(et_stats['fp_rate'] * 100)}% "
                    f"false alarm rate (from {et_resolved} resolved alerts). "
                    f"Be more conservative with these events."
                )

        # Severity guidance (only if FP rate >= 25%)
        by_sev = stats.get("by_severity", {})
        for sev, sev_stats in by_sev.items():
            sev_resolved = sev_stats["true"] + sev_stats["false"]
            if sev_resolved >= 5 and sev_stats["fp_rate"] >= 0.25:
                result["severity_guidance"].append(
                    f"{sev} severity alerts have a {int(sev_stats['fp_rate'] * 100)}% "
                    f"false alarm rate (from {sev_resolved} resolved). "
                    f"Raise the bar before assigning {sev} severity."
                )

        return result

    def _is_on_cooldown(self, track_id):
        """Check if this track_id is on alert cooldown."""
        key = f"stang:cooldown:{self.camera_id}:{track_id}"

        # Try Redis first
        if self.redis is not None:
            try:
                return self.redis.exists(key) > 0
            except Exception:
                pass

        # Fallback to in-memory
        expires = self._memory_cooldowns.get(track_id, 0)
        return time.time() < expires

    def _set_cooldown(self, track_id):
        """Set cooldown for a track_id."""
        key = f"stang:cooldown:{self.camera_id}:{track_id}"
        ttl = int(self._cooldown_seconds)

        # Try Redis first
        if self.redis is not None:
            try:
                self.redis.setex(key, ttl, "1")
                return
            except Exception:
                pass

        # Fallback to in-memory
        self._memory_cooldowns[track_id] = time.time() + self._cooldown_seconds

    def _find_video(self, event_type, event_data):
        """
        Find the video clip saved by the pipeline's clip handler.

        Saves to: data/events/{camera_id}/evt_{YYYYMMDD_HHMMSS}_track{id}_{event_type}.mp4
        """
        try:
            ts = datetime.fromisoformat(event_data["timestamp"])
            ts_str = ts.strftime("%Y%m%d_%H%M%S")
            track_id = event_data.get("track_id", 0)
            cam_id = event_data.get("camera_id", self.camera_id)
            filename = f"evt_{ts_str}_track{track_id}_{event_type}.mp4"

            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

            # Camera-scoped path (new)
            path = os.path.join(project_root, "data", "events", cam_id, filename)
            if os.path.exists(path):
                return path

            # CWD-relative camera-scoped
            cwd_path = os.path.join("data", "events", cam_id, filename)
            if os.path.exists(cwd_path):
                return cwd_path

            # Legacy non-scoped path (backward compat)
            legacy = os.path.join(project_root, "data", "events", filename)
            if os.path.exists(legacy):
                return legacy

        except Exception:
            pass

        return None

    def _find_snapshot(self, event_type, event_data):
        """
        Find the snapshot image saved by the pipeline's snapshot handler.

        Track summaries: data/events/{camera_id}/summary_track{id}.jpg
        Lifecycle events: data/events/{camera_id}/evt_{ts}_track{id}_{type}.jpg
        """
        try:
            track_id = event_data.get("track_id", 0)
            cam_id = event_data.get("camera_id", self.camera_id)
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

            if event_type == "track_summary":
                filename = f"summary_track{track_id}.jpg"
            else:
                ts = datetime.fromisoformat(event_data["timestamp"])
                ts_str = ts.strftime("%Y%m%d_%H%M%S")
                filename = f"evt_{ts_str}_track{track_id}_{event_type}.jpg"

            # Camera-scoped path (new)
            path = os.path.join(project_root, "data", "events", cam_id, filename)
            if os.path.exists(path):
                return path

            # CWD-relative camera-scoped
            cwd_path = os.path.join("data", "events", cam_id, filename)
            if os.path.exists(cwd_path):
                return cwd_path

            # Legacy non-scoped path (backward compat)
            legacy = os.path.join(project_root, "data", "events", filename)
            if os.path.exists(legacy):
                return legacy

        except Exception:
            pass

        return None

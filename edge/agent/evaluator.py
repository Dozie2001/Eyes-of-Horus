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

import logging
import os
import queue
import threading
import time
from datetime import datetime

logger = logging.getLogger(__name__)

from agent.prompts import SYSTEM_PROMPT, build_user_prompt
from agent.registry import build_registry
from agent.decisions import DecisionStorage
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
                 escalation=None, camera_id="cam1", profile_storage=None,
                 frame_buffer=None, source_fps=15):
        """
        Args:
            config: StangWatchConfig
            storage: EventStorage for track history lookups
            scene_memory: SceneMemory for current scene context (or None)
            redis_client: Redis connection for cooldown tracking (or None)
            escalation: EscalationManager instance (or None for simple sends)
            camera_id: identifier for this camera
            profile_storage: CameraProfileStorage for per-camera context (or None)
            frame_buffer: deque of recent frames from the pipeline (for alert clips)
            source_fps: camera FPS (for writing clips at correct speed)
        """
        self.config = config
        self.storage = storage
        self.scene_memory = scene_memory
        self.redis = redis_client
        self.camera_id = camera_id
        self._profile_storage = profile_storage
        self._frame_buffer = frame_buffer
        self._source_fps = source_fps

        # Provider registry (replaces hardcoded if/elif chains)
        self._registry = build_registry(config)

        # Decision storage (same database)
        db_path = str(
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                config.storage.db_path,
            )
        )
        self._decisions = DecisionStorage(db_path)

        # Escalation manager (optional — if configured, routes alerts through chain)
        self._escalation = escalation

        # Work queue (maxsize prevents unbounded memory if Ollama is slow)
        self._queue = queue.Queue(maxsize=50)

        # Worker thread
        self._running = False
        self._thread = None

        # Cooldown tracking
        self._cooldown_seconds = config.agent.cooldown_seconds
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
        logger.info(f"[{self.camera_id}] EvalAgent started (model: {self.config.agent.model})")

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
            logger.info("EvalAgent stopped")

    @property
    def decision_storage(self):
        """Expose decision storage for API endpoints."""
        return self._decisions

    @property
    def registry(self):
        """Expose provider registry for API endpoints."""
        return self._registry

    def _enqueue(self, event_type, event_data):
        """Add an event to the evaluation queue (non-blocking).

        Snapshots the frame buffer at enqueue time so we can write a clip
        later if the AI decides to alert. This captures the right moment
        (not 1-3s later when inference finishes).
        """
        frames_snapshot = None
        if self._frame_buffer and len(self._frame_buffer) > 10:
            frames_snapshot = list(self._frame_buffer)

        item = (event_type, event_data, frames_snapshot)
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(item)
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

            event_type, event_data, frames_snapshot = item

            # Check cooldown — skip if we recently evaluated this track
            track_id = event_data.get("track_id", 0)
            if self._is_on_cooldown(track_id):
                continue

            # Check if any text provider is available
            if self._registry.get_text() is None:
                logger.warning(f"  AGENT: No text provider available, skipping {event_type} for Track #{track_id}")
                continue

            # Step 1: Find video clip and snapshot
            video_path = self._find_video(event_type, event_data)
            snapshot_path = self._find_snapshot(event_type, event_data)

            # Step 2: Vision perceives — describe what the camera shows
            # Prefer multi-frame video analysis over single snapshot
            visual_description = None
            if video_path:
                visual_description = self._call_with_fallback(
                    self._registry.get_vision, "describe_video", video_path,
                )
            if visual_description is None and snapshot_path:
                visual_description = self._call_with_fallback(
                    self._registry.get_vision, "describe", snapshot_path,
                )

            # Step 3: Build context (now includes visual description + cross-camera)
            scene_summary = None
            cross_camera = None
            if self.scene_memory is not None:
                scene_summary = self.scene_memory.get_scene_summary()
                cross_camera = self.scene_memory.get_all_cameras_summary()

            track_history = self.storage.get_by_track(track_id, limit=10,
                                                       camera_id=self.camera_id)

            feedback_context = self._get_feedback_context(event_type)

            camera_profile = None
            if self._profile_storage:
                camera_profile = self._profile_storage.get_profile(self.camera_id)

            user_prompt = build_user_prompt(
                event_type, event_data, scene_summary, track_history,
                visual_description=visual_description,
                cross_camera_context=cross_camera,
                feedback_context=feedback_context,
                camera_profile=camera_profile,
            )

            # Step 4: Text model reasons — decides severity
            start_ms = time.time()
            result = self._call_with_fallback(
                self._registry.get_text, "evaluate", SYSTEM_PROMPT, user_prompt,
            )
            eval_ms = int((time.time() - start_ms) * 1000)

            if result is None:
                logger.error(f"  AGENT: Evaluation failed for {event_type} Track #{track_id}")
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

                # Write alert clip from the frame buffer snapshot (if no clip already exists)
                if video_path is None and frames_snapshot:
                    video_path = self._write_alert_clip(
                        event_type, event_data, frames_snapshot,
                    )

                # Route through escalation if configured, otherwise fan-out via registry
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
                    logger.debug(f"  AGENT: ALERT {severity.upper()} | {event_type} Track #{track_id} | {status} | {eval_ms}ms")
                else:
                    # Fan-out to all available alert senders
                    senders = self._registry.get_all_alerts()
                    if senders:
                        for sender, cb in senders:
                            try:
                                sent = sender.send_alert(
                                    event_type=event_type,
                                    track_id=track_id,
                                    severity=severity,
                                    reason=reason,
                                    recommendation=recommendation,
                                    description=visual_description or "",
                                    snapshot_path=snapshot_path,
                                    video_path=video_path,
                                )
                                if sent:
                                    cb.record_success()
                                else:
                                    cb.record_failure()
                                status = "sent" if sent else "FAILED"
                                logger.debug(f"  AGENT: ALERT {severity.upper()} | {event_type} Track #{track_id} | {cb.name}: {status} | {eval_ms}ms")
                            except Exception as e:
                                cb.record_failure()
                                logger.warning(f"  AGENT: Alert sender {cb.name} failed: {e}")
                    else:
                        logger.debug(f"  AGENT: ALERT {severity.upper()} | {event_type} Track #{track_id} | No alert channel configured | {eval_ms}ms")
            else:
                logger.debug(f"  AGENT: {severity} | {event_type} Track #{track_id} | {reason[:60]} | {eval_ms}ms")

    def _call_with_fallback(self, get_fn, method_name, *args):
        """
        Call a provider method with automatic fallback via the registry.

        1. Gets best available provider from registry
        2. Calls the method
        3. Records success/failure on the circuit breaker
        4. On failure, retries with next available provider

        Args:
            get_fn: registry method (e.g. self._registry.get_text)
            method_name: method to call on the provider (e.g. "evaluate")
            *args: arguments to pass to the method

        Returns:
            Method result, or None if all providers fail
        """
        tried = set()
        while True:
            pair = get_fn(skip=tried)
            if pair is None:
                return None

            provider, cb = pair
            try:
                result = getattr(provider, method_name)(*args)
                if result is not None:
                    cb.record_success()
                    return result
                # None result counts as failure for fallback purposes
                cb.record_failure()
                tried.add(cb.name)
            except Exception as e:
                logger.warning(f"  AGENT: {cb.name}.{method_name} failed: {e}")
                cb.record_failure()
                tried.add(cb.name)

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

    def _write_alert_clip(self, event_type, event_data, frames):
        """
        Write a video clip from a frame buffer snapshot. Called only when
        the AI decides to alert and no clip already exists.

        Writes synchronously (we're already in the worker thread, not the
        pipeline thread) so the clip is ready before sending to Telegram.

        Returns:
            str: path to the written clip, or None on failure
        """
        import cv2

        try:
            ts = datetime.fromisoformat(event_data["timestamp"])
            ts_str = ts.strftime("%Y%m%d_%H%M%S")
            track_id = event_data.get("track_id", 0)
            cam_id = event_data.get("camera_id", self.camera_id)

            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            events_dir = os.path.join(project_root, "data", "events", cam_id)
            os.makedirs(events_dir, exist_ok=True)

            filename = f"alert_{ts_str}_track{track_id}_{event_type}.mp4"
            path = os.path.join(events_dir, filename)
            tmp_path = path.replace(".mp4", ".tmp.mp4")

            output_fps = min(self._source_fps, 15)
            step = max(1, round(self._source_fps / output_fps))

            h, w = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(tmp_path, fourcc, output_fps, (w, h))

            for i, frame in enumerate(frames):
                if i % step == 0:
                    writer.write(frame)

            writer.release()
            os.rename(tmp_path, path)
            logger.debug(f"  CLIP: Alert clip saved {path} ({len(frames)} frames)")
            return path

        except Exception as e:
            logger.error(f"  CLIP: Failed to write alert clip: {e}")
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.remove(tmp_path)
            return None

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

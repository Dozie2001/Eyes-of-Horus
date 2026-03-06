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
from agent.prompts import SYSTEM_PROMPT, build_user_prompt
from agent.decisions import DecisionStorage
from agent.telegram import TelegramSender
from events.tracker import (
    EVENT_APPEARED, EVENT_LOITERING, EVENT_COMPANION,
    EVENT_OBJECTS_CHANGED, EVENT_RETURNED,
)


# Events worth evaluating (high-signal)
EVAL_EVENTS = [
    EVENT_APPEARED,
    EVENT_LOITERING,
    EVENT_COMPANION,
    EVENT_OBJECTS_CHANGED,
    EVENT_RETURNED,
]


class EvalAgent:
    """
    AI agent that evaluates detection events and sends alerts.

    Architecture:
    - Event handlers (on bus) enqueue events (non-blocking, <1ms)
    - Worker thread dequeues and calls Ollama (1-3 seconds per eval)
    - Never blocks the detection pipeline
    """

    def __init__(self, config, storage, scene_memory=None, redis_client=None):
        """
        Args:
            config: StangWatchConfig
            storage: EventStorage for track history lookups
            scene_memory: SceneMemory for current scene context (or None)
            redis_client: Redis connection for cooldown tracking (or None)
        """
        self.config = config
        self.storage = storage
        self.scene_memory = scene_memory
        self.redis = redis_client

        # Ollama client
        self._ollama = OllamaClient(
            model=config.agent.model,
            host=config.agent.ollama_host,
            timeout=config.agent.timeout_seconds,
        )

        # Decision storage (same database)
        db_path = str(
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                config.storage.db_path,
            )
        )
        self._decisions = DecisionStorage(db_path)

        # Telegram sender
        self._telegram = TelegramSender(
            bot_token=config.secrets.telegram_bot_token,
            chat_id=config.secrets.telegram_chat_id,
        )

        # Work queue (maxsize prevents unbounded memory if Ollama is slow)
        self._queue = queue.Queue(maxsize=50)

        # Worker thread
        self._running = False
        self._thread = None

        # Cooldown tracking
        self._cooldown_seconds = config.agent.cooldown_seconds
        # In-memory fallback if Redis is unavailable
        self._memory_cooldowns = {}

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
            name="eval-agent",
        )
        self._thread.start()
        print(f"EvalAgent started (model: {self.config.agent.model})")

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

            # Build context
            scene_summary = None
            if self.scene_memory is not None:
                scene_summary = self.scene_memory.get_scene_summary()

            track_history = self.storage.get_by_track(track_id, limit=10)

            user_prompt = build_user_prompt(
                event_type, event_data, scene_summary, track_history,
            )

            # Evaluate with Ollama
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
            self._decisions.save_decision(
                event_type=event_type,
                track_id=track_id,
                alert=alert,
                severity=severity,
                reason=reason,
                recommendation=recommendation,
                eval_duration_ms=eval_ms,
            )

            if alert and severity in ("medium", "high"):
                # Set cooldown so we don't spam
                self._set_cooldown(track_id)

                # Find snapshot
                snapshot_path = self._find_snapshot(event_type, event_data)

                # Send Telegram alert
                if self._telegram.is_configured():
                    sent = self._telegram.send_alert(
                        event_type=event_type,
                        track_id=track_id,
                        severity=severity,
                        reason=reason,
                        recommendation=recommendation,
                        snapshot_path=snapshot_path,
                    )
                    status = "sent" if sent else "FAILED"
                    print(f"  AGENT: ALERT {severity.upper()} | {event_type} Track #{track_id} | Telegram: {status} | {eval_ms}ms")
                else:
                    print(f"  AGENT: ALERT {severity.upper()} | {event_type} Track #{track_id} | Telegram not configured | {eval_ms}ms")
            else:
                print(f"  AGENT: {severity} | {event_type} Track #{track_id} | {reason[:60]} | {eval_ms}ms")

    def _is_on_cooldown(self, track_id):
        """Check if this track_id is on alert cooldown."""
        key = f"stang:cooldown:{track_id}"

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
        key = f"stang:cooldown:{track_id}"
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

    def _find_snapshot(self, event_type, event_data):
        """
        Find the snapshot image saved by the pipeline's snapshot handler.

        The snapshot handler saves to:
            data/events/evt_{YYYYMMDD_HHMMSS}_track{id}_{event_type}.jpg
        """
        try:
            ts = datetime.fromisoformat(event_data["timestamp"])
            ts_str = ts.strftime("%Y%m%d_%H%M%S")
            track_id = event_data.get("track_id", 0)
            filename = f"evt_{ts_str}_track{track_id}_{event_type}.jpg"

            # Try project-root-relative path (same as snapshot handler)
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            path = os.path.join(project_root, "data", "events", filename)

            if os.path.exists(path):
                return path

            # Also try CWD-relative (if server runs from backend/)
            cwd_path = os.path.join("data", "events", filename)
            if os.path.exists(cwd_path):
                return cwd_path

        except Exception:
            pass

        return None

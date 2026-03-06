"""
Redis-backed event bus for StangWatch.

Drop-in replacement for pyee EventEmitter. Same emit()/on() interface.
Publishes events to Redis Streams (persistent, crash-safe) AND fires
local pyee handlers (backward compatible).

This means:
- EventTracker doesn't change (still calls bus.emit())
- EventStorage doesn't change (still calls bus.on())
- Events survive process crashes (Redis Streams are persistent)
- Multiple processes can consume events (agent, Telegram, dashboard)

Usage:
    from events.redis_bus import RedisBus
    from redis_client import get_redis

    bus = RedisBus(get_redis(), camera_id="front_gate")
    bus.on("appeared", lambda data: print(data))  # local handler
    bus.emit("appeared", {"track_id": 1, ...})     # writes to Redis + fires handler
"""

import json
from pyee.base import EventEmitter


class RedisBus:
    """
    Dual event bus: Redis Streams + local pyee handlers.

    Redis Stream key: stang:events:{camera_id}
    Each entry has: type, data (JSON), camera_id, ts
    """

    def __init__(self, redis_client, camera_id="cam1",
                 stream_prefix="stang:events"):
        self.redis = redis_client
        self.camera_id = camera_id
        self.stream = f"{stream_prefix}:{camera_id}"
        self._local = EventEmitter()

        # Cap stream at 10,000 entries to prevent unbounded growth
        self._maxlen = 10000

    def emit(self, event_type, data):
        """
        Publish an event to Redis Stream + fire local handlers.

        Args:
            event_type: e.g. "appeared", "loitering"
            data: dict from EventTracker._event_data()
        """
        # 1. Write to Redis Stream (persistent)
        try:
            self.redis.xadd(
                self.stream,
                {
                    "type": event_type,
                    "data": json.dumps(data, default=str),
                    "camera_id": self.camera_id,
                },
                maxlen=self._maxlen,
            )
        except Exception as e:
            # Redis down? Log but don't crash the pipeline
            print(f"Redis publish failed: {e}")

        # 2. Fire local pyee handlers (backward compatible)
        self._local.emit(event_type, data)

    def on(self, event_type, handler):
        """
        Register a local handler. Same interface as pyee.

        These handlers fire immediately in the emitting thread.
        For cross-process consumers, use Redis consumer groups instead.
        """
        self._local.on(event_type, handler)

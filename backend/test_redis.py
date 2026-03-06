"""
Test Redis integration for StangWatch.

Verifies:
1. Redis connection works
2. RedisBus publishes events to Redis Streams
3. SceneMemory writes and reads scene state
4. Events flow through the full path (emit → stream → local handler)

Usage:
    cd backend
    python test_redis.py
"""

import json
import sys
import time

from config import get_config
from redis_client import get_redis, reset_redis
from events.redis_bus import RedisBus
from agent.memory import SceneMemory


def _connect():
    """Get Redis connection using config (.env credentials)."""
    cfg = get_config()
    return get_redis(
        host=cfg.redis.host,
        port=cfg.redis.port,
        db=cfg.redis.db,
        username=cfg.redis.username,
        password=cfg.redis.password,
    )


def test_connection():
    """Test basic Redis connectivity."""
    print("1. Testing Redis connection...")
    r = _connect()
    assert r.ping(), "Redis PING failed"
    print("   PONG — connected!")


def test_redis_bus():
    """Test RedisBus publishes to Redis Stream + fires local handler."""
    print("\n2. Testing RedisBus...")
    r = _connect()
    bus = RedisBus(r, camera_id="test_cam")

    # Clean up any previous test data
    r.delete(bus.stream)

    # Track local handler calls
    received = []
    bus.on("appeared", lambda data: received.append(data))

    # Emit an event
    test_data = {
        "track_id": 42,
        "timestamp": "2025-01-15T10:30:00",
        "bbox": [100, 200, 300, 400],
        "confidence": 0.95,
    }
    bus.emit("appeared", test_data)

    # Verify local handler fired
    assert len(received) == 1, f"Expected 1 local handler call, got {len(received)}"
    assert received[0]["track_id"] == 42
    print("   Local handler fired correctly")

    # Verify Redis Stream has the event
    entries = r.xrange(bus.stream)
    assert len(entries) >= 1, "No entries in Redis Stream"
    last_entry = entries[-1]
    entry_data = last_entry[1]
    assert entry_data["type"] == "appeared"
    assert entry_data["camera_id"] == "test_cam"

    parsed = json.loads(entry_data["data"])
    assert parsed["track_id"] == 42
    print(f"   Redis Stream has {len(entries)} entry(ies)")
    print(f"   Stream key: {bus.stream}")

    # Clean up
    r.delete(bus.stream)
    print("   RedisBus works!")


def test_scene_memory():
    """Test SceneMemory write and read."""
    print("\n3. Testing SceneMemory...")
    r = _connect()
    memory = SceneMemory(r, camera_id="test_cam", ttl_seconds=10)

    # Mock detections (what YOLO + ByteTrack returns)
    detections = [
        {
            "track_id": 1,
            "bbox": [100, 150, 200, 400],
            "confidence": 0.92,
            "nearby_objects": ["backpack"],
        },
        {
            "track_id": 2,
            "bbox": [400, 100, 500, 350],
            "confidence": 0.87,
            "nearby_objects": [],
        },
    ]

    # Mock tracker with tracks
    class MockTrack:
        def __init__(self, state, duration, movement):
            self.state = state
            self._duration = duration
            self._movement = movement
        def duration_seconds(self):
            return self._duration
        def recent_movement(self):
            return self._movement

    class MockTracker:
        def __init__(self):
            self.tracks = {
                1: MockTrack("loitering", 45.2, 2.1),
                2: MockTrack("moving", 8.5, 15.3),
            }
            self.departed_tracks = {}

    tracker = MockTracker()
    memory.update_scene(detections, tracker)

    # Read it back
    summary = memory.get_scene_summary()

    assert summary["camera_id"] == "test_cam"
    assert summary["people_count"] == 2, f"Expected 2 people, got {summary['people_count']}"
    assert len(summary["people"]) == 2
    assert len(summary["objects"]) == 1  # one backpack

    # Check person details
    person_1 = next(p for p in summary["people"] if p["track_id"] == 1)
    assert person_1["state"] == "loitering"
    assert person_1["duration"] == 45.2

    print(f"   People: {summary['people_count']}")
    print(f"   Objects: {summary['objects']}")
    print(f"   Stats: {summary['stats']}")

    # Clean up
    r.delete(f"stang:scene:test_cam:people")
    r.delete(f"stang:scene:test_cam:stats")
    r.delete(f"stang:scene:test_cam:objects")
    print("   SceneMemory works!")


def test_multiple_events():
    """Test multiple event types through RedisBus."""
    print("\n4. Testing multiple event types...")
    r = _connect()
    bus = RedisBus(r, camera_id="test_cam")
    r.delete(bus.stream)

    events_fired = []

    bus.on("appeared", lambda d: events_fired.append(("appeared", d)))
    bus.on("loitering", lambda d: events_fired.append(("loitering", d)))
    bus.on("departed", lambda d: events_fired.append(("departed", d)))

    bus.emit("appeared", {"track_id": 1, "timestamp": "2025-01-15T10:30:00"})
    bus.emit("loitering", {"track_id": 1, "timestamp": "2025-01-15T10:35:00", "duration": 300})
    bus.emit("departed", {"track_id": 1, "timestamp": "2025-01-15T10:40:00"})

    assert len(events_fired) == 3
    stream_len = r.xlen(bus.stream)
    assert stream_len == 3, f"Expected 3 stream entries, got {stream_len}"

    print(f"   {len(events_fired)} events fired locally")
    print(f"   {stream_len} entries in Redis Stream")

    # Clean up
    r.delete(bus.stream)
    print("   Multiple events work!")


if __name__ == "__main__":
    print("=" * 50)
    print("StangWatch Redis Integration Tests")
    print("=" * 50)

    try:
        test_connection()
        test_redis_bus()
        test_scene_memory()
        test_multiple_events()

        print("\n" + "=" * 50)
        print("ALL TESTS PASSED")
        print("=" * 50)

    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

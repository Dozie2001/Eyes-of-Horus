"""
Test script for EventStorage.
Tests database creation, event saving, querying, and bus integration.
No camera or YOLO needed — pure data operations.

Run: python test_storage.py (from backend/)
"""

import os
import sys
import tempfile
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from pyee.base import EventEmitter
from events.storage import EventStorage


def test_storage():
    # Use a temp file so tests don't pollute real data
    db_path = os.path.join(tempfile.gettempdir(), "test_stangwatch.db")

    # Clean up from previous runs (SQLite creates -wal and -shm files too)
    for ext in ["", "-wal", "-shm"]:
        path = db_path + ext
        if os.path.exists(path):
            os.remove(path)

    print("=== StangWatch Storage Test ===\n")

    storage = EventStorage(db_path)
    print(f"Database created: {db_path}")

    # --- Test 1: Save an event ---
    fake_event = {
        "track_id": 1,
        "timestamp": datetime.now().isoformat(),
        "first_seen": datetime.now().isoformat(),
        "duration_seconds": 5.2,
        "bbox": [100.0, 200.0, 300.0, 400.0],
        "movement": "stationary",
        "is_quiet_hours": False,
        "nearby_objects": ["backpack"],
    }
    row_id = storage.save_event("appeared", fake_event, snapshot_path="data/events/test.jpg")
    assert row_id == 1, f"Expected row_id=1, got {row_id}"
    print("PASS: save_event (appeared)")

    # --- Test 2: Save event with extra fields (companion) ---
    companion_event = dict(fake_event)
    companion_event["track_id"] = 2
    companion_event["near_track_id"] = 1
    storage.save_event("companion", companion_event)
    print("PASS: save_event with extra fields (companion)")

    # --- Test 3: Save another event for same track (should update track, not create new) ---
    loiter_event = dict(fake_event)
    loiter_event["track_id"] = 1  # same person
    loiter_event["duration_seconds"] = 305.0
    storage.save_event("loitering", loiter_event)
    print("PASS: save_event for existing track (loitering)")

    # --- Test 4: Query recent ---
    recent = storage.get_recent(limit=10)
    assert len(recent) == 3, f"Expected 3 events, got {len(recent)}"
    assert recent[0]["event_type"] == "loitering"  # newest first
    print("PASS: get_recent")

    # --- Test 5: Extra fields preserved ---
    companion_events = storage.get_by_type("companion")
    assert len(companion_events) == 1
    assert companion_events[0]["extra"]["near_track_id"] == 1
    print("PASS: extra fields preserved (near_track_id)")

    # --- Test 6: Query by type ---
    appeared = storage.get_by_type("appeared")
    assert len(appeared) == 1
    assert appeared[0]["snapshot_path"] == "data/events/test.jpg"
    print("PASS: get_by_type + snapshot_path")

    # --- Test 7: Query by track ---
    track1_events = storage.get_by_track(bytetrack_id=1)
    assert len(track1_events) == 2  # appeared + loitering for track 1
    print("PASS: get_by_track")

    # --- Test 8: Count by type ---
    counts = storage.count_by_type()
    assert counts["appeared"] == 1
    assert counts["companion"] == 1
    assert counts["loitering"] == 1
    print("PASS: count_by_type")

    # --- Test 9: Bus integration ---
    bus = EventEmitter()
    storage.subscribe(bus)

    # Emit an event through the bus — should auto-save
    bus.emit("departed", {
        "track_id": 3,
        "timestamp": datetime.now().isoformat(),
        "first_seen": datetime.now().isoformat(),
        "duration_seconds": 12.0,
        "bbox": [50.0, 50.0, 200.0, 400.0],
        "movement": "moving",
        "is_quiet_hours": True,
        "nearby_objects": [],
    })

    departed = storage.get_by_type("departed")
    assert len(departed) == 1
    assert departed[0]["is_quiet_hours"] is True
    print("PASS: bus integration (event auto-saved)")

    # --- Test 10: JSON fields round-trip ---
    event = storage.get_recent(limit=1)[0]
    assert isinstance(event["bbox"], list), "bbox should be a list"
    assert isinstance(event["nearby_objects"], list), "nearby_objects should be a list"
    assert isinstance(event["extra"], dict), "extra should be a dict"
    assert isinstance(event["is_quiet_hours"], bool), "is_quiet_hours should be bool"
    print("PASS: JSON round-trip (all types correct)")

    # --- Cleanup ---
    for ext in ["", "-wal", "-shm"]:
        path = db_path + ext
        if os.path.exists(path):
            os.remove(path)

    print(f"\n=== All 10 tests passed ===")


if __name__ == "__main__":
    test_storage()

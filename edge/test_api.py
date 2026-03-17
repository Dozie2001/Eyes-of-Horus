"""
Tests for the StangWatch FastAPI backend.

Uses TestClient — no real server needed, no YOLO, no camera.
Creates a temp database with seed data for testing.

Run: cd backend && python -m pytest test_api.py -v
"""

import os
import tempfile
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from config import get_config, reset_config
from events.storage import EventStorage


# --- Fixtures ---

@pytest.fixture(scope="module")
def seeded_storage():
    """Create a temp database with seed events for testing."""
    db_path = os.path.join(tempfile.gettempdir(), "test_api_stangwatch.db")
    # Clean up from previous runs
    for ext in ["", "-wal", "-shm"]:
        p = db_path + ext
        if os.path.exists(p):
            os.remove(p)

    storage = EventStorage(db_path)

    # Seed: 2 events for track 1, 1 event for track 2
    base_event = {
        "track_id": 1,
        "timestamp": datetime(2026, 3, 3, 14, 0, 0).isoformat(),
        "first_seen": datetime(2026, 3, 3, 13, 55, 0).isoformat(),
        "duration_seconds": 5.0,
        "bbox": [100.0, 200.0, 300.0, 400.0],
        "movement": "stationary",
        "is_quiet_hours": False,
        "nearby_objects": [],
    }
    storage.save_event("appeared", base_event)

    loiter = dict(base_event)
    loiter["duration_seconds"] = 305.0
    loiter["timestamp"] = datetime(2026, 3, 3, 14, 5, 0).isoformat()
    storage.save_event("loitering", loiter, snapshot_path="data/events/test_snap.jpg")

    departed = dict(base_event)
    departed["track_id"] = 2
    departed["timestamp"] = datetime(2026, 3, 3, 14, 10, 0).isoformat()
    departed["is_quiet_hours"] = True
    storage.save_event("departed", departed)

    yield storage, db_path

    # Cleanup
    for ext in ["", "-wal", "-shm"]:
        p = db_path + ext
        if os.path.exists(p):
            os.remove(p)


@pytest.fixture(scope="module")
def client(seeded_storage):
    """Create a TestClient with the seeded storage injected."""
    storage, db_path = seeded_storage

    from main import app

    # TestClient triggers lifespan which creates its own storage.
    # We override AFTER entering the context so our seeded DB is used.
    reset_config()
    with TestClient(app) as c:
        app.state.storage = storage
        app.state.config = get_config()
        yield c

    reset_config()


# --- Tests ---

class TestHealth:
    def test_returns_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert "site_name" in data
        assert "site_id" in data

    def test_event_count_matches_seed(self, client):
        r = client.get("/health")
        assert r.json()["event_count"] == 3


class TestGetEvents:
    def test_returns_all_seeded_events(self, client):
        events = client.get("/events").json()
        assert len(events) == 3

    def test_newest_first(self, client):
        events = client.get("/events").json()
        # departed at 14:10 is newest, appeared at 14:00 is oldest
        assert events[0]["event_type"] == "departed"
        assert events[-1]["event_type"] == "appeared"

    def test_limit_parameter(self, client):
        events = client.get("/events?limit=1").json()
        assert len(events) == 1

    def test_limit_rejects_zero(self, client):
        r = client.get("/events?limit=0")
        assert r.status_code == 422

    def test_limit_rejects_negative(self, client):
        r = client.get("/events?limit=-1")
        assert r.status_code == 422

    def test_limit_rejects_too_large(self, client):
        r = client.get("/events?limit=1000")
        assert r.status_code == 422


class TestGetEventsByType:
    def test_filter_by_type(self, client):
        events = client.get("/events/type/loitering").json()
        assert len(events) == 1
        assert events[0]["event_type"] == "loitering"

    def test_empty_result_for_unused_type(self, client):
        events = client.get("/events/type/returned").json()
        assert events == []

    def test_invalid_type_returns_400(self, client):
        r = client.get("/events/type/explosion")
        assert r.status_code == 400
        assert "Invalid event_type" in r.json()["detail"]


class TestGetEventsByTrack:
    def test_filter_by_track(self, client):
        events = client.get("/events/track/1").json()
        assert len(events) == 2  # appeared + loitering

    def test_unknown_track_returns_empty(self, client):
        events = client.get("/events/track/999").json()
        assert events == []


class TestEventsSummary:
    def test_structure(self, client):
        r = client.get("/events/summary")
        assert r.status_code == 200
        data = r.json()
        assert "counts" in data
        assert "total" in data
        assert data["total"] == 3

    def test_counts_match_seed(self, client):
        counts = client.get("/events/summary").json()["counts"]
        assert counts["appeared"] == 1
        assert counts["loitering"] == 1
        assert counts["departed"] == 1


class TestSnapshots:
    def test_missing_file_returns_404(self, client):
        r = client.get("/snapshots/nonexistent.jpg")
        assert r.status_code == 404

    def test_path_traversal_blocked(self, client):
        r = client.get("/snapshots/../../../etc/passwd")
        assert r.status_code != 200


class TestEventDictShape:
    def test_has_expected_fields(self, client):
        events = client.get("/events?limit=1").json()
        event = events[0]
        expected_keys = {
            "id", "event_type", "track_id", "timestamp",
            "duration_seconds", "bbox", "movement", "is_quiet_hours",
            "nearby_objects", "snapshot_path", "extra", "created_at",
        }
        assert set(event.keys()) == expected_keys

    def test_snapshot_path_preserved(self, client):
        events = client.get("/events/type/loitering").json()
        assert events[0]["snapshot_path"] == "data/events/test_snap.jpg"

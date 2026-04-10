"""
Unit tests for the zone checking system.

Run from the edge/ directory:
    python -m detection.test_zones
"""

import sys


def test_point_in_polygon_basic():
    from detection.zones import point_in_polygon

    square = [[0, 0], [100, 0], [100, 100], [0, 100]]
    assert point_in_polygon((50, 50), square) is True
    assert point_in_polygon((0, 0), square) is True        # corner
    assert point_in_polygon((150, 50), square) is False
    assert point_in_polygon((-10, 50), square) is False
    assert point_in_polygon((50, -1), square) is False
    print("test_point_in_polygon_basic OK")


def test_point_in_polygon_invalid_inputs():
    from detection.zones import point_in_polygon

    assert point_in_polygon((50, 50), []) is False
    assert point_in_polygon((50, 50), [[0, 0]]) is False
    assert point_in_polygon((50, 50), [[0, 0], [1, 1]]) is False
    assert point_in_polygon((50, 50), "not a polygon") is False
    print("test_point_in_polygon_invalid_inputs OK")


def test_zone_checker_polygon_with_rules():
    from detection.zones import ZoneChecker

    zones = [
        {
            "name": "restricted",
            "zone_type": "polygon",
            "points": [[0, 0], [100, 0], [100, 100], [0, 100]],
            "severity_override": "high",
            "active_hours_start": "21:00",
            "active_hours_end": "06:00",
            "allowed_object_types": ["person"],
            "alert_on_entry": True,
            "alert_on_dwell_seconds": 30.0,
        }
    ]
    checker = ZoneChecker(zones)

    assert checker.has_zones() is True
    assert checker.zone_names() == ["restricted"]

    hits = checker.check_point((50, 50))
    assert len(hits) == 1
    hit = hits[0]
    assert hit["name"] == "restricted"
    assert hit["zone_type"] == "polygon"
    assert hit["severity_override"] == "high"
    assert hit["active_hours_start"] == "21:00"
    assert hit["active_hours_end"] == "06:00"
    assert hit["allowed_object_types"] == ["person"]
    assert hit["alert_on_entry"] is True
    assert hit["alert_on_dwell_seconds"] == 30.0

    assert checker.check_point((200, 200)) == []
    print("test_zone_checker_polygon_with_rules OK")


def test_zone_checker_bbox_api():
    from detection.zones import ZoneChecker

    zones = [{"name": "g", "points": [[0, 0], [100, 0], [100, 100], [0, 100]]}]
    checker = ZoneChecker(zones)

    # bbox = [x1, y1, x2, y2] with centroid (50, 50)
    assert len(checker.check_bbox([10, 10, 90, 90])) == 1
    assert len(checker.check_bbox([150, 150, 200, 200])) == 0
    print("test_zone_checker_bbox_api OK")


def test_zone_checker_skips_invalid_zones():
    from detection.zones import ZoneChecker

    zones = [
        {"name": "too_few_points", "points": [[0, 0], [1, 1]]},
        {"name": "no_points", "points": []},
        {"name": "line_too_few", "zone_type": "line", "points": [[0, 0]]},
        {"name": "valid", "points": [[0, 0], [100, 0], [100, 100], [0, 100]]},
    ]
    checker = ZoneChecker(zones)
    assert checker.zone_names() == ["valid"]
    print("test_zone_checker_skips_invalid_zones OK")


def test_zone_checker_empty():
    from detection.zones import ZoneChecker

    checker = ZoneChecker([])
    assert checker.has_zones() is False
    assert checker.zone_names() == []
    assert checker.check_point((50, 50)) == []
    assert checker.check_bbox([0, 0, 100, 100]) == []
    assert checker.update(None) == []
    print("test_zone_checker_empty OK")


def test_zone_checker_line_zone_registration():
    from detection.zones import ZoneChecker

    zones = [
        {"name": "perimeter", "zone_type": "line", "points": [[0, 50], [200, 50]]},
        {"name": "gate", "zone_type": "polygon",
         "points": [[0, 0], [100, 0], [100, 100], [0, 100]]},
    ]
    checker = ZoneChecker(zones)
    names = checker.zone_names()
    assert "perimeter" in names
    assert "gate" in names

    # Polygon check should still work
    assert len(checker.check_point((50, 50))) == 1
    print("test_zone_checker_line_zone_registration OK")


def test_zone_checker_line_crossing_via_supervision():
    """End-to-end: feed synthetic sv.Detections to a line zone and verify
    crossings are reported."""
    import numpy as np
    import supervision as sv

    from detection.zones import ZoneChecker

    checker = ZoneChecker(
        [{"name": "perimeter", "zone_type": "line", "points": [[0, 100], [200, 100]]}]
    )

    # Frame 1: detection above the line (y center = 50)
    d1 = sv.Detections(
        xyxy=np.array([[80, 30, 120, 70]], dtype=np.float32),
        confidence=np.array([0.9], dtype=np.float32),
        class_id=np.array([0], dtype=np.int32),
        tracker_id=np.array([1], dtype=np.int32),
    )
    checker.update(d1)

    # Frame 2: same track moved below the line (y center = 150)
    d2 = sv.Detections(
        xyxy=np.array([[80, 130, 120, 170]], dtype=np.float32),
        confidence=np.array([0.9], dtype=np.float32),
        class_id=np.array([0], dtype=np.int32),
        tracker_id=np.array([1], dtype=np.int32),
    )
    events = checker.update(d2)

    assert isinstance(events, list)
    crossed = any(e["name"] == "perimeter" for e in events)
    assert crossed, f"expected perimeter crossing event, got {events}"
    print("test_zone_checker_line_crossing_via_supervision OK")


def test_event_tracker_zone_integration():
    """EventTracker should include rich zone dicts in emitted events."""
    from datetime import datetime
    from events.tracker import EventTracker
    from pyee.base import EventEmitter

    bus = EventEmitter()
    zones = [
        {
            "name": "gate",
            "points": [[0, 0], [1000, 0], [1000, 1000], [0, 1000]],
            "severity_override": "high",
        }
    ]
    tracker = EventTracker(event_bus=bus, zones=zones, camera_id="testcam")

    received: list[dict] = []
    bus.on("appeared", lambda data: received.append(data))

    detections = [{"track_id": 1, "bbox": [100, 100, 300, 500], "nearby_objects": []}]
    tracker.update(detections, datetime.now())

    assert len(received) == 1
    data = received[0]
    assert data["track_id"] == 1
    assert len(data["zones"]) == 1
    assert data["zones"][0]["name"] == "gate"
    assert data["zones"][0]["severity_override"] == "high"
    print("test_event_tracker_zone_integration OK")


if __name__ == "__main__":
    tests = [
        test_point_in_polygon_basic,
        test_point_in_polygon_invalid_inputs,
        test_zone_checker_polygon_with_rules,
        test_zone_checker_bbox_api,
        test_zone_checker_skips_invalid_zones,
        test_zone_checker_empty,
        test_zone_checker_line_zone_registration,
        test_zone_checker_line_crossing_via_supervision,
        test_event_tracker_zone_integration,
    ]

    failed = 0
    for test in tests:
        try:
            test()
        except AssertionError as e:
            print(f"FAIL: {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR: {test.__name__}: {type(e).__name__}: {e}")
            failed += 1

    if failed:
        print(f"\n{failed} test(s) failed.")
        sys.exit(1)
    print(f"\nAll {len(tests)} zone tests passed.")

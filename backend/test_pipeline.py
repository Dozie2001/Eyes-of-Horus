"""
Integration test for the full detection + tracking + event pipeline.

Runs for 15 seconds on the webcam:
  Camera → Detector (YOLO + ByteTrack) → EventTracker → Event Bus

Outputs:
  - data/pipeline_test.mp4  — video with bounding boxes and track IDs
  - data/events/evt_*.jpg   — snapshot saved for each event
  - Terminal prints of every event as it happens

Stand in front of the camera to trigger events:
  - "appeared" fires when you enter the frame
  - "loitering" fires if you stand still for 10 seconds (lowered for testing)
  - "moving" fires when you start moving after standing still
  - "departed" fires when you leave the frame (~2 seconds after leaving)
"""

import sys
import os
import time
import cv2
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from capture.camera import Camera
from detection.detector import Detector
from events.bus import event_bus
from events.tracker import EventTracker
from utils import draw_boxes, filter_overlapping


# Global reference to current frame so event handlers can save snapshots
_current_frame = None
_event_count = 0


def save_event_snapshot(event_type, data):
    """Save a snapshot whenever an event fires."""
    global _current_frame, _event_count

    if _current_frame is None:
        return

    _event_count += 1
    os.makedirs("data/events", exist_ok=True)

    # Draw boxes on the snapshot
    bbox = data.get("bbox")
    if bbox:
        detections = [{"bbox": bbox, "confidence": 1.0, "label": f"#{data.get('track_id', '?')}"}]
        frame = draw_boxes(_current_frame, detections)
    else:
        frame = _current_frame.copy()

    # Add event label at the top of the image
    label = f"{event_type.upper()} | Track #{data.get('track_id', '?')} | {data.get('duration_seconds', 0):.1f}s"
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    path = f"data/events/evt_{_event_count:03d}_{event_type}.jpg"
    cv2.imwrite(path, frame)
    print(f"    Snapshot: {path}")


def main():
    print("=== StangWatch Pipeline Test (with video + snapshots) ===")
    print("This test runs for 15 seconds.")
    print("Stand in front of the camera, stand still, then move or leave.")
    print()

    # --- Subscribe to ALL events: print + save snapshot ---
    def on_event(event_type):
        def handler(data):
            track_id = data.get("track_id", "?")
            duration = data.get("duration_seconds", 0)
            movement = data.get("movement", "?")
            quiet = data.get("is_quiet_hours", False)

            print(f"\n  EVENT: {event_type.upper()}")
            print(f"    Track ID:    {track_id}")
            print(f"    Time:        {data.get('timestamp', '?')}")
            print(f"    Duration:    {duration}s")
            print(f"    Movement:    {movement}")
            print(f"    Quiet hours: {quiet}")

            if "near_track_id" in data:
                print(f"    Near person: Track #{data['near_track_id']}")
            if "objects_after" in data:
                print(f"    Objects now:  {data['objects_after']}")
                print(f"    Objects were: {data['objects_before']}")

            # Save snapshot for this event
            save_event_snapshot(event_type, data)

        return handler

    event_types = [
        "appeared", "loitering", "moving",
        "companion", "departed", "objects_changed", "returned",
    ]
    for evt in event_types:
        event_bus.on(evt, on_event(evt))

    # --- Set up components ---
    camera = Camera(source=0)
    if not camera.connect():
        return

    detector = Detector()  # confidence stays at 0.5 (default)
    detector.load()

    tracker = EventTracker(
        event_bus=event_bus,
        loiter_threshold=10,       # 10 seconds for testing
        quiet_hours=None,
        stationary_threshold=5.0,
        departure_seconds=3.0,     # 3 actual seconds — works at any FPS
    )

    camera.warm_up()

    # --- Set up video writer ---
    os.makedirs("data", exist_ok=True)
    video_path = "data/pipeline_test.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_path, fourcc, 15.0, (camera.width, camera.height))
    # Writing at 15fps (half of capture rate) to keep file size reasonable

    print("Starting detection loop...")
    print(f"Video saving to: {video_path}")
    print("(Events will appear below as they happen)\n")

    # --- Main loop ---
    global _current_frame
    start_time = time.time()
    frame_count = 0
    write_every = 2  # write every 2nd frame to get ~15fps output

    while time.time() - start_time < 15:
        frame = camera.read_frame()
        if frame is None:
            continue

        timestamp = datetime.now()
        _current_frame = frame  # make available to event handlers

        # Detect + track
        detections = detector.track_people(frame)

        # Fix 1: Remove duplicate detections (same person detected twice)
        detections = filter_overlapping(detections)

        # Feed to event tracker (emits events via bus)
        tracker.update(detections, timestamp)

        # Draw boxes on frame for video
        annotated = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            tid = det.get("track_id", "?")
            conf = det["confidence"]

            # Green box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Label with track ID and confidence
            label = f"ID:{tid} {conf}"
            cv2.putText(annotated, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Timestamp overlay
        time_str = timestamp.strftime("%H:%M:%S")
        cv2.putText(annotated, time_str, (10, camera.height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Write to video (every 2nd frame)
        if frame_count % write_every == 0:
            out.write(annotated)

        frame_count += 1

    # --- Cleanup ---
    out.release()
    camera.release()

    elapsed = time.time() - start_time
    print(f"\n\n=== Test Complete ===")
    print(f"Processed {frame_count} frames in {elapsed:.1f} seconds")
    print(f"Average FPS: {frame_count / elapsed:.1f}")
    print(f"Active tracks: {len(tracker.tracks)}")
    print(f"Departed tracks: {len(tracker.departed_tracks)}")
    print(f"Events fired: {_event_count}")
    print(f"Video saved: {video_path}")
    print(f"Snapshots saved: data/events/")
    print("Done.")


if __name__ == "__main__":
    main()

"""
Utility functions for StangWatch.
These are stateless operations — no classes needed, just input in, output out.
"""

import cv2
import os


def save_snapshot(frame, path="data/snapshot.jpg"):
    """
    Save a frame as a JPEG image file.

    Args:
        frame: numpy array from Camera.read_frame()
        path: where to save the file
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, frame)
    print(f"Snapshot saved to: {path}")


def draw_boxes(frame, detections):
    """
    Draw bounding boxes on a frame for each detection.
    Returns a new frame with boxes drawn (does not modify the original).

    Args:
        frame: numpy array from Camera.read_frame()
        detections: list of dicts from Detector.detect_people()

    Returns:
        new frame with green boxes and labels drawn on it
    """
    # Make a copy so we don't modify the original frame
    annotated = frame.copy()

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        conf = det["confidence"]
        label = f"{det['label']} {conf}"

        # Green box around the person
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Label above the box
        cv2.putText(
            annotated, label, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
        )

    return annotated

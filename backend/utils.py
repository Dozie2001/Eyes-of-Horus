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


def filter_overlapping(detections, iou_threshold=0.5):
    """
    Remove duplicate detections of the same person.
    If two bounding boxes overlap more than iou_threshold,
    keep only the one with higher confidence.

    Args:
        detections: list of detection dicts (each has "bbox" and "confidence")
        iou_threshold: overlap ratio above which two boxes are considered duplicates.
            0.5 = boxes overlap by 50% or more → same person.

    Returns:
        filtered list with duplicates removed
    """
    if len(detections) <= 1:
        return detections

    # Sort by confidence (highest first) — we keep the most confident detection
    sorted_dets = sorted(detections, key=lambda d: d["confidence"], reverse=True)
    keep = []

    for det in sorted_dets:
        # Check if this box overlaps too much with any box we're already keeping
        is_duplicate = False
        for kept in keep:
            if _iou(det["bbox"], kept["bbox"]) > iou_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            keep.append(det)

    return keep


def _iou(box1, box2):
    """
    Calculate Intersection over Union between two [x1, y1, x2, y2] boxes.
    Returns 0.0 (no overlap) to 1.0 (identical boxes).
    """
    # Find the overlap rectangle
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Area of overlap
    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # Area of each box
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Union = both areas minus the overlap (so we don't count it twice)
    union = area1 + area2 - intersection

    if union == 0:
        return 0.0
    return intersection / union

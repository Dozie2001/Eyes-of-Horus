"""
Object detection via Roboflow Inference Server HTTP API.

Replaces the in-process ultralytics Detector. Sends frames as base64
to the Roboflow Docker container and parses detection results.

Tracking is done client-side via supervision's ByteTrack (Roboflow
has no server-side tracking state).

Usage:
    detector = RoboflowDetector(server_url="http://localhost:9001")
    detector.load()  # verifies server is reachable
    detections = detector.track_people(frame)
    # [{"bbox": [x1,y1,x2,y2], "confidence": 0.87, "label": "person",
    #   "track_id": 1, "nearby_objects": ["backpack"]}]
"""

import base64
import logging
import math

import cv2
import httpx
import numpy as np
import supervision as sv

logger = logging.getLogger(__name__)

# COCO class names we care about (Roboflow returns class names, not IDs)
PERSON_CLASS = "person"
OBJECT_CLASSES = {"backpack", "handbag", "suitcase", "laptop", "cell phone"}


class RoboflowDetector:
    """
    HTTP client to Roboflow Inference Server for object detection.

    Same public interface as the original Detector class so PipelineRunner
    can swap them without changes.
    """

    def __init__(
        self,
        server_url: str = "http://localhost:9001",
        model_id: str = "yolo26s-640",
        confidence_threshold: float = 0.5,
        object_association_distance: float = 200.0,
        api_key: str = "",
    ):
        self.server_url = server_url.rstrip("/")
        self.model_id = model_id
        self.confidence_threshold = confidence_threshold
        self.object_association_distance = object_association_distance
        self.api_key = api_key

        self._client = None
        self._tracker = None

    def load(self):
        """Verify server is reachable and initialize tracker."""
        self._client = httpx.Client(timeout=30.0)
        self._tracker = sv.ByteTrack()

        # Health check
        try:
            resp = self._client.get(f"{self.server_url}/")
            resp.raise_for_status()
            logger.info(
                f"RoboflowDetector connected to {self.server_url} "
                f"(model: {self.model_id})"
            )
        except Exception as e:
            raise ConnectionError(
                f"Cannot reach Roboflow Inference Server at {self.server_url}: {e}"
            )

    def detect_people(self, frame: np.ndarray) -> list[dict]:
        """One-shot person detection. No tracking, no object association."""
        predictions = self._infer(frame)

        return [
            {
                "bbox": self._to_xyxy(p),
                "confidence": round(p["confidence"], 2),
                "label": "person",
            }
            for p in predictions
            if p["class"] == PERSON_CLASS
        ]

    def track_people(self, frame: np.ndarray) -> list[dict]:
        """
        Detection + ByteTrack tracking + object association.

        Same return format as the original Detector.track_people().
        """
        predictions = self._infer(frame)

        if not predictions:
            # Still update tracker with empty detections to age out old tracks
            empty = sv.Detections.empty()
            self._tracker.update_with_detections(empty)
            return []

        # Separate people and objects
        people_preds = [p for p in predictions if p["class"] == PERSON_CLASS]
        object_preds = [p for p in predictions if p["class"] in OBJECT_CLASSES]

        if not people_preds:
            empty = sv.Detections.empty()
            self._tracker.update_with_detections(empty)
            return []

        # Build supervision Detections for people (for ByteTrack)
        bboxes = np.array([self._to_xyxy(p) for p in people_preds])
        confidences = np.array([p["confidence"] for p in people_preds])

        sv_detections = sv.Detections(
            xyxy=bboxes,
            confidence=confidences,
        )

        # Run ByteTrack to assign persistent track IDs
        tracked = self._tracker.update_with_detections(sv_detections)

        # Build output in the same format as the original Detector
        people = []
        for i in range(len(tracked)):
            person = {
                "bbox": tracked.xyxy[i].tolist(),
                "confidence": round(float(tracked.confidence[i]), 2),
                "label": "person",
                "nearby_objects": [],
            }
            if tracked.tracker_id is not None:
                person["track_id"] = int(tracked.tracker_id[i])
            people.append(person)

        # Associate objects with nearest person
        for obj_pred in object_preds:
            obj_bbox = self._to_xyxy(obj_pred)
            obj_center = self._center(obj_bbox)
            nearest_person = None
            nearest_dist = float("inf")

            for person in people:
                person_center = self._center(person["bbox"])
                dist = self._distance(obj_center, person_center)
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_person = person

            if nearest_person and nearest_dist < self.object_association_distance:
                nearest_person["nearby_objects"].append(obj_pred["class"])

        return people

    def _infer(self, frame: np.ndarray) -> list[dict]:
        """Send frame to Roboflow and return raw predictions list."""
        if self._client is None:
            logger.error("RoboflowDetector not loaded. Call load() first.")
            return []

        # Encode frame as JPEG then base64
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

        payload = {
            "model_id": self.model_id,
            "image": {"type": "base64", "value": b64},
            "confidence": self.confidence_threshold,
        }
        if self.api_key:
            payload["api_key"] = self.api_key

        try:
            resp = self._client.post(
                f"{self.server_url}/infer/object_detection",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("predictions", [])

        except Exception as e:
            logger.error(f"Roboflow inference failed: {e}")
            return []

    def _to_xyxy(self, prediction: dict) -> list[float]:
        """Convert Roboflow center-format to [x1, y1, x2, y2]."""
        cx = prediction["x"]
        cy = prediction["y"]
        w = prediction["width"]
        h = prediction["height"]
        return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]

    def _center(self, bbox: list[float]) -> tuple[float, float]:
        """Center point of a bounding box [x1, y1, x2, y2]."""
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

    def _distance(self, pos1: tuple, pos2: tuple) -> float:
        """Euclidean distance between two (x, y) points."""
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        return math.sqrt(dx * dx + dy * dy)


# --- Test ---
if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from capture.camera import Camera
    from utils import save_snapshot, draw_boxes

    print("=== StangWatch Roboflow Detection Test ===")

    camera = Camera(source=0)
    if not camera.connect():
        exit(1)

    detector = RoboflowDetector()
    detector.load()

    camera.warm_up()
    frame = camera.read_frame()

    if frame is None:
        print("ERROR: Could not read frame")
        exit(1)

    print("Running detection via Roboflow API...")
    detections = detector.track_people(frame)

    if detections:
        for det in detections:
            tid = det.get("track_id", "N/A")
            conf = det["confidence"]
            objects = det.get("nearby_objects", [])
            print(f"  Track #{tid}: confidence={conf}, objects={objects}")
    else:
        print("No people detected.")

    annotated = draw_boxes(frame, detections)
    save_snapshot(annotated, "data/detection_test_roboflow.jpg")

    camera.release()
    print("Done.")

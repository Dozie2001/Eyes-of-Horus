"""
Person detection using YOLO11s with ByteTrack tracking.
Also detects carried objects (backpack, handbag, suitcase, laptop, phone)
and associates them with nearby people.
"""

from ultralytics import YOLO
import math

# COCO class IDs for objects we care about
PERSON_CLASS = 0
OBJECT_CLASSES = [24, 26, 28, 63, 67]  # backpack, handbag, suitcase, laptop, cell phone

# Human-readable names for these classes
OBJECT_NAMES = {
    24: "backpack",
    26: "handbag",
    28: "suitcase",
    63: "laptop",
    67: "cell phone",
}

# All classes we detect (person + carried objects)
ALL_CLASSES = [PERSON_CLASS] + OBJECT_CLASSES


class Detector:
    """
    Wraps the YOLO model for person detection, tracking, and object association.

    Detects people (with ByteTrack tracking) AND carried objects.
    Objects are associated with the nearest person.

    Usage:
        detector = Detector()
        detector.load()
        detections = detector.track_people(frame)
        # [{"bbox": [...], "confidence": 0.87, "label": "person",
        #   "track_id": 1, "nearby_objects": ["backpack"]}]
    """

    def __init__(self, model_name="yolo11s.pt", confidence_threshold=0.5,
                 object_association_distance=200):
        """
        Args:
            model_name: YOLO model to load.
            confidence_threshold: minimum confidence (0.0-1.0) to count as detection.
            object_association_distance: max pixels between a person and an object
                to consider the object "carried by" or "near" that person.
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.object_association_distance = object_association_distance
        self.model = None

    def load(self):
        """Load the YOLO model. Call once at startup."""
        print(f"Loading model: {self.model_name}")
        self.model = YOLO(self.model_name)
        print("Model loaded.")

    def detect_people(self, frame):
        """One-shot person detection. No tracking, no object association."""
        if self.model is None:
            print("ERROR: Model not loaded. Call load() first.")
            return []

        results = self.model(
            frame,
            classes=[PERSON_CLASS],
            conf=self.confidence_threshold,
            verbose=False,
        )

        return self._parse_people(results)

    def track_people(self, frame):
        """
        Detection + ByteTrack tracking + object association.

        1. Tracks people across frames (ByteTrack gives persistent track IDs)
        2. Detects carried objects (backpack, handbag, etc.)
        3. Associates each object with the nearest person

        Returns list of person detections, each with a "nearby_objects" field.
        """
        if self.model is None:
            print("ERROR: Model not loaded. Call load() first.")
            return []

        # Single pass: detect ALL classes (person + objects) with tracking
        results = self.model.track(
            frame,
            classes=ALL_CLASSES,
            conf=self.confidence_threshold,
            tracker="bytetrack.yaml",
            persist=True,
            verbose=False,
        )

        # Separate people from objects
        people = []
        objects = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                class_id = int(box.cls[0].item())
                bbox = box.xyxy[0].tolist()
                conf = round(box.conf[0].item(), 2)

                if class_id == PERSON_CLASS:
                    person = {
                        "bbox": bbox,
                        "confidence": conf,
                        "label": "person",
                        "nearby_objects": [],
                    }
                    if box.id is not None:
                        person["track_id"] = int(box.id[0].item())
                    people.append(person)
                elif class_id in OBJECT_NAMES:
                    objects.append({
                        "bbox": bbox,
                        "confidence": conf,
                        "label": OBJECT_NAMES[class_id],
                    })

        # Associate objects with nearest person
        for obj in objects:
            obj_center = self._center(obj["bbox"])
            nearest_person = None
            nearest_dist = float("inf")

            for person in people:
                person_center = self._center(person["bbox"])
                dist = self._distance(obj_center, person_center)
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_person = person

            if nearest_person and nearest_dist < self.object_association_distance:
                nearest_person["nearby_objects"].append(obj["label"])

        return people

    def _parse_people(self, results):
        """Parse YOLO results for person-only detection."""
        detections = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                detections.append({
                    "bbox": box.xyxy[0].tolist(),
                    "confidence": round(box.conf[0].item(), 2),
                    "label": "person",
                })
        return detections

    def _center(self, bbox):
        """Center point of a bounding box [x1, y1, x2, y2]."""
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

    def _distance(self, pos1, pos2):
        """Euclidean distance between two (x, y) points."""
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        return math.sqrt(dx * dx + dy * dy)


# --- Test ---
if __name__ == "__main__":
    import cv2
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from capture.camera import Camera
    from utils import save_snapshot, draw_boxes

    print("=== StangWatch Detection + Object Test ===")

    camera = Camera(source=0)
    if not camera.connect():
        exit(1)

    detector = Detector()
    detector.load()

    camera.warm_up()
    frame = camera.read_frame()

    if frame is None:
        print("ERROR: Could not read frame")
        exit(1)

    print("Running detection with object association...")
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
    save_snapshot(annotated, "data/detection_test.jpg")

    camera.release()
    print("Done.")

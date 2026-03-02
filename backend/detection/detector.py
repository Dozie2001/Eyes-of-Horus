"""
Person detection using YOLO11s.
Loads the model once, then runs detection on frames as they come in.
"""

from ultralytics import YOLO


class Detector:
    """
    Wraps the YOLO model for person detection.

    The model loads once (takes ~2 seconds) and is reused for every frame.
    Only detects people (COCO class 0).

    Usage:
        detector = Detector()
        detections = detector.detect_people(frame)
        # detections = [{"bbox": [x1,y1,x2,y2], "confidence": 0.87, "label": "person"}]
    """

    def __init__(self, model_name="yolo11s.pt", confidence_threshold=0.5):
        """
        Args:
            model_name: which YOLO model to load.
                "yolo11s.pt" is the small model — good balance of speed and accuracy.
                First run downloads it (~25MB), then it's cached locally.
            confidence_threshold: minimum confidence to count as a detection (0.0 to 1.0).
                0.5 means "only report if YOLO is at least 50% sure it's a person".
                Higher = fewer false alarms but might miss some people.
                Lower = catches more people but more false alarms.
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.model = None

    def load(self):
        """
        Load the YOLO model into memory. Call this once at startup.
        First run downloads the model file. After that it loads from cache.
        """
        print(f"Loading model: {self.model_name}")
        self.model = YOLO(self.model_name)
        print("Model loaded.")

    def detect_people(self, frame):
        """
        Run person detection on a single frame.

        Args:
            frame: image frame from Camera.read_frame() (numpy array)

        Returns:
            list of dicts, each with:
                - bbox: [x1, y1, x2, y2] pixel coordinates of the bounding box
                - confidence: how sure YOLO is (0.0 to 1.0)
                - label: always "person" (we only detect people)
        """
        if self.model is None:
            print("ERROR: Model not loaded. Call load() first.")
            return []

        # Run YOLO on the frame
        # classes=[0] = only detect "person" (class 0 in COCO dataset)
        # verbose=False = don't print logs for every frame
        results = self.model(
            frame,
            classes=[0],
            conf=self.confidence_threshold,
            verbose=False,
        )

        detections = []

        for result in results:
            for box in result.boxes:
                detection = {
                    "bbox": box.xyxy[0].tolist(),                  # [x1, y1, x2, y2]
                    "confidence": round(box.conf[0].item(), 2),    # e.g. 0.87
                    "label": "person",
                }
                detections.append(detection)

        return detections


# --- Test ---
if __name__ == "__main__":
    import cv2
    import os
    import sys

    # Add parent directory to path so we can import sibling modules
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from capture.camera import Camera
    from utils import save_snapshot, draw_boxes

    print("=== StangWatch Detection Test ===")

    # Step 1: Open webcam
    camera = Camera(source=0)
    if not camera.connect():
        exit(1)

    # Step 2: Load YOLO model
    detector = Detector()
    detector.load()

    # Step 3: Warm up camera and grab a frame
    camera.warm_up()
    frame = camera.read_frame()

    if frame is None:
        print("ERROR: Could not read frame")
        exit(1)

    # Step 4: Run detection
    print("Running person detection...")
    detections = detector.detect_people(frame)

    # Step 5: Report results
    if detections:
        print(f"Detected {len(detections)} person(s):")
        for i, det in enumerate(detections):
            bbox = det["bbox"]
            conf = det["confidence"]
            print(f"  Person {i+1}: confidence={conf}, "
                  f"position=({int(bbox[0])},{int(bbox[1])}) to ({int(bbox[2])},{int(bbox[3])})")
    else:
        print("No people detected in frame.")

    # Step 6: Save image with bounding boxes drawn on it
    annotated = draw_boxes(frame, detections)
    save_snapshot(annotated, "data/detection_test.jpg")

    camera.release()
    print("Done.")

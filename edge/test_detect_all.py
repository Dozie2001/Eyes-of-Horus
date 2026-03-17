"""
Diagnostic: what does YOLO see in a single frame?
Shows ALL detected objects (all 80 classes) at a lower confidence threshold.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from ultralytics import YOLO
from capture.camera import Camera

camera = Camera(source=0)
if not camera.connect():
    exit(1)

model = YOLO("yolo11s.pt")

camera.warm_up()

# Grab a few frames to get a good one
for _ in range(10):
    frame = camera.read_frame()

print("=== What YOLO sees (ALL classes, confidence >= 0.25) ===\n")

# Detect ALL classes at a low confidence threshold
results = model(frame, conf=0.25, verbose=False)

for result in results:
    if result.boxes is None:
        print("Nothing detected.")
        continue

    for box in result.boxes:
        class_id = int(box.cls[0].item())
        class_name = model.names[class_id]
        conf = round(box.conf[0].item(), 2)
        bbox = [round(v, 1) for v in box.xyxy[0].tolist()]
        print(f"  {class_name:15s} (class {class_id:2d})  confidence={conf}  bbox={bbox}")

print("\n--- Our target objects ---")
print("  24=backpack, 26=handbag, 28=suitcase, 63=laptop, 67=cell phone")

# Save the frame so you can see what YOLO saw
import cv2
os.makedirs("data", exist_ok=True)
cv2.imwrite("data/yolo_sees_this.jpg", frame)
print(f"\nFrame saved: data/yolo_sees_this.jpg  (open it to see what YOLO was looking at)")

camera.release()

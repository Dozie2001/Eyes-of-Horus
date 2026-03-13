#!/usr/bin/env python3
"""
Test V720 camera through the Camera class pipeline.

Run while connected to the camera's WiFi hotspot.

Usage:
    cd ~/stang
    source .venv/bin/activate
    python scripts/test_v720_pipeline.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.capture.camera import Camera
from backend.utils import save_snapshot

camera = Camera(source="v720://192.168.169.1:6123")

if not camera.connect():
    print("Failed to connect. Are you on the camera's WiFi hotspot?")
    sys.exit(1)

camera.warm_up()

frame = camera.read_frame()
if frame is not None:
    print(f"Got frame: {frame.shape}")
    save_snapshot(frame, "data/v720_pipeline_test.jpg")
    print("Saved to data/v720_pipeline_test.jpg")
else:
    print("ERROR: No frame received")

camera.release()
print("Done.")

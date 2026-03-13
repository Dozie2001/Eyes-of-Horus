#!/usr/bin/env python3
"""
Test script for V720 camera connection.

Run while connected to the camera's AP mode WiFi hotspot.
Shows live video and saves test snapshots.

Usage:
    cd ~/stang
    source .venv/bin/activate
    python scripts/test_v720.py

Press 's' to save a snapshot, 'q' to quit.
"""

import sys
import os
import time
import io
from datetime import datetime
from threading import Lock

import cv2
import numpy as np
from PIL import Image

# Add the V720 tool's source to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vendor", "a9-v720", "src"))

from netcl_tcp import netcl_tcp
from v720_ap import v720_ap
import cmd_udp

HOST = "192.168.169.1"
PORT = 6123
SNAPSHOT_DIR = os.path.join(os.path.dirname(__file__), "..", "test_videos")


def main():
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)

    print("=" * 50)
    print("StangWatch V720 Camera Test")
    print("=" * 50)
    print(f"Connecting to {HOST}:{PORT}...")

    try:
        sock = netcl_tcp(HOST, PORT)
        sock.__enter__()
    except Exception as e:
        print(f"\nFailed to connect: {e}")
        print("Make sure you're connected to the camera's WiFi hotspot.")
        return

    print("Connected! Initializing camera...")
    cam = v720_ap(sock)
    cam.init_live_motion()
    print("Camera initialized. Starting live stream...")
    print()
    print("Controls:")
    print("  's' = save snapshot")
    print("  'q' = quit")
    print()

    # Frame state
    frame_lock = Lock()
    current_frame = None
    frame_count = 0
    fps_time = time.time()

    cv2.namedWindow("StangWatch - V720 Test")

    try:
        sync = False
        jpeg_buffer = bytearray()

        def on_frame(cmd, data: bytearray):
            nonlocal sync, jpeg_buffer, current_frame, frame_count, fps_time

            if cmd != cmd_udp.P2P_UDP_CMD_JPEG:
                return

            if not sync:
                # Look for JPEG start marker
                start = data.find(b'\xff\xd8')
                if start != -1:
                    jpeg_buffer.extend(data[start:])
                    sync = True
            else:
                # Look for JPEG end marker
                end = data.find(b'\xff\xd9')
                if end != -1:
                    jpeg_buffer.extend(data[:end + 2])

                    # Decode the complete JPEG frame
                    try:
                        img = np.array(Image.open(io.BytesIO(jpeg_buffer)))
                        # PIL gives RGB, OpenCV wants BGR
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                        # Calculate FPS
                        now = time.time()
                        fps = 1.0 / (now - fps_time) if (now - fps_time) > 0 else 0
                        fps_time = now
                        frame_count += 1

                        # Add overlay text
                        cv2.putText(img, f"FPS: {fps:.1f}", (10, 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                        cv2.putText(img, f"Frame: {frame_count}", (10, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                        cv2.putText(img, datetime.now().strftime("%H:%M:%S"), (10, 75),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

                        with frame_lock:
                            current_frame = img.copy()

                        cv2.imshow("StangWatch - V720 Test", img)
                        key = cv2.waitKey(1) & 0xFF

                        if key == ord('s'):
                            fname = f"v720_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                            path = os.path.join(SNAPSHOT_DIR, fname)
                            cv2.imwrite(path, img)
                            print(f"  Snapshot saved: {path}")
                            print(f"  Resolution: {img.shape[1]}x{img.shape[0]}")

                        elif key == ord('q'):
                            raise KeyboardInterrupt

                    except Exception as e:
                        pass  # Skip corrupted frames silently

                    jpeg_buffer.clear()
                    sync = False
                else:
                    jpeg_buffer.extend(data)

        cam.cap_live(on_frame)

    except KeyboardInterrupt:
        print(f"\nStopping. Total frames received: {frame_count}")

    finally:
        cv2.destroyAllWindows()
        try:
            sock.__exit__(None, None, None)
        except Exception:
            pass

    # Print summary
    print()
    print("=" * 50)
    print("RESULTS — copy this back to Claude:")
    print("=" * 50)
    if frame_count > 0:
        print(f"SUCCESS: Received {frame_count} frames")
        if current_frame is not None:
            print(f"Resolution: {current_frame.shape[1]}x{current_frame.shape[0]}")
        snapshots = [f for f in os.listdir(SNAPSHOT_DIR) if f.startswith("v720_snapshot_")]
        if snapshots:
            print(f"Snapshots saved: {len(snapshots)}")
            for s in snapshots:
                print(f"  {os.path.join(SNAPSHOT_DIR, s)}")
    else:
        print("NO FRAMES received. Camera may not have started streaming.")
    print("=" * 50)


if __name__ == "__main__":
    main()

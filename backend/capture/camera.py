"""
Video capture pipeline for StangWatch.
Reads frames from a webcam, video file, RTSP stream, or V720 camera.

Architecture:
    BaseCamera      — shared interface and state (width, height, fps, etc.)
    OpenCVCamera    — webcam, video file, RTSP (anything OpenCV can open)
    V720Camera      — V720/A9 cameras (proprietary protocol over TCP)
    Camera()        — factory function that returns the right subclass
"""

import io
import sys
import os
import threading
import time

import cv2
import numpy as np
from PIL import Image


class BaseCamera:
    """
    Shared interface for all camera types.

    Subclasses must implement: connect(), read_frame(), release().
    Optionally override: warm_up().

    Properties available after connect():
        width, height, fps, is_connected
    """

    def __init__(self):
        self.width = 0
        self.height = 0
        self.fps = 0.0
        self.is_connected = False

    def connect(self):
        """Open the video source. Returns True on success."""
        raise NotImplementedError

    def read_frame(self):
        """
        Grab the next frame.xw
        """
        raise NotImplementedError

    def release(self):
        """Close the video source and free resources."""
        raise NotImplementedError

    def warm_up(self, frames=30):
        """
        Read and discard frames to let the camera adjust.
        Default implementation works for frame-by-frame sources.
        """
        if not self.is_connected:
            print("ERROR: Camera not connected. Call connect() first.")
            return

        print(f"Warming up camera ({frames} frames)...")
        for _ in range(frames):
            self.read_frame()


class OpenCVCamera(BaseCamera):
    """
    Usage:
        cam = OpenCVCamera(source=0)          # webcam
        cam = OpenCVCamera(source="file.mp4") # video file
        cam = OpenCVCamera(source="rtsp://...") # RTSP stream
    """

    def __init__(self, source=0):
        super().__init__()
        self.source = source
        self.cap = None

    def connect(self):
        self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            print(f"ERROR: Could not open video source: {self.source}")
            self.is_connected = False
            return False

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.is_connected = True

        print(f"Connected to source: {self.source}")
        print(f"  Resolution: {self.width}x{self.height}")
        print(f"  FPS: {self.fps}")

        return True

    def read_frame(self):
        if not self.is_connected:
            return None

        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.is_connected = False
            print("Camera released.")


class V720Camera(BaseCamera):
    """
    Handles V720/A9 cameras using the proprietary protocol.

    The V720 streams MJPEG frames via a callback (push model).
    This class bridges to the pull model (read_frame()) using a
    background thread and a lock-protected frame buffer.

    Usage:
        cam = V720Camera(host="192.168.169.1", port=6123)
    """

    def __init__(self, host="192.168.169.1", port=6123):
        super().__init__()
        self.host = host
        self.port = port

        self._sock = None
        self._cam = None
        self._thread = None
        self._running = False
        self._frame = None
        self._lock = threading.Lock()

    def connect(self):
        # Add the V720 library to path
        v720_src = os.path.join(
            os.path.dirname(__file__), "..", "..", "vendor", "a9-v720", "src"
        )
        v720_src = os.path.abspath(v720_src)
        if v720_src not in sys.path:
            sys.path.insert(0, v720_src)

        try:
            from netcl_tcp import netcl_tcp
            from v720_ap import v720_ap
        except ImportError as e:
            print(f"ERROR: V720 library not found: {e}")
            print(f"  Expected at: {v720_src}")
            self.is_connected = False
            return False

        print(f"Connecting to V720 camera at {self.host}:{self.port}...")
        try:
            self._sock = netcl_tcp(self.host, self.port)
            self._sock.__enter__()
        except Exception as e:
            print(f"ERROR: Could not connect to V720 camera: {e}")
            self.is_connected = False
            return False

        self._cam = v720_ap(self._sock)
        self._cam.init_live_motion()

        self.width = 640
        self.height = 480
        self.fps = 10.0
        self.is_connected = True

        # Start streaming in background thread
        self._running = True
        self._thread = threading.Thread(
            target=self._stream_loop,
            daemon=True,
            name="v720-stream",
        )
        self._thread.start()

        print(f"Connected to V720 camera: {self.host}:{self.port}")
        print(f"  Resolution: {self.width}x{self.height}")
        print(f"  FPS: ~{self.fps}")

        return True

    def warm_up(self, frames=30):
        """Wait for the V720 stream to start delivering frames."""
        if not self.is_connected:
            print("ERROR: Camera not connected. Call connect() first.")
            return

        print("V720 warming up (waiting for frames)...")
        for _ in range(50):  # wait up to 5 seconds
            if self._frame is not None:
                print("V720 streaming.")
                return
            time.sleep(0.1)
        print("WARNING: No V720 frames received during warm-up")

    def read_frame(self):
        if not self.is_connected:
            return None

        with self._lock:
            if self._frame is not None:
                return self._frame.copy()
            return None

    def release(self):
        self._running = False
        self.is_connected = False

        try:
            self._sock.__exit__(None, None, None)
        except Exception:
            pass

        if self._thread is not None:
            self._thread.join(timeout=3)

        self._cam = None
        self._sock = None
        self._frame = None
        print("V720 camera released.")

    def _stream_loop(self):
        """Background thread: receive V720 MJPEG frames and store the latest."""
        try:
            import cmd_udp
        except ImportError:
            print("ERROR: cmd_udp not importable")
            return

        sync = False
        jpeg_buffer = bytearray()

        def on_frame(cmd, data: bytearray):
            nonlocal sync, jpeg_buffer

            if cmd != cmd_udp.P2P_UDP_CMD_JPEG:
                return

            if not sync:
                start = data.find(b'\xff\xd8')
                if start != -1:
                    jpeg_buffer.extend(data[start:])
                    sync = True
            else:
                end = data.find(b'\xff\xd9')
                if end != -1:
                    jpeg_buffer.extend(data[:end + 2])

                    try:
                        img = np.array(Image.open(io.BytesIO(jpeg_buffer)))
                        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                        with self._lock:
                            self._frame = img_bgr

                        # Update resolution from actual frame
                        self.height, self.width = img_bgr.shape[:2]
                    except Exception:
                        pass  # skip corrupted frames

                    jpeg_buffer.clear()
                    sync = False
                else:
                    jpeg_buffer.extend(data)

        try:
            self._cam.cap_live(on_frame)
        except Exception as e:
            if self._running:
                print(f"V720 stream error: {e}")


def _parse_v720_url(source):
    """Parse 'v720://host:port' into (host, port)."""
    addr = source.replace("v720://", "")
    if ":" in addr:
        host, port_str = addr.rsplit(":", 1)
        return host, int(port_str)
    return addr, 6123


def Camera(source=0):
    """
    Factory function — returns the right camera subclass for the source.

    Args:
        source: what to read from
            0               -> Mac webcam (default)
            "file.mp4"      -> video file path
            "rtsp://..."    -> RTSP camera stream
            "v720://ip:port" -> V720/A9 camera

    Returns:
        OpenCVCamera or V720Camera instance
    """
    if isinstance(source, str) and source.startswith("v720://"):
        host, port = _parse_v720_url(source)
        return V720Camera(host, port)
    return OpenCVCamera(source)


# --- Test ---
if __name__ == "__main__":
    from utils import save_snapshot

    print("=== StangWatch Camera Test ===")

    camera = Camera(source=0)

    if not camera.connect():
        exit(1)

    camera.warm_up()

    frame = camera.read_frame()
    if frame is not None:
        save_snapshot(frame, "data/test_frame.jpg")
        print(f"Frame shape: {frame.shape}")
        print(f"Camera type: {type(camera).__name__}")
        print("Capture pipeline is working!")
    else:
        print("ERROR: Could not read frame")

    camera.release()
    print("Done.")

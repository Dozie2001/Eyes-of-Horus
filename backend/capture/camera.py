"""
Video capture pipeline for StangWatch.
Reads frames from a webcam, video file, or RTSP stream.
"""

import cv2


class Camera:
    """
    Manages a video source connection (webcam, file, or RTSP stream).

    The connection opens once and stays open. You call read_frame()
    repeatedly to get frames. Call release() when done.

    Usage:
        camera = Camera(source=0)       # 0 = webcam
        camera.connect()                # opens the connection
        frame = camera.read_frame()     # grab a frame
        camera.release()                # close when done
    """

    def __init__(self, source=0):
        """
        Args:
            source: what to read from
                0           -> Mac webcam (default)
                "file.mp4"  -> video file path
                "rtsp://..." -> RTSP camera stream
        """
        self.source = source
        self.cap = None       # the OpenCV VideoCapture object (set on connect)
        self.width = 0
        self.height = 0
        self.fps = 0.0
        self.is_connected = False

    def connect(self):
        """
        Open the video source. Returns True if successful, False if not.
        Also reads the stream's resolution and FPS.
        """
        self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            print(f"ERROR: Could not open video source: {self.source}")
            self.is_connected = False
            return False

        # Read stream properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.is_connected = True

        print(f"Connected to source: {self.source}")
        print(f"  Resolution: {self.width}x{self.height}")
        print(f"  FPS: {self.fps}")

        return True

    def warm_up(self, frames=30):
        """
        Read and discard frames to let the camera auto-adjust
        exposure and white balance. Only needed for webcams.

        Args:
            frames: number of frames to skip (30 = ~1 second at 30fps)
        """
        if not self.is_connected:
            print("ERROR: Camera not connected. Call connect() first.")
            return

        print(f"Warming up camera ({frames} frames)...")
        for _ in range(frames):
            self.cap.read()

    def read_frame(self):
        """
        Grab the next frame from the video source.

        Returns:
            frame (numpy array) if successful, None if failed.
            Frame shape is (height, width, 3) in BGR color format.
        """
        if not self.is_connected:
            return None

        ret, frame = self.cap.read()

        if not ret:
            return None

        return frame

    def release(self):
        """Close the video source and free resources."""
        if self.cap is not None:
            self.cap.release()
            self.is_connected = False
            print("Camera released.")


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
        print("Capture pipeline is working!")
    else:
        print("ERROR: Could not read frame")

    camera.release()
    print("Done.")

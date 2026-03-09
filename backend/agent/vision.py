"""
Vision description for StangWatch.

Sends images to a local vision LLM (Ollama) and returns a plain-English
description of what's happening.

Supports two modes:
- Single image: describe a snapshot (fallback)
- Multi-frame: extract key frames from a video clip for temporal context

The description is then fed to the text LLM for severity reasoning:
    Vision perceives → Text reasons.

Usage:
    from agent.vision import VisionDescriber

    describer = VisionDescriber(model="qwen2.5vl:7b")
    if describer.is_available():
        # Preferred: analyze video clip (multiple frames, temporal context)
        desc = describer.describe_video("data/events/clip.mp4")

        # Fallback: analyze single snapshot
        desc = describer.describe("data/events/snapshot.jpg")
"""

import base64
import time

import ollama


VISION_PROMPT = (
    "You are a security camera image analyst. "
    "Describe what you see in this security camera image in 1-2 sentences. "
    "Be factual and specific about:\n"
    "- How many people are visible and where they are positioned\n"
    "- What they appear to be doing (standing, walking, crouching, carrying something)\n"
    "- Any objects they are holding or interacting with\n"
    "- Their general appearance (clothing color, build) if visible\n\n"
    "Do NOT speculate about intent. Do NOT use words like 'suspicious', "
    "'criminal', or 'thief'. Just describe what you see."
)

VIDEO_VISION_PROMPT = (
    "You are analyzing a sequence of frames from a security camera, "
    "shown in chronological order (earliest to latest). "
    "Describe what you see happening across these frames in 2-3 sentences. "
    "Focus on:\n"
    "- How people move between frames (approaching, leaving, standing still)\n"
    "- Any changes in posture or activity (picking something up, crouching, running)\n"
    "- Objects they carry or interact with\n"
    "- Their general appearance and position relative to the scene\n\n"
    "Do NOT speculate about intent. Do NOT use words like 'suspicious', "
    "'criminal', or 'thief'. Just describe what you observe happening."
)


class VisionDescriber:
    """Sends images to an Ollama vision model for description."""

    def __init__(self, model="qwen2.5vl:7b", host="http://localhost:11434",
                 timeout=30.0):
        self.model = model
        self.host = host
        self.timeout = timeout
        self._client = ollama.Client(host=host, timeout=timeout)

        # Availability cache
        self._available = None
        self._checked_at = 0
        self._cache_ttl = 60  # seconds

    def is_available(self):
        """Check if the vision model is pulled and Ollama is reachable."""
        now = time.time()
        if (now - self._checked_at) < self._cache_ttl:
            return self._available

        try:
            models = self._client.list()
            available = [m.model for m in models.models]
            self._available = self.model in available
            if not self._available:
                print(f"Vision model '{self.model}' not found. Available: {available}")
        except Exception as e:
            print(f"Vision model health check failed: {e}")
            self._available = False

        self._checked_at = now
        return self._available

    def describe(self, image_path):
        """
        Describe what's happening in a single snapshot image.

        Args:
            image_path: path to a JPEG snapshot file

        Returns:
            str: 1-2 sentence description, or None if failed
        """
        if not self.is_available():
            return None

        try:
            with open(image_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("utf-8")
        except FileNotFoundError:
            print(f"Vision: snapshot not found: {image_path}")
            return None

        try:
            start = time.time()
            response = self._client.chat(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": VISION_PROMPT,
                    "images": [img_b64],
                }],
                options={"temperature": 0.1},
            )
            elapsed_ms = int((time.time() - start) * 1000)

            description = response.message.content.strip()
            print(f"  VISION: {description[:80]}... ({elapsed_ms}ms)")
            return description

        except Exception as e:
            print(f"Vision description failed: {e}")
            return None

    def describe_video(self, video_path, num_frames=5):
        """
        Describe what's happening across multiple frames from a video clip.

        Extracts key frames evenly spaced through the clip and sends them
        all to the vision model for temporal context.

        Args:
            video_path: path to an MP4 video clip
            num_frames: how many key frames to extract (default 5)

        Returns:
            str: 2-3 sentence description, or None if failed
        """
        if not self.is_available():
            return None

        frames_b64 = self._extract_key_frames(video_path, num_frames)
        if not frames_b64:
            return None

        try:
            start = time.time()
            response = self._client.chat(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": VIDEO_VISION_PROMPT,
                    "images": frames_b64,
                }],
                options={"temperature": 0.1},
            )
            elapsed_ms = int((time.time() - start) * 1000)

            description = response.message.content.strip()
            print(f"  VISION (video, {len(frames_b64)} frames): {description[:80]}... ({elapsed_ms}ms)")
            return description

        except Exception as e:
            print(f"Vision video description failed: {e}")
            return None

    def _extract_key_frames(self, video_path, num_frames):
        """
        Extract evenly-spaced key frames from a video file as base64 JPEGs.

        Returns:
            list of base64-encoded JPEG strings, or empty list on failure
        """
        import cv2

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Vision: could not open video: {video_path}")
                return []

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames < 1:
                cap.release()
                return []

            # Calculate frame indices (evenly spaced)
            if total_frames <= num_frames:
                indices = list(range(total_frames))
            else:
                step = total_frames / num_frames
                indices = [int(i * step) for i in range(num_frames)]

            frames_b64 = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    continue

                # Encode as JPEG and base64
                _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frames_b64.append(base64.b64encode(buf.tobytes()).decode("utf-8"))

            cap.release()
            return frames_b64

        except Exception as e:
            print(f"Vision: frame extraction failed: {e}")
            return []

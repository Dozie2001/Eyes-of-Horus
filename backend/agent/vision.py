"""
Vision description for StangWatch.

Sends images to a vision model and returns a plain-English description.
Supports two backends:
- Ollama (local): qwen2.5vl via local Ollama server
- Gemini (cloud): Google Gemini 2.0 Flash via REST API (free tier)

Supports two modes per backend:
- Single image: describe a snapshot (fallback)
- Multi-frame: extract key frames from a video clip for temporal context

The description is then fed to the text LLM for severity reasoning:
    Vision perceives -> Text reasons.

Usage:
    from agent.vision import VisionDescriber, GeminiVisionDescriber

    # Local (Ollama)
    describer = VisionDescriber(model="qwen2.5vl:7b")

    # Cloud (Gemini Flash — free tier)
    describer = GeminiVisionDescriber(api_key="...")

    if describer.is_available():
        desc = describer.describe_video("data/events/clip.mp4")
        desc = describer.describe("data/events/snapshot.jpg")
"""

import base64
import time

import httpx
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


def _extract_key_frames(video_path, num_frames=5):
    """
    Extract evenly-spaced key frames from a video file as base64 JPEGs.

    Shared by both Ollama and Gemini describers.

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
        """Describe what's happening in a single snapshot image."""
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
        """Describe what's happening across multiple frames from a video clip."""
        if not self.is_available():
            return None

        frames_b64 = _extract_key_frames(video_path, num_frames)
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


class GeminiVisionDescriber:
    """
    Sends images to Google Gemini 2.0 Flash for description.

    Uses the REST API directly (no SDK needed). Free tier: 15 req/min.
    """

    GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

    def __init__(self, api_key, timeout=30.0):
        self.api_key = api_key
        self.timeout = timeout
        self._available = None
        self._checked_at = 0
        self._cache_ttl = 300  # 5 min cache

    def is_available(self):
        """Check if the Gemini API key works."""
        now = time.time()
        if (now - self._checked_at) < self._cache_ttl:
            return self._available

        try:
            # Quick health check — list models
            resp = httpx.get(
                "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash",
                params={"key": self.api_key},
                timeout=10.0,
            )
            self._available = resp.status_code == 200
            if not self._available:
                print(f"Gemini health check failed: HTTP {resp.status_code}")
        except Exception as e:
            print(f"Gemini health check failed: {e}")
            self._available = False

        self._checked_at = now
        return self._available

    def describe(self, image_path):
        """Describe what's happening in a single snapshot image."""
        if not self.is_available():
            return None

        try:
            with open(image_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("utf-8")
        except FileNotFoundError:
            print(f"Vision: snapshot not found: {image_path}")
            return None

        parts = [
            {"text": VISION_PROMPT},
            {"inline_data": {"mime_type": "image/jpeg", "data": img_b64}},
        ]

        return self._call_gemini(parts, "snapshot")

    def describe_video(self, video_path, num_frames=5):
        """Describe what's happening across multiple frames from a video clip."""
        if not self.is_available():
            return None

        frames_b64 = _extract_key_frames(video_path, num_frames)
        if not frames_b64:
            return None

        # Build parts: prompt + all frames
        parts = [{"text": VIDEO_VISION_PROMPT}]
        for frame_b64 in frames_b64:
            parts.append({
                "inline_data": {"mime_type": "image/jpeg", "data": frame_b64}
            })

        return self._call_gemini(parts, f"video, {len(frames_b64)} frames")

    def _call_gemini(self, parts, label):
        """Make a Gemini API call with the given parts."""
        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 256,
            },
        }

        try:
            start = time.time()
            resp = httpx.post(
                self.GEMINI_URL,
                params={"key": self.api_key},
                json=payload,
                timeout=self.timeout,
            )
            elapsed_ms = int((time.time() - start) * 1000)

            if resp.status_code != 200:
                print(f"Gemini API error: HTTP {resp.status_code} — {resp.text[:200]}")
                return None

            result = resp.json()
            text = result["candidates"][0]["content"]["parts"][0]["text"].strip()
            print(f"  VISION ({label}): {text[:80]}... ({elapsed_ms}ms)")
            return text

        except Exception as e:
            print(f"Gemini vision failed: {e}")
            return None

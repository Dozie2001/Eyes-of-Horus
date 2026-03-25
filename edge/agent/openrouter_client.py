"""
OpenRouter client for Eyes of Horus.

Unified AI provider with automatic fallbacks across free models.
Handles both text reasoning (event evaluation, chat) and vision (image description).

Uses OpenAI-compatible API with OpenRouter's fallback routing.
All models are free tier — $0/month.

Usage:
    from agent.openrouter_client import OpenRouterTextClient, OpenRouterVisionClient

    text = OpenRouterTextClient(api_key="sk-or-...")
    result = text.evaluate(system_prompt, user_prompt)

    vision = OpenRouterVisionClient(api_key="sk-or-...")
    description = vision.describe("path/to/image.jpg")
"""

import base64
import json
import logging
import time

import httpx

logger = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_HEADERS_BASE = {
    "HTTP-Referer": "https://eyesofhorus.com",
    "X-Title": "Eyes of Horus",
}
MAX_RETRIES = 3


def _post_with_retry(api_key, payload, timeout):
    """POST to OpenRouter with retry on 429 rate limits."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        **OPENROUTER_HEADERS_BASE,
    }
    resp = None
    for attempt in range(MAX_RETRIES):
        resp = httpx.post(
            OPENROUTER_URL,
            headers=headers,
            json=payload,
            timeout=timeout,
        )
        if resp.status_code != 429:
            return resp
        wait = 2 ** attempt
        logger.info(f"OpenRouter rate limited, retrying in {wait}s (attempt {attempt + 1}/{MAX_RETRIES})")
        time.sleep(wait)
    return resp


# Free text models (max 3 for free tier fallback routing)
TEXT_MODELS = [
    "google/gemma-3-27b-it:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "mistralai/mistral-small-3.1-24b-instruct:free",
]

# Free vision models (max 3 for free tier fallback routing)
VISION_MODELS = [
    "google/gemma-3-27b-it:free",
    "nvidia/nemotron-nano-12b-v2-vl:free",
    "google/gemma-3-12b-it:free",
]


class OpenRouterTextClient:
    """Text reasoning via OpenRouter with automatic fallbacks."""

    def __init__(self, api_key, models=None, timeout=60.0):
        self.api_key = api_key
        self.models = models or TEXT_MODELS
        self.timeout = timeout

        self._healthy = None
        self._health_checked_at = 0
        self._health_cache_ttl = 60

    def is_healthy(self):
        """Check if OpenRouter is reachable. Cached for 60 seconds."""
        now = time.time()
        if (now - self._health_checked_at) < self._health_cache_ttl:
            return self._healthy

        try:
            resp = httpx.get(
                "https://openrouter.ai/api/v1/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=10.0,
            )
            self._healthy = resp.status_code == 200
            if not self._healthy:
                logger.warning(f"OpenRouter health check failed: HTTP {resp.status_code}")
        except Exception as e:
            logger.warning(f"OpenRouter health check failed: {e}")
            self._healthy = False

        self._health_checked_at = now
        return self._healthy

    def evaluate(self, system_prompt, user_prompt):
        """
        Send prompts to OpenRouter and get a JSON decision.

        Uses fallback routing — if primary model is down, tries the next.

        Returns:
            dict with keys: alert, severity, reason, recommendation
            None on failure
        """
        payload = {
            "model": self.models[0],
            "route": "fallback",
            "models": self.models,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        }

        try:
            start = time.time()
            resp = _post_with_retry(self.api_key, payload, self.timeout)
            elapsed_ms = int((time.time() - start) * 1000)

            if resp.status_code != 200:
                logger.warning(f"OpenRouter text error: HTTP {resp.status_code} — {resp.text[:200]}")
                return None

            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            model_used = data.get("model", "unknown")
            result = json.loads(content)

            # Validate required fields
            if "alert" not in result:
                result["alert"] = False
            if "severity" not in result:
                result["severity"] = "ignore"
            if "reason" not in result:
                result["reason"] = "No reason provided"
            if "recommendation" not in result:
                result["recommendation"] = ""

            result["alert"] = bool(result["alert"])
            result["severity"] = str(result["severity"]).lower()

            logger.debug(f"OpenRouter text ({model_used}): {elapsed_ms}ms")
            return result

        except json.JSONDecodeError as e:
            logger.warning(f"OpenRouter returned invalid JSON: {e}")
            return None
        except Exception as e:
            logger.error(f"OpenRouter text failed: {e}")
            return None

    def chat(self, system_prompt, user_message):
        """
        Send a chat message and get a plain text response.
        Used by the Telegram bot for conversational AI.

        Returns:
            str response text, or None on failure
        """
        payload = {
            "model": self.models[0],
            "route": "fallback",
            "models": self.models,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "temperature": 0.3,
        }

        try:
            resp = _post_with_retry(self.api_key, payload, self.timeout)

            if resp.status_code != 200:
                logger.warning(f"OpenRouter chat error: HTTP {resp.status_code}")
                return None

            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()

        except Exception as e:
            logger.error(f"OpenRouter chat failed: {e}")
            return None


class OpenRouterVisionClient:
    """Vision description via OpenRouter with automatic fallbacks."""

    def __init__(self, api_key, models=None, timeout=60.0):
        self.api_key = api_key
        self.models = models or VISION_MODELS
        self.timeout = timeout

        self._available = None
        self._checked_at = 0
        self._cache_ttl = 300

    def is_available(self):
        """Check if OpenRouter is reachable. Cached for 5 minutes."""
        now = time.time()
        if (now - self._checked_at) < self._cache_ttl:
            return self._available

        try:
            resp = httpx.get(
                "https://openrouter.ai/api/v1/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=10.0,
            )
            self._available = resp.status_code == 200
        except Exception as e:
            logger.warning(f"OpenRouter vision health check failed: {e}")
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
            logger.warning(f"Vision: snapshot not found: {image_path}")
            return None

        from agent.vision import VISION_PROMPT

        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": VISION_PROMPT},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{img_b64}"
                }},
            ],
        }]

        return self._call(messages, "snapshot")

    def describe_video(self, video_path, num_frames=5):
        """Describe what's happening across multiple frames from a video clip."""
        if not self.is_available():
            return None

        from agent.vision import _extract_key_frames, VIDEO_VISION_PROMPT

        frames_b64 = _extract_key_frames(video_path, num_frames)
        if not frames_b64:
            return None

        content = [{"type": "text", "text": VIDEO_VISION_PROMPT}]
        for frame_b64 in frames_b64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"},
            })

        messages = [{"role": "user", "content": content}]
        return self._call(messages, f"video, {len(frames_b64)} frames")

    def _call(self, messages, label):
        """Make an OpenRouter vision API call with fallbacks."""
        payload = {
            "model": self.models[0],
            "route": "fallback",
            "models": self.models,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 256,
        }

        try:
            start = time.time()
            resp = _post_with_retry(self.api_key, payload, self.timeout)
            elapsed_ms = int((time.time() - start) * 1000)

            if resp.status_code != 200:
                logger.warning(f"OpenRouter vision error: HTTP {resp.status_code} — {resp.text[:200]}")
                return None

            data = resp.json()
            text = data["choices"][0]["message"]["content"].strip()
            model_used = data.get("model", "unknown")
            logger.info(f"  VISION ({label}, {model_used}): {text[:80]}... ({elapsed_ms}ms)")
            return text

        except Exception as e:
            logger.error(f"OpenRouter vision failed: {e}")
            return None

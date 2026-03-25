"""
Groq text reasoning client for Eyes of Horus.

Uses Groq's OpenAI-compatible API for fast text inference.
Free tier: 30 requests/minute.
"""

import json
import logging
import time

import httpx

logger = logging.getLogger(__name__)

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"


class GroqTextClient:
    """Text reasoning via Groq with retry on rate limit."""

    def __init__(self, api_key, model=GROQ_MODEL, timeout=60.0):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

        self._healthy = None
        self._health_checked_at = 0
        self._health_cache_ttl = 60

    def is_healthy(self):
        """Check if Groq is reachable. Cached for 60 seconds."""
        now = time.time()
        if (now - self._health_checked_at) < self._health_cache_ttl:
            return self._healthy

        try:
            resp = httpx.get(
                "https://api.groq.com/openai/v1/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=10.0,
            )
            self._healthy = resp.status_code == 200
            if not self._healthy:
                logger.warning(f"Groq health check failed: HTTP {resp.status_code}")
        except Exception as e:
            logger.warning(f"Groq health check failed: {e}")
            self._healthy = False

        self._health_checked_at = now
        return self._healthy

    def _post(self, payload):
        """POST to Groq with retry on 429."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        for attempt in range(3):
            resp = httpx.post(
                GROQ_URL,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            if resp.status_code != 429:
                return resp
            wait = 2 ** attempt
            logger.info(f"Groq rate limited, retrying in {wait}s")
            time.sleep(wait)
        return resp

    def evaluate(self, system_prompt, user_prompt):
        """
        Send prompts to Groq and get a JSON decision.

        Returns:
            dict with keys: alert, severity, reason, recommendation
            None on failure
        """
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        }

        try:
            start = time.time()
            resp = self._post(payload)
            elapsed_ms = int((time.time() - start) * 1000)

            if resp.status_code != 200:
                logger.warning(f"Groq text error: HTTP {resp.status_code} — {resp.text[:200]}")
                return None

            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            result = json.loads(content)

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

            logger.debug(f"Groq text: {elapsed_ms}ms")
            return result

        except json.JSONDecodeError as e:
            logger.warning(f"Groq returned invalid JSON: {e}")
            return None
        except Exception as e:
            logger.error(f"Groq text failed: {e}")
            return None

    def chat(self, system_prompt, user_message):
        """
        Send a chat message and get a plain text response.

        Returns:
            str response text, or None on failure
        """
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "temperature": 0.3,
        }

        try:
            resp = self._post(payload)

            if resp.status_code != 200:
                logger.warning(f"Groq chat error: HTTP {resp.status_code}")
                return None

            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()

        except Exception as e:
            logger.error(f"Groq chat failed: {e}")
            return None

"""
Ollama client wrapper for StangWatch AI agent.

Calls a local Ollama model (default: Qwen 2.5 7B) for event evaluation.
Forces JSON output, low temperature for consistent decisions,
and includes health checking with cache to avoid hammering Ollama.

Usage:
    from agent.ollama_client import OllamaClient

    client = OllamaClient(model="qwen2.5:7b")
    if client.is_healthy():
        result = client.evaluate(system_prompt, user_prompt)
        # result = {"alert": True, "severity": "high", "reason": "...", ...}
"""

import json
import time

import ollama


class OllamaClient:
    """Wraps the Ollama Python SDK for event evaluation."""

    def __init__(self, model="qwen2.5:7b", host="http://localhost:11434",
                 timeout=10.0):
        """
        Args:
            model: Ollama model name (must be pulled first)
            host: Ollama server URL
            timeout: request timeout in seconds
        """
        self.model = model
        self.host = host
        self.timeout = timeout
        self._client = ollama.Client(host=host, timeout=timeout)

        # Health check cache (avoid hammering Ollama)
        self._healthy = None
        self._health_checked_at = 0
        self._health_cache_ttl = 30  # seconds

    def is_healthy(self):
        """
        Check if Ollama is reachable and the model is available.
        Caches result for 30 seconds.

        Returns:
            bool: True if Ollama is up and model is loaded
        """
        now = time.time()
        if (now - self._health_checked_at) < self._health_cache_ttl:
            return self._healthy

        try:
            models = self._client.list()
            available = [m.model for m in models.models]
            self._healthy = self.model in available
            if not self._healthy:
                print(f"Ollama model '{self.model}' not found. Available: {available}")
        except Exception as e:
            print(f"Ollama health check failed: {e}")
            self._healthy = False

        self._health_checked_at = now
        return self._healthy

    def evaluate(self, system_prompt, user_prompt):
        """
        Send prompts to Ollama and get a JSON decision.

        Args:
            system_prompt: security agent role definition
            user_prompt: event context for evaluation

        Returns:
            dict with keys: alert, severity, reason, recommendation
            None if Ollama fails or returns invalid JSON
        """
        try:
            response = self._client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                format="json",
                options={"temperature": 0.1},
            )

            content = response.message.content
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

            # Normalize types
            result["alert"] = bool(result["alert"])
            result["severity"] = str(result["severity"]).lower()

            return result

        except json.JSONDecodeError as e:
            print(f"Ollama returned invalid JSON: {e}")
            return None
        except Exception as e:
            print(f"Ollama evaluation failed: {e}")
            return None

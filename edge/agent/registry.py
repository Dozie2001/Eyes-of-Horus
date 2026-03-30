"""
Provider plugin registry with circuit breaker for StangWatch.

Makes it trivial to add new text, vision, or alert providers.
Uses circuit breaker pattern for automatic failover between providers.
Ollama is always registered as the local fallback (priority=100).

Usage:
    from agent.registry import build_registry

    registry = build_registry(config)
    result = registry.get_text()      # (provider, circuit_breaker) or None
    result = registry.get_vision()    # (provider, circuit_breaker) or None
    senders = registry.get_all_alerts()  # [(sender, cb), ...]
"""

import logging
import threading
import time
from typing import Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# Protocols — existing clients already conform, zero changes needed

@runtime_checkable
class TextProvider(Protocol):
    def evaluate(self, system_prompt: str, user_prompt: str) -> Optional[dict]: ...
    def is_healthy(self) -> bool: ...


@runtime_checkable
class VisionProvider(Protocol):
    def describe(self, image_path: str) -> Optional[str]: ...
    def describe_video(self, video_path: str, num_frames: int = 5) -> Optional[str]: ...
    def is_available(self) -> bool: ...


@runtime_checkable
class AlertSender(Protocol):
    def send_alert(self, event_type, track_id, severity, reason, recommendation,
                   description, snapshot_path, video_path,
                   camera_id="", timestamp="") -> bool: ...
    def is_configured(self) -> bool: ...

# Circuit Breaker — state tracker wrapping any provider


class CircuitBreaker:
    """
    Three states:
      CLOSED   — healthy, requests flow through
      OPEN     — tripped after N failures, requests skip for recovery_timeout
      HALF_OPEN — cooldown expired, next request is a probe

    The CB does NOT proxy calls. The registry checks cb.is_available() before
    routing and the caller reports record_success() / record_failure() after.
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(self, name: str, failure_threshold: int = 3,
                 recovery_timeout: float = 60.0):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout

        self._state = self.CLOSED
        self._failure_count = 0
        self._opened_at = 0.0
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        with self._lock:
            if self._state == self.OPEN:
                if time.time() - self._opened_at >= self.recovery_timeout:
                    self._state = self.HALF_OPEN
            return self._state

    def is_available(self) -> bool:
        """True if the circuit is closed or half-open (probe allowed)."""
        return self.state != self.OPEN

    def record_success(self):
        with self._lock:
            self._failure_count = 0
            self._state = self.CLOSED

    def record_failure(self):
        with self._lock:
            self._failure_count += 1
            if self._failure_count >= self.failure_threshold:
                self._state = self.OPEN
                self._opened_at = time.time()
                logger.warning(
                    f"Circuit breaker OPEN for '{self.name}' "
                    f"after {self._failure_count} failures "
                    f"(recovery in {self.recovery_timeout}s)"
                )

    def status(self) -> dict:
        return {
            "state": self.state,
            "failure_count": self._failure_count,
        }


# ---------------------------------------------------------------------------
# Provider Registry
# ---------------------------------------------------------------------------

class ProviderRegistry:
    """
    Manages text, vision, and alert providers with priority-based selection
    and circuit breaker health tracking.
    """

    def __init__(self):
        # Each list holds (name, provider, priority, circuit_breaker) tuples
        self._text: list[tuple[str, object, int, CircuitBreaker]] = []
        self._vision: list[tuple[str, object, int, CircuitBreaker]] = []
        self._alert: list[tuple[str, object, int, CircuitBreaker]] = []

    def register_text(self, name: str, provider, priority: int = 10, **cb_kwargs):
        cb = CircuitBreaker(name, **cb_kwargs)
        self._text.append((name, provider, priority, cb))
        self._text.sort(key=lambda x: x[2])
        logger.info(f"  Registry: text provider '{name}' (priority={priority})")

    def register_vision(self, name: str, provider, priority: int = 10, **cb_kwargs):
        cb = CircuitBreaker(name, **cb_kwargs)
        self._vision.append((name, provider, priority, cb))
        self._vision.sort(key=lambda x: x[2])
        logger.info(f"  Registry: vision provider '{name}' (priority={priority})")

    def register_alert(self, name: str, sender, priority: int = 10, **cb_kwargs):
        cb = CircuitBreaker(name, **cb_kwargs)
        self._alert.append((name, sender, priority, cb))
        self._alert.sort(key=lambda x: x[2])
        logger.info(f"  Registry: alert sender '{name}' (priority={priority})")

    def get_text(self, skip: Optional[set] = None) -> Optional[tuple]:
        """First available text provider by priority + health.

        Args:
            skip: set of provider names to skip (for fallback retries)

        Returns:
            (provider, circuit_breaker) or None
        """
        skip = skip or set()
        for name, provider, _pri, cb in self._text:
            if name in skip:
                continue
            if cb.is_available() and provider.is_healthy():
                return (provider, cb)
        return None

    def get_vision(self, skip: Optional[set] = None) -> Optional[tuple]:
        """First available vision provider by priority + health.

        Args:
            skip: set of provider names to skip (for fallback retries)

        Returns:
            (provider, circuit_breaker) or None
        """
        skip = skip or set()
        for name, provider, _pri, cb in self._vision:
            if name in skip:
                continue
            if cb.is_available() and provider.is_available():
                return (provider, cb)
        return None

    def get_all_alerts(self) -> list[tuple]:
        """All available alert senders (for fan-out).

        Returns:
            list of (sender, circuit_breaker)
        """
        return [
            (sender, cb)
            for _name, sender, _pri, cb in self._alert
            if cb.is_available() and sender.is_configured()
        ]

    def status(self) -> dict:
        """Health of all providers (for /health endpoint)."""
        def _entries(providers, health_fn):
            return {
                name: {
                    "priority": pri,
                    "healthy": getattr(provider, health_fn)(),
                    **cb.status(),
                }
                for name, provider, pri, cb in providers
            }

        return {
            "text": _entries(self._text, "is_healthy"),
            "vision": _entries(self._vision, "is_available"),
            "alert": _entries(self._alert, "is_configured"),
        }


# ---------------------------------------------------------------------------
# Factory — builds registry from existing config (backward compatible)
# ---------------------------------------------------------------------------

def build_registry(config) -> ProviderRegistry:
    """
    Build a ProviderRegistry from StangWatchConfig.

    Reads config.agent.text_provider / vision_provider and secrets
    to register providers with appropriate priorities.

    Priority logic:
      - Provider matching config preference gets priority=1
      - Other available cloud providers get priority=10
      - Ollama always gets priority=100
    """
    from agent.ollama_client import OllamaClient
    from agent.openrouter_client import OpenRouterTextClient, OpenRouterVisionClient
    from agent.groq_text_client import GroqTextClient
    from agent.vision import VisionDescriber, GeminiVisionDescriber, GroqVisionDescriber
    from agent.telegram import TelegramSender

    registry = ProviderRegistry()
    secrets = config.secrets
    agent = config.agent
    timeout = agent.timeout_seconds

    # --- Text providers ---

    preferred_text = agent.text_provider

    # OpenRouter text
    if secrets.openrouter_api_key:
        pri = 1 if preferred_text == "openrouter" else 10
        registry.register_text("openrouter", OpenRouterTextClient(
            api_key=secrets.openrouter_api_key,
            timeout=timeout,
        ), priority=pri)

    # Groq text
    if secrets.groq_api_key:
        pri = 1 if preferred_text == "groq" else 10
        registry.register_text("groq", GroqTextClient(
            api_key=secrets.groq_api_key,
            timeout=timeout,
        ), priority=pri)

    # Ollama text (always last resort)
    registry.register_text("ollama", OllamaClient(
        model=agent.model,
        host=agent.ollama_host,
        timeout=timeout,
    ), priority=100)

    # --- Vision providers ---

    preferred_vision = agent.vision_provider

    # OpenRouter vision
    if secrets.openrouter_api_key:
        pri = 1 if preferred_vision == "openrouter" else 10
        registry.register_vision("openrouter", OpenRouterVisionClient(
            api_key=secrets.openrouter_api_key,
            timeout=timeout,
        ), priority=pri)

    # Groq vision
    if secrets.groq_api_key:
        pri = 1 if preferred_vision == "groq" else 10
        registry.register_vision("groq", GroqVisionDescriber(
            api_key=secrets.groq_api_key,
            timeout=timeout,
        ), priority=pri)

    # Gemini vision
    if secrets.gemini_api_key:
        pri = 1 if preferred_vision == "gemini" else 10
        registry.register_vision("gemini", GeminiVisionDescriber(
            api_key=secrets.gemini_api_key,
            timeout=timeout,
        ), priority=pri)

    # Ollama vision (always last resort)
    registry.register_vision("ollama", VisionDescriber(
        model=agent.vision_model,
        host=agent.ollama_host,
        timeout=timeout,
    ), priority=100)

    # --- Alert senders ---

    # Telegram (only alert sender for now)
    telegram = TelegramSender(
        bot_token=secrets.telegram_bot_token,
        chat_id=secrets.telegram_chat_id,
    )
    registry.register_alert("telegram", telegram, priority=1)

    return registry

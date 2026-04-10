"""
Alert provider registry — dispatches alerts to all enabled providers.
"""

import logging
from typing import Optional

from agent.alert_providers.base import AlertProvider

logger = logging.getLogger(__name__)


class AlertProviderRegistry:
    """
    Holds a collection of AlertProvider instances and dispatches alerts to
    all that are configured and enabled.
    """

    def __init__(self, providers: list[AlertProvider]):
        self._providers: dict[str, AlertProvider] = {p.name: p for p in providers}

    def get(self, name: str) -> Optional[AlertProvider]:
        """Fetch a provider by name (e.g. 'telegram')."""
        return self._providers.get(name)

    def all(self) -> list[AlertProvider]:
        """Return every registered provider, configured or not."""
        return list(self._providers.values())

    def enabled(self) -> list[AlertProvider]:
        """Return only providers whose is_configured() is True."""
        return [p for p in self._providers.values() if p.is_configured()]

    def broadcast_alert(self, alert_data: dict) -> dict[str, bool]:
        """
        Send the alert via every enabled provider.

        Returns a dict mapping provider name to success (True/False).
        """
        results: dict[str, bool] = {}
        for provider in self.enabled():
            try:
                ok = provider.send_alert(alert_data)
            except Exception as e:
                logger.error(f"{provider.name} broadcast_alert crashed: {e}")
                ok = False
            results[provider.name] = ok
        return results

    def status(self) -> dict[str, dict]:
        """
        Introspection helper used by health endpoints.

        Returns name -> {configured: bool, supports_feedback: bool}
        """
        return {
            name: {
                "configured": p.is_configured(),
                "supports_feedback": p.supports_feedback(),
            }
            for name, p in self._providers.items()
        }

"""
Telegram alert provider — wraps the existing TelegramSender.

The existing TelegramSender in edge/agent/telegram.py is left untouched.
This wrapper adapts it to the AlertProvider interface so Telegram can be
used alongside other providers (WhatsApp, SMS) via the registry.
"""

import logging

from agent.alert_providers.base import AlertProvider
from agent.telegram import TelegramSender

logger = logging.getLogger(__name__)


class TelegramProvider(AlertProvider):
    """
    AlertProvider that delegates to the existing TelegramSender.

    Keeps the underlying sender available as `self.sender` so escalation
    code that already holds a TelegramSender reference can migrate
    incrementally.
    """

    def __init__(self, bot_token: str = "", chat_id: str = "",
                 sender: TelegramSender | None = None):
        """
        Args:
            bot_token: Telegram bot token (ignored if sender is given)
            chat_id: default chat ID for broadcast alerts
            sender: optional pre-built TelegramSender to wrap (avoids
                    constructing a second instance when main.py already
                    built one)
        """
        if sender is not None:
            self.sender = sender
        else:
            self.sender = TelegramSender(bot_token=bot_token, chat_id=chat_id)

    @property
    def name(self) -> str:
        return "telegram"

    def is_configured(self) -> bool:
        return self.sender.is_configured()

    def supports_feedback(self) -> bool:
        return True

    def send_alert(self, alert_data: dict) -> bool:
        if not self.is_configured():
            return False

        # Accept both the canonical keys used by AlertProvider callers and
        # the legacy names the existing TelegramSender expects.
        description = (
            alert_data.get("description")
            or alert_data.get("visual_description", "")
        )
        video_path = (
            alert_data.get("video_path")
            or alert_data.get("clip_path")
        )

        try:
            result = self.sender.send_alert(
                event_type=alert_data.get("event_type", "event"),
                track_id=alert_data.get("track_id", 0),
                severity=alert_data.get("severity", "medium"),
                reason=alert_data.get("reason", ""),
                recommendation=alert_data.get("recommendation", ""),
                description=description,
                snapshot_path=alert_data.get("snapshot_path"),
                video_path=video_path,
                camera_id=alert_data.get("camera_id", ""),
                timestamp=alert_data.get("timestamp", ""),
            )
            # TelegramSender returns a dict on success, False on failure
            return bool(result)
        except Exception as e:
            logger.error(f"TelegramProvider.send_alert failed: {e}")
            return False

    def send_escalation(self, alert_id: int, alert_data: dict, role: str) -> bool:
        """
        Escalation in the existing codebase is handled by EscalationManager,
        which already owns the TelegramSender directly. This method is a
        convenience for the registry to send a basic text escalation notice
        without going through the full EscalationManager flow.
        """
        if not self.is_configured():
            return False

        try:
            text = (
                f"Escalation #{alert_id} ({role}): "
                f"{alert_data.get('severity', 'medium').upper()} — "
                f"{alert_data.get('reason', 'no reason')}"
            )
            self.sender.send_text(self.sender.chat_id, text)
            return True
        except Exception as e:
            logger.error(f"TelegramProvider.send_escalation failed: {e}")
            return False

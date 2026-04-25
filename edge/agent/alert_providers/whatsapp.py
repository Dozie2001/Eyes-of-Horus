"""
WhatsApp alert provider — stub.

Not implemented yet. Planned integration via WhatsApp Business Cloud API
or Twilio. When ready, this class will mirror TelegramProvider's feedback
support (inline buttons, acknowledge/dismiss callbacks).
"""

import logging

from agent.alert_providers.base import AlertProvider

logger = logging.getLogger(__name__)


class WhatsAppProvider(AlertProvider):
    """WhatsApp provider. Returns False for now."""

    def __init__(self, access_token: str = "", phone_number_id: str = "",
                 recipient: str = ""):
        self.access_token = access_token
        self.phone_number_id = phone_number_id
        self.recipient = recipient

    @property
    def name(self) -> str:
        return "whatsapp"

    def is_configured(self) -> bool:
        return False

    def supports_feedback(self) -> bool:
        return True

    def send_alert(self, alert_data: dict) -> bool:
        logger.debug("WhatsAppProvider.send_alert not implemented")
        return False

    def send_escalation(self, alert_id: int, alert_data: dict, role: str) -> bool:
        logger.debug("WhatsAppProvider.send_escalation not implemented")
        return False

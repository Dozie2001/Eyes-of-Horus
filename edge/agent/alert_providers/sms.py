"""
SMS alert provider — stub.

Not implemented yet. Planned integration via Termii (DND-bypass transactional
route for Nigerian networks). SMS cannot support inline feedback buttons —
`supports_feedback()` stays False.
"""

import logging

from agent.alert_providers.base import AlertProvider

logger = logging.getLogger(__name__)


class SMSProvider(AlertProvider):
    """Stub SMS provider. Returns False from all send methods."""

    def __init__(self, api_key: str = "", sender_id: str = "",
                 recipient: str = ""):
        self.api_key = api_key
        self.sender_id = sender_id
        self.recipient = recipient

    @property
    def name(self) -> str:
        return "sms"

    def is_configured(self) -> bool:
        return False  # stub — never configured

    def supports_feedback(self) -> bool:
        return False  # SMS has no feedback buttons

    def send_alert(self, alert_data: dict) -> bool:
        logger.debug("SMSProvider.send_alert not implemented")
        return False

    def send_escalation(self, alert_id: int, alert_data: dict, role: str) -> bool:
        logger.debug("SMSProvider.send_escalation not implemented")
        return False

"""
Abstract base class for alert providers.

Any alert delivery channel (Telegram, WhatsApp, SMS, USSD, etc.) must
implement this interface. The registry dispatches alerts to all enabled
providers in parallel.
"""

from abc import ABC, abstractmethod


class AlertProvider(ABC):
    """Abstract alert delivery channel."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique provider name used in config and logs (e.g. 'telegram')."""
        ...

    @abstractmethod
    def is_configured(self) -> bool:
        """Return True if credentials/settings are complete and provider can send."""
        ...

    @abstractmethod
    def send_alert(self, alert_data: dict) -> bool:
        """
        Send a single alert message.

        Args:
            alert_data: dict with keys like event_type, track_id, severity,
                        reason, recommendation, snapshot_path, clip_path,
                        camera_id, timestamp, visual_description

        Returns:
            True on success, False on failure (never raises).
        """
        ...

    @abstractmethod
    def send_escalation(self, alert_id: int, alert_data: dict, role: str) -> bool:
        """
        Send an escalation notice to a specific role's members.

        Args:
            alert_id: the escalation row ID (used for Acknowledge / Dismiss
                      callbacks on providers that support feedback)
            alert_data: same shape as send_alert
            role: role name the escalation is targeting (e.g. 'guard', 'admin')

        Returns:
            True on success, False on failure.
        """
        ...

    def supports_feedback(self) -> bool:
        """
        Whether this provider can handle Acknowledge / False Alarm feedback.

        Telegram and WhatsApp support inline buttons. SMS does not.
        Default is False — override in subclasses that support it.
        """
        return False

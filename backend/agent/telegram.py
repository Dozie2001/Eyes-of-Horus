"""
Telegram alert sender for StangWatch.

Uses httpx to call the Telegram Bot API directly (no SDK dependency).
Sends text messages and optional snapshot photos.

Usage:
    from agent.telegram import TelegramSender

    sender = TelegramSender(bot_token="123:ABC", chat_id="456")
    if sender.is_configured():
        sender.send_alert(
            event_type="loitering",
            track_id=5,
            severity="high",
            reason="Person near warehouse door during quiet hours",
            recommendation="Check camera feed",
            snapshot_path="data/events/evt_20250315_022000_track5_loitering.jpg",
        )
"""

import httpx


class TelegramSender:
    """Sends alert messages to Telegram via Bot API."""

    def __init__(self, bot_token="", chat_id=""):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self._base_url = f"https://api.telegram.org/bot{bot_token}"

    def is_configured(self):
        """Check if Telegram credentials are set."""
        return bool(self.bot_token) and bool(self.chat_id)

    def send_alert(self, event_type, track_id, severity, reason,
                   recommendation="", snapshot_path=None):
        """
        Send an alert to Telegram with optional snapshot.

        Args:
            event_type: e.g. "loitering", "appeared"
            track_id: ByteTrack person ID
            severity: "low", "medium", "high"
            reason: AI explanation of why this is suspicious
            recommendation: what the operator should do
            snapshot_path: optional path to event snapshot image

        Returns:
            bool: True if sent successfully
        """
        if not self.is_configured():
            return False

        severity_icon = {"low": "🟡", "medium": "🟠", "high": "🔴"}.get(
            severity, "⚪"
        )

        text = (
            f"{severity_icon} *StangWatch Alert*\n"
            f"\n"
            f"*Event:* {event_type.upper()}\n"
            f"*Severity:* {severity}\n"
            f"*Person:* Track \\#{track_id}\n"
            f"\n"
            f"*Why:* {_escape_markdown(reason)}\n"
        )

        if recommendation:
            text += f"\n*Action:* {_escape_markdown(recommendation)}\n"

        try:
            if snapshot_path:
                return self._send_photo(text, snapshot_path)
            else:
                return self._send_message(text)
        except Exception as e:
            print(f"Telegram send failed: {e}")
            return False

    def _send_message(self, text):
        """Send a text-only message."""
        response = httpx.post(
            f"{self._base_url}/sendMessage",
            json={
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": "Markdown",
            },
            timeout=10,
        )
        if not response.json().get("ok"):
            print(f"Telegram API error: {response.text}")
            return False
        return True

    def _send_photo(self, caption, photo_path):
        """Send a photo with caption."""
        try:
            with open(photo_path, "rb") as f:
                response = httpx.post(
                    f"{self._base_url}/sendPhoto",
                    data={
                        "chat_id": self.chat_id,
                        "caption": caption,
                        "parse_mode": "Markdown",
                    },
                    files={"photo": (photo_path.split("/")[-1], f, "image/jpeg")},
                    timeout=15,
                )
            if not response.json().get("ok"):
                print(f"Telegram API error: {response.text}")
                return False
            return True
        except FileNotFoundError:
            print(f"Snapshot not found: {photo_path}, sending text-only")
            return self._send_message(caption)


def _escape_markdown(text):
    """Escape Markdown special characters for Telegram."""
    for char in ("_", "*", "[", "]", "(", ")", "~", "`", ">", "#", "+",
                 "-", "=", "|", "{", "}", ".", "!"):
        text = text.replace(char, f"\\{char}")
    return text

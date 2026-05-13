"""
WhatsApp Cloud API (Meta Graph) alert sender for Eyes of Horus.

Implements the AlertSender Protocol (agent.registry.AlertSender) so the
existing registry/escalation plumbing can fan out alerts to WhatsApp in
parallel with Telegram. v1 broadcasts to a flat list of E.164 recipients
configured via WHATSAPP_RECIPIENTS in .env.

Outbound messages are sent as `interactive` button messages so the
operator can tap Acknowledge / False Alarm directly in WhatsApp. Reply
button IDs reuse the existing `ack:{alert_id}` / `dismiss:{alert_id}`
payloads, which the webhook router translates into EscalationManager
state changes.

Snapshots and video clips are uploaded to /{phone_number_id}/media first,
then attached as a header on the interactive message via media_id.

Usage:
    from agent.whatsapp import WhatsAppSender

    sender = WhatsAppSender(
        access_token="EAAG...",
        phone_number_id="123456789012345",
        recipients=["+2348012345678"],
        app_secret="...",
    )
    if sender.is_configured():
        sender.send_alert_with_button(
            recipient="+2348012345678",
            event_type="loitering",
            track_id=5,
            severity="high",
            reason="Someone's been near the warehouse door for 4 minutes.",
            snapshot_path="data/events/evt_loiter.jpg",
            ack_callback_data="ack:42",
            dismiss_callback_data="dismiss:42",
        )
"""

import hashlib
import hmac
import logging
import mimetypes
import os

import httpx

from agent.telegram import format_friendly_time

logger = logging.getLogger(__name__)

GRAPH_API_VERSION = "v24.0"
GRAPH_BASE = f"https://graph.facebook.com/{GRAPH_API_VERSION}"

# WhatsApp interactive button title limit is 20 characters.
ACK_BUTTON_TITLE = "Acknowledge"
DISMISS_BUTTON_TITLE = "False Alarm"

# WhatsApp interactive body text limit is 1024 characters.
BODY_TEXT_LIMIT = 1024


class WhatsAppSender:
    """Sends alert messages to WhatsApp via Meta's Cloud API."""

    def __init__(self, access_token="", phone_number_id="", recipients=None,
                 app_secret=""):
        self.access_token = access_token
        self.phone_number_id = phone_number_id
        self.recipients = list(recipients) if recipients else []
        self.app_secret = app_secret

    # --- Public AlertSender Protocol ---

    def is_configured(self):
        """Check if WhatsApp credentials and at least one recipient are set."""
        return bool(self.access_token and self.phone_number_id and self.recipients)

    def send_alert(self, severity, reason,
                   recommendation="", description="", snapshot_path=None,
                   video_path=None, camera_id="", timestamp=""):
        """
        Send a plain alert (no buttons) to every configured recipient.

        Returns True if at least one recipient succeeded.
        """
        if not self.is_configured():
            return False

        body = self._format_body(severity, reason, recommendation, description,
                                 camera_id, timestamp)
        header_media_id, header_kind = self._maybe_upload_media(
            video_path, snapshot_path
        )

        any_ok = False
        for recipient in self.recipients:
            result = self._send_interactive(
                recipient=recipient,
                body_text=body,
                ack_id=None,
                dismiss_id=None,
                header_media_id=header_media_id,
                header_kind=header_kind,
            )
            if result:
                any_ok = True
        return any_ok

    def send_alert_with_button(self, recipient, severity,
                               reason, recommendation="", description="",
                               snapshot_path=None, video_path=None,
                               ack_callback_data="", dismiss_callback_data="",
                               camera_id="", timestamp=""):
        """
        Send a single alert with Acknowledge / False Alarm reply buttons.

        Mirrors TelegramSender.send_alert_with_button but takes a phone
        number instead of a chat_id. Returns
            {"ok": True, "wa_message_id": "..."}
        on success or False on failure.
        """
        if not self.is_configured():
            return False

        body = self._format_body(severity, reason, recommendation, description,
                                 camera_id, timestamp)
        header_media_id, header_kind = self._maybe_upload_media(
            video_path, snapshot_path
        )

        return self._send_interactive(
            recipient=recipient,
            body_text=body,
            ack_id=ack_callback_data or None,
            dismiss_id=dismiss_callback_data or None,
            header_media_id=header_media_id,
            header_kind=header_kind,
        )

    def verify_signature(self, raw_body: bytes, header_value: str) -> bool:
        """
        Validate an inbound webhook payload's X-Hub-Signature-256 header.

        The header format is 'sha256=<hex>'. Returns False if app_secret is
        unset or the signature does not match — callers should reject the
        request in that case.
        """
        if not self.app_secret or not header_value:
            return False
        prefix = "sha256="
        received = header_value[len(prefix):] if header_value.startswith(prefix) else header_value
        expected = hmac.new(
            self.app_secret.encode("utf-8"),
            raw_body,
            hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(expected, received)

    # --- Internal helpers ---

    def _headers_json(self):
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
        }

    def _headers_auth(self):
        return {"Authorization": f"Bearer {self.access_token}"}

    def _format_body(self, severity, reason, recommendation, description,
                     camera_id, timestamp):
        """Build the plain-text body for an interactive message."""
        label = {"low": "Heads Up", "medium": "Alert", "high": "Urgent"}.get(
            severity, "Notice"
        )
        parts = [f"*{label}*", "", reason]
        if description:
            parts.append("")
            parts.append(f"📸 {description}")
        if recommendation:
            parts.append("")
            parts.append(f"💡 {recommendation}")

        footer_bits = []
        if camera_id:
            footer_bits.append(camera_id)
        if timestamp:
            footer_bits.append(format_friendly_time(timestamp))
        if footer_bits:
            parts.append("")
            parts.append("📍 " + " · ".join(footer_bits))

        body = "\n".join(parts)
        if len(body) > BODY_TEXT_LIMIT:
            body = body[: BODY_TEXT_LIMIT - 1] + "…"
        return body

    def _maybe_upload_media(self, video_path, snapshot_path):
        """
        Upload the best available media for the alert, preferring video.

        Returns (media_id, kind) where kind is 'video' or 'image', or
        (None, None) if no media is available or upload fails.
        """
        if video_path and os.path.exists(video_path):
            media_id = self._upload_media(video_path, mime_hint="video/mp4")
            if media_id:
                return media_id, "video"
        if snapshot_path and os.path.exists(snapshot_path):
            media_id = self._upload_media(snapshot_path, mime_hint="image/jpeg")
            if media_id:
                return media_id, "image"
        return None, None

    def _upload_media(self, file_path, mime_hint=None):
        """
        Upload a local file to /{phone_number_id}/media and return media_id.

        Returns the media_id string on success, or None on failure.
        """
        guessed, _ = mimetypes.guess_type(file_path)
        mime = guessed or mime_hint or "application/octet-stream"
        url = f"{GRAPH_BASE}/{self.phone_number_id}/media"
        try:
            with open(file_path, "rb") as f:
                response = httpx.post(
                    url,
                    headers=self._headers_auth(),
                    data={"messaging_product": "whatsapp", "type": mime},
                    files={"file": (os.path.basename(file_path), f, mime)},
                    timeout=60,
                )
            data = response.json()
            media_id = data.get("id")
            if not media_id:
                logger.warning(f"WhatsApp media upload failed: {response.text}")
                return None
            return media_id
        except Exception as e:
            logger.warning(f"WhatsApp media upload error: {e}")
            return None

    def _send_interactive(self, recipient, body_text, ack_id=None,
                          dismiss_id=None, header_media_id=None,
                          header_kind=None):
        """
        Send an interactive button message. If neither ack_id nor dismiss_id
        is provided, falls back to a plain text or media message so callers
        can still deliver something without buttons.
        """
        url = f"{GRAPH_BASE}/{self.phone_number_id}/messages"

        buttons = []
        if ack_id:
            buttons.append({
                "type": "reply",
                "reply": {"id": ack_id, "title": ACK_BUTTON_TITLE},
            })
        if dismiss_id:
            buttons.append({
                "type": "reply",
                "reply": {"id": dismiss_id, "title": DISMISS_BUTTON_TITLE},
            })

        if buttons:
            interactive = {
                "type": "button",
                "body": {"text": body_text},
                "action": {"buttons": buttons},
            }
            if header_media_id and header_kind in ("image", "video"):
                interactive["header"] = {
                    "type": header_kind,
                    header_kind: {"id": header_media_id},
                }
            payload = {
                "messaging_product": "whatsapp",
                "recipient_type": "individual",
                "to": recipient,
                "type": "interactive",
                "interactive": interactive,
            }
        elif header_media_id and header_kind:
            payload = {
                "messaging_product": "whatsapp",
                "recipient_type": "individual",
                "to": recipient,
                "type": header_kind,
                header_kind: {"id": header_media_id, "caption": body_text},
            }
        else:
            payload = {
                "messaging_product": "whatsapp",
                "recipient_type": "individual",
                "to": recipient,
                "type": "text",
                "text": {"body": body_text},
            }

        try:
            response = httpx.post(
                url,
                headers=self._headers_json(),
                json=payload,
                timeout=30,
            )
            data = response.json()
            messages = data.get("messages") or []
            if not messages:
                logger.warning(
                    f"WhatsApp send to {recipient} failed: {response.text}"
                )
                return False
            return {"ok": True, "wa_message_id": messages[0].get("id", "")}
        except Exception as e:
            logger.warning(f"WhatsApp send to {recipient} error: {e}")
            return False

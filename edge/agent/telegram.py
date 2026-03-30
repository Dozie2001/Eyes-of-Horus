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
            reason="Someone's been standing near the warehouse door for about 4 minutes. It's 2AM and they have a bag — worth checking.",
            recommendation="Take a look at the warehouse entrance and see who this is.",
            snapshot_path="data/events/evt_20250315_022000_track5_loitering.jpg",
            camera_id="warehouse-east",
            timestamp="2025-03-15T02:20:00",
        )
"""

import logging
from datetime import datetime

import httpx

logger = logging.getLogger(__name__)


def format_friendly_time(iso_str):
    """
    Convert ISO timestamp to human-friendly format.

    '2026-03-28T14:03:22.123456' -> '28th of March 2026 at around 2:03PM'
    """
    try:
        dt = datetime.fromisoformat(iso_str)
    except (ValueError, TypeError):
        return str(iso_str)

    day = dt.day
    if 11 <= day <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")

    time_str = dt.strftime("%-I:%M%p").replace("AM", "AM").replace("PM", "PM")

    return f"{day}{suffix} of {dt.strftime('%B %Y')} at around {time_str}"


def format_duration_natural(seconds):
    """Convert seconds to natural language. 65 -> 'about a minute', 240 -> 'about 4 minutes'."""
    if seconds is None or seconds <= 0:
        return "a moment"
    seconds = float(seconds)
    if seconds < 10:
        return "a few seconds"
    if seconds < 45:
        return f"about {int(seconds)} seconds"
    if seconds < 90:
        return "about a minute"
    if seconds < 3600:
        mins = round(seconds / 60)
        return f"about {mins} minutes"
    hours = round(seconds / 3600, 1)
    if hours == int(hours):
        hours = int(hours)
    return f"about {hours} hours"


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
                   recommendation="", description="", snapshot_path=None,
                   video_path=None, chat_id=None, reply_markup=None,
                   camera_id="", timestamp=""):
        """
        Send an alert to Telegram with optional video clip or snapshot.

        Prefers video > photo > text-only.

        Args:
            event_type: e.g. "loitering", "appeared"
            track_id: ByteTrack person ID
            severity: "low", "medium", "high"
            reason: AI explanation of why this is suspicious
            recommendation: what the operator should do
            description: visual description from vision LLM
            snapshot_path: optional path to event snapshot image
            video_path: optional path to event video clip (preferred over snapshot)
            chat_id: override recipient (default: self.chat_id)
            reply_markup: optional inline keyboard dict for buttons
            camera_id: camera name for the footer
            timestamp: ISO timestamp for friendly time display

        Returns:
            dict with "ok" bool and optional "message_id" int, or False on failure
        """
        if not self.is_configured():
            return False

        target = chat_id or self.chat_id

        severity_label = {"low": "🟡 Heads Up", "medium": "🟠 Alert", "high": "🔴 Urgent"}.get(
            severity, "⚪ Notice"
        )

        text = f"*{_escape_markdown(severity_label)}*\n\n"
        text += f"{_escape_markdown(reason)}\n"

        if description:
            text += f"\n📸 {_escape_markdown(description)}\n"

        if recommendation:
            text += f"\n💡 {_escape_markdown(recommendation)}\n"

        # Footer: camera + friendly timestamp
        footer_parts = []
        if camera_id:
            footer_parts.append(camera_id)
        if timestamp:
            footer_parts.append(format_friendly_time(timestamp))
        if footer_parts:
            text += f"\n📍 {_escape_markdown(' · '.join(footer_parts))}\n"

        try:
            if video_path:
                return self._send_video(text, video_path, chat_id=target,
                                        reply_markup=reply_markup)
            elif snapshot_path:
                return self._send_photo(text, snapshot_path, chat_id=target,
                                        reply_markup=reply_markup)
            else:
                return self._send_message(text, chat_id=target,
                                          reply_markup=reply_markup)
        except Exception as e:
            logger.warning(f"Telegram send failed: {e}")
            return False

    def send_alert_with_button(self, chat_id, event_type, track_id, severity,
                               reason, recommendation="", description="",
                               snapshot_path=None, video_path=None,
                               ack_callback_data="",
                               dismiss_callback_data="",
                               camera_id="", timestamp=""):
        """
        Send an alert with inline Acknowledge and False Alarm buttons.

        Returns:
            dict with "ok" and "message_id", or False on failure.
        """
        reply_markup = {
            "inline_keyboard": [[
                {"text": "✅ Acknowledge", "callback_data": ack_callback_data},
                {"text": "❌ False Alarm", "callback_data": dismiss_callback_data},
            ]]
        }
        return self.send_alert(
            event_type=event_type,
            track_id=track_id,
            severity=severity,
            reason=reason,
            recommendation=recommendation,
            description=description,
            snapshot_path=snapshot_path,
            video_path=video_path,
            chat_id=chat_id,
            reply_markup=reply_markup,
            camera_id=camera_id,
            timestamp=timestamp,
        )

    def answer_callback(self, callback_query_id, text=""):
        """Answer a callback query (acknowledge the button press in Telegram UI)."""
        try:
            response = httpx.post(
                f"{self._base_url}/answerCallbackQuery",
                json={
                    "callback_query_id": callback_query_id,
                    **({"text": text} if text else {}),
                },
                timeout=30,
            )
            return response.json().get("ok", False)
        except Exception as e:
            logger.warning(f"Telegram answerCallbackQuery failed: {e}")
            return False

    def edit_message(self, chat_id, message_id, new_text):
        """Edit an existing message's text."""
        try:
            response = httpx.post(
                f"{self._base_url}/editMessageText",
                json={
                    "chat_id": chat_id,
                    "message_id": message_id,
                    "text": new_text,
                    "parse_mode": "Markdown",
                },
                timeout=30,
            )
            return response.json().get("ok", False)
        except Exception as e:
            logger.warning(f"Telegram editMessageText failed: {e}")
            return False

    def edit_message_caption(self, chat_id, message_id, new_caption):
        """Edit an existing photo message's caption."""
        try:
            response = httpx.post(
                f"{self._base_url}/editMessageCaption",
                json={
                    "chat_id": chat_id,
                    "message_id": message_id,
                    "caption": new_caption,
                    "parse_mode": "Markdown",
                },
                timeout=30,
            )
            return response.json().get("ok", False)
        except Exception as e:
            logger.warning(f"Telegram editMessageCaption failed: {e}")
            return False

    def send_text(self, chat_id, text):
        """Send a plain text reply to a chat (for bot command responses)."""
        try:
            response = httpx.post(
                f"{self._base_url}/sendMessage",
                json={
                    "chat_id": chat_id,
                    "text": text,
                    "parse_mode": "Markdown",
                },
                timeout=30,
            )
            result = response.json()
            if result.get("ok"):
                return True

            # Markdown parsing failed — retry without parse_mode
            logger.debug(f"Markdown send failed, retrying as plain text")
            response = httpx.post(
                f"{self._base_url}/sendMessage",
                json={
                    "chat_id": chat_id,
                    "text": text,
                },
                timeout=30,
            )
            return response.json().get("ok", False)
        except Exception as e:
            logger.warning(f"Telegram send_text failed: {e}")
            return False

    def get_updates(self, offset=0):
        """
        Poll for updates (callback queries + messages) from Telegram.

        Args:
            offset: pass last update_id + 1 to only get new updates

        Returns:
            list of update dicts, or empty list on failure
        """
        try:
            params = {
                "timeout": 0,  # non-blocking
                "allowed_updates": ["callback_query", "message"],
            }
            if offset:
                params["offset"] = offset
            response = httpx.post(
                f"{self._base_url}/getUpdates",
                json=params,
                timeout=30,
            )
            data = response.json()
            if data.get("ok"):
                return data.get("result", [])
            return []
        except Exception as e:
            logger.warning(f"Telegram getUpdates failed: {e}")
            return []

    def _send_message(self, text, chat_id=None, reply_markup=None):
        """Send a text-only message."""
        target = chat_id or self.chat_id
        payload = {
            "chat_id": target,
            "text": text,
            "parse_mode": "Markdown",
        }
        if reply_markup:
            payload["reply_markup"] = reply_markup

        response = httpx.post(
            f"{self._base_url}/sendMessage",
            json=payload,
            timeout=10,
        )
        data = response.json()
        if not data.get("ok"):
            logger.warning(f"Telegram API error: {response.text}")
            return False
        return {"ok": True, "message_id": data["result"]["message_id"]}

    def _send_photo(self, caption, photo_path, chat_id=None, reply_markup=None):
        """Send a photo with caption."""
        import json as json_mod

        target = chat_id or self.chat_id
        try:
            form_data = {
                "chat_id": target,
                "caption": caption,
                "parse_mode": "Markdown",
            }
            if reply_markup:
                form_data["reply_markup"] = json_mod.dumps(reply_markup)

            with open(photo_path, "rb") as f:
                response = httpx.post(
                    f"{self._base_url}/sendPhoto",
                    data=form_data,
                    files={"photo": (photo_path.split("/")[-1], f, "image/jpeg")},
                    timeout=15,
                )
            data = response.json()
            if not data.get("ok"):
                logger.warning(f"Telegram API error: {response.text}")
                return False
            return {"ok": True, "message_id": data["result"]["message_id"]}
        except FileNotFoundError:
            logger.warning(f"Snapshot not found: {photo_path}, sending text-only")
            return self._send_message(caption, chat_id=target,
                                      reply_markup=reply_markup)


    def _send_video(self, caption, video_path, chat_id=None, reply_markup=None):
        """Send a video clip with caption."""
        import json as json_mod

        target = chat_id or self.chat_id
        try:
            form_data = {
                "chat_id": target,
                "caption": caption,
                "parse_mode": "Markdown",
            }
            if reply_markup:
                form_data["reply_markup"] = json_mod.dumps(reply_markup)

            with open(video_path, "rb") as f:
                response = httpx.post(
                    f"{self._base_url}/sendVideo",
                    data=form_data,
                    files={"video": (video_path.split("/")[-1], f, "video/mp4")},
                    timeout=30,
                )
            data = response.json()
            if not data.get("ok"):
                logger.warning(f"Telegram sendVideo error: {response.text}")
                # Fall back to photo if video send fails
                return self._send_photo(caption, video_path.replace(".mp4", ".jpg"),
                                        chat_id=target, reply_markup=reply_markup)
            return {"ok": True, "message_id": data["result"]["message_id"]}
        except FileNotFoundError:
            logger.warning(f"Video not found: {video_path}, trying snapshot fallback")
            snapshot = video_path.replace(".mp4", ".jpg")
            return self._send_photo(caption, snapshot, chat_id=target,
                                    reply_markup=reply_markup)


def _escape_markdown(text):
    """Escape Markdown special characters for Telegram MarkdownV1.

    V1 only uses: *bold*, _italic_, `code`, [link](url).
    Only these four characters need escaping in body text.
    """
    for char in ("_", "*", "`", "["):
        text = text.replace(char, f"\\{char}")
    return text

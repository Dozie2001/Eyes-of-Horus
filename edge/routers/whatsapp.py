"""
WhatsApp Cloud API webhook endpoints.

Two routes:
- GET  /whatsapp/webhook — Meta's verification handshake. Returns the
  hub.challenge value when hub.mode=subscribe and hub.verify_token matches
  the configured WHATSAPP_VERIFY_TOKEN.
- POST /whatsapp/webhook — Inbound message events. It handles
  interactive button replies whose IDs follow the existing
  `ack:{alert_id}` / `dismiss:{alert_id}` convention; other event shapes
  (status callbacks, text messages, templates) are intentionally ignored.

Signature validation uses HMAC-SHA256 over the raw request body with the
WhatsApp app secret. Always returns 200 from POST on success so Meta does
not retry; auth failures return 401/403 as appropriate.
"""

import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, PlainTextResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/whatsapp/webhook")
async def whatsapp_verify(request: Request):
    """Meta verification handshake. Returns hub.challenge as plain text."""
    config = request.app.state.config
    expected_token = config.secrets.whatsapp_verify_token

    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge", "")

    if mode == "subscribe" and token and token == expected_token:
        return PlainTextResponse(challenge, status_code=200)

    return PlainTextResponse("forbidden", status_code=403)


@router.post("/whatsapp/webhook")
async def whatsapp_event(request: Request):
    """Handle inbound WhatsApp events (button replies only in v1)."""
    whatsapp = getattr(request.app.state, "whatsapp", None)
    escalation = getattr(request.app.state, "escalation", None)

    raw_body = await request.body()
    signature = request.headers.get("X-Hub-Signature-256", "")

    if whatsapp is None or not whatsapp.verify_signature(raw_body, signature):
        logger.warning("Rejected WhatsApp webhook with bad signature")
        return JSONResponse({"error": "invalid signature"}, status_code=401)

    if escalation is None:
        return JSONResponse({"status": "ignored"}, status_code=200)

    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"status": "ignored"}, status_code=200)

    for entry in payload.get("entry", []) or []:
        for change in entry.get("changes", []) or []:
            value = change.get("value", {}) or {}
            contacts = value.get("contacts", []) or []
            for message in value.get("messages", []) or []:
                if message.get("type") != "interactive":
                    continue
                interactive = message.get("interactive") or {}
                if interactive.get("type") != "button_reply":
                    continue
                reply = interactive.get("button_reply") or {}
                button_id = reply.get("id", "")
                if ":" not in button_id:
                    continue

                action, _, alert_id_str = button_id.partition(":")
                try:
                    alert_id = int(alert_id_str)
                except ValueError:
                    continue
                if action not in ("ack", "dismiss"):
                    continue

                username = _resolve_username(message, contacts)

                try:
                    if action == "ack":
                        escalation.acknowledge(alert_id, username=username)
                    else:
                        escalation.dismiss(alert_id, username=username)
                except Exception as e:
                    logger.warning(
                        f"WhatsApp webhook {action} on alert #{alert_id} failed: {e}"
                    )

    return JSONResponse({"status": "ok"}, status_code=200)


def _resolve_username(message: dict, contacts: list) -> str:
    """Build a human-ish identifier for the WhatsApp sender."""
    sender_number = message.get("from", "") or ""
    for contact in contacts:
        if contact.get("wa_id") == sender_number:
            profile = contact.get("profile") or {}
            name = profile.get("name")
            if name:
                return f"wa:{name}"
            break
    return f"wa:{sender_number}" if sender_number else "wa:unknown"

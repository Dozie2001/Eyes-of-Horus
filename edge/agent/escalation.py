"""
Escalation policy engine for StangWatch.

Routes alerts to the right people based on severity, tracks acknowledgments
via Telegram inline buttons, and escalates to higher roles on timeout.

Also handles Telegram bot commands for role management:
    /invite guard  — admin generates an invite code
    /join SW-A7X2  — anyone redeems an invite code to join
    /revoke @user  — admin removes someone's access
    /whoami        — check your current role
    /members       — admin lists all active members

Usage:
    from agent.escalation import EscalationManager

    manager = EscalationManager(config, telegram, db_path, role_storage)
    manager.start()
    manager.escalate(...)
    manager.stop()
"""

import logging
import threading
import time
from datetime import datetime, timedelta

from agent.decisions import DecisionStorage

logger = logging.getLogger(__name__)

CHAT_SYSTEM_PROMPT = """You are Eyes of Horus, a friendly AI assistant for a CCTV monitoring system.

You chat naturally with security staff via Telegram. You can:
- Answer questions about what cameras see, recent events, and alerts
- Have normal conversations (greetings, small talk, clarifications)
- Explain alerts and give recommendations when asked

How to respond:
- If someone says "hey", "hello", "good morning" etc — reply naturally and briefly. Do NOT dump camera data unless asked.
- If someone asks about cameras, events, or security — use the data provided to answer.
- If someone says "tell me more" or follows up on an alert — give details about the most recent alert.
- Keep responses SHORT. 1-3 sentences for casual chat. A few bullet points max for data queries.
- Do not speculate about criminal intent. Describe facts only.
- Do not repeat the entire event history unless specifically asked for it."""


class EscalationManager:
    """
    Routes alerts through escalation chains and handles acknowledgment.

    Two background threads:
    1. Timeout checker (every 30s) — escalates unacknowledged alerts
    2. Telegram poller (every 3s) — receives button presses + bot commands
    """

    def __init__(self, config, telegram, db_path, role_storage, site_id="default",
                 whatsapp=None):
        """
        Args:
            config: StangWatchConfig (reads config.escalation)
            telegram: TelegramSender instance
            db_path: path to SQLite database
            role_storage: RoleStorage instance for looking up role members
            site_id: site identifier for multi-site support
            whatsapp: optional WhatsAppSender. When configured, every alert
                that escalates via Telegram also broadcasts to every number
                in whatsapp.recipients (It is recipient-list-based, not
                role-aware).
        """
        from agent.escalation_storage import EscalationStorage

        self._esc_config = config.escalation
        self._telegram = telegram
        self._whatsapp = whatsapp
        self._storage = EscalationStorage(db_path, site_id=site_id)
        self._decisions = DecisionStorage(db_path, site_id=site_id)
        self._roles_db = role_storage
        self._scene_memories = {}  # camera_id -> SceneMemory (for /ask, /scene, /compare)
        self._clip_index = None    # ClipIndex (for /find)

        # Build lookup tables from config
        self._roles = {}
        for role in self._esc_config.roles:
            self._roles[role.name] = role

        self._policies = {}  # severity -> EscalationPolicy
        for policy in self._esc_config.policies:
            self._policies[policy.severity] = policy

        # Valid role names (for invite validation)
        self._valid_role_names = set(self._roles.keys())

        # Background threads
        self._running = False
        self._timeout_thread = None
        self._poller_thread = None
        self._update_offset = 0  # Telegram getUpdates offset

    def is_configured(self):
        """Check if escalation is enabled and has at least one policy."""
        return (
            self._esc_config.enabled
            and len(self._roles) > 0
            and len(self._policies) > 0
        )

    @property
    def storage(self):
        """Expose escalation storage for API endpoints."""
        return self._storage

    @property
    def role_storage(self):
        """Expose role storage for API endpoints."""
        return self._roles_db

    def register_scene_memory(self, camera_id, scene_memory):
        """Register a SceneMemory instance for a camera."""
        self._scene_memories[camera_id] = scene_memory

    def register_clip_index(self, clip_index):
        """Register a ClipIndex instance for /find command."""
        self._clip_index = clip_index

    def start(self):
        """Start background threads for timeout checking and Telegram polling."""
        if self._running or not self.is_configured():
            return

        # Flush old Telegram updates so we don't replay stale callbacks
        # from previous server sessions (which would auto-acknowledge alerts)
        self._flush_old_updates()

        self._running = True

        self._timeout_thread = threading.Thread(
            target=self._check_timeouts_loop,
            daemon=True,
            name="escalation-timeouts",
        )
        self._timeout_thread.start()

        self._poller_thread = threading.Thread(
            target=self._poll_updates_loop,
            daemon=True,
            name="escalation-poller",
        )
        self._poller_thread.start()

        logger.info("EscalationManager started (timeout checker + Telegram poller)")

    def _flush_old_updates(self):
        """Skip all pending Telegram updates on startup.

        Without this, restarting the server replays old button presses
        and auto-acknowledges alerts the user didn't actually press.
        """
        try:
            updates = self._telegram.get_updates(offset=0)
            if updates:
                last_id = max(u.get("update_id", 0) for u in updates)
                self._update_offset = last_id + 1
                logger.debug(f"  ESCALATION: Flushed {len(updates)} old Telegram updates")
            else:
                logger.debug("  ESCALATION: No old Telegram updates to flush")
        except Exception as e:
            logger.warning(f"  ESCALATION: Could not flush old updates: {e}")

    def stop(self):
        """Stop background threads."""
        self._running = False
        if self._timeout_thread is not None:
            self._timeout_thread.join(timeout=5)
        if self._poller_thread is not None:
            self._poller_thread.join(timeout=5)
        logger.info("EscalationManager stopped")

    def escalate(self, decision_id, event_type, track_id, severity,
                 reason, recommendation="", description="",
                 snapshot_path=None, video_path=None, camera_id="cam1"):
        """
        Start the escalation chain for an alert.

        Looks up the policy for this severity, sends to ALL active members
        of the first role in the chain, and saves to DB as pending.
        """
        policy = self._policies.get(severity)
        if policy is None:
            logger.debug(f"  ESCALATION: No policy for severity '{severity}', skipping")
            return None

        chain = policy.chain
        if not chain:
            return None

        # Calculate timeout for first step
        first_step = chain[0]
        next_esc_at = None
        if first_step.timeout_minutes > 0:
            next_esc_at = datetime.now() + timedelta(minutes=first_step.timeout_minutes)

        # Save to DB
        alert_id = self._storage.save_alert(
            decision_id=decision_id,
            event_type=event_type,
            track_id=track_id,
            severity=severity,
            max_level=len(chain) - 1,
            snapshot_path=snapshot_path or "",
            video_path=video_path or "",
            next_escalation_at=next_esc_at,
            camera_id=camera_id,
        )

        # Send to all active members of the first role
        role_name = first_step.role
        members = self._roles_db.get_active_members(role_name)
        if not members:
            logger.debug(f"  ESCALATION: No active members for role '{role_name}'")
            return alert_id

        msg_ids = {}
        for member in members:
            chat_id = member["telegram_chat_id"]
            result = self._telegram.send_alert_with_button(
                chat_id=chat_id,
                event_type=event_type,
                track_id=track_id,
                severity=severity,
                reason=reason,
                recommendation=recommendation,
                description=description,
                snapshot_path=snapshot_path,
                video_path=video_path,
                ack_callback_data=f"ack:{alert_id}",
                dismiss_callback_data=f"dismiss:{alert_id}",
                camera_id=camera_id,
            )
            if result and isinstance(result, dict) and result.get("message_id"):
                msg_ids[chat_id] = result["message_id"]

        # Save message IDs for later editing on acknowledgment
        if msg_ids:
            self._storage.update_message_ids(alert_id, msg_ids)

        # WhatsApp broadcast (flat recipient list, no role routing).
        # Reuses the same ack:/dismiss: payload IDs so the webhook can call
        # acknowledge() / dismiss() on this manager without branching.
        if self._whatsapp is not None and self._whatsapp.is_configured():
            wa_sent = 0
            for recipient in self._whatsapp.recipients:
                try:
                    result = self._whatsapp.send_alert_with_button(
                        recipient=recipient,
                        event_type=event_type,
                        track_id=track_id,
                        severity=severity,
                        reason=reason,
                        recommendation=recommendation,
                        description=description,
                        snapshot_path=snapshot_path,
                        video_path=video_path,
                        ack_callback_data=f"ack:{alert_id}",
                        dismiss_callback_data=f"dismiss:{alert_id}",
                        camera_id=camera_id,
                    )
                    if result:
                        wa_sent += 1
                except Exception as e:
                    logger.warning(f"  ESCALATION: WhatsApp send to {recipient} failed: {e}")
            if wa_sent:
                logger.debug(f"  ESCALATION: Alert #{alert_id} also sent to {wa_sent} WhatsApp recipient(s)")

        timeout_str = f", escalates in {first_step.timeout_minutes}m" if first_step.timeout_minutes > 0 else ""
        logger.debug(f"  ESCALATION: Alert #{alert_id} sent to {len(members)} {role_name}(s) ({severity}){timeout_str}")

        return alert_id

    def acknowledge(self, alert_id, username="api"):
        """
        Manually acknowledge an alert (via API or Telegram button).

        Returns the updated alert dict, or None if not found.
        """
        alert_data = self._storage.mark_acknowledged(alert_id, username=username)
        if alert_data is None:
            return None

        self._edit_messages_acknowledged(alert_data, username)
        logger.debug(f"  ESCALATION: Alert #{alert_id} acknowledged by {username}")
        return alert_data

    def dismiss(self, alert_id, username="api"):
        """
        Manually dismiss an alert as false alarm (via API or Telegram button).

        Returns the updated alert dict, or None if not found.
        """
        alert_data = self._storage.mark_dismissed(alert_id, username=username)
        if alert_data is None:
            return None

        self._edit_messages_dismissed(alert_data, username)
        logger.debug(f"  ESCALATION: Alert #{alert_id} dismissed as false alarm by {username}")
        return alert_data

    # --- Background loops ---

    def _check_timeouts_loop(self):
        """Background: check for expired pending alerts every 30 seconds."""
        while self._running:
            try:
                self._check_timeouts()
            except Exception as e:
                logger.error(f"  ESCALATION: Timeout check error: {e}")
            for _ in range(30):
                if not self._running:
                    return
                time.sleep(1)

    def _check_timeouts(self):
        """Find expired alerts and escalate to next role in chain."""
        expired = self._storage.get_pending_expired()

        for alert_data in expired:
            alert_id = alert_data["id"]
            severity = alert_data["severity"]
            current_level = alert_data["current_level"]

            policy = self._policies.get(severity)
            if policy is None:
                continue

            chain = policy.chain
            next_level = current_level + 1

            if next_level >= len(chain):
                self._storage.advance_level(alert_id, next_escalation_at=None)
                logger.debug(f"  ESCALATION: Alert #{alert_id} reached end of chain")
                continue

            next_step = chain[next_level]
            next_esc_at = None
            if next_step.timeout_minutes > 0:
                next_esc_at = datetime.now() + timedelta(minutes=next_step.timeout_minutes)

            # Send to all active members of the next role
            members = self._roles_db.get_active_members(next_step.role)
            if not members:
                logger.debug(f"  ESCALATION: No active members for role '{next_step.role}', skipping")
                self._storage.advance_level(alert_id, next_escalation_at=next_esc_at)
                continue

            # Look up original AI reason from the decision that triggered this alert
            original_reason = "Unacknowledged alert"
            original_recommendation = "Please acknowledge this alert"
            decision_id = alert_data.get("decision_id")
            if decision_id:
                decision = self._decisions.get_by_id(decision_id)
                if decision:
                    original_reason = decision.get("reason", original_reason)
                    original_recommendation = decision.get("recommendation", original_recommendation)

            # Build escalated reason: keep the original natural language,
            # add a short note about why it was escalated
            prev_role = chain[current_level].role if current_level < len(chain) else "previous role"
            escalated_reason = (
                f"{original_reason}\n\n"
                f"⬆ Escalated to {next_step.role} — no response from {prev_role}."
            )

            new_msg_ids = {}
            for member in members:
                chat_id = member["telegram_chat_id"]
                result = self._telegram.send_alert_with_button(
                    chat_id=chat_id,
                    event_type=alert_data["event_type"],
                    track_id=alert_data["track_id"],
                    severity=severity,
                    reason=escalated_reason,
                    recommendation=original_recommendation,
                    snapshot_path=alert_data.get("snapshot_path") or None,
                    video_path=alert_data.get("video_path") or None,
                    ack_callback_data=f"ack:{alert_id}",
                    dismiss_callback_data=f"dismiss:{alert_id}",
                    camera_id=alert_data.get("camera_id", ""),
                )
                if result and isinstance(result, dict) and result.get("message_id"):
                    new_msg_ids[chat_id] = result["message_id"]

            self._storage.advance_level(
                alert_id,
                next_escalation_at=next_esc_at,
                new_message_ids=new_msg_ids,
            )

            timeout_str = f", escalates in {next_step.timeout_minutes}m" if next_step.timeout_minutes > 0 else ""
            logger.debug(f"  ESCALATION: Alert #{alert_id} escalated to {len(members)} {next_step.role}(s){timeout_str}")

    # --- Telegram polling (callbacks + commands) ---

    def _poll_updates_loop(self):
        """Background: poll Telegram for callbacks and commands every 3 seconds."""
        while self._running:
            try:
                self._poll_updates()
            except Exception as e:
                logger.error(f"  ESCALATION: Poll error: {e}")
            for _ in range(3):
                if not self._running:
                    return
                time.sleep(1)

    def _poll_updates(self):
        """Process callback queries and text commands from Telegram."""
        if not self._telegram.is_configured():
            return

        updates = self._telegram.get_updates(offset=self._update_offset)

        for update in updates:
            update_id = update.get("update_id", 0)
            if update_id >= self._update_offset:
                self._update_offset = update_id + 1

            # Handle callback queries (button presses)
            callback = update.get("callback_query")
            if callback is not None:
                self._handle_callback(callback)
                continue

            message = update.get("message")
            if message is not None:
                text = message.get("text", "")
                if text.startswith("/"):
                    self._handle_command(message)
                elif text.strip():
                    self._handle_chat(message)

    def _handle_callback(self, callback):
        """Process an acknowledge or dismiss button press."""
        callback_id = callback.get("id")
        data = callback.get("data", "")
        user = callback.get("from", {})
        username = user.get("username") or user.get("first_name", "unknown")

        # Parse action:alert_id format
        if ":" not in data:
            return
        action, alert_id_str = data.split(":", 1)
        if action not in ("ack", "dismiss"):
            return

        try:
            alert_id = int(alert_id_str)
        except ValueError:
            return

        alert_data = self._storage.get_by_id(alert_id)
        if alert_data is None:
            self._telegram.answer_callback(callback_id, text="Alert not found")
            return
        if alert_data["status"] != "pending":
            self._telegram.answer_callback(
                callback_id,
                text=f"Already {alert_data['status']}",
            )
            return

        if action == "ack":
            updated = self._storage.mark_acknowledged(alert_id, username=username)
            if updated is None:
                self._telegram.answer_callback(callback_id, text="Error acknowledging")
                return
            self._telegram.answer_callback(
                callback_id,
                text=f"Acknowledged by @{username}",
            )
            self._edit_messages_acknowledged(updated, username)
            logger.debug(f"  ESCALATION: Alert #{alert_id} acknowledged by @{username} via button")

        elif action == "dismiss":
            updated = self._storage.mark_dismissed(alert_id, username=username)
            if updated is None:
                self._telegram.answer_callback(callback_id, text="Error dismissing")
                return
            self._telegram.answer_callback(
                callback_id,
                text=f"Dismissed by @{username}",
            )
            self._edit_messages_dismissed(updated, username)
            logger.debug(f"  ESCALATION: Alert #{alert_id} dismissed as false alarm by @{username} via button")

    def _handle_command(self, message):
        """Route a bot command to the appropriate handler."""
        text = message.get("text", "").strip()
        chat_id = str(message.get("chat", {}).get("id", ""))
        user = message.get("from", {})
        username = user.get("username") or user.get("first_name", "unknown")

        parts = text.split(None, 1)
        command = parts[0].lower().split("@")[0]  # strip @botname suffix
        args = parts[1].strip() if len(parts) > 1 else ""

        logger.info(f"Bot command: {command} from {username} ({chat_id})")

        if command == "/start":
            self._cmd_start(chat_id, username)
        elif command == "/help":
            self._cmd_help(chat_id)
        elif command == "/invite":
            self._cmd_invite(chat_id, username, args)
        elif command == "/join":
            self._cmd_join(chat_id, username, args)
        elif command == "/revoke":
            self._cmd_revoke(chat_id, username, args)
        elif command == "/whoami":
            self._cmd_whoami(chat_id)
        elif command == "/members":
            self._cmd_members(chat_id)
        elif command == "/ask":
            self._cmd_ask(chat_id, username, args)
        elif command == "/scene":
            self._cmd_scene(chat_id, username)
        elif command == "/compare":
            self._cmd_compare(chat_id, username, args)
        elif command == "/find":
            self._cmd_find(chat_id, username, args)

    # --- Bot command handlers ---

    def _cmd_help(self, chat_id):
        """Handle /help — show available commands."""
        member = self._roles_db.get_member_by_chat_id(chat_id)
        role = member["role"] if member else "unregistered"

        lines = [
            "Eyes of Horus Bot Commands:",
            "",
            "/whoami - Show your role and permissions",
            "/scene - What all cameras see right now",
            "/ask <question> - Ask the AI anything about the cameras",
            "  Example: /ask is anyone near the gate?",
            "/compare - Compare activity across cameras",
            "/find <description> - Search clips by description",
            "  Example: /find someone with a bag near the gate",
        ]

        if role in ("admin", "owner"):
            lines.extend([
                "",
                "Admin commands:",
                "/invite <role> - Generate invite code (guard/supervisor/admin)",
                "/invite <role> @user - Invite by username",
                "/revoke @user - Remove someone's access",
                "/members - List all active members",
            ])

        lines.extend([
            "",
            "You can also just type a message and the AI will answer.",
            "Example: Who was on camera in the last hour?",
        ])

        self._telegram.send_text(chat_id, "\n".join(lines))

    def _cmd_start(self, chat_id, username):
        """
        Handle /start — activates a pending membership if one exists.

        Flow: admin does /invite guard @user → user sends /start → activated.
        If no pending membership, just show a welcome message.
        """
        # Try to activate a pending membership
        activated = self._roles_db.activate_by_username(username, chat_id)
        if activated is not None:
            self._telegram.send_text(
                chat_id,
                f"Welcome! You've been activated as *{activated['role']}*.\n"
                f"You will now receive alerts based on your role.",
            )
            logger.debug(f"  ROLES: @{username} activated as {activated['role']} via /start")
            return

        # Check if already a member
        member = self._roles_db.get_member_by_chat_id(chat_id)
        if member is not None:
            self._telegram.send_text(
                chat_id,
                f"You're already registered as *{member['role']}*.\n"
                f"Use /whoami for details.",
            )
            return

        # Not a member, no pending invite
        self._telegram.send_text(
            chat_id,
            "Welcome to Eyes of Horus.\n\n"
            "If you have an invite code, send:\n"
            "`/join <code>`\n\n"
            "If an admin invited you by username, "
            "your account has been activated.",
        )

    def _get_role_level(self, chat_id):
        """Get the config-defined level for this user's role, or 0."""
        member = self._roles_db.get_member_by_chat_id(chat_id)
        if member is None:
            return 0
        role_config = self._roles.get(member["role"])
        if role_config is None:
            return 0
        return role_config.level

    def _cmd_invite(self, chat_id, username, args):
        """
        Handle /invite — admin only. Two forms:
            /invite guard          → generates an invite code
            /invite guard @user    → pre-registers @user, activated on /start
        """
        level = self._get_role_level(chat_id)
        if level < 3:  # admin required
            self._telegram.send_text(chat_id, "Only admins can create invite codes.")
            return

        parts = args.strip().split()
        if not parts:
            roles_list = ", ".join(sorted(self._valid_role_names))
            self._telegram.send_text(
                chat_id,
                f"Usage:\n"
                f"  `/invite <role>` — generate a code\n"
                f"  `/invite <role> @username` — invite directly\n\n"
                f"Roles: {roles_list}",
            )
            return

        role_name = parts[0].lower()
        target_username = parts[1] if len(parts) > 1 else None

        if role_name not in self._valid_role_names:
            roles_list = ", ".join(sorted(self._valid_role_names))
            self._telegram.send_text(
                chat_id,
                f"Unknown role '{role_name}'. Available: {roles_list}",
            )
            return

        # Direct invite by username
        if target_username and target_username.startswith("@"):
            result = self._roles_db.add_pending_member(
                role=role_name,
                username=target_username,
                invited_by=chat_id,
            )
            if result is None:
                self._telegram.send_text(
                    chat_id,
                    f"{target_username} is already registered or pending.",
                )
                return

            self._telegram.send_text(
                chat_id,
                f"Invited {target_username} as *{role_name}*.\n\n"
                f"Tell them to open this bot and send `/start` to activate.",
            )
            logger.debug(f"  ROLES: @{username} invited {target_username} as {role_name} (pending /start)")
            return

        # Code-based invite
        invite = self._roles_db.create_invite(role_name, created_by_chat_id=chat_id)
        self._telegram.send_text(
            chat_id,
            f"Invite code for *{role_name}*:\n\n"
            f"`{invite['code']}`\n\n"
            f"Share this with the person. They send:\n"
            f"`/join {invite['code']}`\n\n"
            f"Expires in 24 hours. Single use.",
        )
        logger.debug(f"  ROLES: @{username} created invite {invite['code']} for {role_name}")

    def _cmd_join(self, chat_id, username, args):
        """Handle /join <code> — anyone with a valid code."""
        code = args.strip().upper()
        if not code:
            self._telegram.send_text(
                chat_id,
                "Usage: `/join <code>`\nExample: `/join SW-A7X2`",
            )
            return

        result = self._roles_db.redeem_invite(code, chat_id, username=username)
        if "error" in result:
            self._telegram.send_text(chat_id, f"Could not join: {result['error']}")
            return

        membership = result["membership"]
        self._telegram.send_text(
            chat_id,
            f"Welcome! You are now registered as *{membership['role']}*.\n"
            f"You will receive alerts based on your role.",
        )
        logger.debug(f"  ROLES: @{username} joined as {membership['role']} (code: {code})")

    def _cmd_revoke(self, chat_id, username, args):
        """Handle /revoke <@username> — admin only."""
        level = self._get_role_level(chat_id)
        if level < 3:
            self._telegram.send_text(chat_id, "Only admins can revoke access.")
            return

        target = args.strip()
        if not target:
            self._telegram.send_text(
                chat_id,
                "Usage: `/revoke @username`",
            )
            return

        revoked = self._roles_db.revoke_by_username(target)
        if revoked is None:
            self._telegram.send_text(
                chat_id,
                f"No active member found with username '{target}'.",
            )
            return

        self._telegram.send_text(
            chat_id,
            f"Revoked access for @{revoked['telegram_username']} "
            f"(was {revoked['role']}). They will no longer receive alerts.",
        )
        logger.debug(f"  ROLES: @{username} revoked @{revoked['telegram_username']}")

    def _cmd_whoami(self, chat_id):
        """Handle /whoami — show current role."""
        member = self._roles_db.get_member_by_chat_id(chat_id)
        if member is None:
            self._telegram.send_text(
                chat_id,
                "You are not registered. Ask an admin for an invite code.",
            )
            return

        role_config = self._roles.get(member["role"])
        abilities = ", ".join(role_config.abilities) if role_config else "unknown"
        self._telegram.send_text(
            chat_id,
            f"*Role:* {member['role']}\n"
            f"*Level:* {role_config.level if role_config else '?'}\n"
            f"*Abilities:* {abilities}\n"
            f"*Since:* {member['created_at'][:10]}",
        )

    def _cmd_members(self, chat_id):
        """Handle /members — admin only, list all active members."""
        level = self._get_role_level(chat_id)
        if level < 3:
            self._telegram.send_text(chat_id, "Only admins can list members.")
            return

        members = self._roles_db.get_all_members()
        active = [m for m in members if m["status"] == "active"]

        if not active:
            self._telegram.send_text(chat_id, "No active members.")
            return

        lines = ["*Active members:*\n"]
        for m in active:
            name = f"@{m['telegram_username']}" if m["telegram_username"] else m["telegram_chat_id"]
            lines.append(f"  {m['role']} — {name}")

        self._telegram.send_text(chat_id, "\n".join(lines))

    def _ensure_ai_clients(self):
        """Initialize cloud text + Ollama clients on first use."""
        if hasattr(self, '_cloud_text'):
            return

        from config import get_config
        config = get_config()

        # Cloud text provider (Groq or OpenRouter)
        self._cloud_text = None
        tp = config.agent.text_provider
        if tp == "groq" and config.secrets.groq_api_key:
            from agent.groq_text_client import GroqTextClient
            self._cloud_text = GroqTextClient(
                api_key=config.secrets.groq_api_key,
                timeout=config.agent.timeout_seconds,
            )
        elif tp == "openrouter" and config.secrets.openrouter_api_key:
            from agent.openrouter_client import OpenRouterTextClient
            self._cloud_text = OpenRouterTextClient(
                api_key=config.secrets.openrouter_api_key,
                timeout=config.agent.timeout_seconds,
            )

        # Ollama fallback
        from agent.ollama_client import OllamaClient
        self._ollama = OllamaClient(
            model=config.agent.model,
            host=config.agent.ollama_host,
            timeout=config.agent.timeout_seconds,
        )

    # --- Conversational AI ---

    def _handle_chat(self, message):
        """Route natural language messages to the AI assistant."""
        text = message.get("text", "").strip()
        chat_id = str(message.get("chat", {}).get("id", ""))
        user = message.get("from", {})
        username = user.get("username") or user.get("first_name", "unknown")

        member = self._roles_db.get_member_by_chat_id(chat_id)
        if member is None:
            return  # ignore messages from non-members

        # Detect if this is a security-related question or casual chat
        security_keywords = [
            "camera", "alert", "event", "detect", "person", "people",
            "who", "what", "anyone", "somebody", "activity", "movement",
            "gate", "door", "entrance", "zone", "restricted", "track",
            "last", "recent", "today", "tonight", "hour", "ago",
            "tell me more", "explain", "why", "details", "summary",
            "scene", "status", "compare", "quiet hours",
        ]
        is_security_query = any(kw in text.lower() for kw in security_keywords)

        if is_security_query:
            self._telegram.send_text(chat_id, "Checking...")
        else:
            self._telegram.send_text(chat_id, "...")

        from agent.prompts import build_chat_prompt
        import os

        # Only load heavy context for security queries
        scene_context = None
        recent_events = None
        recent_alerts = None
        last_alert = None

        if is_security_query:
            # Gather scene context
            scene_ctx = {}
            for cam_id, memory in self._scene_memories.items():
                try:
                    scene_ctx[cam_id] = memory.get_scene_summary()
                except Exception:
                    pass
            scene_context = scene_ctx if scene_ctx else None

            # Get recent events + alerts
            try:
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                db_path = os.path.join(project_root, "data", "stangwatch.db")

                from events.storage import EventStorage
                event_store = EventStorage(db_path)
                recent_events = event_store.get_recent(limit=10)
                recent_alerts = self._decisions.get_alerts_only(limit=5)
                last_alert = recent_alerts[0] if recent_alerts else None
            except Exception as e:
                logger.debug(f"Could not load events for chat: {e}")

        prompt = build_chat_prompt(
            message=text,
            role=member["role"],
            scene_context=scene_context,
            recent_events=recent_events,
            recent_alerts=recent_alerts,
            last_alert=last_alert,
        )

        try:
            self._ensure_ai_clients()

            # Try cloud provider first, fall back to Ollama
            answer = None
            if self._cloud_text and self._cloud_text.is_healthy():
                answer = self._cloud_text.chat(CHAT_SYSTEM_PROMPT, prompt)
            if answer is None and self._ollama.is_healthy():
                response = self._ollama._client.chat(
                    model=self._ollama.model,
                    messages=[
                        {"role": "system", "content": CHAT_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    options={"temperature": 0.3},
                )
                answer = response.message.content.strip()

            if answer is None:
                self._telegram.send_text(chat_id, "AI is currently unavailable. Try again later.")
                return

            if len(answer) > 3500:
                answer = answer[:3500] + "..."
            self._telegram.send_text(chat_id, answer)

        except Exception as e:
            logger.error(f"Chat failed: {e}")
            self._telegram.send_text(chat_id, "Could not process that right now. Try /scene or /ask instead.")

    # --- Interactive AI commands ---

    def _cmd_ask(self, chat_id, username, question):
        """Handle /ask <question> — anyone with a role can ask about what cameras see."""
        member = self._roles_db.get_member_by_chat_id(chat_id)
        if member is None:
            self._telegram.send_text(chat_id, "You need to be registered to ask questions.")
            return

        if not question:
            self._telegram.send_text(
                chat_id,
                "Usage: `/ask <your question>`\n\nExample: `/ask is anyone near the gate?`",
            )
            return

        self._telegram.send_text(chat_id, "Thinking...")

        from agent.prompts import build_interactive_prompt

        scene_context = {}
        for cam_id, memory in self._scene_memories.items():
            try:
                scene_context[cam_id] = memory.get_scene_summary()
            except Exception:
                pass

        prompt = build_interactive_prompt(
            question=question,
            role=member["role"],
            scene_context=scene_context if scene_context else None,
        )

        try:
            self._ensure_ai_clients()

            ask_system = "You are Eyes of Horus, an AI security camera assistant. Answer questions about what the cameras currently see. Be factual and concise. Do not speculate about intent."

            # Try cloud provider first, fall back to Ollama
            answer = None
            if self._cloud_text and self._cloud_text.is_healthy():
                answer = self._cloud_text.chat(ask_system, prompt)
            if answer is None and self._ollama.is_healthy():
                response = self._ollama._client.chat(
                    model=self._ollama.model,
                    messages=[
                        {"role": "system", "content": ask_system},
                        {"role": "user", "content": prompt},
                    ],
                    options={"temperature": 0.3},
                )
                answer = response.message.content.strip()

            if answer is None:
                self._telegram.send_text(chat_id, "AI is currently unavailable. Try again later.")
                return

            if len(answer) > 3500:
                answer = answer[:3500] + "..."
            self._telegram.send_text(chat_id, answer)

        except Exception as e:
            logger.error(f"Ask command failed: {e}")
            self._telegram.send_text(chat_id, "Could not get an answer right now. Try again.")

    def _cmd_scene(self, chat_id, username):
        """Handle /scene — get current scene summary from all cameras."""
        member = self._roles_db.get_member_by_chat_id(chat_id)
        if member is None:
            self._telegram.send_text(chat_id, "You need to be registered.")
            return

        if not self._scene_memories:
            self._telegram.send_text(chat_id, "Scene memory is not available (Redis may be down).")
            return

        lines = ["*Current Scene Report*\n"]

        for cam_id, memory in self._scene_memories.items():
            try:
                summary = memory.get_scene_summary()
                people_count = summary.get("people_count", 0)
                objects = summary.get("objects", [])

                if people_count > 0:
                    lines.append(f"*{cam_id}*: {people_count} person(s)")
                    for p in summary.get("people", []):
                        tid = p.get("track_id", "?")
                        dur = p.get("duration", 0)
                        mvmt = p.get("movement_30f", 0)
                        lines.append(f"  Track #{tid}: {dur}s on camera, movement={mvmt}")
                    if objects:
                        lines.append(f"  Objects: {', '.join(objects)}")
                else:
                    lines.append(f"*{cam_id}*: Empty")
            except Exception:
                lines.append(f"*{cam_id}*: Data unavailable")

        self._telegram.send_text(chat_id, "\n".join(lines))

    def _cmd_compare(self, chat_id, username, args):
        """Handle /compare — compare what different cameras see."""
        member = self._roles_db.get_member_by_chat_id(chat_id)
        if member is None:
            self._telegram.send_text(chat_id, "You need to be registered.")
            return

        if not self._scene_memories:
            self._telegram.send_text(chat_id, "Scene memory is not available.")
            return

        cam_ids = list(self._scene_memories.keys())
        if len(cam_ids) < 2:
            self._telegram.send_text(chat_id, "Need at least 2 cameras to compare.")
            return

        if args:
            requested = [c.strip() for c in args.split(",")]
            cam_ids = [c for c in requested if c in self._scene_memories]
            if len(cam_ids) < 2:
                available = ", ".join(self._scene_memories.keys())
                self._telegram.send_text(chat_id, f"Could not find those cameras. Available: {available}")
                return

        lines = ["*Camera Comparison*\n"]
        total_people = 0

        for cam_id in cam_ids:
            try:
                summary = self._scene_memories[cam_id].get_scene_summary()
                people = summary.get("people_count", 0)
                total_people += people
                objects = summary.get("objects", [])
                obj_str = f", objects: {', '.join(objects)}" if objects else ""
                lines.append(f"*{cam_id}*: {people} person(s){obj_str}")
            except Exception:
                lines.append(f"*{cam_id}*: Data unavailable")

        lines.append(f"\n*Total across cameras:* {total_people} person(s)")
        self._telegram.send_text(chat_id, "\n".join(lines))

    def _cmd_find(self, chat_id, username, args):
        """Handle /find — search clips by natural language description."""
        member = self._roles_db.get_member_by_chat_id(chat_id)
        if member is None:
            self._telegram.send_text(chat_id, "You need to be registered.")
            return

        if not args:
            self._telegram.send_text(
                chat_id,
                "Tell me what you're looking for.\n"
                "Example: /find someone with a bag near the gate"
            )
            return

        if not self._clip_index:
            self._telegram.send_text(chat_id, "Clip search is not configured yet.")
            return

        self._telegram.send_text(chat_id, "Searching...")

        results = self._clip_index.search(query=args, limit=5)

        if not results:
            self._telegram.send_text(chat_id, "No matching clips found.")
            return

        from agent.telegram import format_friendly_time

        for i, r in enumerate(results, 1):
            reason = r.get("reason", "No description")
            severity = r.get("severity", "?")
            camera = r.get("camera_id", "?")
            ts = r.get("edge_created_at", "")
            friendly_time = format_friendly_time(ts) if ts else "unknown time"
            video_url = r.get("video_url", "")
            snapshot_url = r.get("snapshot_url", "")
            score = r.get("similarity", 0)

            text = (
                f"*{i}.* {reason}\n"
                f"Camera: {camera} | {severity}\n"
                f"{friendly_time}\n"
                f"Match: {score:.0%}"
            )

            # Send with video if available, otherwise snapshot, otherwise text
            if video_url:
                self._telegram.send_text(chat_id, f"{text}\n\n[Watch clip]({video_url})")
            elif snapshot_url:
                self._telegram.send_text(chat_id, f"{text}\n\n[View snapshot]({snapshot_url})")
            else:
                self._telegram.send_text(chat_id, text)

    # --- Helpers ---

    def _edit_messages_acknowledged(self, alert_data, username):
        """Edit all Telegram messages for an alert to show acknowledgment."""
        ack_time = datetime.now().strftime("%-I:%M%p").lower()
        ack_text = f"✅ *Acknowledged* by @{username} at {ack_time}"

        message_ids = alert_data.get("telegram_message_ids", {})
        for chat_id, msg_id in message_ids.items():
            ok = self._telegram.edit_message_caption(chat_id, msg_id, ack_text)
            if not ok:
                self._telegram.edit_message(chat_id, msg_id, ack_text)

    def _edit_messages_dismissed(self, alert_data, username):
        """Edit all Telegram messages for an alert to show dismissal."""
        dismiss_time = datetime.now().strftime("%-I:%M%p").lower()
        dismiss_text = f"❌ *Dismissed as false alarm* by @{username} at {dismiss_time}"

        message_ids = alert_data.get("telegram_message_ids", {})
        for chat_id, msg_id in message_ids.items():
            ok = self._telegram.edit_message_caption(chat_id, msg_id, dismiss_text)
            if not ok:
                self._telegram.edit_message(chat_id, msg_id, dismiss_text)

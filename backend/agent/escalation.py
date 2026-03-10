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

import threading
import time
from datetime import datetime, timedelta


class EscalationManager:
    """
    Routes alerts through escalation chains and handles acknowledgment.

    Two background threads:
    1. Timeout checker (every 30s) — escalates unacknowledged alerts
    2. Telegram poller (every 3s) — receives button presses + bot commands
    """

    def __init__(self, config, telegram, db_path, role_storage):
        """
        Args:
            config: StangWatchConfig (reads config.escalation)
            telegram: TelegramSender instance
            db_path: path to SQLite database
            role_storage: RoleStorage instance for looking up role members
        """
        from agent.escalation_storage import EscalationStorage

        self._esc_config = config.escalation
        self._telegram = telegram
        self._storage = EscalationStorage(db_path)
        self._roles_db = role_storage

        # Build lookup tables from config
        self._roles = {}  # name -> EscalationRole (config structure)
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

    def start(self):
        """Start background threads for timeout checking and Telegram polling."""
        if self._running or not self.is_configured():
            return

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

        print("EscalationManager started (timeout checker + Telegram poller)")

    def stop(self):
        """Stop background threads."""
        self._running = False
        if self._timeout_thread is not None:
            self._timeout_thread.join(timeout=5)
        if self._poller_thread is not None:
            self._poller_thread.join(timeout=5)
        print("EscalationManager stopped")

    def escalate(self, decision_id, event_type, track_id, severity,
                 reason, recommendation="", description="",
                 snapshot_path=None, video_path=None):
        """
        Start the escalation chain for an alert.

        Looks up the policy for this severity, sends to ALL active members
        of the first role in the chain, and saves to DB as pending.
        """
        policy = self._policies.get(severity)
        if policy is None:
            print(f"  ESCALATION: No policy for severity '{severity}', skipping")
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
        )

        # Send to all active members of the first role
        role_name = first_step.role
        members = self._roles_db.get_active_members(role_name)
        if not members:
            print(f"  ESCALATION: No active members for role '{role_name}'")
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
            )
            if result and isinstance(result, dict) and result.get("message_id"):
                msg_ids[chat_id] = result["message_id"]

        # Save message IDs for later editing on acknowledgment
        if msg_ids:
            self._storage.update_message_ids(alert_id, msg_ids)

        timeout_str = f", escalates in {first_step.timeout_minutes}m" if first_step.timeout_minutes > 0 else ""
        print(f"  ESCALATION: Alert #{alert_id} sent to {len(members)} {role_name}(s) ({severity}){timeout_str}")

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
        print(f"  ESCALATION: Alert #{alert_id} acknowledged by {username}")
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
        print(f"  ESCALATION: Alert #{alert_id} dismissed as false alarm by {username}")
        return alert_data

    # --- Background loops ---

    def _check_timeouts_loop(self):
        """Background: check for expired pending alerts every 30 seconds."""
        while self._running:
            try:
                self._check_timeouts()
            except Exception as e:
                print(f"  ESCALATION: Timeout check error: {e}")
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
                print(f"  ESCALATION: Alert #{alert_id} reached end of chain")
                continue

            next_step = chain[next_level]
            next_esc_at = None
            if next_step.timeout_minutes > 0:
                next_esc_at = datetime.now() + timedelta(minutes=next_step.timeout_minutes)

            # Send to all active members of the next role
            members = self._roles_db.get_active_members(next_step.role)
            if not members:
                print(f"  ESCALATION: No active members for role '{next_step.role}', skipping")
                self._storage.advance_level(alert_id, next_escalation_at=next_esc_at)
                continue

            new_msg_ids = {}
            for member in members:
                chat_id = member["telegram_chat_id"]
                result = self._telegram.send_alert_with_button(
                    chat_id=chat_id,
                    event_type=alert_data["event_type"],
                    track_id=alert_data["track_id"],
                    severity=severity,
                    reason=f"[ESCALATED] Unacknowledged alert — escalated to {next_step.role}",
                    recommendation="Please acknowledge this alert",
                    snapshot_path=alert_data.get("snapshot_path") or None,
                    video_path=alert_data.get("video_path") or None,
                    ack_callback_data=f"ack:{alert_id}",
                    dismiss_callback_data=f"dismiss:{alert_id}",
                )
                if result and isinstance(result, dict) and result.get("message_id"):
                    new_msg_ids[chat_id] = result["message_id"]

            self._storage.advance_level(
                alert_id,
                next_escalation_at=next_esc_at,
                new_message_ids=new_msg_ids,
            )

            timeout_str = f", escalates in {next_step.timeout_minutes}m" if next_step.timeout_minutes > 0 else ""
            print(f"  ESCALATION: Alert #{alert_id} escalated to {len(members)} {next_step.role}(s){timeout_str}")

    # --- Telegram polling (callbacks + commands) ---

    def _poll_updates_loop(self):
        """Background: poll Telegram for callbacks and commands every 3 seconds."""
        while self._running:
            try:
                self._poll_updates()
            except Exception as e:
                print(f"  ESCALATION: Poll error: {e}")
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

            # Handle text messages (bot commands)
            message = update.get("message")
            if message is not None:
                text = message.get("text", "")
                if text.startswith("/"):
                    self._handle_command(message)

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
            print(f"  ESCALATION: Alert #{alert_id} acknowledged by @{username} via button")

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
            print(f"  ESCALATION: Alert #{alert_id} dismissed as false alarm by @{username} via button")

    def _handle_command(self, message):
        """Route a bot command to the appropriate handler."""
        text = message.get("text", "").strip()
        chat_id = str(message.get("chat", {}).get("id", ""))
        user = message.get("from", {})
        username = user.get("username") or user.get("first_name", "unknown")

        parts = text.split(None, 1)
        command = parts[0].lower().split("@")[0]  # strip @botname suffix
        args = parts[1].strip() if len(parts) > 1 else ""

        if command == "/start":
            self._cmd_start(chat_id, username)
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

    # --- Bot command handlers ---

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
            print(f"  ROLES: @{username} activated as {activated['role']} via /start")
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
            "Welcome to StangWatch.\n\n"
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
            print(f"  ROLES: @{username} invited {target_username} as {role_name} (pending /start)")
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
        print(f"  ROLES: @{username} created invite {invite['code']} for {role_name}")

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
        print(f"  ROLES: @{username} joined as {membership['role']} (code: {code})")

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
        print(f"  ROLES: @{username} revoked @{revoked['telegram_username']}")

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

    # --- Helpers ---

    def _edit_messages_acknowledged(self, alert_data, username):
        """Edit all Telegram messages for an alert to show acknowledgment."""
        ack_time = datetime.now().strftime("%H:%M")
        ack_text = f"✅ *Acknowledged* by @{username} at {ack_time}"

        message_ids = alert_data.get("telegram_message_ids", {})
        for chat_id, msg_id in message_ids.items():
            ok = self._telegram.edit_message_caption(chat_id, msg_id, ack_text)
            if not ok:
                self._telegram.edit_message(chat_id, msg_id, ack_text)

    def _edit_messages_dismissed(self, alert_data, username):
        """Edit all Telegram messages for an alert to show dismissal."""
        dismiss_time = datetime.now().strftime("%H:%M")
        dismiss_text = f"❌ *Dismissed as false alarm* by @{username} at {dismiss_time}"

        message_ids = alert_data.get("telegram_message_ids", {})
        for chat_id, msg_id in message_ids.items():
            ok = self._telegram.edit_message_caption(chat_id, msg_id, dismiss_text)
            if not ok:
                self._telegram.edit_message(chat_id, msg_id, dismiss_text)

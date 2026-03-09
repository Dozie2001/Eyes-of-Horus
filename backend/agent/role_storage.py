"""
Role membership and invite code storage for StangWatch.
Manages who is assigned to each escalation role (guard, supervisor, admin)
and handles invite codes for self-registration via the Telegram bot.
Tables:
- role_membership: who is in what role (active/revoked)
- invite_code: short-lived codes for joining via /join command
Uses the same stangwatch.db database as other storage classes.
"""

import os
import secrets
import string
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import event as sa_event
from sqlmodel import SQLModel, Field, Session, create_engine, select


class RoleMembership(SQLModel, table=True):
    """One person assigned to one role."""
    __tablename__ = "role_membership"

    id: int | None = Field(default=None, primary_key=True)
    role: str = Field(index=True)               # guard, supervisor, admin
    telegram_chat_id: str = Field(index=True)
    telegram_username: str = ""
    invited_by: str = ""
    status: str = Field(default="active", index=True)  # active | pending_start | revoked
    created_at: datetime = Field(default_factory=datetime.now)
    revoked_at: Optional[datetime] = None


class InviteCode(SQLModel, table=True):
    """A short-lived invite code for joining via Telegram bot."""
    __tablename__ = "invite_code"

    id: int | None = Field(default=None, primary_key=True)
    code: str = Field(index=True, unique=True)
    role: str                                 
    created_by_chat_id: str
    expires_at: datetime
    used_by_chat_id: str = "" 
    used_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.now)


def _generate_code() -> str:
    """Generate a 6-char alphanumeric invite code like 'SW-A7X2'."""
    chars = string.ascii_uppercase + string.digits
    suffix = "".join(secrets.choice(chars) for _ in range(4))
    return f"SW-{suffix}"


class RoleStorage:
    """Manages role memberships and invite codes in SQLite."""

    def __init__(self, db_path="data/stangwatch.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        self.engine = create_engine(
            f"sqlite:///{db_path}",
            echo=False,
        )

        @sa_event.listens_for(self.engine, "connect")
        def set_sqlite_wal(dbapi_conn, connection_record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.close()

        SQLModel.metadata.create_all(self.engine)


    def bootstrap_admin(self, chat_id, username="owner"):
        """
        Register the initial admin if no admin exists yet.

        Called on startup with TELEGRAM_CHAT_ID from .env.
        Idempotent — does nothing if an active admin already exists.

        Returns:
            True if a new admin was created, False if one already exists.
        """
        with Session(self.engine) as session:
            existing = session.exec(
                select(RoleMembership)
                .where(RoleMembership.role == "admin")
                .where(RoleMembership.status == "active")
            ).first()
            if existing is not None:
                return False

            member = RoleMembership(
                role="admin",
                telegram_chat_id=chat_id,
                telegram_username=username,
                invited_by="system",
                status="active",
            )
            session.add(member)
            session.commit()
            return True

    # --- Membership ---

    def get_active_members(self, role):
        """Get all active members for a role."""
        with Session(self.engine) as session:
            members = session.exec(
                select(RoleMembership)
                .where(RoleMembership.role == role)
                .where(RoleMembership.status == "active")
            ).all()
            return [self._member_to_dict(m) for m in members]

    def get_all_members(self):
        """Get all members (active and revoked), newest first."""
        with Session(self.engine) as session:
            members = session.exec(
                select(RoleMembership)
                .order_by(RoleMembership.created_at.desc())
            ).all()
            return [self._member_to_dict(m) for m in members]

    def get_member_by_chat_id(self, chat_id):
        """Get the active membership for a chat_id (if any)."""
        with Session(self.engine) as session:
            member = session.exec(
                select(RoleMembership)
                .where(RoleMembership.telegram_chat_id == str(chat_id))
                .where(RoleMembership.status == "active")
            ).first()
            if member is None:
                return None
            return self._member_to_dict(member)

    def add_member(self, role, chat_id, username="", invited_by=""):
        """
        Add a new active member.

        Returns:
            dict of the new membership, or None if already an active member.
        """
        with Session(self.engine) as session:
            # Check for existing active membership
            existing = session.exec(
                select(RoleMembership)
                .where(RoleMembership.telegram_chat_id == str(chat_id))
                .where(RoleMembership.status == "active")
            ).first()
            if existing is not None:
                return None  # already a member

            member = RoleMembership(
                role=role,
                telegram_chat_id=str(chat_id),
                telegram_username=username,
                invited_by=str(invited_by),
                status="active",
            )
            session.add(member)
            session.commit()
            session.refresh(member)
            return self._member_to_dict(member)

    def add_pending_member(self, role, username, invited_by=""):
        """
        Pre-register a member by username (no chat_id yet).

        Status is 'pending_start' — activated when they /start the bot.

        Returns:
            dict of the pending membership, or None if already active/pending.
        """
        username = username.lstrip("@")
        with Session(self.engine) as session:
            # Check for existing active or pending membership
            existing = session.exec(
                select(RoleMembership)
                .where(RoleMembership.telegram_username == username)
                .where(RoleMembership.status.in_(["active", "pending_start"]))
            ).first()
            if existing is not None:
                return None

            member = RoleMembership(
                role=role,
                telegram_chat_id="",  # filled in when they /start
                telegram_username=username,
                invited_by=str(invited_by),
                status="pending_start",
            )
            session.add(member)
            session.commit()
            session.refresh(member)
            return self._member_to_dict(member)

    def activate_by_username(self, username, chat_id):
        """
        Activate a pending_start membership when the user sends /start.

        Fills in their chat_id and flips status to active.

        Returns:
            dict of the activated membership, or None if no pending membership found.
        """
        username = username.lstrip("@")
        with Session(self.engine) as session:
            member = session.exec(
                select(RoleMembership)
                .where(RoleMembership.telegram_username == username)
                .where(RoleMembership.status == "pending_start")
            ).first()
            if member is None:
                return None

            member.telegram_chat_id = str(chat_id)
            member.status = "active"
            session.add(member)
            session.commit()
            session.refresh(member)
            return self._member_to_dict(member)

    def revoke_member(self, member_id):
        """
        Revoke a membership by ID.

        Returns:
            dict of the revoked membership, or None if not found.
        """
        with Session(self.engine) as session:
            member = session.get(RoleMembership, member_id)
            if member is None:
                return None
            member.status = "revoked"
            member.revoked_at = datetime.now()
            session.add(member)
            session.commit()
            session.refresh(member)
            return self._member_to_dict(member)

    def revoke_by_username(self, username):
        """
        Revoke an active membership by Telegram username.

        Returns:
            dict of the revoked membership, or None if not found.
        """
        # Strip leading @ if present
        username = username.lstrip("@")
        with Session(self.engine) as session:
            member = session.exec(
                select(RoleMembership)
                .where(RoleMembership.telegram_username == username)
                .where(RoleMembership.status == "active")
            ).first()
            if member is None:
                return None
            member.status = "revoked"
            member.revoked_at = datetime.now()
            session.add(member)
            session.commit()
            session.refresh(member)
            return self._member_to_dict(member)

    # --- Invite codes ---

    def create_invite(self, role, created_by_chat_id, expires_hours=24):
        """
        Create a new invite code for a role.

        Returns:
            dict with the invite code details.
        """
        code = _generate_code()
        expires_at = datetime.now() + timedelta(hours=expires_hours)

        with Session(self.engine) as session:
            invite = InviteCode(
                code=code,
                role=role,
                created_by_chat_id=str(created_by_chat_id),
                expires_at=expires_at,
            )
            session.add(invite)
            session.commit()
            session.refresh(invite)
            return self._invite_to_dict(invite)

    def redeem_invite(self, code, chat_id, username=""):
        """
        Redeem an invite code — adds the user as a member.

        Returns:
            dict with {"membership": ..., "invite": ...} on success,
            or {"error": "reason"} on failure.
        """
        with Session(self.engine) as session:
            invite = session.exec(
                select(InviteCode)
                .where(InviteCode.code == code)
            ).first()

            if invite is None:
                return {"error": "Invalid invite code"}

            if invite.used_by_chat_id:
                return {"error": "Invite code already used"}

            if datetime.now() > invite.expires_at:
                return {"error": "Invite code expired"}

            # Check if already an active member
            existing = session.exec(
                select(RoleMembership)
                .where(RoleMembership.telegram_chat_id == str(chat_id))
                .where(RoleMembership.status == "active")
            ).first()
            if existing is not None:
                return {"error": f"Already registered as {existing.role}"}

            # Mark invite as used
            invite.used_by_chat_id = str(chat_id)
            invite.used_at = datetime.now()
            session.add(invite)

            # Create membership
            member = RoleMembership(
                role=invite.role,
                telegram_chat_id=str(chat_id),
                telegram_username=username,
                invited_by=invite.created_by_chat_id,
                status="active",
            )
            session.add(member)
            session.commit()
            session.refresh(member)
            session.refresh(invite)

            return {
                "membership": self._member_to_dict(member),
                "invite": self._invite_to_dict(invite),
            }

    def get_pending_invites(self):
        """Get all unused, non-expired invite codes."""
        now = datetime.now()
        with Session(self.engine) as session:
            invites = session.exec(
                select(InviteCode)
                .where(InviteCode.used_by_chat_id == "")
                .where(InviteCode.expires_at > now)
                .order_by(InviteCode.created_at.desc())
            ).all()
            return [self._invite_to_dict(i) for i in invites]

    # --- Serialization ---

    def _member_to_dict(self, member):
        return {
            "id": member.id,
            "role": member.role,
            "telegram_chat_id": member.telegram_chat_id,
            "telegram_username": member.telegram_username,
            "invited_by": member.invited_by,
            "status": member.status,
            "created_at": member.created_at.isoformat(),
            "revoked_at": member.revoked_at.isoformat() if member.revoked_at else None,
        }

    def _invite_to_dict(self, invite):
        return {
            "id": invite.id,
            "code": invite.code,
            "role": invite.role,
            "created_by_chat_id": invite.created_by_chat_id,
            "expires_at": invite.expires_at.isoformat(),
            "used_by_chat_id": invite.used_by_chat_id,
            "used_at": invite.used_at.isoformat() if invite.used_at else None,
            "created_at": invite.created_at.isoformat(),
        }

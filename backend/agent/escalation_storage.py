"""
Escalation state storage for StangWatch.

Tracks pending escalation alerts in SQLite so the EscalationManager
can check timeouts, advance levels, and mark acknowledgments.

Uses the same stangwatch.db database as EventStorage and DecisionStorage.
"""

import json
import os
from datetime import datetime
from typing import Optional

from sqlalchemy import Column, event as sa_event
from sqlalchemy.types import Text
from sqlmodel import SQLModel, Field, Session, create_engine, select


class EscalationAlert(SQLModel, table=True):
    """One row per escalated alert."""
    __tablename__ = "escalation_alert"

    id: int | None = Field(default=None, primary_key=True)
    decision_id: int = Field(index=True)        # links to AgentDecision.id
    event_type: str
    track_id: int
    severity: str                                # "medium" or "high"
    current_level: int = 0                       # position in the chain (0, 1, 2...)
    max_level: int = 0                           # total chain length - 1
    status: str = Field(default="pending", index=True)  # pending | acknowledged | expired
    acknowledged_by: str = ""                    # Telegram username
    acknowledged_at: Optional[datetime] = None
    telegram_message_ids: str = ""               # JSON: {"chat_id": message_id}
    snapshot_path: str = ""
    video_path: str = ""
    next_escalation_at: Optional[datetime] = Field(default=None, index=True)
    created_at: datetime = Field(default_factory=datetime.now, index=True)

    def get_message_ids(self) -> dict:
        """Parse telegram_message_ids JSON string."""
        if not self.telegram_message_ids:
            return {}
        return json.loads(self.telegram_message_ids)

    def set_message_ids(self, data: dict):
        """Serialize telegram_message_ids to JSON string."""
        self.telegram_message_ids = json.dumps(data)


class EscalationStorage:
    """Manages SQLite storage for escalation alerts."""

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

    def save_alert(self, decision_id, event_type, track_id, severity,
                   max_level, snapshot_path="", video_path="",
                   next_escalation_at=None):
        """
        Save a new escalation alert as pending.

        Returns:
            int: the alert row ID
        """
        alert = EscalationAlert(
            decision_id=decision_id,
            event_type=event_type,
            track_id=track_id,
            severity=severity,
            current_level=0,
            max_level=max_level,
            status="pending",
            snapshot_path=snapshot_path or "",
            video_path=video_path or "",
            next_escalation_at=next_escalation_at,
        )
        with Session(self.engine) as session:
            session.add(alert)
            session.commit()
            session.refresh(alert)
            return alert.id

    def get_pending_expired(self, now=None):
        """Get pending alerts whose timeout has passed."""
        if now is None:
            now = datetime.now()
        with Session(self.engine) as session:
            alerts = session.exec(
                select(EscalationAlert)
                .where(EscalationAlert.status == "pending")
                .where(EscalationAlert.next_escalation_at != None)
                .where(EscalationAlert.next_escalation_at <= now)
            ).all()
            return [self._to_dict(a) for a in alerts]

    def mark_acknowledged(self, alert_id, username=""):
        """Mark an alert as acknowledged."""
        with Session(self.engine) as session:
            alert = session.get(EscalationAlert, alert_id)
            if alert is None:
                return None
            alert.status = "acknowledged"
            alert.acknowledged_by = username
            alert.acknowledged_at = datetime.now()
            alert.next_escalation_at = None  # cancel further escalation
            session.add(alert)
            session.commit()
            session.refresh(alert)
            return self._to_dict(alert)

    def advance_level(self, alert_id, next_escalation_at=None,
                      new_message_ids=None):
        """Move to next role in chain."""
        with Session(self.engine) as session:
            alert = session.get(EscalationAlert, alert_id)
            if alert is None:
                return None
            alert.current_level += 1
            alert.next_escalation_at = next_escalation_at
            if new_message_ids:
                # Merge new message IDs with existing
                existing = alert.get_message_ids()
                existing.update(new_message_ids)
                alert.set_message_ids(existing)
            # If we've reached max level with no more timeout, mark expired
            if alert.current_level >= alert.max_level and next_escalation_at is None:
                alert.next_escalation_at = None
            session.add(alert)
            session.commit()
            session.refresh(alert)
            return self._to_dict(alert)

    def update_message_ids(self, alert_id, message_ids):
        """Update the telegram_message_ids for an alert."""
        with Session(self.engine) as session:
            alert = session.get(EscalationAlert, alert_id)
            if alert is None:
                return None
            existing = alert.get_message_ids()
            existing.update(message_ids)
            alert.set_message_ids(existing)
            session.add(alert)
            session.commit()

    def get_pending(self, limit=50):
        """Get currently pending (unacknowledged) alerts."""
        with Session(self.engine) as session:
            alerts = session.exec(
                select(EscalationAlert)
                .where(EscalationAlert.status == "pending")
                .order_by(EscalationAlert.created_at.desc())
                .limit(limit)
            ).all()
            return [self._to_dict(a) for a in alerts]

    def get_recent(self, limit=50):
        """Get all escalation alerts, newest first."""
        with Session(self.engine) as session:
            alerts = session.exec(
                select(EscalationAlert)
                .order_by(EscalationAlert.created_at.desc())
                .limit(limit)
            ).all()
            return [self._to_dict(a) for a in alerts]

    def get_by_id(self, alert_id):
        """Get a single alert by ID."""
        with Session(self.engine) as session:
            alert = session.get(EscalationAlert, alert_id)
            if alert is None:
                return None
            return self._to_dict(alert)

    def _to_dict(self, alert):
        """Convert an EscalationAlert to a plain dict."""
        return {
            "id": alert.id,
            "decision_id": alert.decision_id,
            "event_type": alert.event_type,
            "track_id": alert.track_id,
            "severity": alert.severity,
            "current_level": alert.current_level,
            "max_level": alert.max_level,
            "status": alert.status,
            "acknowledged_by": alert.acknowledged_by,
            "acknowledged_at": alert.acknowledged_at.isoformat() if alert.acknowledged_at else None,
            "telegram_message_ids": alert.get_message_ids(),
            "snapshot_path": alert.snapshot_path,
            "video_path": alert.video_path,
            "next_escalation_at": alert.next_escalation_at.isoformat() if alert.next_escalation_at else None,
            "created_at": alert.created_at.isoformat(),
        }

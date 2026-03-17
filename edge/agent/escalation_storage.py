"""
Escalation state storage for StangWatch.

Tracks pending escalation alerts in SQLite so the EscalationManager
can check timeouts, advance levels, and mark acknowledgments.

Uses the same stangwatch.db database as EventStorage and DecisionStorage.
"""

import json
import os
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import Column, event as sa_event, text
from sqlalchemy.types import Text
from sqlmodel import SQLModel, Field, Session, create_engine, select


class EscalationAlert(SQLModel, table=True):
    """One row per escalated alert."""
    __tablename__ = "escalation_alert"

    id: int | None = Field(default=None, primary_key=True)
    camera_id: str = Field(default="cam1", index=True)
    decision_id: int = Field(index=True)        # links to AgentDecision.id
    event_type: str
    track_id: int
    severity: str                                # "medium" or "high"
    current_level: int = 0                       # position in the chain (0, 1, 2...)
    max_level: int = 0                           # total chain length - 1
    status: str = Field(default="pending", index=True)  # pending | acknowledged | expired | dismissed
    outcome: str = Field(default="unresolved")   # unresolved | true_alert | false_alarm
    acknowledged_by: str = ""                    # Telegram username
    acknowledged_at: Optional[datetime] = None
    dismissed_by: str = ""                       # Telegram username (who pressed False Alarm)
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
        self._migrate()

    def _migrate(self):
        """Add columns that don't exist yet (SQLModel create_all won't do this)."""
        with self.engine.connect() as conn:
            existing = {row[1] for row in conn.execute(text("PRAGMA table_info(escalation_alert)"))}
            migrations = [
                ("outcome", "TEXT NOT NULL DEFAULT 'unresolved'"),
                ("dismissed_by", "TEXT NOT NULL DEFAULT ''"),
                ("camera_id", "TEXT NOT NULL DEFAULT 'cam1'"),
            ]
            for col_name, col_def in migrations:
                if col_name not in existing:
                    conn.execute(text(f"ALTER TABLE escalation_alert ADD COLUMN {col_name} {col_def}"))
            conn.commit()

    def save_alert(self, decision_id, event_type, track_id, severity,
                   max_level, snapshot_path="", video_path="",
                   next_escalation_at=None, camera_id="cam1"):
        """
        Save a new escalation alert as pending.

        Returns:
            int: the alert row ID
        """
        alert = EscalationAlert(
            camera_id=camera_id,
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
        """Mark an alert as acknowledged (confirmed real)."""
        with Session(self.engine) as session:
            alert = session.get(EscalationAlert, alert_id)
            if alert is None:
                return None
            alert.status = "acknowledged"
            alert.outcome = "true_alert"
            alert.acknowledged_by = username
            alert.acknowledged_at = datetime.now()
            alert.next_escalation_at = None  # cancel further escalation
            session.add(alert)
            session.commit()
            session.refresh(alert)
            return self._to_dict(alert)

    def mark_dismissed(self, alert_id, username=""):
        """Mark an alert as a false alarm."""
        with Session(self.engine) as session:
            alert = session.get(EscalationAlert, alert_id)
            if alert is None:
                return None
            alert.status = "dismissed"
            alert.outcome = "false_alarm"
            alert.dismissed_by = username
            alert.acknowledged_at = datetime.now()  # reuse as "resolved at" timestamp
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

    def get_pending(self, limit=50, camera_id=None):
        """Get currently pending (unacknowledged) alerts."""
        with Session(self.engine) as session:
            stmt = select(EscalationAlert).where(EscalationAlert.status == "pending")
            if camera_id:
                stmt = stmt.where(EscalationAlert.camera_id == camera_id)
            alerts = session.exec(
                stmt.order_by(EscalationAlert.created_at.desc()).limit(limit)
            ).all()
            return [self._to_dict(a) for a in alerts]

    def get_recent(self, limit=50, camera_id=None):
        """Get all escalation alerts, newest first."""
        with Session(self.engine) as session:
            stmt = select(EscalationAlert)
            if camera_id:
                stmt = stmt.where(EscalationAlert.camera_id == camera_id)
            alerts = session.exec(
                stmt.order_by(EscalationAlert.created_at.desc()).limit(limit)
            ).all()
            return [self._to_dict(a) for a in alerts]

    def get_by_id(self, alert_id):
        """Get a single alert by ID."""
        with Session(self.engine) as session:
            alert = session.get(EscalationAlert, alert_id)
            if alert is None:
                return None
            return self._to_dict(alert)

    def get_outcome_stats(self, days=30, camera_id=None):
        """Get alert outcome statistics for the last N days."""
        cutoff = datetime.now() - timedelta(days=days)
        cam_filter = " AND camera_id = :camera_id" if camera_id else ""
        params = {"cutoff": cutoff.isoformat()}
        if camera_id:
            params["camera_id"] = camera_id
        with self.engine.connect() as conn:
            # Overall totals
            rows = conn.execute(text(
                "SELECT outcome, COUNT(*) as cnt FROM escalation_alert "
                f"WHERE created_at >= :cutoff{cam_filter} GROUP BY outcome"
            ), params).fetchall()

            totals = {"unresolved": 0, "true_alert": 0, "false_alarm": 0}
            for outcome, cnt in rows:
                if outcome in totals:
                    totals[outcome] = cnt
            total = sum(totals.values())

            # By event_type
            et_rows = conn.execute(text(
                "SELECT event_type, outcome, COUNT(*) as cnt FROM escalation_alert "
                f"WHERE created_at >= :cutoff{cam_filter} GROUP BY event_type, outcome"
            ), params).fetchall()

            by_event_type = {}
            for event_type, outcome, cnt in et_rows:
                if event_type not in by_event_type:
                    by_event_type[event_type] = {"total": 0, "true": 0, "false": 0, "unresolved": 0}
                entry = by_event_type[event_type]
                entry["total"] += cnt
                if outcome == "true_alert":
                    entry["true"] += cnt
                elif outcome == "false_alarm":
                    entry["false"] += cnt
                else:
                    entry["unresolved"] += cnt

            for entry in by_event_type.values():
                resolved = entry["true"] + entry["false"]
                entry["fp_rate"] = round(entry["false"] / resolved, 2) if resolved > 0 else 0.0

            # By severity
            sev_rows = conn.execute(text(
                "SELECT severity, outcome, COUNT(*) as cnt FROM escalation_alert "
                f"WHERE created_at >= :cutoff{cam_filter} GROUP BY severity, outcome"
            ), params).fetchall()

            by_severity = {}
            for severity, outcome, cnt in sev_rows:
                if severity not in by_severity:
                    by_severity[severity] = {"total": 0, "true": 0, "false": 0, "unresolved": 0}
                entry = by_severity[severity]
                entry["total"] += cnt
                if outcome == "true_alert":
                    entry["true"] += cnt
                elif outcome == "false_alarm":
                    entry["false"] += cnt
                else:
                    entry["unresolved"] += cnt

            for entry in by_severity.values():
                resolved = entry["true"] + entry["false"]
                entry["fp_rate"] = round(entry["false"] / resolved, 2) if resolved > 0 else 0.0

            resolved_total = totals["true_alert"] + totals["false_alarm"]
            return {
                "period_days": days,
                "total_alerts": total,
                "true_alerts": totals["true_alert"],
                "false_alarms": totals["false_alarm"],
                "unresolved": totals["unresolved"],
                "false_positive_rate": round(totals["false_alarm"] / resolved_total, 2) if resolved_total > 0 else 0.0,
                "by_event_type": by_event_type,
                "by_severity": by_severity,
            }

    def get_decisions_with_outcomes(self, days=30, camera_id=None, limit=200):
        """
        JOIN agent_decision + escalation_alert to return decisions enriched
        with their escalation outcome (true_alert / false_alarm / unresolved).

        Returns list of dicts with decision fields + outcome fields.
        """
        cutoff = datetime.now() - timedelta(days=days)
        cam_filter = " AND d.camera_id = :camera_id" if camera_id else ""
        params = {"cutoff": cutoff.isoformat(), "limit": limit}
        if camera_id:
            params["camera_id"] = camera_id

        query = text(
            "SELECT d.id, d.camera_id, d.event_type, d.track_id, "
            "d.alert, d.severity, d.reason, d.recommendation, "
            "d.eval_duration_ms, d.created_at, "
            "e.id AS escalation_id, e.outcome, e.status AS escalation_status, "
            "e.acknowledged_by, e.dismissed_by "
            "FROM agent_decision d "
            "LEFT JOIN escalation_alert e ON e.decision_id = d.id "
            f"WHERE d.created_at >= :cutoff{cam_filter} "
            "ORDER BY d.created_at DESC "
            "LIMIT :limit"
        )

        with self.engine.connect() as conn:
            rows = conn.execute(query, params).fetchall()
            results = []
            for row in rows:
                results.append({
                    "id": row[0],
                    "camera_id": row[1],
                    "event_type": row[2],
                    "track_id": row[3],
                    "alert": bool(row[4]),
                    "severity": row[5],
                    "reason": row[6],
                    "recommendation": row[7],
                    "eval_duration_ms": row[8],
                    "created_at": row[9],
                    "escalation_id": row[10],
                    "outcome": row[11],
                    "escalation_status": row[12],
                    "acknowledged_by": row[13] or "",
                    "dismissed_by": row[14] or "",
                })
            return results

    def _to_dict(self, alert):
        """Convert an EscalationAlert to a plain dict."""
        return {
            "id": alert.id,
            "camera_id": alert.camera_id,
            "decision_id": alert.decision_id,
            "event_type": alert.event_type,
            "track_id": alert.track_id,
            "severity": alert.severity,
            "current_level": alert.current_level,
            "max_level": alert.max_level,
            "status": alert.status,
            "outcome": alert.outcome,
            "acknowledged_by": alert.acknowledged_by,
            "acknowledged_at": alert.acknowledged_at.isoformat() if alert.acknowledged_at else None,
            "dismissed_by": alert.dismissed_by,
            "telegram_message_ids": alert.get_message_ids(),
            "snapshot_path": alert.snapshot_path,
            "video_path": alert.video_path,
            "next_escalation_at": alert.next_escalation_at.isoformat() if alert.next_escalation_at else None,
            "created_at": alert.created_at.isoformat(),
        }

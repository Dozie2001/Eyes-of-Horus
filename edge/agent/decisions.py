"""
Agent decision storage for StangWatch.

Stores every AI evaluation decision (alert or not) in SQLite for auditability.
Uses the same stangwatch.db database as EventStorage.

Usage:
    from agent.decisions import DecisionStorage

    store = DecisionStorage("data/stangwatch.db")
    store.save_decision(
        event_type="loitering",
        track_id=5,
        alert=True,
        severity="high",
        reason="Person standing near warehouse door during quiet hours",
        recommendation="Check camera feed immediately",
        eval_duration_ms=1450,
    )

    recent = store.get_recent(limit=20)
    alerts = store.get_alerts_only(limit=20)
"""

import os
from datetime import datetime

from sqlalchemy import Column, event as sa_event
from sqlalchemy.types import JSON
from sqlmodel import SQLModel, Field, Session, create_engine, select


class AgentDecision(SQLModel, table=True):
    """One row per AI evaluation decision."""
    __tablename__ = "agent_decision"

    id: int | None = Field(default=None, primary_key=True)
    site_id: str = Field(default="default", index=True)
    camera_id: str = Field(default="cam1", index=True)
    event_type: str = Field(index=True)
    track_id: int = Field(index=True)
    alert: bool = Field(index=True)
    severity: str  # ignore, low, medium, high
    reason: str
    recommendation: str = ""
    eval_duration_ms: int = 0
    created_at: datetime = Field(default_factory=datetime.now, index=True)
    synced_at: datetime | None = None


class DecisionStorage:
    """
    Manages SQLite storage for agent decisions.

    Uses the same database as EventStorage. The AgentDecision table
    gets created automatically if it doesn't exist.
    """

    def __init__(self, db_path="data/stangwatch.db", site_id="default"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.site_id = site_id

        self.engine = create_engine(
            f"sqlite:///{db_path}",
            echo=False,
        )

        # Enable WAL mode for crash safety
        @sa_event.listens_for(self.engine, "connect")
        def set_sqlite_wal(dbapi_conn, connection_record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.close()

        # Create only the AgentDecision table if it doesn't exist
        SQLModel.metadata.create_all(self.engine)
        self._migrate()

    def _migrate(self):
        """Add columns that don't exist yet (SQLModel create_all won't do this)."""
        from sqlalchemy import text

        with self.engine.connect() as conn:
            existing = {row[1] for row in conn.execute(text("PRAGMA table_info(agent_decision)"))}
            migrations = [
                ("site_id", "TEXT NOT NULL DEFAULT 'default'"),
                ("synced_at", "TEXT"),
            ]
            for col_name, col_def in migrations:
                if col_name not in existing:
                    conn.execute(text(f"ALTER TABLE agent_decision ADD COLUMN {col_name} {col_def}"))
            conn.commit()

    def save_decision(self, event_type, track_id, alert, severity, reason,
                      recommendation="", eval_duration_ms=0, camera_id="cam1",
                      site_id=None):
        """
        Save one evaluation decision.

        Returns:
            int: the decision row ID
        """
        decision = AgentDecision(
            site_id=site_id or self.site_id,
            camera_id=camera_id,
            event_type=event_type,
            track_id=track_id,
            alert=alert,
            severity=severity,
            reason=reason,
            recommendation=recommendation,
            eval_duration_ms=eval_duration_ms,
        )
        with Session(self.engine) as session:
            session.add(decision)
            session.commit()
            session.refresh(decision)
            return decision.id

    def get_recent(self, limit=50, camera_id=None):
        """Get the most recent decisions, newest first."""
        with Session(self.engine) as session:
            stmt = select(AgentDecision)
            if camera_id:
                stmt = stmt.where(AgentDecision.camera_id == camera_id)
            decisions = session.exec(
                stmt.order_by(AgentDecision.created_at.desc()).limit(limit)
            ).all()
            return [self._to_dict(d) for d in decisions]

    def get_by_id(self, decision_id):
        """Get a single decision by ID."""
        with Session(self.engine) as session:
            decision = session.get(AgentDecision, decision_id)
            if decision is None:
                return None
            return self._to_dict(decision)

    def get_alerts_only(self, limit=50, camera_id=None):
        """Get only decisions where alert=True."""
        with Session(self.engine) as session:
            stmt = select(AgentDecision).where(AgentDecision.alert == True)
            if camera_id:
                stmt = stmt.where(AgentDecision.camera_id == camera_id)
            decisions = session.exec(
                stmt.order_by(AgentDecision.created_at.desc()).limit(limit)
            ).all()
            return [self._to_dict(d) for d in decisions]

    def _to_dict(self, decision):
        """Convert an AgentDecision to a plain dict."""
        return {
            "id": decision.id,
            "camera_id": decision.camera_id,
            "event_type": decision.event_type,
            "track_id": decision.track_id,
            "alert": decision.alert,
            "severity": decision.severity,
            "reason": decision.reason,
            "recommendation": decision.recommendation,
            "eval_duration_ms": decision.eval_duration_ms,
            "created_at": decision.created_at.isoformat(),
            "site_id": decision.site_id,
            "synced_at": decision.synced_at.isoformat() if decision.synced_at else None,
        }

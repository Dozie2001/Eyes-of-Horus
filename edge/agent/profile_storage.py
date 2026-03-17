"""
Camera profile storage for StangWatch.

Stores per-camera descriptions and schedules in SQLite. The AI agent uses
these profiles to understand what's "normal" for each camera — reducing
false alarms by providing context about expected activity patterns.

Usage:
    from agent.profile_storage import CameraProfileStorage

    store = CameraProfileStorage("data/stangwatch.db")
    store.upsert_profile("front_gate", description="Faces main entrance",
                         schedule={"weekday": {"start": "08:00", "end": "18:00"}})

    profile = store.get_profile("front_gate")
    is_normal = store.is_normal_hours("front_gate", datetime.now())
"""

import json
import os
from datetime import datetime, time

from sqlalchemy import event as sa_event
from sqlmodel import SQLModel, Field, Session, create_engine, select


class CameraProfile(SQLModel, table=True):
    """One row per camera — operator-written description + schedule."""
    __tablename__ = "camera_profile"

    id: int | None = Field(default=None, primary_key=True)
    site_id: str = Field(default="default", index=True)
    camera_id: str = Field(unique=True, index=True)
    description: str = ""
    schedule_json: str = ""  # JSON: {"weekday": {"start": "08:00", "end": "18:00"}, ...}
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class CameraProfileStorage:
    """
    Manages SQLite storage for camera profiles.

    Uses the same database as EventStorage/DecisionStorage.
    """

    def __init__(self, db_path="data/stangwatch.db", site_id="default"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.site_id = site_id

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
        from sqlalchemy import text

        with self.engine.connect() as conn:
            existing = {row[1] for row in conn.execute(text("PRAGMA table_info(camera_profile)"))}
            if "site_id" not in existing:
                conn.execute(text("ALTER TABLE camera_profile ADD COLUMN site_id TEXT NOT NULL DEFAULT 'default'"))
            conn.commit()

    def get_profile(self, camera_id):
        """
        Get profile for a camera.

        Returns:
            dict with camera_id, description, schedule, created_at, updated_at
            or None if no profile exists
        """
        with Session(self.engine) as session:
            stmt = select(CameraProfile).where(
                CameraProfile.camera_id == camera_id
            )
            profile = session.exec(stmt).first()
            if profile is None:
                return None
            return self._to_dict(profile)

    def upsert_profile(self, camera_id, description=None, schedule=None):
        """
        Create or update a camera profile.

        Args:
            camera_id: camera identifier (matches config.yaml)
            description: operator-written description of what the camera sees
            schedule: dict like {"weekday": {"start": "08:00", "end": "18:00"}, ...}

        Returns:
            dict of the saved profile
        """
        with Session(self.engine) as session:
            stmt = select(CameraProfile).where(
                CameraProfile.camera_id == camera_id
            )
            profile = session.exec(stmt).first()

            if profile is None:
                profile = CameraProfile(
                    camera_id=camera_id,
                    description=description or "",
                    schedule_json=json.dumps(schedule) if schedule else "",
                )
                session.add(profile)
            else:
                if description is not None:
                    profile.description = description
                if schedule is not None:
                    profile.schedule_json = json.dumps(schedule)
                profile.updated_at = datetime.now()

            session.commit()
            session.refresh(profile)
            return self._to_dict(profile)

    def get_all_profiles(self):
        """Get all camera profiles."""
        with Session(self.engine) as session:
            profiles = session.exec(
                select(CameraProfile).order_by(CameraProfile.camera_id)
            ).all()
            return [self._to_dict(p) for p in profiles]

    def delete_profile(self, camera_id):
        """
        Delete a camera profile.

        Returns:
            True if deleted, False if not found
        """
        with Session(self.engine) as session:
            stmt = select(CameraProfile).where(
                CameraProfile.camera_id == camera_id
            )
            profile = session.exec(stmt).first()
            if profile is None:
                return False
            session.delete(profile)
            session.commit()
            return True

    def is_normal_hours(self, camera_id, timestamp=None):
        """
        Check if the given timestamp falls within normal hours for a camera.

        Args:
            camera_id: camera identifier
            timestamp: datetime to check (defaults to now)

        Returns:
            True if within normal hours, False if outside, None if no schedule
        """
        with Session(self.engine) as session:
            stmt = select(CameraProfile).where(
                CameraProfile.camera_id == camera_id
            )
            profile = session.exec(stmt).first()
            if profile is None or not profile.schedule_json:
                return None

            schedule = json.loads(profile.schedule_json)
            return self._check_schedule(schedule, timestamp or datetime.now())

    @staticmethod
    def _check_schedule(schedule, timestamp):
        """
        Check if timestamp falls within the schedule.

        Args:
            schedule: {"weekday": {"start": "08:00", "end": "18:00"}, "weekend": {...}}
            timestamp: datetime to check

        Returns:
            True if within hours, False if outside, None if no entry for this day
        """
        day_of_week = timestamp.weekday()  # 0=Mon, 6=Sun
        is_weekend = day_of_week >= 5

        if is_weekend:
            hours = schedule.get("weekend")
        else:
            hours = schedule.get("weekday")

        if hours is None:
            return None

        start_str = hours.get("start")
        end_str = hours.get("end")
        if not start_str or not end_str:
            return None

        start = time.fromisoformat(start_str)
        end = time.fromisoformat(end_str)
        current = timestamp.time()

        if start <= end:
            return start <= current <= end
        else:
            # Overnight range (e.g. 22:00 - 06:00)
            return current >= start or current <= end

    def _to_dict(self, profile):
        """Convert a CameraProfile to a plain dict."""
        schedule = None
        if profile.schedule_json:
            try:
                schedule = json.loads(profile.schedule_json)
            except (json.JSONDecodeError, TypeError):
                schedule = None

        return {
            "site_id": profile.site_id,
            "camera_id": profile.camera_id,
            "description": profile.description,
            "schedule": schedule,
            "created_at": profile.created_at.isoformat(),
            "updated_at": profile.updated_at.isoformat(),
        }

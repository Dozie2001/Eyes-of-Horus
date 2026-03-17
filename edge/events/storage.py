"""
SQLite event storage for StangWatch using SQLModel.

Stores all detection events in a normalized schema:
  - tracks table: one row per tracked person
  - events table: one row per event, linked to a track

Uses WAL mode for crash safety (survives power outages).
SQLModel means these same models work as FastAPI response models later.
Switching to Postgres = change the connection URL, nothing else.

Usage:
    from events.storage import EventStorage
    from events.bus import event_bus

    storage = EventStorage("data/stangwatch.db")
    storage.subscribe(event_bus)  # auto-saves all events

    # Query later
    recent = storage.get_recent(limit=20)
"""

import os
from datetime import datetime

from sqlalchemy import Column, event as sa_event
from sqlalchemy.types import JSON
from sqlmodel import SQLModel, Field, Session, create_engine, select


# --- Models ---

class Track(SQLModel, table=True):
    """One row per tracked person (scoped by camera)."""
    id: int | None = Field(default=None, primary_key=True)
    camera_id: str = Field(default="cam1", index=True)
    bytetrack_id: int = Field(index=True)
    first_seen: datetime
    last_seen: datetime


class Event(SQLModel, table=True):
    """One row per detection event."""
    id: int | None = Field(default=None, primary_key=True)
    camera_id: str = Field(default="cam1", index=True)
    event_type: str = Field(index=True)
    track_id: int = Field(foreign_key="track.id", index=True)
    timestamp: datetime = Field(index=True)
    duration_seconds: float
    bbox: list = Field(sa_column=Column(JSON, nullable=False))
    is_quiet_hours: bool = False
    nearby_objects: list = Field(sa_column=Column(JSON, nullable=False, default=[]))
    snapshot_path: str | None = None
    extra: dict = Field(sa_column=Column(JSON, nullable=False, default={}))
    created_at: datetime = Field(default_factory=datetime.now)


# All event types (must match events/tracker.py)
ALL_EVENT_TYPES = [
    "appeared", "departed", "returned", "companion",
    "objects_changed", "track_summary",
]


# --- Storage class ---

class EventStorage:
    """
    Manages SQLite storage for tracks and events.

    Auto-creates database and tables on init.
    Subscribes to the event bus for automatic persistence.
    Provides query methods for the dashboard API.
    """

    def __init__(self, db_path="data/stangwatch.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

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

        SQLModel.metadata.create_all(self.engine)

    def save_event(self, event_type, event_data, snapshot_path=None):
        """
        Save one event to the database.

        Automatically creates or updates the associated track.

        Args:
            event_type: one of ALL_EVENT_TYPES
            event_data: dict from EventTracker._event_data()
            snapshot_path: optional path to snapshot image

        Returns:
            int: the event row ID
        """
        bytetrack_id = event_data["track_id"]
        camera_id = event_data.get("camera_id", "cam1")
        event_timestamp = datetime.fromisoformat(event_data["timestamp"])
        first_seen = datetime.fromisoformat(event_data["first_seen"])

        with Session(self.engine) as session:
            # Get or create track (scoped by camera_id + bytetrack_id)
            track = session.exec(
                select(Track)
                .where(Track.camera_id == camera_id)
                .where(Track.bytetrack_id == bytetrack_id)
            ).first()

            if track is None:
                track = Track(
                    camera_id=camera_id,
                    bytetrack_id=bytetrack_id,
                    first_seen=first_seen,
                    last_seen=event_timestamp,
                )
                session.add(track)
                session.flush()  # get track.id before creating event
            else:
                track.last_seen = event_timestamp

            # Separate known fields from extra
            known_fields = {
                "track_id", "camera_id", "timestamp", "first_seen",
                "duration_seconds", "bbox", "is_quiet_hours", "nearby_objects",
            }
            extra = {k: v for k, v in event_data.items() if k not in known_fields}

            event = Event(
                camera_id=camera_id,
                event_type=event_type,
                track_id=track.id,
                timestamp=event_timestamp,
                duration_seconds=event_data["duration_seconds"],
                bbox=event_data["bbox"],
                is_quiet_hours=event_data.get("is_quiet_hours", False),
                nearby_objects=event_data.get("nearby_objects", []),
                snapshot_path=snapshot_path,
                extra=extra,
            )
            session.add(event)
            session.commit()
            session.refresh(event)

            return event.id

    def subscribe(self, event_bus):
        """
        Subscribe to all event types on the bus.
        Each event auto-saves to SQLite when it fires.
        """
        for event_type in ALL_EVENT_TYPES:
            def make_handler(et):
                def handler(event_data):
                    self.save_event(et, event_data)
                return handler
            event_bus.on(event_type, make_handler(event_type))

    # --- Query methods ---

    def get_recent(self, limit=50, camera_id=None):
        """Get the most recent events, newest first."""
        with Session(self.engine) as session:
            stmt = select(Event)
            if camera_id:
                stmt = stmt.where(Event.camera_id == camera_id)
            events = session.exec(
                stmt.order_by(Event.timestamp.desc()).limit(limit)
            ).all()
            return [self._event_to_dict(e) for e in events]

    def get_by_type(self, event_type, limit=50, camera_id=None):
        """Get recent events of a specific type."""
        with Session(self.engine) as session:
            stmt = select(Event).where(Event.event_type == event_type)
            if camera_id:
                stmt = stmt.where(Event.camera_id == camera_id)
            events = session.exec(
                stmt.order_by(Event.timestamp.desc()).limit(limit)
            ).all()
            return [self._event_to_dict(e) for e in events]

    def get_by_track(self, bytetrack_id, limit=50, camera_id=None):
        """Get all events for a specific tracked person (by ByteTrack ID)."""
        with Session(self.engine) as session:
            # Look up the track first
            stmt = select(Track).where(Track.bytetrack_id == bytetrack_id)
            if camera_id:
                stmt = stmt.where(Track.camera_id == camera_id)
            track = session.exec(stmt).first()

            if track is None:
                return []

            events = session.exec(
                select(Event)
                .where(Event.track_id == track.id)
                .order_by(Event.timestamp.desc())
                .limit(limit)
            ).all()
            return [self._event_to_dict(e) for e in events]

    def count_by_type(self, camera_id=None):
        """Get event counts grouped by type. Useful for dashboard summary."""
        with Session(self.engine) as session:
            results = {}
            for et in ALL_EVENT_TYPES:
                stmt = select(Event).where(Event.event_type == et)
                if camera_id:
                    stmt = stmt.where(Event.camera_id == camera_id)
                count = len(session.exec(stmt).all())
                if count > 0:
                    results[et] = count
            return results

    def _event_to_dict(self, event):
        """Convert an Event model to a plain dict."""
        return {
            "id": event.id,
            "camera_id": event.camera_id,
            "event_type": event.event_type,
            "track_id": event.track_id,
            "timestamp": event.timestamp.isoformat(),
            "duration_seconds": event.duration_seconds,
            "bbox": event.bbox,
            "is_quiet_hours": event.is_quiet_hours,
            "nearby_objects": event.nearby_objects,
            "snapshot_path": event.snapshot_path,
            "extra": event.extra,
            "created_at": event.created_at.isoformat(),
        }

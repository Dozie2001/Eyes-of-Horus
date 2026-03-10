"""
FastAPI backend for StangWatch.

Serves the event API and runs the detection pipeline in a background thread.
The pipeline writes events to SQLite, the API reads and serves them.

Run:
    cd backend
    uvicorn main:app --reload --port 8000
"""

from contextlib import asynccontextmanager
from pathlib import Path

from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import get_config
from events.storage import EventStorage, ALL_EVENT_TYPES
from agent.decisions import DecisionStorage
from agent.telegram import TelegramSender
from agent.escalation import EscalationManager
from agent.role_storage import RoleStorage
from pipeline.runner import PipelineRunner


# Project root (stang/) — same convention as config module
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize storage, escalation manager, and start detection pipeline."""
    config = get_config()
    db_path = str(_PROJECT_ROOT / config.storage.db_path)

    storage = EventStorage(db_path)
    decisions = DecisionStorage(db_path)
    app.state.storage = storage
    app.state.decisions = decisions
    app.state.config = config

    role_storage = RoleStorage(db_path)
    app.state.role_storage = role_storage

    escalation = None
    if config.escalation.enabled:
        if config.secrets.telegram_chat_id:
            created = role_storage.bootstrap_admin(
                chat_id=config.secrets.telegram_chat_id,
                username="owner",
            )
            if created:
                print(f"Bootstrapped admin from TELEGRAM_CHAT_ID: {config.secrets.telegram_chat_id}")

        telegram = TelegramSender(
            bot_token=config.secrets.telegram_bot_token,
            chat_id=config.secrets.telegram_chat_id,
        )
        escalation = EscalationManager(config, telegram, db_path, role_storage)
        escalation.start()
    app.state.escalation = escalation

    runner = PipelineRunner(config, escalation=escalation)
    app.state.pipeline = runner

    if config.cameras:
        runner.start(storage)

    yield

    runner.stop()
    if escalation is not None:
        escalation.stop()


app = FastAPI(
    title="StangWatch API",
    description="AI CCTV monitoring",
    version="0.2.0",
    lifespan=lifespan,
)

# CORS for Next.js dashboard (local dev + LAN access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["GET", "PUT", "POST", "DELETE"],
    allow_headers=["*"],
)


# --- Endpoints ---

@app.get("/health")
def health():
    config = app.state.config
    storage = app.state.storage
    pipeline = app.state.pipeline
    counts = storage.count_by_type()
    return {
        "status": "ok",
        "site_id": config.site.id,
        "site_name": config.site.name,
        "event_count": sum(counts.values()),
        "pipeline_status": pipeline.status,
    }


@app.get("/pipeline/status")
def pipeline_status():
    """Current state of the detection pipeline."""
    runner = app.state.pipeline
    return {
        "status": runner.status,
        "fps": runner.fps,
        "frame_count": runner.frame_count,
        "active_tracks": runner.active_tracks,
        "error": runner.error,
    }


@app.get("/events")
def get_events(limit: int = Query(default=50, ge=1, le=500)):
    return app.state.storage.get_recent(limit=limit)


@app.get("/events/summary")
def get_events_summary():
    counts = app.state.storage.count_by_type()
    return {
        "counts": counts,
        "total": sum(counts.values()),
    }


@app.get("/events/type/{event_type}")
def get_events_by_type(
    event_type: str,
    limit: int = Query(default=50, ge=1, le=500),
):
    if event_type not in ALL_EVENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid event_type '{event_type}'. Must be one of: {ALL_EVENT_TYPES}",
        )
    return app.state.storage.get_by_type(event_type, limit=limit)


@app.get("/events/track/{bytetrack_id}")
def get_events_by_track(
    bytetrack_id: int,
    limit: int = Query(default=50, ge=1, le=500),
):
    return app.state.storage.get_by_track(bytetrack_id, limit=limit)



@app.get("/agent/decisions")
def get_agent_decisions(limit: int = Query(default=50, ge=1, le=500)):
    """Recent AI agent evaluation decisions (all, including non-alerts)."""
    return app.state.decisions.get_recent(limit=limit)


@app.get("/agent/alerts")
def get_agent_alerts(limit: int = Query(default=50, ge=1, le=500)):
    """Only decisions where the agent flagged an alert."""
    return app.state.decisions.get_alerts_only(limit=limit)


# --- Config endpoints ---

class QuietHoursUpdate(BaseModel):
    start: Optional[str] = None  # "22:00" or null to disable
    end: Optional[str] = None    # "06:00" or null to disable


@app.get("/config/quiet-hours")
def get_quiet_hours():
    """Get current quiet hours setting."""
    runner = app.state.pipeline
    tracker = runner._tracker
    if tracker is None or tracker.quiet_hours is None:
        return {"start": None, "end": None, "active": False}
    return {
        "start": tracker.quiet_hours["start"],
        "end": tracker.quiet_hours["end"],
        "active": True,
    }


@app.put("/config/quiet-hours")
def set_quiet_hours(body: QuietHoursUpdate):
    """Update quiet hours without restarting. Send null to disable."""
    runner = app.state.pipeline
    tracker = runner._tracker
    if tracker is None:
        raise HTTPException(status_code=503, detail="Pipeline not running")

    if body.start is None or body.end is None:
        tracker.quiet_hours = None
        return {"start": None, "end": None, "active": False}

    # Validate time format
    from datetime import time
    try:
        time.fromisoformat(body.start)
        time.fromisoformat(body.end)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid time format. Use HH:MM (e.g. '22:00')",
        )

    tracker.quiet_hours = {"start": body.start, "end": body.end}
    return {
        "start": body.start,
        "end": body.end,
        "active": True,
    }


# --- Role management endpoints ---

@app.get("/roles/members")
def get_role_members():
    """List all role members (active and revoked)."""
    return app.state.role_storage.get_all_members()


@app.get("/roles/members/active")
def get_active_role_members():
    """List active members, grouped by role."""
    role_storage = app.state.role_storage
    config = app.state.config

    result = {}
    for role in config.escalation.roles:
        result[role.name] = role_storage.get_active_members(role.name)
    return result


class InviteRequest(BaseModel):
    role: str


@app.post("/roles/invite")
def create_invite(body: InviteRequest):
    """Create an invite code for a role (admin action)."""
    config = app.state.config
    valid_roles = {r.name for r in config.escalation.roles}
    if body.role not in valid_roles:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid role '{body.role}'. Must be one of: {sorted(valid_roles)}",
        )
    invite = app.state.role_storage.create_invite(
        role=body.role,
        created_by_chat_id="api",
    )
    return invite


@app.get("/roles/invites")
def get_pending_invites():
    """List pending (unused, non-expired) invite codes."""
    return app.state.role_storage.get_pending_invites()


@app.delete("/roles/members/{member_id}")
def revoke_member(member_id: int):
    """Revoke a member's access by membership ID."""
    result = app.state.role_storage.revoke_member(member_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Member #{member_id} not found")
    return result


# --- Escalation endpoints ---

@app.get("/escalation/pending")
def get_escalation_pending(limit: int = Query(default=50, ge=1, le=500)):
    """Currently unacknowledged escalation alerts."""
    escalation = app.state.escalation
    if escalation is None:
        return []
    return escalation.storage.get_pending(limit=limit)


@app.get("/escalation/recent")
def get_escalation_recent(limit: int = Query(default=50, ge=1, le=500)):
    """All escalation alerts (for dashboard), newest first."""
    escalation = app.state.escalation
    if escalation is None:
        return []
    return escalation.storage.get_recent(limit=limit)


@app.put("/escalation/{alert_id}/acknowledge")
def acknowledge_escalation(alert_id: int):
    """Manually acknowledge an escalation alert via API (backup for Telegram button)."""
    escalation = app.state.escalation
    if escalation is None:
        raise HTTPException(status_code=503, detail="Escalation not enabled")

    result = escalation.acknowledge(alert_id, username="api")
    if result is None:
        raise HTTPException(status_code=404, detail=f"Alert #{alert_id} not found")
    return result


@app.put("/escalation/{alert_id}/dismiss")
def dismiss_escalation(alert_id: int):
    """Dismiss an escalation alert as false alarm via API (backup for Telegram button)."""
    escalation = app.state.escalation
    if escalation is None:
        raise HTTPException(status_code=503, detail="Escalation not enabled")

    result = escalation.dismiss(alert_id, username="api")
    if result is None:
        raise HTTPException(status_code=404, detail=f"Alert #{alert_id} not found")
    return result


# --- Metrics endpoints ---

@app.get("/metrics/alert-quality")
def get_alert_quality_metrics(days: int = Query(default=30, ge=1, le=365)):
    """Alert outcome statistics — false positive rates by event type and severity."""
    escalation = app.state.escalation
    if escalation is None:
        raise HTTPException(status_code=503, detail="Escalation not enabled")

    return escalation.storage.get_outcome_stats(days=days)


# --- Static file mount for snapshots ---
_snapshot_dir = _PROJECT_ROOT / "data" / "events"
_snapshot_dir.mkdir(parents=True, exist_ok=True)
app.mount("/snapshots", StaticFiles(directory=str(_snapshot_dir)), name="snapshots")

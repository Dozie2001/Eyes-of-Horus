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
from agent.profile_storage import CameraProfileStorage
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

    profile_storage = CameraProfileStorage(db_path)
    app.state.profile_storage = profile_storage

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

    runners = {}
    enabled_cameras = [c for c in config.cameras if c.enabled]
    for cam_cfg in enabled_cameras:
        runner = PipelineRunner(config, escalation=escalation,
                                camera_config=cam_cfg,
                                profile_storage=profile_storage)
        runner.start(storage)
        runners[cam_cfg.name] = runner

    app.state.runners = runners
    app.state.pipeline = next(iter(runners.values()))

    yield

    for runner in runners.values():
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
    runners = app.state.runners
    counts = storage.count_by_type()
    camera_status = {cam_id: r.status for cam_id, r in runners.items()}
    return {
        "status": "ok",
        "site_id": config.site.id,
        "site_name": config.site.name,
        "event_count": sum(counts.values()),
        "cameras": camera_status,
    }


@app.get("/pipeline/status")
def pipeline_status():
    """Current state of all detection pipelines (one per camera)."""
    runners = app.state.runners
    cameras = {}
    for cam_id, runner in runners.items():
        cameras[cam_id] = {
            "status": runner.status,
            "fps": runner.fps,
            "frame_count": runner.frame_count,
            "active_tracks": runner.active_tracks,
            "error": runner.error,
        }
    return {"cameras": cameras}


@app.get("/events")
def get_events(
    limit: int = Query(default=50, ge=1, le=500),
    camera_id: Optional[str] = Query(default=None),
):
    return app.state.storage.get_recent(limit=limit, camera_id=camera_id)


@app.get("/events/summary")
def get_events_summary(camera_id: Optional[str] = Query(default=None)):
    counts = app.state.storage.count_by_type(camera_id=camera_id)
    return {
        "counts": counts,
        "total": sum(counts.values()),
    }


@app.get("/events/type/{event_type}")
def get_events_by_type(
    event_type: str,
    limit: int = Query(default=50, ge=1, le=500),
    camera_id: Optional[str] = Query(default=None),
):
    if event_type not in ALL_EVENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid event_type '{event_type}'. Must be one of: {ALL_EVENT_TYPES}",
        )
    return app.state.storage.get_by_type(event_type, limit=limit, camera_id=camera_id)


@app.get("/events/track/{bytetrack_id}")
def get_events_by_track(
    bytetrack_id: int,
    limit: int = Query(default=50, ge=1, le=500),
    camera_id: Optional[str] = Query(default=None),
):
    return app.state.storage.get_by_track(bytetrack_id, limit=limit, camera_id=camera_id)



@app.get("/agent/decisions")
def get_agent_decisions(
    limit: int = Query(default=50, ge=1, le=500),
    camera_id: Optional[str] = Query(default=None),
):
    """Recent AI agent evaluation decisions (all, including non-alerts)."""
    return app.state.decisions.get_recent(limit=limit, camera_id=camera_id)


@app.get("/agent/alerts")
def get_agent_alerts(
    limit: int = Query(default=50, ge=1, le=500),
    camera_id: Optional[str] = Query(default=None),
):
    """Only decisions where the agent flagged an alert."""
    return app.state.decisions.get_alerts_only(limit=limit, camera_id=camera_id)


@app.get("/agent/feedback")
def get_agent_feedback(
    days: int = Query(default=30, ge=1, le=365),
    camera_id: Optional[str] = Query(default=None),
    limit: int = Query(default=200, ge=1, le=1000),
):
    """Decisions enriched with escalation outcomes — for feedback analysis."""
    escalation = app.state.escalation
    if escalation is None:
        return []
    return escalation.storage.get_decisions_with_outcomes(
        days=days, camera_id=camera_id, limit=limit,
    )


# --- Config endpoints ---

class QuietHoursUpdate(BaseModel):
    start: Optional[str] = None  # "22:00" or null to disable
    end: Optional[str] = None    # "06:00" or null to disable


@app.get("/config/quiet-hours")
def get_quiet_hours():
    """Get current quiet hours setting (from first running pipeline)."""
    for runner in app.state.runners.values():
        tracker = runner._tracker
        if tracker is not None:
            if tracker.quiet_hours is None:
                return {"start": None, "end": None, "active": False}
            return {
                "start": tracker.quiet_hours["start"],
                "end": tracker.quiet_hours["end"],
                "active": True,
            }
    return {"start": None, "end": None, "active": False}


@app.put("/config/quiet-hours")
def set_quiet_hours(body: QuietHoursUpdate):
    """Update quiet hours on ALL cameras without restarting. Send null to disable."""
    runners = app.state.runners
    any_updated = False

    # Validate time format
    if body.start is not None and body.end is not None:
        from datetime import time
        try:
            time.fromisoformat(body.start)
            time.fromisoformat(body.end)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid time format. Use HH:MM (e.g. '22:00')",
            )

    for runner in runners.values():
        tracker = runner._tracker
        if tracker is None:
            continue
        if body.start is None or body.end is None:
            tracker.quiet_hours = None
        else:
            tracker.quiet_hours = {"start": body.start, "end": body.end}
        any_updated = True

    if not any_updated:
        raise HTTPException(status_code=503, detail="No pipelines running")

    if body.start is None or body.end is None:
        return {"start": None, "end": None, "active": False}
    return {"start": body.start, "end": body.end, "active": True}


# --- Camera profile endpoints ---

class CameraProfileUpdate(BaseModel):
    description: str = ""
    schedule: dict | None = None  # {"weekday": {"start": "08:00", "end": "18:00"}, ...}


@app.get("/cameras/profiles")
def list_camera_profiles():
    """List all camera profiles."""
    return app.state.profile_storage.get_all_profiles()


@app.get("/cameras/profiles/{camera_id}")
def get_camera_profile(camera_id: str):
    """Get profile for a specific camera."""
    profile = app.state.profile_storage.get_profile(camera_id)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"No profile for camera '{camera_id}'")
    return profile


@app.put("/cameras/profiles/{camera_id}")
def upsert_camera_profile(camera_id: str, body: CameraProfileUpdate):
    """Create or update a camera profile."""
    return app.state.profile_storage.upsert_profile(
        camera_id=camera_id,
        description=body.description,
        schedule=body.schedule,
    )


@app.delete("/cameras/profiles/{camera_id}")
def delete_camera_profile(camera_id: str):
    """Delete a camera profile."""
    deleted = app.state.profile_storage.delete_profile(camera_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"No profile for camera '{camera_id}'")
    return {"deleted": camera_id}


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
def get_escalation_pending(
    limit: int = Query(default=50, ge=1, le=500),
    camera_id: Optional[str] = Query(default=None),
):
    """Currently unacknowledged escalation alerts."""
    escalation = app.state.escalation
    if escalation is None:
        return []
    return escalation.storage.get_pending(limit=limit, camera_id=camera_id)


@app.get("/escalation/recent")
def get_escalation_recent(
    limit: int = Query(default=50, ge=1, le=500),
    camera_id: Optional[str] = Query(default=None),
):
    """All escalation alerts (for dashboard), newest first."""
    escalation = app.state.escalation
    if escalation is None:
        return []
    return escalation.storage.get_recent(limit=limit, camera_id=camera_id)


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
def get_alert_quality_metrics(
    days: int = Query(default=30, ge=1, le=365),
    camera_id: Optional[str] = Query(default=None),
):
    """Alert outcome statistics — false positive rates by event type and severity."""
    escalation = app.state.escalation
    if escalation is None:
        raise HTTPException(status_code=503, detail="Escalation not enabled")

    return escalation.storage.get_outcome_stats(days=days, camera_id=camera_id)


# --- Static file mount for snapshots ---
_snapshot_dir = _PROJECT_ROOT / "data" / "events"
_snapshot_dir.mkdir(parents=True, exist_ok=True)
app.mount("/snapshots", StaticFiles(directory=str(_snapshot_dir)), name="snapshots")

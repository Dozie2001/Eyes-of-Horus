"""
FastAPI backend for StangWatch.

Serves the event API and runs the detection pipeline in a background thread.
The pipeline writes events to SQLite, the API reads and serves them.

Run:
    cd edge
    uvicorn main:app --reload --port 8000
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import get_config
from logging_setup import configure_logging
from events.storage import EventStorage, ALL_EVENT_TYPES
from agent.decisions import DecisionStorage
from agent.telegram import TelegramSender
from agent.escalation import EscalationManager
from agent.role_storage import RoleStorage
from agent.profile_storage import CameraProfileStorage
from pipeline.runner import PipelineRunner
from detection.roboflow_segmenter import RoboflowSegmenter


# Project root (stang/) — same convention as config module
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize storage, escalation manager, and start detection pipeline."""
    config = get_config()
    configure_logging(
        log_level=config.site.log_level,
        environment=config.site.environment,
    )
    db_path = str(_PROJECT_ROOT / config.storage.db_path)
    site_id = config.site.id

    storage = EventStorage(db_path, site_id=site_id)
    decisions = DecisionStorage(db_path, site_id=site_id)
    app.state.storage = storage
    app.state.decisions = decisions
    app.state.config = config

    role_storage = RoleStorage(db_path, site_id=site_id)
    app.state.role_storage = role_storage

    profile_storage = CameraProfileStorage(db_path, site_id=site_id)
    app.state.profile_storage = profile_storage

    escalation = None
    if config.escalation.enabled:
        if config.secrets.telegram_chat_id:
            created = role_storage.bootstrap_admin(
                chat_id=config.secrets.telegram_chat_id,
                username="owner",
            )
            if created:
                logger.info(f"Bootstrapped admin from TELEGRAM_CHAT_ID: {config.secrets.telegram_chat_id}")

        telegram = TelegramSender(
            bot_token=config.secrets.telegram_bot_token,
            chat_id=config.secrets.telegram_chat_id,
        )
        escalation = EscalationManager(config, telegram, db_path, role_storage, site_id=site_id)
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

    # SAM3 segmenter (Roboflow — serverless or local GPU Docker)
    segmenter = None
    try:
        seg_cfg = config.roboflow.segmentation
        segmenter = RoboflowSegmenter(
            provider=seg_cfg.provider,
            api_key=config.roboflow.api_key,
            local_url=seg_cfg.local_url,
            serverless_url=seg_cfg.serverless_url,
            confidence_threshold=seg_cfg.confidence_threshold,
            timeout_seconds=seg_cfg.timeout_seconds,
        )
        segmenter.load()
        logger.info(
            f"Roboflow segmenter ready (provider={seg_cfg.provider})"
        )
    except Exception as e:
        logger.warning(f"Roboflow segmenter not available: {e}")
        segmenter = None
    app.state.segmenter = segmenter

    # Clip search index (Supabase pgvector — optional)
    clip_index = None
    if config.supabase.enabled:
        try:
            from agent.clip_index import ClipIndex
            clip_index = ClipIndex(config)
            if clip_index.is_ready():
                logger.info("Clip search index ready")
            else:
                logger.warning("Clip search index not ready (Supabase or embedding provider missing)")
                clip_index = None
        except Exception as e:
            logger.warning(f"Clip search index init failed: {e}")
    app.state.clip_index = clip_index

    # Wire clip index into escalation manager for /find command
    if clip_index and escalation:
        escalation.register_clip_index(clip_index)

    yield

    for runner in runners.values():
        runner.stop()
    if escalation is not None:
        escalation.stop()
    if segmenter is not None:
        segmenter.close()


app = FastAPI(
    title="Eyes of Horus API",
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

    # Provider status from the first runner's eval agent registry
    providers = {}
    for runner in runners.values():
        agent = getattr(runner, "_eval_agent", None)
        if agent is not None:
            providers = agent.registry.status()
            break

    return {
        "status": "ok",
        "site_id": config.site.id,
        "site_name": config.site.name,
        "event_count": sum(counts.values()),
        "cameras": camera_status,
        "providers": providers,
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



@app.get("/cameras/{camera_id}/frame")
def camera_frame(camera_id: str):
    """Single JPEG snapshot from the running pipeline. Used by the zone editor
    to get a stable frame for click-to-segment."""
    from fastapi.responses import Response

    runners = app.state.runners
    runner = runners.get(camera_id)
    if runner is None:
        raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found")

    jpeg_bytes = runner.get_current_frame()
    if jpeg_bytes is None:
        raise HTTPException(status_code=503, detail=f"Camera '{camera_id}' has no frame yet")

    return Response(content=jpeg_bytes, media_type="image/jpeg")



class ZoneCreateRequest(BaseModel):
    name: str
    points: list[list[float]]
    zone_type: str = "polygon"
    severity_override: Optional[str] = None
    active_hours_start: Optional[str] = None
    active_hours_end: Optional[str] = None
    allowed_object_types: Optional[list[str]] = None
    alert_on_entry: bool = False
    alert_on_dwell_seconds: Optional[float] = None


def _refresh_tracker_zones(camera_id: str, zones: list):
    """Push updated zones to the running EventTracker so they take effect
    without restarting the pipeline."""
    runners = getattr(app.state, "runners", {})
    runner = runners.get(camera_id)
    if runner is None:
        return
    tracker = getattr(runner, "_tracker", None)
    if tracker is not None:
        tracker.refresh_zones(zones)


@app.get("/cameras/{camera_id}/zones")
def list_zones(camera_id: str):
    """List all zones for a camera."""
    return app.state.profile_storage.get_zones(camera_id)


@app.post("/cameras/{camera_id}/zones")
def create_zone(camera_id: str, body: ZoneCreateRequest):
    """Create or update a zone. If a zone with the same name exists, it is replaced."""
    zone = body.model_dump(exclude_none=True)
    zones = app.state.profile_storage.add_zone(camera_id, zone)
    _refresh_tracker_zones(camera_id, zones)
    return {"created": body.name, "zone_count": len(zones)}


@app.delete("/cameras/{camera_id}/zones/{zone_name}")
def delete_zone(camera_id: str, zone_name: str):
    """Delete a zone by name."""
    removed, zones = app.state.profile_storage.remove_zone(camera_id, zone_name)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Zone '{zone_name}' not found")
    _refresh_tracker_zones(camera_id, zones)
    return {"deleted": zone_name, "zone_count": len(zones)}


# --- Live camera stream ---

@app.get("/cameras/{camera_id}/stream")
async def camera_stream(camera_id: str):
    """MJPEG stream of the live camera feed.

    Returns a multipart/x-mixed-replace response — the browser (or an <img> tag)
    renders each JPEG frame as it arrives, creating a live video effect.

    Frame rate is capped at ~10fps to keep bandwidth reasonable.
    """
    runners = app.state.runners
    runner = runners.get(camera_id)
    if runner is None:
        raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found")
    if runner.status != "running":
        raise HTTPException(status_code=503, detail=f"Camera '{camera_id}' is {runner.status}")

    async def generate():
        while True:
            jpeg_bytes = runner.get_current_frame()
            if jpeg_bytes is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + jpeg_bytes
                    + b"\r\n"
                )
            await asyncio.sleep(0.1)  # ~10fps

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# --- Clip search ---

class ClipSearchRequest(BaseModel):
    query: str
    limit: int = 10
    camera_id: Optional[str] = None
    after: Optional[str] = None  # ISO date string

@app.post("/clips/search")
def search_clips(req: ClipSearchRequest):
    """Search clips using natural language (powered by Supabase pgvector)."""
    clip_index = getattr(app.state, "clip_index", None)
    if clip_index is None or not clip_index.is_ready():
        raise HTTPException(status_code=503, detail="Clip search not configured")

    results = clip_index.search(
        query=req.query,
        limit=req.limit,
        camera_id=req.camera_id,
        after=req.after,
    )
    return {"query": req.query, "count": len(results), "results": results}


@app.post("/clips/reembed")
def reembed_clips():
    """Re-embed all clip decisions with the current embedding provider.

    Use after switching embedding providers. May take a few minutes for large datasets.
    """
    clip_index = getattr(app.state, "clip_index", None)
    if clip_index is None or not clip_index.is_ready():
        raise HTTPException(status_code=503, detail="Clip search not configured")

    stats = clip_index.reembed_all()
    return stats


# --- Zone segmentation (SAM3 proxy) ---

class ZonePoint(BaseModel):
    x: float
    y: float
    positive: bool = True


class ZoneSegmentRequest(BaseModel):
    """
    Request a SAM3 segmentation to create a zone polygon.

    Supply exactly one image source and exactly one prompt type.

    Image source (choose one):
      - camera_id: grab the latest frame from a running pipeline
      - image_base64: a base64-encoded JPEG/PNG sent by the dashboard

    Prompt (choose one):
      - text: natural language concept (uses /sam3/concept_segment)
      - point: click coordinate (uses /sam3/visual_segment)
    """
    camera_id: Optional[str] = None
    image_base64: Optional[str] = None
    text: Optional[str] = None
    point: Optional[ZonePoint] = None
    confidence: Optional[float] = None


@app.post("/zones/segment")
def zone_segment(req: ZoneSegmentRequest, request: Request):
    """
    Proxy to Roboflow SAM3 for zone creation.

    Returns a list of polygons ready to be saved as a zone. Each polygon is
    a list of [x, y] points in the original image coordinate space.
    """
    # --- validate prompt before touching expensive resources ---
    if req.text and req.point:
        raise HTTPException(
            status_code=400, detail="Provide either text or point, not both"
        )
    if not req.text and not req.point:
        raise HTTPException(
            status_code=400, detail="Provide either text or point"
        )
    if not req.image_base64 and not req.camera_id:
        raise HTTPException(
            status_code=400,
            detail="Provide either camera_id or image_base64"
        )

    segmenter = getattr(request.app.state, "segmenter", None)
    if segmenter is None:
        raise HTTPException(
            status_code=503,
            detail="Segmentation not available. Check ROBOFLOW_API_KEY and "
                   "roboflow.segmentation config."
        )

    # --- resolve image source ---
    import cv2
    import numpy as np
    import base64 as _b64

    frame = None
    if req.image_base64:
        try:
            raw = _b64.b64decode(req.image_base64)
            arr = np.frombuffer(raw, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception:
            raise HTTPException(
                status_code=400,
                detail="image_base64 is not a valid base64-encoded image"
            )
        if frame is None:
            raise HTTPException(
                status_code=400, detail="Could not decode image_base64"
            )
    else:
        runners = getattr(request.app.state, "runners", {})
        runner = runners.get(req.camera_id)
        if runner is None:
            raise HTTPException(
                status_code=404,
                detail=f"Camera '{req.camera_id}' not found"
            )
        current = getattr(runner, "_current_frame", None)
        if current is None:
            raise HTTPException(
                status_code=503,
                detail=f"Camera '{req.camera_id}' has no current frame yet"
            )
        frame = current.copy()

    # --- run segmentation ---
    try:
        if req.text:
            polygons = segmenter.segment_by_text(
                frame, text=req.text, confidence=req.confidence
            )
            prompt_type = "text"
        else:
            polygons = segmenter.segment_by_click(
                frame, point=(req.point.x, req.point.y), positive=req.point.positive
            )
            prompt_type = "click"
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Segmentation failed: {e}")

    return {
        "prompt_type": prompt_type,
        "polygon_count": len(polygons),
        "polygons": polygons,
        "image_size": {"width": int(frame.shape[1]), "height": int(frame.shape[0])},
    }


# --- Static file mount for snapshots ---
_snapshot_dir = _PROJECT_ROOT / "data" / "events"
_snapshot_dir.mkdir(parents=True, exist_ok=True)
app.mount("/snapshots", StaticFiles(directory=str(_snapshot_dir)), name="snapshots")

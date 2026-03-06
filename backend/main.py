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

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from config import get_config
from events.storage import EventStorage, ALL_EVENT_TYPES
from pipeline.runner import PipelineRunner


# Project root (stang/) — same convention as config module
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize storage and start detection pipeline."""
    config = get_config()
    db_path = str(_PROJECT_ROOT / config.storage.db_path)

    storage = EventStorage(db_path)
    app.state.storage = storage
    app.state.config = config

    # Start detection pipeline in background thread
    runner = PipelineRunner(config)
    app.state.pipeline = runner

    if config.cameras:
        runner.start(storage)

    yield

    runner.stop()


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
    allow_methods=["GET"],
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
    runner = app.state.pipeline
    if runner._eval_agent is None:
        return []
    return runner._eval_agent.decision_storage.get_recent(limit=limit)


@app.get("/agent/alerts")
def get_agent_alerts(limit: int = Query(default=50, ge=1, le=500)):
    """Only decisions where the agent flagged an alert."""
    runner = app.state.pipeline
    if runner._eval_agent is None:
        return []
    return runner._eval_agent.decision_storage.get_alerts_only(limit=limit)


# --- Static file mount for snapshots ---
_snapshot_dir = _PROJECT_ROOT / "data" / "events"
_snapshot_dir.mkdir(parents=True, exist_ok=True)
app.mount("/snapshots", StaticFiles(directory=str(_snapshot_dir)), name="snapshots")

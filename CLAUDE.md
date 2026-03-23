# Project: StangWatch — AI CCTV Monitoring Agent for Nigeria

## What This Is
An AI-powered CCTV monitoring system that watches camera feeds, detects unusual activity, and sends Telegram alerts with human-readable explanations. Built for the Nigerian security market (warehouses, residential, farms).

The system is a true AI agent: it perceives (camera + YOLO), reasons (vision LLM describes the scene, text LLM evaluates severity), acts (sends Telegram alerts with escalation chains), and learns (tracks alert outcomes for quality metrics).

## Hard Rules — NEVER Break These

### No Hallucinations
- If you don't know something, say so. Do NOT guess or make up APIs, libraries, or features.
- If unsure about a library's API, read its docs or search first.
- When uncertain about an approach, ASK the user before implementing.
- Do NOT assume features exist in YOLO, Ollama, or any library — verify first.

### No System-Level Installs
- NEVER install anything system-level on the Mac (no homebrew, no system packages, no global binaries) without asking permision
- Only use: Python pip packages, npm packages, and Docker containers.
- The development Mac must stay clean.

### Incremental Development
- Build one piece at a time. Get it working before moving to the next.
- Do NOT build the whole system at once.
- Each component must be testable independently.
- Ask before making architectural decisions.

### User Must Be Consulted on Every Design Decision
- This is a LEARNING EXPERIENCE for the user. They must understand every choice.
- Before implementing anything, EXPLAIN what we're about to do and WHY.
- ASK the user before making any design decision, no matter how small.
- Walk the user through setup steps — don't just run commands silently.
- If something needs to be installed or configured, tell the user FIRST and let them approve.
- No silent decisions. No "I went ahead and did X." Always ask.

### No Assumptions
- Do NOT assume what the user wants. Ask if unclear.
- Do NOT add features that weren't requested.
- Do NOT over-engineer. Build the simplest thing that works and if an iteration may be better let the user say so.
- Do NOT add error handling for scenarios that can't happen yet.

### No Money Required for Testing
- All testing must work WITHOUT physical cameras.
- Phone camera (IP Webcam app / IP4K) as secondary live test input.
- V720/A9 camera via AP mode hotspot for real CCTV testing.
- Docker RTSP server (MediaMTX) only when explicitly requested.

## Architecture — Layered Intelligence

```
Video File / Phone Camera / V720 Camera / RTSP Stream
    |
    v
Camera class (OpenCV + V720 protocol — holds connection, reads frames)
    |
    v
PipelineRunner (background thread — orchestrates the loop)
    |
    v
Detector class (YOLO11s — person + object detection + ByteTrack)
    |
    v
EventTracker class (tracks detections over time, emits events via bus)
    |
    v
Event Bus (pyee in-process, or Redis Streams for distributed)
    |
    v
EventStorage (SQLite — persists all events)
    |
    v
EvalAgent — Two-stage local LLM reasoning:
    |
    |-- Stage 1: Vision perceives (Qwen 2.5 VL 7B via Ollama)
    |   "Describe what you see in this security camera image"
    |   Supports single snapshot AND multi-frame video analysis
    |
    |-- Stage 2: Text reasons (Qwen 2.5 7B via Ollama)
    |   Receives: event data + visual description + scene context + track history
    |   Returns: {alert, severity, reason, recommendation}
    |
    |-- Optional: Claude Vision API (paid, cloud)
    |   Future upgrade for higher-quality descriptions on flagged events
    |
    v
EscalationManager (role-based alert chains with timeouts)
    |
    v
TelegramSender (alerts with snapshots + video clips + inline buttons)
    |                Acknowledge / False Alarm feedback loop
    v
DecisionStorage + EscalationStorage (SQLite — full audit trail)
```

## Design Principles

### The System Flags Unusual Patterns, NOT "Crime"
- We detect statistical anomalies: presence during unusual hours, loitering, objects out of place
- We do NOT claim to detect theft, intent, or criminality
- The human (guard/owner) makes the judgment call
- Agent descriptions are factual: "person with backpack at warehouse door at 2am" — not "thief detected"

### What Each AI Layer Can Actually Do
- **YOLO11s + ByteTrack**: detects and tracks objects per frame (person, bag, vehicle, knife, etc.) — no understanding of actions or intent
- **Qwen 2.5 VL (vision LLM, local)**: describes what it sees in camera images/video — "two people standing near a doorway, one carrying a bag" — runs locally via Ollama, free, 24/7
- **Qwen 2.5 (text LLM, local)**: reasons about structured event data + visual description, applies rules, decides severity and whether to alert — runs locally via Ollama, free, 24/7
- **Claude Vision API (cloud, optional)**: higher-quality image descriptions for flagged events — paid, future upgrade for critical alerts only
- **Action recognition (v2)**: SlowFast/VideoMAE models to detect actions (walking, running, carrying) — future upgrade

### Offline-First
- Works fully without internet (YOLO + Ollama both run locally)
- Alerts queue locally when offline, flush when internet returns
- SMS fallback (Termii) for critical alerts when Telegram unavailable (production)

## Tech Stack — Locked In

| Layer | Technology | Install Method |
|-------|-----------|---------------|
| Detection model | YOLO11s (ultralytics) + ByteTrack | pip |
| Video processing | OpenCV | pip (opencv-python) |
| Backend API | FastAPI + uvicorn | pip |
| Database | SQLite with WAL mode | built into Python |
| Local vision LLM | Qwen 2.5 VL 7B via Ollama | Docker or native |
| Local text LLM | Qwen 2.5 7B via Ollama | Docker or native |
| Cloud LLM (optional) | Claude Vision API | pip (anthropic) — future |
| Event bus | pyee (in-process) or Redis Streams | pip |
| Scene memory | Redis hashes with TTL | Docker or native |
| Alerts | Telegram Bot API (direct HTTP) | pip (requests) |
| Dashboard | Next.js | npm — not yet built |
| V720 camera | a9-v720 vendor library | vendored (vendor/a9-v720) |

## Code Architecture

### Pipeline (`backend/pipeline/`)
- **PipelineRunner** — runs Camera -> Detector -> EventTracker loop in a background thread. Saves snapshots + 5-second video clips for events. Initializes EvalAgent.

### Capture (`backend/capture/`)
- **Camera** — manages video connection (OpenCV for webcam/file/RTSP, V720 protocol for V720 cameras). Supports `v720://host:port` URL scheme.

### Detection (`backend/detection/`)
- **Detector** — loads YOLO11s, runs person detection + ByteTrack tracking + nearby object association.

### Events (`backend/events/`)
- **EventTracker** — tracks people across frames. Emits factual events only: `appeared`, `departed`, `returned`, `companion`, `objects_changed`, `track_summary` (every 60s per active track). No state machine, no pre-filtering — the AI decides what's suspicious from raw movement data (avg_movement, total_distance, position_spread).
- **EventStorage** — SQLite storage for all events. Subscribes to the event bus.
- **event_bus / RedisBus** — pyee in-process event bus (default) or Redis Streams for distributed setups.

### Agent (`backend/agent/`)
- **EvalAgent** — subscribes to event bus, evaluates events in a background thread. Two-stage LLM: vision describes, text reasons.
- **VisionDescriber** — sends images/video frames to Ollama vision model (Qwen 2.5 VL). Supports single snapshot and multi-frame video analysis.
- **OllamaClient** — wraps Ollama SDK for text evaluation. Forces JSON output, low temperature.
- **prompts.py** — system prompt (security analyst role) + user prompt builder (4 layers: event, visual description, scene context, track history).
- **DecisionStorage** — SQLite storage for every AI decision (alert or not), for auditability.
- **SceneMemory** — Redis-backed short-term memory. Stores current scene state (who's on camera, objects visible) with TTL auto-expiry.
- **TelegramSender** — sends alerts with snapshots, video clips, and inline keyboard buttons (Acknowledge + False Alarm).
- **EscalationManager** — role-based escalation chains (guard -> supervisor -> admin) with configurable timeouts. Handles Telegram callback buttons.
- **EscalationStorage** — SQLite storage for escalation alerts. Tracks outcomes (true_alert, false_alarm, unresolved) for quality metrics.
- **RoleStorage** — SQLite storage for role memberships and invite codes. Supports bootstrap admin, invite-based onboarding.

### Utilities (`backend/utils.py`)
- `save_snapshot()` — save frame as image
- `draw_boxes()` — draw bounding boxes on frame
- `filter_overlapping()` — remove duplicate/overlapping detections

### Configuration (`config.yaml` + `backend/config/`)
- YAML-based config: cameras, detection thresholds, tracking params, agent settings, escalation policies, Redis, storage
- Secrets from `.env`: `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, `REDIS_PASSWORD`

### API (`backend/main.py`)
- `GET /health` — system status
- `GET /pipeline/status` — detection loop stats (FPS, frame count, active tracks)
- `GET /events` — recent events
- `GET /events/summary` — event counts by type
- `GET /events/type/{type}` — events filtered by type
- `GET /events/track/{id}` — events for a specific tracked person
- `GET /agent/decisions` — all AI evaluation decisions
- `GET /agent/alerts` — only alert decisions
- `GET/PUT /config/quiet-hours` — runtime quiet hours config
- `GET /roles/members` — all role members
- `GET /roles/members/active` — active members by role
- `POST /roles/invite` — create invite code
- `GET /roles/invites` — pending invite codes
- `DELETE /roles/members/{id}` — revoke member
- `GET /escalation/pending` — unacknowledged alerts
- `GET /escalation/recent` — all escalation alerts
- `PUT /escalation/{id}/acknowledge` — acknowledge alert
- `PUT /escalation/{id}/dismiss` — dismiss as false alarm
- `GET /metrics/alert-quality` — false positive rates by event type/severity
- `/snapshots/` — static file mount for event images

## Event Model — AI Decides Everything

The EventTracker emits **factual events only** — it does not judge what is suspicious. Every event includes raw measurement data. The AI agent (Ollama) receives all events and decides severity.

### Event Types

| Event | When Emitted | Key Data |
|-------|-------------|----------|
| `appeared` | New person detected | bbox, timestamp |
| `departed` | Person gone for N seconds | duration, total_distance, position_spread |
| `returned` | Departed person came back | previous duration |
| `companion` | New person near existing one | near_track_id, distance |
| `objects_changed` | Objects near a person changed | objects_before, objects_after |
| `track_summary` | Every 60s per active track | avg_movement_30f/150f, total_distance, position_spread, nearby_objects, is_quiet_hours |

### Track Summary Payload (sent to AI)
```
avg_movement_30f: 3.2    # px/frame ~1s — 0-5=still, 5-20=slow, 20+=walking
avg_movement_150f: 8.7   # px/frame ~5s — longer-term trend
total_distance: 1250.5   # cumulative pixels traveled
position_spread: 45.2    # max distance from centroid (roaming radius)
duration_seconds: 240.0
frames_tracked: 7200
nearby_objects: ["backpack"]
is_quiet_hours: true
companion_track_ids: [3]
```

### Expected Volume
~7 events per 5 minutes per person (1 appeared + ~4 summaries@60s + 1 departed + ~1 objects_changed). Down from 4,200+ with the old state machine approach.

## Alert Format

Each alert includes:
1. **Snapshot** image + **video clip** (5-second pre-event buffer)
2. **Visual description** from vision LLM (what the camera shows)
3. **Agent reasoning** from text LLM (why it was flagged)
4. **Structured data**: timestamp, camera, detection type, duration, confidence
5. **Inline buttons**: Acknowledge / False Alarm (feedback loop)
6. **Language**: English + Pidgin English (configurable)

## MVP Scope — Phase 1

### Built:
1. Camera class (video capture — webcam, file, RTSP, V720)
2. Detector class (YOLO11s person detection + ByteTrack + object association)
3. Utility functions (save_snapshot, draw_boxes, filter_overlapping)
4. EventTracker class (tracks detections, emits events)
5. Event bus (pyee + Redis Streams option)
6. EventStorage (SQLite)
7. FastAPI backend with full REST API
8. PipelineRunner (background detection loop)
9. EvalAgent with two-stage LLM (vision + text via Ollama)
10. Telegram alerts with snapshots + video clips
11. Escalation system (role-based chains, timeouts, Telegram callbacks)
12. Role management (memberships, invite codes, bootstrap admin)
13. Feedback loop (Acknowledge + False Alarm buttons)
14. Alert quality metrics (false positive rates)
15. Scene memory (Redis-backed current state)
16. Decision audit trail (every AI evaluation logged)

### Not yet built (v2+):
- Multi-camera support (beyond 1 stream)
- Zone drawing on camera view
- Vehicle/animal detection classes beyond COCO defaults
- SMS/WhatsApp alert fallback
- Cloud sync
- Farm/warehouse/residential deployment profiles
- Next.js dashboard (event log, config UI)
- User authentication on dashboard
- Action recognition models (SlowFast/VideoMAE)
- Claude Vision API integration (higher-quality cloud descriptions)
- Docker containerization

## Project Structure

```
stang/
├── CLAUDE.md              # This file — project rules and architecture
├── config.yaml            # Runtime configuration (cameras, thresholds, agent, escalation)
├── requirements.txt       # Python dependencies
├── .env                   # Secrets (gitignored): TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
├── .env.example           # Template for .env
├── .gitignore
├── yolo11s.pt             # YOLO model weights
├── backend/               # Python backend (all server code)
│   ├── __init__.py
│   ├── main.py            # FastAPI app + all REST endpoints
│   ├── utils.py           # Stateless utility functions
│   ├── redis_client.py    # Redis connection helper
│   ├── config/            # YAML config loader + dataclasses
│   │   └── __init__.py
│   ├── capture/           # Video input
│   │   ├── __init__.py
│   │   └── camera.py      # Camera class (OpenCV + V720)
│   ├── detection/         # Object detection
│   │   ├── __init__.py
│   │   └── detector.py    # Detector class (YOLO11s + ByteTrack)
│   ├── events/            # Event tracking + storage
│   │   ├── __init__.py
│   │   ├── tracker.py     # EventTracker (periodic summaries, no state machine)
│   │   ├── storage.py     # EventStorage (SQLite)
│   │   ├── bus.py         # In-process event bus (pyee)
│   │   └── redis_bus.py   # Redis Streams event bus
│   ├── agent/             # AI agent (evaluation + alerting)
│   │   ├── __init__.py
│   │   ├── evaluator.py   # EvalAgent (two-stage LLM evaluation)
│   │   ├── vision.py      # VisionDescriber (Ollama vision model)
│   │   ├── ollama_client.py  # OllamaClient (text LLM wrapper)
│   │   ├── prompts.py     # System + user prompt templates
│   │   ├── decisions.py   # DecisionStorage (SQLite audit trail)
│   │   ├── memory.py      # SceneMemory (Redis short-term state)
│   │   ├── telegram.py    # TelegramSender (alerts + buttons)
│   │   ├── escalation.py  # EscalationManager (role-based chains)
│   │   ├── escalation_storage.py  # EscalationStorage (SQLite + metrics)
│   │   └── role_storage.py  # RoleStorage (memberships + invites)
│   ├── pipeline/          # Detection loop orchestration
│   │   ├── __init__.py
│   │   └── runner.py      # PipelineRunner (background thread)
│   ├── alerts/            # (legacy — alert logic moved to agent/)
│   │   └── __init__.py
│   └── data/events/       # Event snapshots + video clips (gitignored)
├── scripts/               # Standalone utility scripts
│   ├── test_v720.py       # V720 camera standalone test (live view)
│   ├── test_v720_pipeline.py  # V720 camera pipeline integration test
│   ├── diagnose_network.py    # Network diagnostic for camera discovery
│   └── find_camera.py     # Port scanner for finding cameras
├── vendor/                # Third-party vendored code
│   └── a9-v720/           # V720/A9 camera protocol library (github.com/intx82/a9-v720)
├── data/                  # Runtime data (gitignored)
│   └── events/            # Event snapshots
├── test_videos/           # Downloaded test videos (gitignored)
└── dashboard/             # Next.js frontend (not yet built)
```

## Nigerian Context

- Offline-first: system must work without internet
- Power-aware: 10-second video segments so power loss loses minimal data
- SQLite WAL mode for crash recovery
- Telegram (primary) — works on low bandwidth, free bot API
- SMS via Termii (production fallback) — DND-bypass route for Nigerian networks
- USSD via Africa's Talking (last resort fallback)
- Alert language: English + Pidgin English (configurable)
- 60%+ of Nigerian mobile users are on DND — must use transactional SMS routes

## Testing Strategy

1. **Unit tests**: Each component testable independently
2. **Video file tests**: Downloaded CCTV-style footage from Pexels/Pixabay
3. **Live tests**: Phone camera via IP Webcam app, V720 camera via AP mode
4. **Integration tests**: Full pipeline from video -> detection -> event -> agent -> alert

## Git Workflow

- Main branch: `main`
- Feature branches for each component
- Commit messages: clear, descriptive
- No secrets in git (API keys in .env, gitignored)

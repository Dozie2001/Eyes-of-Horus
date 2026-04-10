# Project: StangWatch — AI CCTV Monitoring Agent for Nigeria

## What This Is
An AI-powered CCTV monitoring system that watches camera feeds, detects unusual activity, and sends alerts with human-readable explanations. Built for the Nigerian security market (warehouses, residential, farms).

The system is a true AI agent: it perceives (camera + YOLO), reasons (vision LLM describes the scene, text LLM evaluates severity), acts (sends alerts via pluggable providers with escalation chains), and learns (tracks alert outcomes for quality metrics).

## Hard Rules — NEVER Break These

### No Hallucinations
- If you don't know something, say so. Do NOT guess or make up APIs, libraries, or features.
- If unsure about a library's API, read its docs or search first.
- When uncertain about an approach, ASK the user before implementing.
- Do NOT assume features exist in YOLO, Ollama, Roboflow, or any library — verify first.

### No System-Level Installs
- NEVER install anything system-level on the Mac (no homebrew, no system packages, no global binaries) without asking permission
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

## Deployment Modes

StangWatch supports two deployment topologies. The pipeline code is identical — only the infrastructure topology differs.

### Mode A: Edge (Fully Local)
Everything runs on-premise on a local server/device. No internet required except for alert delivery.

```
Cameras (RTSP / V720 / USB)
    |
    v
Edge Server (local)
    ├── FastAPI (backend)
    ├── Roboflow Docker (YOLO26 + SAM3)
    ├── Ollama (Qwen VL for vision + text LLM)
    ├── Redis (scene memory + event bus)
    ├── SQLite + local disk (source of truth)
    └── Alert providers (Telegram/WhatsApp/SMS) ──► Internet (outbound only)
         Optional: SyncWorker ──► Supabase (cloud backup)
```

- SQLite + local disk is the **source of truth** for all data
- Clips/snapshots stored on local disk (`data/events/`)
- Dashboard accessible on LAN, or via optional cloud sync
- Internet only needed for: alert delivery + optional cloud sync

### Mode B: Cloud (Cameras Remote)
Cameras are on-premise, connected via VPN/router to a cloud server running all compute.

```
On-premise                          Cloud Server
┌──────────────────┐               ┌──────────────────────────┐
│ Cameras + Router │               │ FastAPI (backend)        │
│ └── VPN/RTSP ────┼──► Internet ──┼► Roboflow Docker (YOLO26 + SAM3)
└──────────────────┘               │ Ollama (Qwen VL + text LLM)
                                   │ Redis (scene memory + bus)
                                   │ Supabase (primary storage)
                                   │ Alert providers           │
                                   └────────────┬─────────────┘
                                                │
                                   ┌────────────▼─────────────┐
                                   │ Supabase                  │
                                   │ ├── Postgres (events,     │
                                   │ │   decisions, zones)     │
                                   │ ├── Storage (clips)       │
                                   │ ├── pgvector (embeddings) │
                                   │ └── Auth (users/orgs)     │
                                   └────────────┬─────────────┘
                                                │
                                   ┌────────────▼─────────────┐
                                   │ Next.js Dashboard (hosted)│
                                   │ ├── Multi-site aggregation│
                                   │ ├── Zone editor (SAM3)   │
                                   │ ├── Clip search           │
                                   │ └── Alert history         │
                                   └──────────────────────────┘
```

- Supabase is the **primary storage** (no local persistence needed)
- SyncWorker writes directly to Supabase
- Dashboard reads from Supabase, multi-site aggregation across locations
- All compute (detection, LLM, segmentation) runs on the cloud server

### What's The Same in Both Modes
- Roboflow Docker, Ollama, Redis always run local to the compute server
- Pipeline code (Camera → Detector → EventTracker → EvalAgent) is identical
- Alert provider system works the same way
- Zone system works the same way

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
RoboflowDetector class (YOLO26 via Roboflow HTTP API + client-side ByteTrack)
    |
    v
EventTracker class (tracks detections over time, emits zone-aware events via bus)
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
    |   Receives: event data + visual description + scene context + track history + zone info
    |   Returns: {alert, severity, reason, recommendation}
    |
    |-- Optional: Cloud LLM providers (Groq, OpenRouter)
    |   Faster/higher-quality inference when internet available
    |
    v
EscalationManager (role-based alert chains with timeouts)
    |
    v
AlertProviders (pluggable: Telegram, WhatsApp, SMS)
    |   Alerts with snapshots + video clips + inline buttons
    |   Acknowledge / False Alarm feedback loop
    v
DecisionStorage + EscalationStorage (SQLite — full audit trail)
    |
    v (optional, async)
SyncWorker → Supabase (cloud backup + dashboard data layer)
    ├── Storage (clips/snapshots)
    ├── Postgres (events, decisions, zones, escalation)
    └── pgvector (embeddings via Jina for clip search)
```

## Design Principles

### The System Flags Unusual Patterns, NOT "Crime"
- We detect statistical anomalies: presence during unusual hours, loitering, objects out of place
- We do NOT claim to detect theft, intent, or criminality
- The human (guard/owner) makes the judgment call
- Agent descriptions are factual: "person with backpack at warehouse door at 2am" — not "thief detected"

### What Each AI Layer Can Actually Do
- **YOLO26 + ByteTrack** (via Roboflow Docker): detects and tracks objects per frame (person, bag, vehicle, knife, etc.) — no understanding of actions or intent. Also supports instance segmentation variants.
- **SAM3** (via Roboflow Docker): segment anything by click, text prompt, or object selection. Used for zone drawing. Three endpoints: `concept_segment` (text), `visual_segment` (click), `embed_image` (cache).
- **Qwen 2.5 VL (vision LLM, local)**: describes what it sees in camera images/video — "two people standing near a doorway, one carrying a bag" — runs locally via Ollama, free, 24/7
- **Qwen 2.5 (text LLM, local)**: reasons about structured event data + visual description, applies rules, decides severity and whether to alert — runs locally via Ollama, free, 24/7
- **Cloud LLM providers (optional)**: Groq, OpenRouter for faster/better inference when internet available
- **Action recognition (v2)**: SlowFast/VideoMAE models to detect actions (walking, running, carrying) — future upgrade

### Edge-First
- All compute runs locally: detection (Roboflow Docker), inference (Ollama), state (Redis), storage (SQLite + disk)
- Internet only required for: alert delivery (Telegram/WhatsApp/SMS) and optional cloud sync
- Alerts queue locally when offline, flush when internet returns
- Cloud sync (Supabase) is an optional overlay — the system works fully without it
- Dashboard can run on LAN (edge mode) or cloud-hosted reading from Supabase

### Docker is a Hard Dependency
- Roboflow inference server and Ollama both run as Docker containers
- No fallback to in-process ultralytics — all detection goes through Roboflow HTTP API
- This simplifies the stack: one inference server handles YOLO26, SAM3, and future models

## Tech Stack

| Layer | Technology | Runs As | Notes |
|-------|-----------|---------|-------|
| Detection | YOLO26 via Roboflow Inference API | Docker (`roboflow/inference-server:latest`, port 9001) | Hard dependency. Model ID: `yolo26s-640`. No API key needed for pre-trained models |
| Segmentation | SAM3 via Roboflow Inference API | Same Docker container | Endpoints: `/sam3/concept_segment`, `/sam3/visual_segment`, `/sam3/embed_image` |
| Tracking | ByteTrack (client-side) | In-process (Python) | Roboflow has no server-side tracking — ByteTrack stays in the pipeline |
| Video processing | OpenCV | pip (opencv-python) | Camera capture, frame handling |
| Backend API | FastAPI + uvicorn | pip | Edge API server |
| Database | SQLite with WAL mode | built into Python | Events, decisions, zones, escalation, roles |
| Local vision LLM | Qwen 2.5 VL 7B via Ollama | Docker or native (port 11434) | Vision description for alerts |
| Local text LLM | Qwen 2.5 7B via Ollama | Docker or native | Severity reasoning, JSON output |
| Cloud LLM (optional) | Groq / OpenRouter | API calls | Faster/better when internet available |
| Event bus | pyee (in-process) or Redis Streams | pip / Docker | pyee default, Redis for distributed |
| Scene memory | Redis hashes with TTL | Docker or native (port 6379) | Ephemeral real-time state |
| Alerts | Pluggable providers | pip (requests/httpx) | Telegram (built), WhatsApp + SMS (planned) |
| Cloud storage | Supabase Storage | Cloud service | Clip/snapshot backup + remote access |
| Cloud database | Supabase Postgres | Cloud service | Dashboard data layer, multi-site aggregation |
| Embeddings | Jina embeddings-v3 → Supabase pgvector | Cloud service | 1024-dim vectors for clip search |
| Dashboard | Next.js | npm (cloud-hosted) | Multi-site, zone editor, clip search, auth |
| V720 camera | a9-v720 vendor library | vendored (vendor/a9-v720) | V720/A9 camera protocol |

### Docker Containers (required)
```bash
# Roboflow Inference Server (YOLO26 + SAM3)
docker run -it --rm -p 9001:9001 --gpus=all roboflow/inference-server:latest

# Ollama (vision + text LLM)
# Already running natively or via Docker

# Redis
# Already running natively or via Docker
```

## Zone System (SAM3-Powered)

Zones are named regions on a camera view with configurable alert rules. SAM3 enables three methods for creating zones:

### Zone Creation Methods
1. **Click-to-segment** — user clicks a point on the camera frame → SAM3 `visual_segment` → polygon mask
2. **Text/concept** — user types "gate", "driveway", "warehouse door" → SAM3 `concept_segment` → polygon mask
3. **Object-based** — user selects a detected object (e.g., a door, fence) → SAM3 segments around it → polygon mask

### Zone Definition
Each zone includes:
- **Name**: human-readable label ("Front Gate", "Parking Lot")
- **Polygon**: list of (x, y) points defining the zone boundary (from SAM3 mask)
- **Camera ID**: which camera this zone belongs to
- **Rules**: severity override, active hours, allowed object types
- **Stored in**: SQLite locally, synced to Supabase for dashboard access

### Zone-Aware Detection
- Each detection's bbox centroid is checked against saved zone polygons
- Events include which zone(s) the person is in
- AI agent receives zone context for severity reasoning (e.g., "person in restricted zone during quiet hours")

### Zone Editing Flow (Dashboard)
1. Dashboard grabs a frame from the edge camera (via FastAPI `/cameras/{id}/frame`)
2. User draws/clicks/types to create zone → request sent to edge FastAPI `/zones/segment` proxy
3. Edge proxies to local Roboflow Docker SAM3 endpoint → returns mask/polygon
4. User names the zone, configures rules
5. Zone saved to Supabase → edge pulls zone config on next sync cycle

## Alert Provider System

Alerts are delivered via pluggable providers. The system supports multiple providers simultaneously.

### Built
- **Telegram** — primary provider. Bot API with inline keyboard buttons (Acknowledge / False Alarm). Works on low bandwidth.

### Planned
- **WhatsApp** — via WhatsApp Business API or Twilio
- **SMS** — via Termii (DND-bypass transactional route for Nigerian networks)
- **USSD** — via Africa's Talking (last resort fallback)

### Provider Interface
Each provider implements: `send_alert(alert_data)`, `send_escalation(escalation_data)`, handles callback/feedback if supported.

### Alert Format
Each alert includes:
1. **Snapshot** image + **video clip** (5-second pre-event buffer)
2. **Visual description** from vision LLM (what the camera shows)
3. **Agent reasoning** from text LLM (why it was flagged)
4. **Structured data**: timestamp, camera, detection type, duration, confidence, zone
5. **Inline buttons**: Acknowledge / False Alarm (feedback loop, provider-dependent)
6. **Language**: English + Pidgin English (configurable)

## Code Architecture

### Pipeline (`edge/pipeline/`)
- **PipelineRunner** — runs Camera -> Detector -> EventTracker loop in a background thread. Saves snapshots + video clips for events. Initializes EvalAgent. Loads zone definitions.

### Capture (`edge/capture/`)
- **Camera** — manages video connection (OpenCV for webcam/file/RTSP, V720 protocol for V720 cameras). Supports `v720://host:port` URL scheme.

### Detection (`edge/detection/`)
- **RoboflowDetector** — HTTP client to Roboflow Inference Server. Sends frames as base64, receives detection results. Replaces in-process ultralytics.
- **ByteTrack** — client-side tracking (Roboflow has no server-side tracking). Assigns persistent track IDs across frames.
- **Object association** — associates detected objects (backpack, bag, etc.) with nearest person.

### Events (`edge/events/`)
- **EventTracker** — tracks people across frames. Emits factual events only: `appeared`, `departed`, `returned`, `companion`, `objects_changed`, `track_summary` (every 60s per active track). Zone-aware: events include which zone(s) the detection is in.
- **EventStorage** — SQLite storage for all events. Subscribes to the event bus.
- **event_bus / RedisBus** — pyee in-process event bus (default) or Redis Streams for distributed setups.

### Agent (`edge/agent/`)
- **EvalAgent** — subscribes to event bus, evaluates events in a background thread. Two-stage LLM: vision describes, text reasons.
- **VisionDescriber** — sends images/video frames to Ollama vision model (Qwen 2.5 VL). Supports single snapshot and multi-frame video analysis.
- **OllamaClient** — wraps Ollama SDK for text evaluation. Forces JSON output, low temperature.
- **prompts.py** — system prompt (security analyst role) + user prompt builder (4 layers: event, visual description, scene context, track history + zone info).
- **DecisionStorage** — SQLite storage for every AI decision (alert or not), for auditability.
- **SceneMemory** — Redis-backed short-term memory. Stores current scene state (who's on camera, objects visible) with TTL auto-expiry.
- **AlertProviders** — pluggable alert delivery (Telegram built, WhatsApp/SMS planned).
- **TelegramSender** — sends alerts with snapshots, video clips, and inline keyboard buttons (Acknowledge + False Alarm).
- **EscalationManager** — role-based escalation chains (guard -> supervisor -> admin) with configurable timeouts. Handles callback buttons.
- **EscalationStorage** — SQLite storage for escalation alerts. Tracks outcomes (true_alert, false_alarm, unresolved) for quality metrics.
- **RoleStorage** — SQLite storage for role memberships and invite codes. Supports bootstrap admin, invite-based onboarding.
- **ClipUploader** — uploads clips/snapshots to Supabase Storage (async background).
- **ClipIndex** — embeds decisions via Jina, stores vectors in Supabase pgvector for natural language clip search.
- **embeddings.py** — 7 embedding providers (Jina default, Ollama offline fallback, HuggingFace, Cohere, Google, Voyage, OpenAI).

### Sync (`edge/sync/`)
- **SyncWorker** — background thread that pushes events, decisions, escalation, and media to Supabase when online. Pulls zone definitions from Supabase (dashboard → edge). Sends heartbeats.

### Cloud API (`cloud/`)
- **cloud/main.py** — FastAPI app for cloud-side endpoints (sync receiver, dashboard API proxy)
- **cloud/sync/receiver.py** — receives batch uploads from edge SyncWorker, writes to Supabase
- **cloud/proxy/vision.py** — proxies vision requests if needed

### Utilities (`edge/utils.py`)
- `save_snapshot()` — save frame as image
- `draw_boxes()` — draw bounding boxes on frame
- `filter_overlapping()` — remove duplicate/overlapping detections

### Configuration (`config.yaml` + `edge/config/`)
- YAML-based config: cameras, detection thresholds, tracking params, agent settings, escalation policies, Redis, storage, Roboflow, Supabase
- Secrets from `.env`: `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, `REDIS_PASSWORD`, `SUPABASE_URL`, `SUPABASE_SERVICE_KEY`, `JINA_API_KEY`

### API (`edge/main.py`)
- `GET /health` — system status (includes Roboflow + Ollama connectivity)
- `GET /pipeline/status` — detection loop stats (FPS, frame count, active tracks)
- `GET /cameras/{id}/frame` — current camera frame (JPEG, for zone editor)
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
- `POST /zones/segment` — proxy to Roboflow SAM3 for zone creation
- `GET/POST/PUT/DELETE /zones` — zone CRUD
- `POST /clips/search` — natural language clip search via embeddings
- `/snapshots/` — static file mount for event images

## Supabase Schema (Cloud Data Layer)

### Existing Tables
- `organization` — customer accounts
- `org_member` — dashboard users (Supabase Auth)
- `edge_device` — registered edge devices per org (API key, heartbeat, status)
- `cloud_event` — events synced from edge
- `cloud_decision` — AI decisions synced from edge (includes `embedding` vector(1024) + `embedding_model`)
- `cloud_escalation` — escalation alerts synced from edge
- pgvector extension + HNSW index for semantic clip search
- `search_clips()` + `match_clips()` RPC functions for hybrid/semantic search
- Supabase Storage bucket `CLIPS` for media files

### Planned Tables
- `cloud_zone` — zone definitions (polygon, rules, camera_id) — created in dashboard, pulled by edge
- `cloud_camera_profile` — camera metadata, SAM3 embeddings cache

## Event Model — AI Decides Everything

The EventTracker emits **factual events only** — it does not judge what is suspicious. Every event includes raw measurement data + zone context. The AI agent (Ollama) receives all events and decides severity.

### Event Types

| Event | When Emitted | Key Data |
|-------|-------------|----------|
| `appeared` | New person detected | bbox, timestamp, zone |
| `departed` | Person gone for N seconds | duration, total_distance, position_spread, zone |
| `returned` | Departed person came back | previous duration, zone |
| `companion` | New person near existing one | near_track_id, distance, zone |
| `objects_changed` | Objects near a person changed | objects_before, objects_after, zone |
| `track_summary` | Every 60s per active track | avg_movement_30f/150f, total_distance, position_spread, nearby_objects, is_quiet_hours, zone |

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
zone: "front_gate"       # which zone the person is in (if any)
```

### Expected Volume
~7 events per 5 minutes per person (1 appeared + ~4 summaries@60s + 1 departed + ~1 objects_changed). Down from 4,200+ with the old state machine approach.

## MVP Scope — Phase 1

### Built:
1. Camera class (video capture — webcam, file, RTSP, V720)
2. Detector class (YOLO11s person detection + ByteTrack + object association) — being replaced by RoboflowDetector
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
17. Clip upload to Supabase Storage
18. Clip search via Jina embeddings + Supabase pgvector
19. Cloud sync schema (Supabase tables for events, decisions, escalation, orgs, devices)

### In Progress (Phase 2 — current):
- RoboflowDetector (YOLO26 via Roboflow Docker HTTP API, replacing ultralytics)
- SAM3 integration for zone segmentation
- Zone system (storage, zone-aware events, rules)
- Next.js dashboard (cloud-hosted, multi-site, zone editor)
- SyncWorker (edge → Supabase push/pull)
- Alert provider plugin system (WhatsApp, SMS alongside Telegram)

### Not yet built (v3+):
- Multi-camera support (beyond 1 stream)
- Vehicle/animal detection classes beyond COCO defaults
- Farm/warehouse/residential deployment profiles
- User authentication on dashboard (Supabase Auth)
- Action recognition models (SlowFast/VideoMAE)
- USSD alert fallback (Africa's Talking)

## Project Structure

```
stang/
├── CLAUDE.md              # This file — project rules and architecture
├── config.yaml            # Runtime configuration
├── requirements.txt       # Python dependencies
├── .env                   # Secrets (gitignored)
├── .env.example           # Template for .env
├── .gitignore
├── edge/                  # Edge server (Python backend — all pipeline code)
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
│   │   └── detector.py    # RoboflowDetector (HTTP client) + ByteTrack
│   ├── events/            # Event tracking + storage
│   │   ├── __init__.py
│   │   ├── tracker.py     # EventTracker (zone-aware, periodic summaries)
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
│   │   ├── role_storage.py  # RoleStorage (memberships + invites)
│   │   ├── clip_upload.py # ClipUploader (Supabase Storage)
│   │   ├── clip_index.py  # ClipIndex (Jina + pgvector search)
│   │   └── embeddings.py  # Embedding providers (Jina, Ollama, etc.)
│   ├── pipeline/          # Detection loop orchestration
│   │   ├── __init__.py
│   │   └── runner.py      # PipelineRunner (background thread)
│   ├── sync/              # Cloud sync
│   │   ├── __init__.py
│   │   └── worker.py      # SyncWorker (edge → Supabase)
│   └── data/events/       # Event snapshots + video clips (gitignored)
├── cloud/                 # Cloud API (sync receiver, dashboard proxy)
│   ├── __init__.py
│   ├── main.py            # FastAPI cloud endpoints
│   ├── sync/
│   │   └── receiver.py    # Batch sync receiver
│   └── proxy/
│       └── vision.py      # Vision proxy
├── dashboard/             # Next.js frontend (cloud-hosted)
│   └── (not yet built)
├── supabase/              # Supabase migrations
│   └── migrations/
│       ├── 001_cloud_tables.sql    # Orgs, devices, events, decisions, escalation
│       ├── 002_rls_policies.sql    # Row-level security
│       └── 003_vector_search.sql   # pgvector extension + search functions
├── scripts/               # Standalone utility scripts
├── vendor/                # Third-party vendored code
│   └── a9-v720/           # V720/A9 camera protocol library
├── docker/                # Docker configs
├── data/                  # Runtime data (gitignored)
│   └── events/            # Event snapshots
└── test_videos/           # Downloaded test videos (gitignored)
```

## Nigerian Context

- Edge-first: system must work without internet (all compute local)
- Power-aware: 10-second video segments so power loss loses minimal data
- SQLite WAL mode for crash recovery
- Alert providers (pluggable):
  - Telegram (primary) — works on low bandwidth, free bot API
  - WhatsApp (planned) — high adoption in Nigeria
  - SMS via Termii (planned) — DND-bypass transactional route for Nigerian networks
  - USSD via Africa's Talking (future, last resort fallback)
- Alert language: English + Pidgin English (configurable)
- 60%+ of Nigerian mobile users are on DND — must use transactional SMS routes

## Testing Strategy

1. **Unit tests**: Each component testable independently
2. **Video file tests**: Downloaded CCTV-style footage from Pexels/Pixabay
3. **Live tests**: Phone camera via IP Webcam app, V720 camera via AP mode
4. **Integration tests**: Full pipeline from video -> detection -> event -> agent -> alert
5. **Roboflow tests**: Test Roboflow Docker API connectivity + model inference separately

## Git Workflow

- Main branch: `main`
- Feature branches for each component
- Commit messages: clear, descriptive
- No secrets in git (API keys in .env, gitignored)

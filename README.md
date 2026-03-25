# Eyes of Horus

**AI-powered CCTV Agent.**

Security cameras record everything. Nobody watches. Footage only gets checked *after* something goes wrong.

Eyes of Horus changes that. It watches your cameras 24/7 using AI, detects unusual activity in real time, and sends you a Telegram alert with a snapshot and a plain-English explanation of what it sees. You decide what to do.

It runs on a laptop. It works offline. No cloud subscription required.

---

## How It Works

```
Camera Feed (webcam, IP camera, or video file)
       |
       v
  YOLO 11s  ──────────────  Person + object detection, frame by frame
       |
       v
  Event Tracker  ─────────  Tracks people over time: appeared, loitering,
       |                     departed, returned, companion, objects changed
       v
  AI Evaluation Agent  ───  Local LLM (Ollama) decides: is this worth alerting?
       |
       v
  Vision Description  ────  Vision LLM describes the snapshot in plain English
       |
       v
  Escalation Engine  ─────  Routes alert to the right people based on severity
       |                     Guard → Supervisor → Admin (with timeouts)
       v
  Telegram Alert  ─────────  Snapshot + explanation + Acknowledge button
```

**The system flags unusual patterns, not "crime."** It detects statistical anomalies — presence during unusual hours, loitering, objects out of place. The human (guard, owner, operator) makes the judgment call.


## Alert Example

When the system detects something unusual, you get a Telegram message like this:

```
🟠 Alert

Event: LOITERING
Severity: medium
Person: Track #5

Why: Person detected during quiet hours, stationary for 45 seconds
     near entry point with backpack

Description: A person in dark clothing is standing at the warehouse
             door holding a large bag, looking at the entrance.

Action: Check camera feed — person has been stationary near entry
        for extended period during quiet hours

              [ ✅ Acknowledge ]
```

If nobody presses Acknowledge within 5 minutes, the alert escalates to the next person in the chain.

<!-- TODO: Replace with actual screenshot -->

---

## Features

- **Real-time person detection** — YOLO 11s with ByteTrack multi-person tracking
- **AI severity assessment** — local LLM evaluates every event, only alerts when it matters
- **Vision-based scene description** — AI describes what it sees in the snapshot (local or cloud)
- **Escalation chains** — guard gets it first, supervisor if no response, then admin
- **Telegram acknowledgment** — inline button confirms someone is handling it
- **Role-based access** — invite people via bot commands, revoke access anytime
- **Offline-first** — YOLO + Ollama both run locally, no internet needed for core function
- **Crash-safe storage** — SQLite with WAL mode, minimal data loss on power failure
- **REST API** — full FastAPI backend for events, decisions, config, and role management
- Zone Drwaing

---

## Prerequisites

Before setting up Eyes of Horus, you need:

| Requirement | Why | How to Get It |
|-------------|-----|---------------|
| **Python 3.11+** | Runs the backend | [python.org](https://www.python.org/downloads/) or your package manager |
| **Ollama** | Runs AI models locally | [ollama.com](https://ollama.com/download) |
| **Redis** (optional) | Scene memory + event streaming | [redis.io](https://redis.io/download) or Redis Cloud free tier |
| **A Telegram account** | Receive alerts | [telegram.org](https://telegram.org/) |
| **A camera** (or video file) | Something to watch | Webcam, IP camera, phone, or a downloaded test video |

**Hardware:**
- Any modern laptop or desktop with at least 8GB RAM
- A GPU helps but is not required — YOLO and Ollama run on CPU too (slower)
- For production: a dedicated machine with 16GB+ RAM and a GPU is recommended

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/eyes-of-horus.git
cd eyes-of-horus

# 2. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Pull the AI models
ollama pull qwen2.5:7b        # text reasoning (decisions)
ollama pull qwen2.5vl:7b      # vision (image descriptions)

# 5. Set up your environment
cp .env.example .env
# Edit .env with your Telegram bot token and chat ID

# 6. Run
cd backend
uvicorn main:app --port 8000
```

The system will start detecting immediately using your webcam. Stand in front of the camera during quiet hours to trigger an alert.

---

## Setup Guide (Step by Step)

If you're new to this, follow every step. Each one explains **what** you're doing and **why**.

### 1. Install Python

You need Python 3.11 or newer.

**macOS:**
```bash
# Check if you already have it
python3 --version

# If not, install via Homebrew
brew install python@3.11
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip
```

**Windows:**
Download from [python.org](https://www.python.org/downloads/). During installation, check "Add Python to PATH".

### 2. Install Ollama

Ollama runs AI models locally on your machine. No API keys, no cloud, no cost.

**macOS / Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows:**
Download from [ollama.com/download](https://ollama.com/download).

After installing, verify it's running:
```bash
ollama --version
```

### 3. Pull the AI Models

Eyes of Horus uses two models:

```bash
# Text model — evaluates events and decides severity (~5GB)
ollama pull qwen2.5:7b

# Vision model — describes what it sees in snapshots (~6GB)
ollama pull qwen2.5vl:7b
```

This will take a few minutes depending on your internet speed. The models are downloaded once and cached locally.

### 4. Clone and Install

```bash
git clone https://github.com/YOUR_USERNAME/eyes-of-horus.git
cd eyes-of-horus

# Create an isolated Python environment
python3 -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate

# Install all Python packages
pip install -r requirements.txt
```

### 5. Create a Telegram Bot

You need a Telegram bot to receive alerts. This is free.

1. Open Telegram and search for **@BotFather**
2. Send `/newbot`
3. Choose a name (e.g., "Eyes of Horus Alerts")
4. Choose a username (e.g., `eyes_of_horus_bot`) — must end in `bot`
5. BotFather gives you a **token** like `123456789:ABCdef...` — copy this

Now get your **chat ID** (so the bot knows where to send alerts):

1. Search for **@userinfobot** on Telegram
2. Send it any message
3. It replies with your chat ID (a number like `964136226`) — copy this

### 6. Configure Environment

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` and fill in your values:

```env
# Telegram Bot (from BotFather)
TELEGRAM_BOT_TOKEN=123456789:ABCdef_your_token_here
TELEGRAM_CHAT_ID=your_chat_id_here

# Optional: Anthropic API key for Claude Vision (premium image descriptions)
ANTHROPIC_API_KEY=

# Optional: Redis (leave empty to use in-process event bus)
REDIS_HOST=
REDIS_PORT=
REDIS_USERNAME=
REDIS_PASSWORD=
```

### 7. Configure the System

Edit `config.yaml` to match your setup:

```yaml
site:
  name: "My Office"        # Name shown in alerts

cameras:
  - name: "webcam"
    source: "0"             # 0 = built-in webcam
    enabled: true

tracking:
  quiet_hours:
    start: "22:00"          # When quiet hours begin
    end: "06:00"            # When quiet hours end
  loiter_threshold: 300.0   # Seconds before loitering alert (5 min)

agent:
  model: "qwen2.5:7b"      # Must match what you pulled in Ollama
```

**Key settings to adjust:**
- `quiet_hours` — the time window when person detection triggers alerts. Outside this window, only high-severity events alert.
- `loiter_threshold` — how long someone must stand still before it's considered loitering. Use `10.0` for testing, `300.0` for production.

### 8. Run

```bash
cd backend
uvicorn main:app --port 8000
```

You should see:
```
Bootstrapped admin from TELEGRAM_CHAT_ID: ...
EscalationManager started (timeout checker + Telegram poller)
Pipeline running: source=0, FPS=30
EvalAgent started (model: qwen2.5:7b)
```

The API is now live at `http://localhost:8000`. Check health:
```bash
curl http://localhost:8000/health
```

### 9. Test It

1. Make sure quiet hours cover the current time (edit `config.yaml` or use the API)
2. Stand in front of your webcam
3. Wait 10-15 seconds for detection + AI evaluation
4. Check Telegram — you should receive an alert with an Acknowledge button

---

## Testing Without a Camera

You don't need a physical camera to test. Two options:

### Option A: Use a Video File

Download a CCTV-style video from [Pexels](https://www.pexels.com/search/videos/cctv/) or [Pixabay](https://pixabay.com/videos/search/security%20camera/).

```yaml
# In config.yaml, point the camera to a file:
cameras:
  - name: "test"
    source: "test_videos/warehouse.mp4"
    enabled: true
```

### Option B: Use Your Phone

1. Install **IP Webcam** (Android) or **IP4K** (iOS) on your phone
2. Start the camera server — it shows a URL like `http://192.168.1.5:8080`
3. Point Eyes of Horus at it:

```yaml
cameras:
  - name: "phone"
    source: "http://192.168.1.5:8080/video"
    enabled: true
```

---

## Role Management

Eyes of Horus uses Telegram for role-based access control. The person who sets up the system is automatically the **admin**.

### Roles

| Role | Level | Gets Alerts | Can Do |
|------|-------|------------|--------|
| **Guard** | 1 | Medium severity | Acknowledge alerts |
| **Supervisor** | 2 | Medium + escalated | Acknowledge alerts |
| **Admin** | 3 | High + unacknowledged | Acknowledge, invite, revoke, configure |

### Adding People

**By invite code** (when you don't know their Telegram username):
```
You:    /invite guard
Bot:    Invite code: SW-A7X2. Share this with the person.
Guard:  /join SW-A7X2
Bot:    Welcome! You are now registered as guard.
```

**By username** (when you know who they are):
```
You:    /invite guard @emeka
Bot:    Invited @emeka as guard. Tell them to send /start.
Emeka:  /start
Bot:    Welcome! You've been activated as guard.
```

### Bot Commands

| Command | Who | What |
|---------|-----|------|
| `/start` | Anyone | Activate a pending invite |
| `/join <code>` | Anyone | Join using an invite code |
| `/invite <role>` | Admin | Create an invite code |
| `/invite <role> @user` | Admin | Invite someone directly |
| `/revoke @user` | Admin | Remove someone's access |
| `/whoami` | Members | Check your role |
| `/members` | Admin | List all active members |

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | System status |
| GET | `/pipeline/status` | Detection pipeline state, FPS, active tracks |
| GET | `/events` | Recent detection events |
| GET | `/events/summary` | Event counts by type |
| GET | `/agent/decisions` | AI evaluation decisions (all) |
| GET | `/agent/alerts` | Only decisions that triggered alerts |
| GET | `/escalation/pending` | Unacknowledged escalation alerts |
| GET | `/escalation/recent` | All escalation alerts |
| PUT | `/escalation/{id}/acknowledge` | Manually acknowledge an alert |
| GET | `/config/quiet-hours` | Current quiet hours setting |
| PUT | `/config/quiet-hours` | Update quiet hours at runtime |
| GET | `/roles/members` | All role members |
| GET | `/roles/members/active` | Active members by role |
| POST | `/roles/invite` | Create invite code via API |
| DELETE | `/roles/members/{id}` | Revoke a member |

---

## Architecture

```
eyes-of-horus/
├── backend/
│   ├── main.py                 # FastAPI server + pipeline startup
│   ├── utils.py                # Stateless utilities
│   ├── capture/
│   │   └── camera.py           # Camera connection + frame reading
│   ├── detection/
│   │   └── detector.py         # YOLO model + ByteTrack tracking
│   ├── events/
│   │   ├── tracker.py          # EventTracker — detections → events
│   │   ├── storage.py          # SQLite event storage
│   │   ├── bus.py              # In-process event bus (pyee)
│   │   └── redis_bus.py        # Redis Streams event bus
│   ├── agent/
│   │   ├── evaluator.py        # EvalAgent — AI decision engine
│   │   ├── ollama_client.py    # Ollama API client
│   │   ├── prompts.py          # System + user prompts for LLM
│   │   ├── decisions.py        # Decision storage (SQLite)
│   │   ├── memory.py           # Scene memory (Redis)
│   │   ├── telegram.py         # Telegram Bot API client
│   │   ├── escalation.py       # Escalation policy engine
│   │   ├── escalation_storage.py  # Escalation state (SQLite)
│   │   └── role_storage.py     # Role membership + invite codes
│   ├── pipeline/
│   │   └── runner.py           # Detection loop orchestrator
│   └── config/
│       └── __init__.py         # Pydantic config from YAML + .env
├── config.yaml                 # Non-secret settings
├── .env                        # Secrets (gitignored)
├── requirements.txt            # Python dependencies
└── README.md                   # You are here
```

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Detection | YOLO 11s (ultralytics) | Person + object detection |
| Tracking | ByteTrack (via ultralytics) | Multi-person tracking across frames |
| Video | OpenCV | Camera connection + frame processing |
| Backend | FastAPI + uvicorn | REST API server |
| Database | SQLite (WAL mode) | Events, decisions, escalation state, roles |
| Text AI | Ollama (qwen2.5:7b) | Event evaluation + severity decisions |
| Vision AI | Ollama (qwen2.5vl:7b) | Snapshot image description |
| Vision AI (premium) | Claude Vision API | Higher quality image descriptions |
| Event bus | pyee / Redis Streams | In-process or distributed event routing |
| Alerts | Telegram Bot API | Alert delivery + acknowledgment |

---

## Roadmap

### Built
- [x] Camera capture pipeline (webcam, IP camera, video files)
- [x] YOLO person + object detection with ByteTrack
- [x] Event tracking (appeared, loitering, departed, returned, companion, objects changed)
- [x] AI evaluation agent with local LLM
- [x] SQLite storage for events + decisions
- [x] Telegram alerts with snapshots
- [x] Escalation chains with severity-based routing
- [x] Telegram inline button acknowledgment
- [x] Role management (invite, join, revoke)
- [x] REST API for events, config, roles
- [x] Redis Streams event bus + scene memory
- [x] Quiet hours configuration (runtime adjustable)

### Planned
- [ ] Vision-based image description (Ollama VL + Claude Vision)
- [ ] Next.js dashboard (event log, live view, config)
- [ ] Multi-camera support
- [ ] Zone drawing on camera view
- [ ] SMS fallback alerts (Termii — DND-bypass for Nigerian networks)
- [ ] Action recognition (SlowFast/VideoMAE)
- [ ] Cloud sync for multi-site deployments
- [ ] User authentication on dashboard

---

## Nigerian Context

Eyes of Horus is built with Nigerian infrastructure realities in mind:

- **Offline-first** — core detection and AI run locally. No internet needed.
- **Power-aware** — 10-second video segments, SQLite WAL mode for crash recovery. Power cuts don't lose data.
- **Low bandwidth** — Telegram works on 2G/3G. Snapshots are compressed JPEG.
- **DND-aware** — 60%+ of Nigerian mobile users are on Do Not Disturb. Production SMS uses Termii transactional routes that bypass DND.
- **Runs on available hardware** — designed for laptop-class machines, not expensive servers.

---

## Contributing

Eyes of Horus is open source and contributions are welcome.

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes
4. Test locally
5. Submit a pull request

Please open an issue first for large changes so we can discuss the approach.

---

## Name

**Horus** was the ancient Egyptian god of the sky. His eye — the Eye of Horus — was a symbol of protection, royal power, and good health. It watched over the people.

Eyes of Horus watches over your space the same way. Always alert. Always watching.

---

## License

MIT License. See [LICENSE](LICENSE) for details.

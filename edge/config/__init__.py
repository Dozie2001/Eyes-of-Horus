"""
Configuration module for StangWatch.

Two layers:
- config.yaml  — non-secret settings (cameras, thresholds, site info)
- .env         — secrets only (API keys, camera passwords)

Pydantic validates everything. Call get_config() to load both and get
a typed StangWatchConfig object.
"""

import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, field_validator
import yaml
from dotenv import dotenv_values


# --- Project root (two levels up from this file) ---
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# --- Pydantic models ---

class Secrets(BaseModel):
    """Loaded from .env — API keys, camera passwords, tokens."""
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    anthropic_api_key: str = ""
    gemini_api_key: str = ""
    groq_api_key: str = ""
    openrouter_api_key: str = ""
    openclaw_token: str = ""
    supabase_url: str = ""
    supabase_service_key: str = ""
    jina_api_key: str = ""
    huggingface_api_key: str = ""
    cohere_api_key: str = ""
    google_api_key: str = ""
    voyage_api_key: str = ""
    # Camera credentials are dynamic — accessed via get_camera_credential()

    # Raw env dict kept for camera credential lookups
    _env: dict = {}

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def get_camera_credential(self, camera_name: str, field: str) -> str:
        """
        Look up a camera credential from .env.

        Convention: CAMERA_{NAME}_{FIELD}
        e.g. camera_name="front gate", field="user" -> CAMERA_FRONT_GATE_USER

        Args:
            camera_name: human-readable camera name
            field: "user" or "password"

        Returns:
            The credential string, or "" if not set.
        """
        key = "CAMERA_" + camera_name.upper().replace(" ", "_") + "_" + field.upper()
        return self._env.get(key, "")


class SiteConfig(BaseModel):
    """Site identification and general settings."""
    id: str = "default"
    name: str = "My Location"
    log_level: str = "INFO"
    environment: str = "development"  # "development" | "production"

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v):
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in allowed:
            raise ValueError(f"log_level must be one of {allowed}, got '{v}'")
        return v.upper()

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v):
        allowed = {"development", "production"}
        if v.lower() not in allowed:
            raise ValueError(f"environment must be one of {allowed}, got '{v}'")
        return v.lower()


class TrackingConfig(BaseModel):
    """EventTracker thresholds — can be overridden per camera."""
    model_config = ConfigDict(extra="ignore")  # ignore old loiter_threshold, stationary_threshold

    quiet_hours: Optional[dict] = None
    departure_seconds: float = 3.0
    companion_distance: float = 200.0
    summary_interval: float = 60.0  # seconds between periodic track summaries


class CameraConfig(BaseModel):
    """One camera source."""
    name: str
    source: str  # URL or file path. May contain {user} and {password} placeholders.
    enabled: bool = True
    tracking: Optional[TrackingConfig] = None  # per-camera overrides

    def resolved_source(self, secrets: Secrets) -> str:
        """
        Replace {user} and {password} placeholders in the source URL
        with credentials from .env.

        Returns the source string with credentials injected.
        """
        if "{user}" not in self.source and "{password}" not in self.source:
            return self.source

        user = secrets.get_camera_credential(self.name, "user")
        password = secrets.get_camera_credential(self.name, "password")

        result = self.source.replace("{user}", user)
        result = result.replace("{password}", password)
        return result

    def effective_tracking(self, global_tracking: TrackingConfig) -> TrackingConfig:
        """
        Merge per-camera tracking overrides with global defaults.

        Per-camera values win where set; global values fill the rest.
        """
        if self.tracking is None:
            return global_tracking

        # Start with global values, override with per-camera values
        merged = global_tracking.model_dump()
        overrides = self.tracking.model_dump(exclude_none=True)

        # Only override fields that were explicitly set (not None)
        for key, value in overrides.items():
            if value is not None:
                merged[key] = value

        return TrackingConfig(**merged)


class RoboflowSegmentationConfig(BaseModel):
    """SAM3 segmentation settings — where to run it, how to reach it."""
    provider: str = "serverless"
    local_url: str = "http://localhost:9001"
    serverless_url: str = "https://serverless.roboflow.com"
    confidence_threshold: float = 0.5
    timeout_seconds: float = 120.0


class RoboflowConfig(BaseModel):
    """Roboflow Inference Server settings."""
    server_url: str = "http://localhost:9001"
    model_id: str = "yolo26s-640"
    api_key: str = ""
    segmentation: RoboflowSegmentationConfig = RoboflowSegmentationConfig()


class DetectionConfig(BaseModel):
    """Detection settings."""
    provider: str = "roboflow"  # "roboflow" (default) or "ultralytics" (legacy)
    model_name: str = "yolo11s.pt"  # only used when provider=ultralytics
    confidence_threshold: float = 0.5
    association_distance: float = 200.0

    @field_validator("confidence_threshold")
    @classmethod
    def validate_confidence(cls, v):
        if not 0.0 < v <= 1.0:
            raise ValueError(f"confidence_threshold must be between 0.0 (exclusive) and 1.0, got {v}")
        return v


class RedisConfig(BaseModel):
    enabled: bool = True
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    username: Optional[str] = None
    password: Optional[str] = None


class StorageConfig(BaseModel):
    """SQLite + optional cloud sync."""
    db_path: str = "data/stangwatch.db"
    cloud_sync: bool = False
    cloud_provider: str = ""
    cloud_url: str = ""


class CloudConfig(BaseModel):
    """Cloud sync settings for SaaS mode."""
    enabled: bool = False
    api_url: str = ""        # https://api.eyesofhorus.com
    api_key: str = ""        # device API key (sk_device_xxx)
    sync_interval: int = 30  # seconds between sync cycles
    heartbeat_interval: int = 60  # seconds between heartbeat pushes


class EmbeddingConfig(BaseModel):
    """Embedding provider settings for clip search."""
    provider: str = "jina"          # jina | ollama | huggingface | cohere | google | voyage | openai
    model: str = "jina-embeddings-v3"
    dimensions: int = 1024          # jina-embeddings-v3 outputs 1024
    ollama_host: str = "http://localhost:11434"  # only used when provider=ollama


class SupabaseConfig(BaseModel):
    """Supabase connection for cloud storage + vector search."""
    enabled: bool = False
    url: str = ""
    service_key: str = ""
    storage_bucket: str = "CLIPS"


class AgentConfig(BaseModel):
    """AI evaluation agent settings."""
    enabled: bool = True
    model: str = "qwen2.5:7b"                    # local Ollama model (fallback)
    vision_model: str = "qwen2.5vl:7b"           # local vision LLM (fallback)
    text_provider: str = "openrouter"             # "openrouter" | "ollama"
    vision_provider: str = "openrouter"           # "openrouter" | "groq" | "gemini" | "ollama"
    ollama_host: str = "http://localhost:11434"
    timeout_seconds: float = 60.0
    cooldown_seconds: float = 120.0               # min seconds between alerts for same track


class AlertConfig(BaseModel):
    """Alert delivery settings."""
    cooldown_seconds: float = 60.0
    language: str = "en"

    @field_validator("language")
    @classmethod
    def validate_language(cls, v):
        allowed = {"en", "pidgin"}
        if v not in allowed:
            raise ValueError(f"language must be one of {allowed}, got '{v}'")
        return v


class EscalationRole(BaseModel):
    """One role in the escalation chain (guard, supervisor, admin).

    Defines structure only — who is assigned to each role is stored in the
    role_membership DB table, managed via bot commands or dashboard API.
    """
    name: str
    level: int                              # 1=guard, 2=supervisor, 3=admin
    abilities: list[str] = ["acknowledge"]  # acknowledge, mute_1h, mute_8h, mute_permanent, configure


class EscalationStep(BaseModel):
    """One step in an escalation chain — a role + how long to wait."""
    role: str
    timeout_minutes: int = 0  # 0 = last in chain, no further escalation


class EscalationPolicy(BaseModel):
    """Escalation chain for a given severity level."""
    severity: str             # "medium" or "high"
    chain: list[EscalationStep]


class EscalationConfig(BaseModel):
    """Escalation policy engine settings. Off by default for backward compat."""
    enabled: bool = False
    roles: list[EscalationRole] = []
    policies: list[EscalationPolicy] = []


class StangWatchConfig(BaseModel):
    """Top-level config combining all sections."""
    site: SiteConfig = SiteConfig()
    cameras: list[CameraConfig] = []
    detection: DetectionConfig = DetectionConfig()
    roboflow: RoboflowConfig = RoboflowConfig()
    tracking: TrackingConfig = TrackingConfig()
    redis: RedisConfig = RedisConfig()
    storage: StorageConfig = StorageConfig()
    cloud: CloudConfig = CloudConfig()
    alerts: AlertConfig = AlertConfig()
    agent: AgentConfig = AgentConfig()
    escalation: EscalationConfig = EscalationConfig()
    embeddings: EmbeddingConfig = EmbeddingConfig()
    supabase: SupabaseConfig = SupabaseConfig()
    secrets: Secrets = Secrets()


# --- Singleton cache ---
_cached_config: Optional[StangWatchConfig] = None


def _parse_escalation(raw_esc: dict) -> EscalationConfig:
    """Parse the escalation section from raw YAML dict.

    Roles define structure only (name, level, abilities).
    Who is assigned to each role lives in the role_membership DB table.
    """
    if not raw_esc:
        return EscalationConfig()

    roles = [
        EscalationRole(
            name=r["name"],
            level=r["level"],
            abilities=r.get("abilities", ["acknowledge"]),
        )
        for r in raw_esc.get("roles", [])
    ]
    policies = [
        EscalationPolicy(
            severity=p["severity"],
            chain=[EscalationStep(**s) for s in p.get("chain", [])],
        )
        for p in raw_esc.get("policies", [])
    ]
    return EscalationConfig(
        enabled=raw_esc.get("enabled", False),
        roles=roles,
        policies=policies,
    )


def get_config(
    config_path: Optional[str] = None,
    env_path: Optional[str] = None,
    reload: bool = False,
) -> StangWatchConfig:
    """
    Load config.yaml + .env and return a validated StangWatchConfig.

    Args:
        config_path: path to config.yaml. Defaults to PROJECT_ROOT/config.yaml.
        env_path: path to .env. Defaults to PROJECT_ROOT/.env.
        reload: if True, ignore cached config and reload from disk.

    Returns:
        StangWatchConfig with all sections populated and validated.
    """
    global _cached_config

    if _cached_config is not None and not reload:
        return _cached_config

    # Resolve paths
    if config_path is None:
        config_path = str(_PROJECT_ROOT / "config.yaml")
    if env_path is None:
        env_path = str(_PROJECT_ROOT / ".env")

    # Load YAML (empty dict if file doesn't exist)
    raw = {}
    if os.path.exists(config_path):
        with open(config_path) as f:
            loaded = yaml.safe_load(f)
            if loaded is not None:
                raw = loaded

    # Load .env
    env = {}
    if os.path.exists(env_path):
        env = dotenv_values(env_path)

    # Also include actual environment variables (they override .env file)
    for key in list(env.keys()):
        if key in os.environ:
            env[key] = os.environ[key]
    # Pick up env vars that aren't in the file but are set in the environment
    _env_prefixes = (
        "TELEGRAM_", "ANTHROPIC_", "OPENCLAW_", "CAMERA_", "REDIS_", "CLOUD_",
        "OPENROUTER_", "GROQ_", "GEMINI_", "SUPABASE_", "JINA_", "HUGGINGFACE_",
        "COHERE_", "GOOGLE_", "VOYAGE_", "OPENAI_", "ROBOFLOW_", "OLLAMA_",
        "STANGWATCH_", "LOG_LEVEL",
    )
    for key in os.environ:
        if key.startswith(_env_prefixes) or key == "LOG_LEVEL":
            if key not in env:
                env[key] = os.environ[key]

    # Build secrets
    secrets = Secrets(
        telegram_bot_token=env.get("TELEGRAM_BOT_TOKEN", ""),
        telegram_chat_id=env.get("TELEGRAM_CHAT_ID", ""),
        anthropic_api_key=env.get("ANTHROPIC_API_KEY", ""),
        gemini_api_key=env.get("GEMINI_API_KEY", ""),
        groq_api_key=env.get("GROQ_API_KEY", ""),
        openrouter_api_key=env.get("OPENROUTER_API_KEY", ""),
        openclaw_token=env.get("OPENCLAW_TOKEN", ""),
        supabase_url=env.get("SUPABASE_URL", ""),
        supabase_service_key=env.get("SUPABASE_SERVICE_KEY", ""),
        jina_api_key=env.get("JINA_API_KEY", ""),
        huggingface_api_key=env.get("HUGGINGFACE_API_KEY", ""),
        cohere_api_key=env.get("COHERE_API_KEY", ""),
        google_api_key=env.get("GOOGLE_API_KEY", ""),
        voyage_api_key=env.get("VOYAGE_API_KEY", ""),
    )
    secrets._env = env

    # --- Roboflow segmentation: merge YAML + env, auto-pick provider ---
    raw_seg = raw.get("roboflow", {}).get("segmentation", {})
    seg_provider = (
        env.get("ROBOFLOW_SEGMENTATION_PROVIDER")
        or raw_seg.get("provider")
    )
    # If provider is unset AND an API key is present, default to serverless.
    # Without an API key we can't call serverless, so local is the only option.
    if seg_provider is None:
        seg_provider = "serverless" if env.get("ROBOFLOW_API_KEY") else "local"

    seg_config = RoboflowSegmentationConfig(
        **{
            **raw_seg,
            "provider": seg_provider,
            **({"local_url": env["ROBOFLOW_SEGMENTATION_LOCAL_URL"]}
               if env.get("ROBOFLOW_SEGMENTATION_LOCAL_URL") else {}),
            **({"serverless_url": env["ROBOFLOW_SEGMENTATION_SERVERLESS_URL"]}
               if env.get("ROBOFLOW_SEGMENTATION_SERVERLESS_URL") else {}),
        }
    )

    # Build config from YAML sections
    config = StangWatchConfig(
        site=SiteConfig(**{
            **raw.get("site", {}),
            **({"environment": env["STANGWATCH_ENV"]}
               if env.get("STANGWATCH_ENV") else {}),
            **({"log_level": env["LOG_LEVEL"]}
               if env.get("LOG_LEVEL") else {}),
        }),
        cameras=[CameraConfig(**c) for c in raw.get("cameras", [])],
        detection=DetectionConfig(**raw.get("detection", {})),
        roboflow=RoboflowConfig(**{
            **{k: v for k, v in raw.get("roboflow", {}).items() if k != "segmentation"},
            **({"server_url": env["ROBOFLOW_SERVER_URL"]}
               if env.get("ROBOFLOW_SERVER_URL") else {}),
            **({"api_key": env["ROBOFLOW_API_KEY"]}
               if env.get("ROBOFLOW_API_KEY") else {}),
            "segmentation": seg_config,
        }),
        tracking=TrackingConfig(**raw.get("tracking", {})),
        redis=RedisConfig(
            **{
                **raw.get("redis", {}),
                # .env overrides config.yaml for Redis connection
                **({"host": env["REDIS_HOST"]} if env.get("REDIS_HOST") else {}),
                **({"port": int(env["REDIS_PORT"])} if env.get("REDIS_PORT") else {}),
                **({"username": env["REDIS_USERNAME"]} if env.get("REDIS_USERNAME") else {}),
                **({"password": env["REDIS_PASSWORD"]} if env.get("REDIS_PASSWORD") else {}),
            }
        ),
        storage=StorageConfig(**raw.get("storage", {})),
        cloud=CloudConfig(**{
            **raw.get("cloud", {}),
            **({"api_url": env["CLOUD_API_URL"]} if env.get("CLOUD_API_URL") else {}),
            **({"api_key": env["CLOUD_API_KEY"]} if env.get("CLOUD_API_KEY") else {}),
        }),
        alerts=AlertConfig(**raw.get("alerts", {})),
        agent=AgentConfig(**{
            **raw.get("agent", {}),
            **({"ollama_host": env["OLLAMA_HOST"]}
               if env.get("OLLAMA_HOST") else {}),
        }),
        escalation=_parse_escalation(raw.get("escalation", {})),
        embeddings=EmbeddingConfig(**{
            **raw.get("embeddings", {}),
            **({"ollama_host": env["OLLAMA_HOST"]}
               if env.get("OLLAMA_HOST") else {}),
        }),
        supabase=SupabaseConfig(**{
            **raw.get("supabase", {}),
            # .env overrides config.yaml for Supabase connection
            **({"url": env["SUPABASE_URL"]} if env.get("SUPABASE_URL") else {}),
            **({"service_key": env["SUPABASE_SERVICE_KEY"]} if env.get("SUPABASE_SERVICE_KEY") else {}),
            **({"storage_bucket": env["SUPABASE_STORAGE_BUCKET"]} if env.get("SUPABASE_STORAGE_BUCKET") else {}),
            # Auto-enable if credentials are present
            **({"enabled": True} if env.get("SUPABASE_URL") and env.get("SUPABASE_SERVICE_KEY") else {}),
        }),
        secrets=secrets,
    )

    _cached_config = config
    return config


def reset_config():
    """Clear the cached config. Mainly for testing."""
    global _cached_config
    _cached_config = None

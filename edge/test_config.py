"""
Tests for the StangWatch config module.

Covers: default config, YAML loading, validation errors, secrets,
camera credential resolution, per-camera tracking overrides, singleton caching.
"""

import pytest
import yaml

from config import (
    get_config,
    reset_config,
    StangWatchConfig,
    CameraConfig,
    TrackingConfig,
    Secrets,
)


@pytest.fixture(autouse=True)
def clear_cache():
    """Reset the singleton cache before each test."""
    reset_config()
    yield
    reset_config()


def _write_yaml(path, data):
    with open(path, "w") as f:
        yaml.dump(data, f)


def _write_env(path, lines):
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# --- Default config (no YAML file) ---

class TestDefaults:
    def test_default_config_when_no_yaml(self, tmp_path):
        """get_config() works with no config.yaml — returns all defaults."""
        config = get_config(
            config_path=str(tmp_path / "nonexistent.yaml"),
            env_path=str(tmp_path / "nonexistent.env"),
        )
        assert isinstance(config, StangWatchConfig)
        assert config.site.name == "My Location"
        assert config.site.id == "default"
        assert config.site.log_level == "INFO"
        assert config.cameras == []
        assert config.detection.model_name == "yolo11s.pt"
        assert config.detection.confidence_threshold == 0.5
        assert config.tracking.loiter_threshold == 300.0
        assert config.tracking.quiet_hours is None
        assert config.storage.db_path == "data/stangwatch.db"
        assert config.alerts.language == "en"
        assert config.alerts.cooldown_seconds == 60.0

    def test_default_secrets_are_empty(self, tmp_path):
        config = get_config(
            config_path=str(tmp_path / "nope.yaml"),
            env_path=str(tmp_path / "nope.env"),
        )
        assert config.secrets.telegram_bot_token == ""
        assert config.secrets.anthropic_api_key == ""


# --- Loading a valid config.yaml ---

class TestYAMLLoading:
    def test_loads_site_from_yaml(self, tmp_path):
        _write_yaml(tmp_path / "config.yaml", {
            "site": {"id": "warehouse-01", "name": "Ikeja Warehouse", "log_level": "DEBUG"},
        })
        config = get_config(
            config_path=str(tmp_path / "config.yaml"),
            env_path=str(tmp_path / "nope.env"),
        )
        assert config.site.id == "warehouse-01"
        assert config.site.name == "Ikeja Warehouse"
        assert config.site.log_level == "DEBUG"

    def test_loads_cameras_from_yaml(self, tmp_path):
        _write_yaml(tmp_path / "config.yaml", {
            "cameras": [
                {"name": "front gate", "source": "rtsp://192.168.1.10/stream"},
                {"name": "back door", "source": "test_videos/back.mp4", "enabled": False},
            ],
        })
        config = get_config(
            config_path=str(tmp_path / "config.yaml"),
            env_path=str(tmp_path / "nope.env"),
        )
        assert len(config.cameras) == 2
        assert config.cameras[0].name == "front gate"
        assert config.cameras[0].enabled is True
        assert config.cameras[1].name == "back door"
        assert config.cameras[1].enabled is False

    def test_loads_detection_overrides(self, tmp_path):
        _write_yaml(tmp_path / "config.yaml", {
            "detection": {"confidence_threshold": 0.7, "model_name": "yolo11m.pt"},
        })
        config = get_config(
            config_path=str(tmp_path / "config.yaml"),
            env_path=str(tmp_path / "nope.env"),
        )
        assert config.detection.confidence_threshold == 0.7
        assert config.detection.model_name == "yolo11m.pt"

    def test_loads_tracking_with_quiet_hours(self, tmp_path):
        _write_yaml(tmp_path / "config.yaml", {
            "tracking": {
                "loiter_threshold": 120,
                "quiet_hours": {"start": "22:00", "end": "06:00"},
            },
        })
        config = get_config(
            config_path=str(tmp_path / "config.yaml"),
            env_path=str(tmp_path / "nope.env"),
        )
        assert config.tracking.loiter_threshold == 120.0
        assert config.tracking.quiet_hours == {"start": "22:00", "end": "06:00"}

    def test_empty_yaml_file_uses_defaults(self, tmp_path):
        """An empty YAML file (parses as None) should use all defaults."""
        (tmp_path / "config.yaml").write_text("")
        config = get_config(
            config_path=str(tmp_path / "config.yaml"),
            env_path=str(tmp_path / "nope.env"),
        )
        assert config.site.name == "My Location"
        assert config.cameras == []


# --- Validation errors ---

class TestValidation:
    def test_bad_confidence_too_high(self, tmp_path):
        _write_yaml(tmp_path / "config.yaml", {
            "detection": {"confidence_threshold": 1.5},
        })
        with pytest.raises(Exception, match="confidence_threshold"):
            get_config(
                config_path=str(tmp_path / "config.yaml"),
                env_path=str(tmp_path / "nope.env"),
            )

    def test_bad_confidence_zero(self, tmp_path):
        _write_yaml(tmp_path / "config.yaml", {
            "detection": {"confidence_threshold": 0.0},
        })
        with pytest.raises(Exception, match="confidence_threshold"):
            get_config(
                config_path=str(tmp_path / "config.yaml"),
                env_path=str(tmp_path / "nope.env"),
            )

    def test_bad_log_level(self, tmp_path):
        _write_yaml(tmp_path / "config.yaml", {
            "site": {"log_level": "VERBOSE"},
        })
        with pytest.raises(Exception, match="log_level"):
            get_config(
                config_path=str(tmp_path / "config.yaml"),
                env_path=str(tmp_path / "nope.env"),
            )

    def test_bad_language(self, tmp_path):
        _write_yaml(tmp_path / "config.yaml", {
            "alerts": {"language": "french"},
        })
        with pytest.raises(Exception, match="language"):
            get_config(
                config_path=str(tmp_path / "config.yaml"),
                env_path=str(tmp_path / "nope.env"),
            )


# --- Secret loading ---

class TestSecrets:
    def test_loads_secrets_from_env_file(self, tmp_path):
        _write_env(tmp_path / ".env", [
            "TELEGRAM_BOT_TOKEN=abc123",
            "TELEGRAM_CHAT_ID=999",
            "ANTHROPIC_API_KEY=sk-ant-xxx",
        ])
        config = get_config(
            config_path=str(tmp_path / "nope.yaml"),
            env_path=str(tmp_path / ".env"),
        )
        assert config.secrets.telegram_bot_token == "abc123"
        assert config.secrets.telegram_chat_id == "999"
        assert config.secrets.anthropic_api_key == "sk-ant-xxx"

    def test_env_vars_override_env_file(self, tmp_path, monkeypatch):
        _write_env(tmp_path / ".env", [
            "TELEGRAM_BOT_TOKEN=from_file",
        ])
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "from_env")
        config = get_config(
            config_path=str(tmp_path / "nope.yaml"),
            env_path=str(tmp_path / ".env"),
        )
        assert config.secrets.telegram_bot_token == "from_env"


# --- Camera credential resolution ---

class TestCameraCredentials:
    def test_resolved_source_injects_credentials(self, tmp_path):
        _write_env(tmp_path / ".env", [
            "CAMERA_FRONT_GATE_USER=admin",
            "CAMERA_FRONT_GATE_PASSWORD=secret123",
        ])
        _write_yaml(tmp_path / "config.yaml", {
            "cameras": [{
                "name": "front gate",
                "source": "rtsp://{user}:{password}@192.168.1.100/stream",
            }],
        })
        config = get_config(
            config_path=str(tmp_path / "config.yaml"),
            env_path=str(tmp_path / ".env"),
        )
        cam = config.cameras[0]
        resolved = cam.resolved_source(config.secrets)
        assert resolved == "rtsp://admin:secret123@192.168.1.100/stream"

    def test_resolved_source_no_placeholders(self):
        """Source without placeholders is returned unchanged."""
        cam = CameraConfig(name="test", source="test_videos/clip.mp4")
        secrets = Secrets()
        assert cam.resolved_source(secrets) == "test_videos/clip.mp4"

    def test_missing_credentials_resolve_to_empty(self, tmp_path):
        """If credentials aren't in .env, placeholders become empty strings."""
        _write_yaml(tmp_path / "config.yaml", {
            "cameras": [{
                "name": "back door",
                "source": "rtsp://{user}:{password}@192.168.1.200/stream",
            }],
        })
        config = get_config(
            config_path=str(tmp_path / "config.yaml"),
            env_path=str(tmp_path / "nope.env"),
        )
        cam = config.cameras[0]
        resolved = cam.resolved_source(config.secrets)
        assert resolved == "rtsp://:@192.168.1.200/stream"


# --- Per-camera tracking overrides ---

class TestTrackingOverrides:
    def test_no_override_returns_global(self):
        global_tracking = TrackingConfig(loiter_threshold=300)
        cam = CameraConfig(name="test", source="file.mp4")
        effective = cam.effective_tracking(global_tracking)
        assert effective.loiter_threshold == 300.0

    def test_partial_override_merges(self):
        global_tracking = TrackingConfig(
            loiter_threshold=300,
            stationary_threshold=5.0,
            departure_seconds=3.0,
        )
        cam = CameraConfig(
            name="test",
            source="file.mp4",
            tracking=TrackingConfig(loiter_threshold=120),
        )
        effective = cam.effective_tracking(global_tracking)
        assert effective.loiter_threshold == 120.0  # overridden
        assert effective.stationary_threshold == 5.0  # from global
        assert effective.departure_seconds == 3.0  # from global

    def test_full_override(self):
        global_tracking = TrackingConfig(loiter_threshold=300, departure_seconds=3.0)
        cam = CameraConfig(
            name="test",
            source="file.mp4",
            tracking=TrackingConfig(
                loiter_threshold=60,
                departure_seconds=10.0,
                stationary_threshold=2.0,
                companion_distance=100.0,
            ),
        )
        effective = cam.effective_tracking(global_tracking)
        assert effective.loiter_threshold == 60.0
        assert effective.departure_seconds == 10.0
        assert effective.stationary_threshold == 2.0
        assert effective.companion_distance == 100.0


# --- Singleton caching + reload ---

class TestCaching:
    def test_returns_same_object_on_second_call(self, tmp_path):
        config1 = get_config(
            config_path=str(tmp_path / "nope.yaml"),
            env_path=str(tmp_path / "nope.env"),
        )
        config2 = get_config(
            config_path=str(tmp_path / "nope.yaml"),
            env_path=str(tmp_path / "nope.env"),
        )
        assert config1 is config2

    def test_reload_returns_fresh_object(self, tmp_path):
        config1 = get_config(
            config_path=str(tmp_path / "nope.yaml"),
            env_path=str(tmp_path / "nope.env"),
        )
        config2 = get_config(
            config_path=str(tmp_path / "nope.yaml"),
            env_path=str(tmp_path / "nope.env"),
            reload=True,
        )
        assert config1 is not config2

    def test_reload_picks_up_changes(self, tmp_path):
        _write_yaml(tmp_path / "config.yaml", {
            "site": {"name": "Version 1"},
        })
        config1 = get_config(
            config_path=str(tmp_path / "config.yaml"),
            env_path=str(tmp_path / "nope.env"),
        )
        assert config1.site.name == "Version 1"

        _write_yaml(tmp_path / "config.yaml", {
            "site": {"name": "Version 2"},
        })
        config2 = get_config(
            config_path=str(tmp_path / "config.yaml"),
            env_path=str(tmp_path / "nope.env"),
            reload=True,
        )
        assert config2.site.name == "Version 2"

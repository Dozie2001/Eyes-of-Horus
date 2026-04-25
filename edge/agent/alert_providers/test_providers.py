"""
Unit tests for the alert provider plugin system.

Run from the edge/ directory:
    python -m agent.alert_providers.test_providers
"""

import sys


def test_telegram_unconfigured():
    from agent.alert_providers.telegram import TelegramProvider

    p = TelegramProvider(bot_token="", chat_id="")
    assert p.name == "telegram"
    assert p.is_configured() is False
    assert p.supports_feedback() is True
    # Should return False (not crash) when unconfigured
    assert p.send_alert({"event_type": "appeared"}) is False
    assert p.send_escalation(1, {"severity": "high", "reason": "x"}, "guard") is False
    print("test_telegram_unconfigured OK")


def test_whatsapp_stub():
    from agent.alert_providers.whatsapp import WhatsAppProvider

    p = WhatsAppProvider()
    assert p.name == "whatsapp"
    assert p.is_configured() is False
    assert p.supports_feedback() is True
    assert p.send_alert({}) is False
    assert p.send_escalation(1, {}, "guard") is False
    print("test_whatsapp_stub OK")


def test_sms_stub():
    from agent.alert_providers.sms import SMSProvider

    p = SMSProvider()
    assert p.name == "sms"
    assert p.is_configured() is False
    assert p.supports_feedback() is False
    assert p.send_alert({}) is False
    print("test_sms_stub OK")


def test_registry():
    from agent.alert_providers.registry import AlertProviderRegistry
    from agent.alert_providers.telegram import TelegramProvider
    from agent.alert_providers.whatsapp import WhatsAppProvider
    from agent.alert_providers.sms import SMSProvider

    providers = [
        TelegramProvider(bot_token="", chat_id=""),
        WhatsAppProvider(),
        SMSProvider(),
    ]
    registry = AlertProviderRegistry(providers)

    assert len(registry.all()) == 3
    # None are configured since we gave empty credentials
    assert len(registry.enabled()) == 0

    # get()
    assert registry.get("telegram") is not None
    assert registry.get("nonexistent") is None

    # broadcast_alert on empty enabled list returns empty dict
    results = registry.broadcast_alert({"event_type": "appeared"})
    assert results == {}

    # status() shape
    status = registry.status()
    assert set(status.keys()) == {"telegram", "whatsapp", "sms"}
    for name, info in status.items():
        assert "configured" in info
        assert "supports_feedback" in info
    assert status["telegram"]["supports_feedback"] is True
    assert status["sms"]["supports_feedback"] is False

    print("test_registry OK")


def test_registry_with_fake_enabled_provider():
    """Verify broadcast works when a provider reports configured=True."""
    from agent.alert_providers.base import AlertProvider
    from agent.alert_providers.registry import AlertProviderRegistry

    class FakeProvider(AlertProvider):
        def __init__(self):
            self.sent = []

        @property
        def name(self):
            return "fake"

        def is_configured(self):
            return True

        def send_alert(self, data):
            self.sent.append(data)
            return True

        def send_escalation(self, alert_id, data, role):
            return True

    fake = FakeProvider()
    reg = AlertProviderRegistry([fake])

    assert len(reg.enabled()) == 1
    results = reg.broadcast_alert({"event_type": "departed"})
    assert results == {"fake": True}
    assert len(fake.sent) == 1
    assert fake.sent[0]["event_type"] == "departed"
    print("test_registry_with_fake_enabled_provider OK")


if __name__ == "__main__":
    try:
        test_telegram_unconfigured()
        test_whatsapp_stub()
        test_sms_stub()
        test_registry()
        test_registry_with_fake_enabled_provider()
        print("\nAll alert provider tests passed.")
    except AssertionError as e:
        print(f"FAIL: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        sys.exit(1)

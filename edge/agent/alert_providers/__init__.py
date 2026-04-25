"""
Alert provider plugin system for StangWatch.

Provides a pluggable interface for alert delivery channels. The existing
TelegramSender in edge/agent/telegram.py remains unchanged and in use —
the providers here wrap it and add a uniform interface for future channels
(WhatsApp, SMS, etc.).
"""

from agent.alert_providers.base import AlertProvider
from agent.alert_providers.registry import AlertProviderRegistry
from agent.alert_providers.telegram import TelegramProvider
from agent.alert_providers.whatsapp import WhatsAppProvider
from agent.alert_providers.sms import SMSProvider

__all__ = [
    "AlertProvider",
    "AlertProviderRegistry",
    "TelegramProvider",
    "WhatsAppProvider",
    "SMSProvider",
]

"""Environment-backed configuration for DommerAI."""

from __future__ import annotations

import os

API_KEY = os.environ.get("DOMMER_API_KEY", "dev-key-change-in-prod")
DEFAULT_WEBHOOK_URL = os.environ.get("WEBHOOK_URL", "").strip()
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
APP_VERSION = "2.0.0"


def cors_origins() -> list[str]:
    return [
        origin.strip()
        for origin in os.environ.get("CORS_ORIGINS", "*").split(",")
        if origin.strip()
    ]

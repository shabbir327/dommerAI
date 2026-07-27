"""Environment-backed configuration for DommerAI."""

from __future__ import annotations

import os


# Core DommerAI configuration
API_KEY = os.environ.get(
    "DOMMER_API_KEY",
    "dev-key-change-in-prod",
)

DEFAULT_WEBHOOK_URL = os.environ.get(
    "WEBHOOK_URL",
    "",
).strip()

LOG_LEVEL = os.environ.get(
    "LOG_LEVEL",
    "INFO",
).upper()

APP_VERSION = "2.2.0"


# Main DommerAI Supabase project
SUPABASE_URL = os.environ.get(
    "SUPABASE_URL",
    "",
).strip()

SUPABASE_SERVICE_ROLE_KEY = os.environ.get(
    "SUPABASE_SERVICE_ROLE_KEY",
    "",
).strip()


# COR / DanskGrammatik Hub Supabase project
SUPABASE_URL_COR = os.environ.get(
    "SUPABASE_URL_COR",
    "",
).strip()

SUPABASE_KEY_COR = os.environ.get(
    "SUPABASE_KEY_COR",
    "",
).strip()


def cors_origins() -> list[str]:
    return [
        origin.strip()
        for origin in os.environ.get("CORS_ORIGINS", "*").split(",")
        if origin.strip()
    ]

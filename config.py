"""Environment-backed configuration for DommerAI."""

from __future__ import annotations

import os

API_KEY = os.environ.get("DOMMER_API_KEY", "dev-key-change-in-prod")
DEFAULT_WEBHOOK_URL = os.environ.get("WEBHOOK_URL", "").strip()
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
APP_VERSION = "2.1.2"

# Two separate Supabase projects are intentionally used:
# - DOMMER_*: DKF/EKE knowledge and persisted candidate evaluations
# - SUPABASE_*_COR: COR / ordregister.dk lexical resources
DOMMER_SUPABASE_URL = (
    os.environ.get("DOMMER_SUPABASE_URL")
    or os.environ.get("SUPABASE_URL")
    or ""
).rstrip("/")
DOMMER_SUPABASE_SERVICE_ROLE_KEY = (
    os.environ.get("DOMMER_SUPABASE_SERVICE_ROLE_KEY")
    or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    or ""
)
SUPABASE_URL_COR = os.environ.get("SUPABASE_URL_COR", "").rstrip("/")
SUPABASE_KEY_COR = os.environ.get("SUPABASE_KEY_COR", "")


def cors_origins() -> list[str]:
    return [
        origin.strip()
        for origin in os.environ.get("CORS_ORIGINS", "*").split(",")
        if origin.strip()
    ]

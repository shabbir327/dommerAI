"""Central configuration for DommerAI v1.0."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "DommerAI")
    app_version: str = os.getenv("APP_VERSION", "1.0.0")
    environment: str = os.getenv("ENVIRONMENT", "production")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    dommer_api_key: str = os.getenv("DOMMER_API_KEY", "")
    database_url: str = os.getenv("DATABASE_URL", "")
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    default_webhook_url: str | None = os.getenv("WEBHOOK_URL", "").strip() or None

    webhook_timeout_seconds: float = float(os.getenv("WEBHOOK_TIMEOUT_SECONDS", "10"))
    webhook_retries: int = int(os.getenv("WEBHOOK_RETRIES", "3"))

    def validate(self) -> None:
        missing: list[str] = []
        if not self.dommer_api_key:
            missing.append("DOMMER_API_KEY")
        if not self.database_url:
            missing.append("DATABASE_URL")
        if not self.groq_api_key:
            missing.append("GROQ_API_KEY")
        if missing:
            raise RuntimeError(
                "Missing required environment variables: " + ", ".join(missing)
            )


settings = Settings()

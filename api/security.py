"""Shared API-key authentication dependency."""

from fastapi import HTTPException, Security
from fastapi.security.api_key import APIKeyHeader

from config import API_KEY

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(key: str | None = Security(api_key_header)) -> str:
    if not key or key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")
    return key

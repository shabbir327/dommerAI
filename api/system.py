"""System and knowledge-source health routes."""

from fastapi import APIRouter

from app_state import state
from models import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health() -> HealthResponse:
    counts = state.repository.counts() if state.repository else {}
    sources = state.repository.source_status() if state.repository else {}
    return HealthResponse(
        status="ok",
        scorer_ready=state.scorer is not None,
        knowledge_ready=counts.get("dommer", 0) > 0,
        knowledge_counts=counts,
        knowledge_sources=sources,
    )

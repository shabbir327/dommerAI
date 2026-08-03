"""Consolidated application health route."""

from __future__ import annotations

from fastapi import APIRouter

from app_state import state
from config import APP_VERSION
from models import (
    GrammarHubHealth,
    HealthResponse,
    PersistenceHealth,
    PosTaggerHealth,
    ScorerHealth,
)
from services.scorer import (
    GRADING_PROVIDER,
    GROQ_GRADING_MODEL,
    GROQ_INTERN_MODEL,
    INTERN_PROVIDER,
    PROMPT_VERSION,
)

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health() -> HealthResponse:
    counts = state.repository.counts() if state.repository else {}
    sources = state.repository.source_status() if state.repository else {}

    knowledge_ready = counts.get("dommer", 0) > 0
    scorer_ready = state.scorer is not None
    persistence_ready = state.result_store is not None

    lexical_engine = state.lexical_engine
    lexical_configured = bool(
        lexical_engine is not None and lexical_engine.configured
    )
    integrated = bool(
        scorer_ready
        and lexical_engine is not None
        and getattr(state.scorer, "lexical_engine", None) is lexical_engine
    )

    grammar_data = {
        "status": "not_configured",
        "configured": lexical_configured,
        "database_reachable": False,
        "sample_row_available": False,
        "latency_ms": None,
    }
    grammar_detail = None

    if lexical_configured:
        try:
            grammar_data = await lexical_engine.health()
        except Exception as exc:  # Health must report failures, not crash.
            grammar_data = {
                "status": "error",
                "configured": True,
                "database_reachable": False,
                "sample_row_available": False,
                "latency_ms": None,
            }
            grammar_detail = str(exc)

    grammar_connected = bool(
        grammar_data.get("database_reachable")
        and grammar_data.get("sample_row_available")
    )
    lexical_engine_ready = grammar_connected and integrated

    overall_ok = all(
        (
            scorer_ready,
            knowledge_ready,
            lexical_engine_ready,
            persistence_ready,
        )
    )

    return HealthResponse(
        status="ok" if overall_ok else "degraded",
        app_version=APP_VERSION,
        scorer_ready=scorer_ready,
        knowledge_ready=knowledge_ready,
        lexical_engine_ready=lexical_engine_ready,
        grammar_hub_connected=grammar_connected,
        persistence_ready=persistence_ready,
        scorer=ScorerHealth(
            status="ready" if scorer_ready else "unavailable",
            ready=scorer_ready,
            provider=GRADING_PROVIDER if scorer_ready else None,
            intern_provider=INTERN_PROVIDER if scorer_ready else None,
            model=GROQ_GRADING_MODEL if scorer_ready else None,
            intern_model=GROQ_INTERN_MODEL if scorer_ready else None,
            prompt_version=PROMPT_VERSION if scorer_ready else None,
            grammar_hub_integrated=integrated,
        ),
        grammar_hub=GrammarHubHealth(
            status=str(grammar_data.get("status", "unknown")),
            ready=lexical_engine_ready,
            configured=bool(grammar_data.get("configured", lexical_configured)),
            database_reachable=bool(grammar_data.get("database_reachable", False)),
            sample_row_available=bool(grammar_data.get("sample_row_available", False)),
            latency_ms=grammar_data.get("latency_ms"),
            integrated_into_scorer=integrated,
            detail=grammar_detail,
        ),
        persistence=PersistenceHealth(
            status="ready" if persistence_ready else "unavailable",
            ready=persistence_ready,
            database_client_ready=bool(
                state.repository is not None
                and getattr(state.repository, "client", None) is not None
            ),
        ),
        pos_tagger=PosTaggerHealth(
            status="ready" if (state.pos_tagger is not None and state.pos_tagger.ready) else "unavailable",
            ready=bool(state.pos_tagger is not None and state.pos_tagger.ready),
            model=state.pos_tagger.model_name if (state.pos_tagger is not None and state.pos_tagger.ready) else None,
        ),
        knowledge_counts=counts,
        knowledge_sources=sources,
    )

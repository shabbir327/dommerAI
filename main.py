"""FastAPI entrypoint for DommerAI.

Render can continue to start the service with: uvicorn main:app
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.evaluations import router as evaluations_router
from api.lexicon import router as lexicon_router
from api.system import router as system_router
from app_state import state
from config import APP_VERSION, DEFAULT_WEBHOOK_URL, LOG_LEVEL, cors_origins
from repositories.knowledge_repository import KnowledgeRepository
from repositories.result_store import EvaluationResultStore
from services.examiner_knowledge import ExaminerKnowledgeEngine
from services.lexical_engine import LexicalEngine
from services.pos_tagger import PosTagger
from services.scorer import Scorer

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("dommer.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.repository = KnowledgeRepository()
    state.repository.refresh(force=True)

    examiner_engine = ExaminerKnowledgeEngine(state.repository)

    # Loaded once here, not per-request — this is the expensive part (model
    # weights into memory). If it fails to load, LexicalEngine falls back to
    # its pre-tagger heuristic rather than crashing the whole service.
    state.pos_tagger = PosTagger()
    state.pos_tagger.load()

    state.lexical_engine = LexicalEngine(pos_tagger=state.pos_tagger)
    await state.lexical_engine.start()

    state.scorer = Scorer(examiner_engine, state.lexical_engine)
    state.result_store = EvaluationResultStore(state.repository.client)

    logger.info(
        "Dommer ready — knowledge=%s grammar_hub=%s default_webhook=%s pos_tagger=%s",
        state.repository.counts(),
        "configured" if state.lexical_engine.configured else "not configured",
        DEFAULT_WEBHOOK_URL or "not configured",
        state.pos_tagger.model_name if state.pos_tagger.ready else "unavailable",
    )

    try:
        yield
    finally:
        if state.lexical_engine is not None:
            await state.lexical_engine.close()


app = FastAPI(
    title="Dommer — DanskProeve Writing Evaluator",
    description=(
        "Knowledge-grounded PD2/PD3 evaluator with exact inline grammar locations, "
        "webhook delivery, result polling, and DanskGrammatik Hub lexical services."
    ),
    version=APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins(),
    allow_credentials=False,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

app.include_router(system_router)
app.include_router(evaluations_router)
app.include_router(lexicon_router)

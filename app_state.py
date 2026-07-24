"""Runtime service registry populated during FastAPI lifespan."""

from __future__ import annotations

from repositories.knowledge_repository import KnowledgeRepository
from repositories.result_store import EvaluationResultStore
from services.lexical_engine import LexicalEngine
from services.scorer import Scorer


class AppState:
    repository: KnowledgeRepository | None = None
    scorer: Scorer | None = None
    result_store: EvaluationResultStore | None = None
    lexical_engine: LexicalEngine | None = None


state = AppState()

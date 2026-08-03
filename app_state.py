"""Runtime service registry populated during FastAPI lifespan."""

from __future__ import annotations

from repositories.knowledge_repository import KnowledgeRepository
from repositories.mock_progress_repository import MockProgressStore
from repositories.result_store import EvaluationResultStore
from services.lexical_engine import LexicalEngine
from services.pos_tagger import PosTagger
from services.scorer import Scorer


class AppState:
    repository: KnowledgeRepository | None = None
    scorer: Scorer | None = None
    result_store: EvaluationResultStore | None = None
    mock_progress: MockProgressStore | None = None
    lexical_engine: LexicalEngine | None = None
    pos_tagger: PosTagger | None = None


state = AppState()

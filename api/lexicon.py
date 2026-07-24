"""DanskGrammatik Hub lexical routes."""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field, field_validator

from api.security import require_api_key
from app_state import state
from services.lexical_engine import LexicalEngineError

router = APIRouter()


class TextAnalysisRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=20_000)
    unique_only: bool = True

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("text cannot be empty")
        return value


class LexicalResponse(BaseModel):
    data: dict[str, Any]


def lexical_http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, ValueError):
        return HTTPException(status_code=422, detail=str(exc))
    if isinstance(exc, LexicalEngineError):
        return HTTPException(status_code=503, detail=str(exc))
    return HTTPException(status_code=500, detail="Unexpected lexical engine error.")


def engine():
    if state.lexical_engine is None:
        raise HTTPException(status_code=503, detail="Lexical engine is not ready.")
    return state.lexical_engine


@router.get("/grammar-hub/health", tags=["Grammar Hub"])
async def grammar_hub_health(_: str = Depends(require_api_key)):
    try:
        return await engine().health()
    except Exception as exc:
        raise lexical_http_error(exc) from exc


@router.get("/lexicon/lookup", response_model=LexicalResponse, tags=["Grammar Hub"])
async def lexical_lookup(
    word: str = Query(..., min_length=1),
    normalized_only: bool = False,
    _: str = Depends(require_api_key),
) -> LexicalResponse:
    try:
        return LexicalResponse(data=await engine().lookup_word(word, normalized_only))
    except Exception as exc:
        raise lexical_http_error(exc) from exc


@router.get(
    "/lexicon/lemma/{cor_lemma_id}",
    response_model=LexicalResponse,
    tags=["Grammar Hub"],
)
async def lexical_lemma(
    cor_lemma_id: str,
    _: str = Depends(require_api_key),
) -> LexicalResponse:
    try:
        return LexicalResponse(data=await engine().get_lemma(cor_lemma_id))
    except Exception as exc:
        raise lexical_http_error(exc) from exc


@router.post("/lexicon/analyze", response_model=LexicalResponse, tags=["Grammar Hub"])
async def lexical_analyze(
    request: TextAnalysisRequest,
    _: str = Depends(require_api_key),
) -> LexicalResponse:
    try:
        return LexicalResponse(
            data=await engine().analyze_text(request.text, request.unique_only)
        )
    except Exception as exc:
        raise lexical_http_error(exc) from exc

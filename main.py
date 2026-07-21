"""DommerAI v1.0 FastAPI application."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

import httpx
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader

from config import settings
from database import (
    check_database,
    close_database,
    complete_evaluation,
    create_evaluation,
    evaluation_exists,
    get_evaluation_record,
    mark_processing,
)
from models import (
    AckResponse,
    EvaluationRequest,
    EvaluationStatusResponse,
    HealthResponse,
    WebhookPayload,
)
from scorer import Scorer

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("dommer.api")

scorer: Scorer | None = None
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global scorer

    settings.validate()
    logger.info("Starting %s v%s", settings.app_name, settings.app_version)

    await check_database()
    logger.info("PostgreSQL connection successful")

    scorer = Scorer()
    logger.info("Scorer loaded with model=%s", settings.groq_model)

    yield

    await close_database()
    logger.info("DommerAI shut down")


app = FastAPI(
    title="DommerAI - DanskProeve Writing Evaluator",
    description="Production API for asynchronous PD2/PD3 Danish writing evaluation.",
    version=settings.app_version,
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


async def require_api_key(key: str | None = Security(api_key_header)) -> str:
    if key != settings.dommer_api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")
    return key


async def fire_webhook(
    payload: WebhookPayload,
    destination: str | None,
) -> bool:
    if not destination:
        logger.info("No webhook configured - eval_id=%s", payload.eval_id)
        return False

    body = payload.model_dump(mode="json")
    async with httpx.AsyncClient(timeout=settings.webhook_timeout_seconds) as client:
        for attempt in range(1, settings.webhook_retries + 1):
            try:
                response = await client.post(destination, json=body)
                response.raise_for_status()
                logger.info(
                    "Webhook delivered - eval_id=%s status=%d",
                    payload.eval_id,
                    response.status_code,
                )
                return True
            except Exception as exc:
                logger.warning(
                    "Webhook attempt %d failed - eval_id=%s error=%s",
                    attempt,
                    payload.eval_id,
                    exc,
                )
                if attempt < settings.webhook_retries:
                    await asyncio.sleep(2 ** attempt)

    logger.error("Webhook delivery failed - eval_id=%s", payload.eval_id)
    return False


async def score_and_notify(request: EvaluationRequest) -> None:
    started_at = datetime.now(timezone.utc)

    try:
        await mark_processing(request.eval_id, started_at)

        if scorer is None:
            raise RuntimeError("Scorer is not initialized.")

        payload = await scorer.score(request)

        await complete_evaluation(
            eval_id=request.eval_id,
            status=payload.status,
            completed_at=payload.completed_at,
            rubric=(
                payload.rubrik.model_dump(mode="json")
                if payload.rubrik is not None
                else None
            ),
            overall=payload.overall,
            pass_fail=payload.pass_fail,
            feedback_da=payload.feedback_da,
            errors=[item.model_dump(mode="json") for item in payload.errors],
            word_count=payload.word_count,
            model_name=payload.model_name,
            prompt_version=payload.prompt_version,
            error=payload.error,
        )

        destination = (
            str(request.webhook_url)
            if request.webhook_url is not None
            else settings.default_webhook_url
        )
        await fire_webhook(payload, destination)

    except Exception as exc:
        logger.exception("Background evaluation failed - eval_id=%s", request.eval_id)
        await complete_evaluation(
            eval_id=request.eval_id,
            status="failed",
            completed_at=datetime.now(timezone.utc),
            rubric=None,
            overall=None,
            pass_fail=None,
            feedback_da=None,
            errors=[],
            word_count=None,
            model_name=settings.groq_model,
            prompt_version="v1",
            error=str(exc),
        )


def build_status_response(row: dict[str, Any]) -> EvaluationStatusResponse:
    result: WebhookPayload | None = None

    if row["status"] in {"scored", "failed"}:
        result = WebhookPayload(
            event=(
                "evaluation.completed"
                if row["status"] == "scored"
                else "evaluation.failed"
            ),
            eval_id=row["eval_id"],
            candidate_id=row["candidate_id"],
            status=row["status"],
            submitted_at=row["submitted_at"],
            completed_at=row["completed_at"],
            metadata=row["metadata"] or {},
            webhook_url=row["webhook_url"],
            rubrik=row["rubric"],
            overall=row["overall"],
            pass_fail=row["pass_fail"],
            feedback_da=row["feedback_da"],
            errors=row["errors"] or [],
            word_count=row["word_count"],
            model_name=row["model_name"],
            prompt_version=row["prompt_version"] or "v1",
            error=row["error"],
        )

    return EvaluationStatusResponse(
        eval_id=row["eval_id"],
        status=row["status"],
        candidate_id=row["candidate_id"],
        exam_type=row["exam_type"],
        submitted_at=row["submitted_at"],
        started_at=row["started_at"],
        completed_at=row["completed_at"],
        metadata=row["metadata"] or {},
        webhook_url=row["webhook_url"],
        result=result,
        error=row["error"],
    )


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health() -> HealthResponse:
    database_ready = True
    try:
        await check_database()
    except Exception:
        database_ready = False
        logger.exception("Health database check failed")

    ready = scorer is not None and database_ready
    return HealthResponse(
        status="ok" if ready else "degraded",
        scorer_ready=scorer is not None,
        database_ready=database_ready,
    )


@app.post("/evaluate", response_model=AckResponse, status_code=202, tags=["Scoring"])
async def evaluate(
    request: EvaluationRequest,
    background_tasks: BackgroundTasks,
    _: str = Depends(require_api_key),
) -> AckResponse:
    if await evaluation_exists(request.eval_id):
        raise HTTPException(status_code=409, detail="eval_id already exists")

    try:
        await create_evaluation(
            eval_id=request.eval_id,
            candidate_id=request.candidate_id,
            exam_type=request.exam_type,
            question=request.question,
            question_description=request.question_description,
            answer=request.answer,
            submitted_at=request.submitted_at,
            metadata=request.metadata,
            webhook_url=str(request.webhook_url) if request.webhook_url else None,
        )
    except Exception as exc:
        logger.exception("Could not create evaluation - eval_id=%s", request.eval_id)
        raise HTTPException(status_code=503, detail="Evaluation could not be stored.") from exc

    background_tasks.add_task(score_and_notify, request)

    return AckResponse(
        eval_id=request.eval_id,
        status="pending",
        submitted_at=request.submitted_at,
    )


@app.get(
    "/evaluations/{eval_id}",
    response_model=EvaluationStatusResponse,
    tags=["Scoring"],
)
async def get_evaluation(
    eval_id: str,
    _: str = Depends(require_api_key),
) -> EvaluationStatusResponse:
    row = await get_evaluation_record(eval_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    return build_status_response(row)

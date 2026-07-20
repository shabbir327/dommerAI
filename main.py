import asyncio
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import httpx
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader

from models import (
    AckResponse,
    EvaluationRequest,
    EvaluationStatusResponse,
    HealthResponse,
    WebhookPayload,
)
from scorer import Scorer

API_KEY = os.environ.get("DOMMER_API_KEY")
if not API_KEY:
    raise RuntimeError("DOMMER_API_KEY environment variable is required.")

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("dommer.api")

scorer: Scorer | None = None
results_store: dict[str, EvaluationStatusResponse] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global scorer
    scorer = Scorer()
    logger.info("Dommer API started")
    yield
    logger.info("Dommer API stopped")


app = FastAPI(
    title="Dommer API",
    description="Asynchronous Danish writing evaluation service for PD2 and PD3.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(key: str | None = Security(api_key_header)) -> str:
    if not key or key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")
    return key


async def deliver_webhook(
    webhook_url: str,
    payload: WebhookPayload,
    retries: int = 4,
) -> None:
    async with httpx.AsyncClient(timeout=20.0) as client:
        for attempt in range(1, retries + 1):
            try:
                response = await client.post(webhook_url, json=payload.model_dump(mode="json"))
                response.raise_for_status()
                logger.info("Webhook delivered for eval_id=%s", payload.eval_id)
                return
            except Exception as exc:
                logger.warning(
                    "Webhook attempt %s/%s failed for eval_id=%s: %s",
                    attempt,
                    retries,
                    payload.eval_id,
                    exc,
                )
                if attempt < retries:
                    await asyncio.sleep(2 ** (attempt - 1))

    logger.error("Webhook delivery failed for eval_id=%s", payload.eval_id)


async def score_and_notify(request: EvaluationRequest) -> None:
    assert scorer is not None

    try:
        scored = await scorer.score(request)
        payload = WebhookPayload(
            event="evaluation.completed" if scored.status == "scored" else "evaluation.failed",
            eval_id=request.eval_id,
            candidate_id=request.candidate_id,
            status=scored.status,
            completed_at=datetime.now(timezone.utc),
            rubrik=scored.rubrik,
            overall=scored.overall,
            pass_fail=scored.pass_fail,
            feedback_da=scored.feedback_da,
            errors=scored.errors or [],
            word_count=scored.word_count,
            error=scored.error,
            metadata=request.metadata,
        )
    except Exception as exc:
        logger.exception("Unhandled scoring error for eval_id=%s", request.eval_id)
        payload = WebhookPayload(
            event="evaluation.failed",
            eval_id=request.eval_id,
            candidate_id=request.candidate_id,
            status="failed",
            completed_at=datetime.now(timezone.utc),
            error="Evaluation failed.",
            metadata=request.metadata,
        )

    results_store[request.eval_id] = EvaluationStatusResponse(
        eval_id=request.eval_id,
        status=payload.status,
        result=payload,
    )

    await deliver_webhook(str(request.webhook_url), payload)


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health() -> HealthResponse:
    return HealthResponse(status="ok", scorer_ready=scorer is not None)


@app.post("/evaluate", response_model=AckResponse, status_code=202, tags=["Evaluation"])
async def evaluate(
    request: EvaluationRequest,
    background_tasks: BackgroundTasks,
    _: str = Depends(require_api_key),
) -> AckResponse:
    existing = results_store.get(request.eval_id)
    if existing:
        raise HTTPException(status_code=409, detail="eval_id already exists.")

    results_store[request.eval_id] = EvaluationStatusResponse(
        eval_id=request.eval_id,
        status="pending",
    )
    background_tasks.add_task(score_and_notify, request)
    return AckResponse(eval_id=request.eval_id)


@app.get(
    "/evaluations/{eval_id}",
    response_model=EvaluationStatusResponse,
    tags=["Evaluation"],
)
async def get_evaluation(
    eval_id: str,
    _: str = Depends(require_api_key),
) -> EvaluationStatusResponse:
    result = results_store.get(eval_id)
    if not result:
        raise HTTPException(status_code=404, detail="Evaluation not found.")
    return result

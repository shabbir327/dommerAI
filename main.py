"""FastAPI entrypoint for the DommerAI writing evaluator."""

import asyncio
import logging
import os
from contextlib import asynccontextmanager

import httpx
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader

# Render starts this project with `uvicorn main:app`, so these must be absolute
# imports from the flat repository root, not package-relative imports.
from eke import ExaminerKnowledgeEngine
from knowledge_repository import KnowledgeRepository
from models import (
    AckResponse,
    EvaluationListResponse,
    EvaluationRequest,
    HealthResponse,
    SubmissionStatus,
    WebhookPayload,
)
from result_store import EvaluationResultStore
from scorer import Scorer

API_KEY = os.environ.get("DOMMER_API_KEY", "dev-key-change-in-prod")
DEFAULT_WEBHOOK_URL = os.environ.get("WEBHOOK_URL", "").strip()

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("dommer.api")

scorer: Scorer | None = None
repository: KnowledgeRepository | None = None
result_store: EvaluationResultStore | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global scorer, repository, result_store
    repository = KnowledgeRepository()
    repository.refresh(force=True)
    eke = ExaminerKnowledgeEngine(repository)
    scorer = Scorer(eke)
    result_store = EvaluationResultStore(repository.client)
    logger.info(
        "Dommer ready — knowledge=%s default_webhook=%s",
        repository.counts(),
        DEFAULT_WEBHOOK_URL or "not configured",
    )
    yield


app = FastAPI(
    title="Dommer — DanskProeve Writing Evaluator",
    description=(
        "Knowledge-grounded PD2/PD3 evaluator with exact inline grammar locations, "
        "optional per-request webhooks, environment webhook fallback, and result polling."
    ),
    version="1.5.0",
    lifespan=lifespan,
)

cors_origins = [
    origin.strip()
    for origin in os.environ.get("CORS_ORIGINS", "*").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=False,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(key: str | None = Security(api_key_header)) -> str:
    if not key or key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")
    return key


async def score_store_and_notify(
    request: EvaluationRequest,
    webhook_url: str | None,
) -> None:
    if scorer is None or result_store is None:
        logger.error("Scorer/result store not initialised — eval_id=%s", request.eval_id)
        return

    try:
        payload = await scorer.score(request)
        result_store.save(payload.model_dump(mode="json", exclude_none=True))

        if webhook_url:
            await _fire_webhook(payload, webhook_url)
        else:
            logger.info("No webhook configured — eval_id=%s", request.eval_id)
    except Exception as exc:
        logger.exception("Background evaluation failed — eval_id=%s", request.eval_id)
        result_store.save({
            "eval_id": request.eval_id,
            "status": "failed",
            "error": str(exc),
        })


async def _fire_webhook(
    payload: WebhookPayload,
    webhook_url: str,
    retries: int = 3,
) -> None:
    async with httpx.AsyncClient(timeout=20) as client:
        for attempt in range(1, retries + 1):
            try:
                response = await client.post(
                    webhook_url,
                    json=payload.model_dump(mode="json", exclude_none=True),
                )
                response.raise_for_status()
                logger.info(
                    "Webhook delivered — eval_id=%s url=%s",
                    payload.eval_id,
                    webhook_url,
                )
                return
            except Exception as exc:
                logger.warning(
                    "Webhook attempt %d/%d failed — eval_id=%s error=%s",
                    attempt,
                    retries,
                    payload.eval_id,
                    exc,
                )
                if attempt < retries:
                    await asyncio.sleep(2 ** attempt)
    logger.error("Webhook failed — eval_id=%s", payload.eval_id)


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health() -> HealthResponse:
    counts = repository.counts() if repository else {}
    sources = repository.source_status() if repository else {}
    return HealthResponse(
        status="ok",
        scorer_ready=scorer is not None,
        knowledge_ready=counts.get("dommer", 0) > 0,
        knowledge_counts=counts,
        knowledge_sources=sources,
    )


@app.post(
    "/evaluate",
    response_model=AckResponse,
    status_code=202,
    tags=["Scoring"],
    summary="Submit a writing evaluation",
)
async def evaluate(
    request: EvaluationRequest,
    background_tasks: BackgroundTasks,
    _: str = Depends(require_api_key),
) -> AckResponse:
    if scorer is None or result_store is None:
        raise HTTPException(status_code=503, detail="Scorer is not ready.")

    request_webhook = str(request.webhook_url) if request.webhook_url else None
    effective_webhook = request_webhook or DEFAULT_WEBHOOK_URL or None
    if request_webhook:
        webhook_source = "request"
    elif DEFAULT_WEBHOOK_URL:
        webhook_source = "environment"
    else:
        webhook_source = "none"

    result_store.save({"eval_id": request.eval_id, "status": "pending"})
    background_tasks.add_task(score_store_and_notify, request, effective_webhook)

    return AckResponse(
        eval_id=request.eval_id,
        status="pending",
        webhook_url_used=effective_webhook,
        webhook_source=webhook_source,
    )


@app.get(
    "/evaluation/{eval_id}",
    response_model=WebhookPayload,
    response_model_exclude_none=True,
    tags=["Results"],
    summary="Get one evaluation by eval_id",
)
async def get_evaluation(
    eval_id: str,
    _: str = Depends(require_api_key),
) -> WebhookPayload:
    if result_store is None:
        raise HTTPException(status_code=503, detail="Result store is not ready.")
    result = result_store.get(eval_id.strip())
    if result is None:
        raise HTTPException(status_code=404, detail="Evaluation not found.")
    return WebhookPayload.model_validate(result)


@app.get(
    "/evaluations",
    response_model=EvaluationListResponse,
    tags=["Results"],
    summary="List recent evaluations",
)
async def list_evaluations(
    limit: int = Query(default=20, ge=1, le=100),
    status: SubmissionStatus | None = Query(default=None),
    _: str = Depends(require_api_key),
) -> EvaluationListResponse:
    if result_store is None:
        raise HTTPException(status_code=503, detail="Result store is not ready.")
    items = result_store.list(limit=limit, status=status)
    return EvaluationListResponse(count=len(items), items=items)

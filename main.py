"""FastAPI entrypoint for the DommerAI writing evaluator."""

import asyncio
import logging
import os
from contextlib import asynccontextmanager

import httpx
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader

from .eke import ExaminerKnowledgeEngine
from .knowledge_repository import KnowledgeRepository
from .models import AckResponse, EvaluationRequest, HealthResponse, WebhookPayload
from .scorer import Scorer

API_KEY = os.environ.get("DOMMER_API_KEY", "dev-key-change-in-prod")
WEBHOOK_URL = os.environ.get("WEBHOOK_URL", "https://danskprove.com/webhook")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("dommer.api")

scorer: Scorer | None = None
repository: KnowledgeRepository | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global scorer, repository
    repository = KnowledgeRepository()
    repository.refresh(force=True)
    eke = ExaminerKnowledgeEngine(repository)
    scorer = Scorer(eke)
    logger.info("Dommer ready — knowledge=%s webhook=%s", repository.counts(), WEBHOOK_URL)
    yield


app = FastAPI(
    title="Dommer — DanskProeve Writing Evaluator",
    description="Knowledge-grounded AI scoring engine for PD2/PD3 writing.",
    version="1.1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in os.environ.get("CORS_ORIGINS", "*").split(",")],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(key: str = Security(api_key_header)) -> str:
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")
    return key


async def score_and_notify(request: EvaluationRequest) -> None:
    if scorer is None:
        logger.error("Scorer not initialised — eval_id=%s", request.eval_id)
        return
    payload = await scorer.score(request)
    await _fire_webhook(payload)


async def _fire_webhook(payload: WebhookPayload, retries: int = 3) -> None:
    async with httpx.AsyncClient(timeout=15) as client:
        for attempt in range(1, retries + 1):
            try:
                response = await client.post(WEBHOOK_URL, json=payload.model_dump())
                response.raise_for_status()
                logger.info("Webhook delivered — eval_id=%s", payload.eval_id)
                return
            except Exception as exc:
                logger.warning("Webhook attempt %d failed: %s", attempt, exc)
                if attempt < retries:
                    await asyncio.sleep(2 ** attempt)
    logger.error("Webhook failed — eval_id=%s", payload.eval_id)


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health() -> HealthResponse:
    counts = repository.counts() if repository else {}
    return HealthResponse(
        status="ok",
        scorer_ready=scorer is not None,
        knowledge_ready=counts.get("dommer", 0) > 0,
        knowledge_counts=counts,
    )


@app.post("/evaluate", response_model=AckResponse, status_code=202, tags=["Scoring"])
async def evaluate(
    request: EvaluationRequest,
    background_tasks: BackgroundTasks,
    _: str = Depends(require_api_key),
) -> AckResponse:
    if scorer is None:
        raise HTTPException(status_code=503, detail="Scorer is not ready.")
    background_tasks.add_task(score_and_notify, request)
    return AckResponse(eval_id=request.eval_id, status="pending")

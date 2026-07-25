"""Public evaluation and result-polling routes."""

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query

from api.security import require_api_key
from app_state import state
from config import DEFAULT_WEBHOOK_URL
from models import (
    AckResponse,
    EvaluationListResponse,
    EvaluationRequest,
    SubmissionStatus,
    WebhookPayload,
)
from services.evaluation_service import score_store_and_notify

router = APIRouter()


@router.post(
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
    if state.scorer is None or state.result_store is None:
        raise HTTPException(status_code=503, detail="Scorer is not ready.")

    request_webhook = str(request.webhook_url) if request.webhook_url else None
    effective_webhook = request_webhook or DEFAULT_WEBHOOK_URL or None
    if request_webhook:
        webhook_source = "request"
    elif DEFAULT_WEBHOOK_URL:
        webhook_source = "environment"
    else:
        webhook_source = "none"

    state.result_store.save({"eval_id": request.eval_id, "status": "pending"})
    background_tasks.add_task(score_store_and_notify, request, effective_webhook)

    return AckResponse(
        eval_id=request.eval_id,
        status="pending",
        webhook_url_used=effective_webhook,
        webhook_source=webhook_source,
    )


@router.get(
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
    if state.result_store is None:
        raise HTTPException(status_code=503, detail="Result store is not ready.")
    result = state.result_store.get(eval_id.strip())
    if result is None:
        raise HTTPException(status_code=404, detail="Evaluation not found.")
    return WebhookPayload.model_validate(result)


@router.get(
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
    if state.result_store is None:
        raise HTTPException(status_code=503, detail="Result store is not ready.")
    items = state.result_store.list(limit=limit, status=status)
    return EvaluationListResponse(count=len(items), items=items)

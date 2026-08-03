"""Public evaluation and result-polling routes."""

from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Body, Depends, HTTPException, Query

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
from services.evaluation_service import (
    grade_and_combine_mock,
    score_store_and_notify,
    store_mock_part,
)

router = APIRouter()


EVALUATE_EXAMPLES = {
    "pd2_single_mock": {
        "summary": "PD2 — single mock (question + answer, official grade)",
        "description": (
            "A complete PD2 writing submission graded on the official -3..12 "
            "scale. This is the default mode (submission_mode can be omitted)."
        ),
        "value": {
            "eval_id": "pd2-single-demo-001",
            "exam_type": "PD2",
            "question": "Skriv om din yndlingshøjtid. Hvorfor kan du bedst lide den?",
            "question_description": "Brug adjektiver og beskriv følelser. Minimum 50 ord.",
            "answer": "Min yndlingshøjtid er jul. Jeg elske jul, fordi hele familien samles. Vi pynter en juletræ og synger sange sammen.",
        },
    },
    "pd3_mock_del1": {
        "summary": "PD3 mock — step 1 of 2 (Del 1, the e-mail)",
        "description": (
            "First half of a full PD3 mock. Response comes back "
            "'awaiting_other_part' — no grade yet, no LLM calls spent. Send "
            "the Del 2 example next with the SAME mock_id to trigger grading."
        ),
        "value": {
            "eval_id": "pd3-mock-demo-del1",
            "exam_type": "PD3",
            "submission_mode": "mock",
            "mock_id": "pd3-mock-demo-001",
            "delprove_part": "del1",
            "question": "Skriv et svar til din ven Mia. Tak for mailen. Foreslå, at I mødes, når Mia kommer hjem.",
            "question_description": "Del 1 af PD3 mock. Personlig e-mail.",
            "answer": "Kære Mia\nTak for din e-mail! Køreprøven gik fint. Jeg synes, vi skal mødes, når du kommer hjem.",
        },
    },
    "pd3_mock_del2": {
        "summary": "PD3 mock — step 2 of 2 (Del 2, the essay)",
        "description": (
            "Second half, same mock_id as the Del 1 example. Sending this "
            "triggers background grading of BOTH halves and the deterministic "
            "combination — poll GET /evaluation/pd3-mock-demo-001 for the "
            "final result once status moves past 'pending'."
        ),
        "value": {
            "eval_id": "pd3-mock-demo-del2",
            "exam_type": "PD3",
            "submission_mode": "mock",
            "mock_id": "pd3-mock-demo-001",
            "delprove_part": "del2",
            "question": "Skriv om fordele og ulemper ved fjernarbejde. Introduktion, 2-3 afsnit, konklusion.",
            "question_description": "Del 2 af PD3 mock. Formel skriftlig fremstilling.",
            "answer": "Fjernarbejde er blevet meget mere almindeligt siden pandemien, og det har både fordele og ulemper.",
        },
    },
    "practice_writing": {
        "summary": "Practice drill (Writing Correction tool — pass/fail, no grade)",
        "description": (
            "Standalone practice exercise, not tied to an exam mock. Response "
            "omits 'rubrik' and 'overall' entirely — pass/fail is based on "
            "whether any high-severity error remains unresolved. The 'errors' "
            "list (exact line/char positions for inline highlighting) is "
            "identical in shape to the exam modes — practice mode is lighter "
            "on grading framing, not on error detail."
        ),
        "value": {
            "eval_id": "practice-demo-001",
            "exam_type": "PD2",
            "submission_mode": "practice",
            "question": "Describe your daily routine in 8 sentences using present tense.",
            "question_description": "Week 2, Day 5. Minimum 50 words.",
            "answer": "Jeg vågner kl. 7. Jeg spiser en stor morgenmad. Jeg cykler til arbejde hver dag.",
        },
    },
}


@router.post(
    "/evaluate",
    response_model=AckResponse,
    status_code=202,
    tags=["Scoring"],
    summary="Submit a writing evaluation",
)
async def evaluate(
    request: Annotated[EvaluationRequest, Body(openapi_examples=EVALUATE_EXAMPLES)],
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

    if request.submission_mode == "mock":
        # Deliberately skips the pending-row save below: an abandoned
        # single-part mock should leave no permanently-"pending" row and
        # cost no LLM calls. store_mock_part is synchronous (no LLM calls),
        # so we know synchronously whether to ack "still waiting" or kick
        # off the real grading in the background.
        if state.mock_progress is None:
            raise HTTPException(status_code=503, detail="Mock progress store is not ready.")
        ready = await store_mock_part(request)
        if not ready:
            return AckResponse(
                eval_id=request.eval_id,
                status="awaiting_other_part",
                webhook_url_used=effective_webhook,
                webhook_source=webhook_source,
            )
        background_tasks.add_task(grade_and_combine_mock, request.mock_id, effective_webhook)
        return AckResponse(
            eval_id=request.mock_id,
            status="pending",
            webhook_url_used=effective_webhook,
            webhook_source=webhook_source,
        )

    # "submission" carries the original request fields through to
    # EvaluationResultStore._build_database_row, which reads exam_type /
    # question / answer / webhook_url from exactly this key.
    state.result_store.save({
        "eval_id": request.eval_id,
        "status": "pending",
        "submission": request.model_dump(mode="json", exclude_none=True),
    })
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
    description=(
        "For submission_mode='single' or 'practice', use the eval_id you sent. "
        "For submission_mode='mock', poll using mock_id instead — the combined "
        "result is stored under that id, not under either half's own eval_id."
    ),
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

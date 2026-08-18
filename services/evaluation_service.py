"""Background scoring, persistence, and webhook delivery."""

from __future__ import annotations

import asyncio
import logging

import httpx

from app_state import state
from models import EvaluationRequest, WebhookPayload

logger = logging.getLogger("dommer.evaluation")


async def store_mock_part(request: EvaluationRequest) -> tuple[bool, int]:
    """Synchronous, no LLM calls — just persists this half and reports
    whether its pair has already arrived. Kept separate from the actual
    grading so the API layer can return its 202 ack immediately either way,
    same as the existing single-submission flow, instead of blocking the
    request on up to four LLM calls when this happens to be the second half.

    Returns (ready, generation) — generation is the mock_progress
    generation counter immediately after this save, which the caller must
    pass through to grade_and_combine_mock so it can detect, right before
    saving its result, whether a later resubmission has since superseded it.
    """
    if state.mock_progress is None:
        raise RuntimeError("Mock progress store is not initialised.")

    row = state.mock_progress.save_part(
        mock_id=request.mock_id,
        exam_type=request.exam_type,
        part=request.delprove_part,
        request_dict=request.model_dump(mode="json", exclude_none=True),
    )
    other_part = "del2" if request.delprove_part == "del1" else "del1"
    ready = row.get(other_part) is not None
    generation = int(row.get("generation") or 0)
    return ready, generation


async def grade_and_combine_mock(
    mock_id: str, webhook_url: str | None, generation: int | None = None
) -> None:
    """Runs as a background task once store_mock_part reports both halves
    are present. Grades each half independently (own pragmatisk/diskursiv/
    lingvistisk read, since an email and a formal essay have genuinely
    different register expectations), then combines them deterministically
    via Scorer.combine_mock_grades — never asks an LLM to do that arithmetic.
    An abandoned single-part mock never reaches this function at all: no
    LLM calls are spent on it, by design.

    `generation` is the mock_progress generation counter captured by the
    caller at the moment THIS run was triggered. Since this involves up to
    four LLM calls, a resubmission of either half can trigger a second,
    newer run before this one finishes. Right before saving, this function
    re-checks the current generation — if a newer run has since started,
    this one is stale and must not save, or it could overwrite a newer
    (possibly already-completed) result with an old one purely because it
    happened to finish last. If generation is None (caller didn't supply
    one — e.g. an older client), the check is skipped, preserving prior
    behaviour rather than failing closed.
    """
    if state.mock_progress is None or state.scorer is None or state.result_store is None:
        logger.error("Mock grading dependencies not ready — mock_id=%s", mock_id)
        return

    row: dict | None = None
    try:
        row = state.mock_progress.get(mock_id)
        if row is None:
            logger.error("Mock progress row vanished — mock_id=%s", mock_id)
            return

        del1_payload = None
        if row.get("del1") is not None:
            del1_payload = await state.scorer.score(EvaluationRequest.model_validate(row["del1"]))

        del2_payload = None
        if row.get("del2") is not None:
            del2_payload = await state.scorer.score(EvaluationRequest.model_validate(row["del2"]))

        combined = state.scorer.combine_mock_grades(del1_payload, del2_payload)

        del1_raw = row.get("del1") or {}
        del2_raw = row.get("del2") or {}

        def _combine_field(field: str, fallback: str | None = "(not answered)") -> str | None:
            # evaluations.question/answer are NOT NULL — a combined mock
            # result never had a "the submission" the way a single-mode one
            # does, so this builds one from both halves rather than leaving
            # either column null (which broke the write the same way
            # exam_type's absence did before). question_description is NOT
            # a NOT NULL column though, so it gets fallback=None instead —
            # otherwise the literal string "(not answered)" leaks straight
            # into a frontend that renders this field directly.
            del1_value = str(del1_raw.get(field) or "").strip()
            del2_value = str(del2_raw.get(field) or "").strip()
            if del1_value and del2_value:
                return f"DEL 1: {del1_value}\n\nDEL 2: {del2_value}"
            return del2_value or del1_value or fallback

        # One consistent errors[] contract regardless of submission_mode:
        # each error carries its own position PLUS which part it's relative
        # to, instead of forcing frontend code to know "for mock results,
        # go dig into del1/del2 instead of the top-level field."
        combined_errors: list = []
        if del1_payload and del1_payload.errors:
            combined_errors.extend(
                error.model_copy(update={"part": "del1"}) for error in del1_payload.errors
            )
        if del2_payload and del2_payload.errors:
            combined_errors.extend(
                error.model_copy(update={"part": "del2"}) for error in del2_payload.errors
            )

        del1_feedback = (del1_payload.feedback or "").strip() if del1_payload else ""
        del2_feedback = (del2_payload.feedback or "").strip() if del2_payload else ""
        if del1_feedback and del2_feedback:
            combined_feedback = f"Del 1: {del1_feedback} Del 2: {del2_feedback}"
        else:
            combined_feedback = del2_feedback or del1_feedback or None

        del1_summary = (del1_payload.examiner_summary or "").strip() if del1_payload else ""
        del2_summary = (del2_payload.examiner_summary or "").strip() if del2_payload else ""
        if del1_summary and del2_summary:
            narrative_summary = f"Del 1: {del1_summary} Del 2: {del2_summary}"
        else:
            narrative_summary = del2_summary or del1_summary or combined["combination_reason"]

        final_payload = WebhookPayload(
            eval_id=mock_id,
            status=combined.get("status", "scored"),
            exam_type=row.get("exam_type"),
            question=_combine_field("question"),
            question_description=_combine_field("question_description", fallback=None),
            answer=_combine_field("answer"),
            rubrik=combined["rubrik"],
            overall=combined["overall"],
            pass_fail=combined["pass_fail"],
            feedback=combined_feedback[:2000] if combined_feedback else None,
            examiner_summary=narrative_summary[:1200],
            error=combined["combination_reason"] if combined.get("status") == "failed" else None,
            errors=combined_errors or None,
            del1=combined["del1_result"],
            del2=combined["del2_result"],
            del1_word_count=del1_payload.word_count if del1_payload else None,
            del2_word_count=del2_payload.word_count if del2_payload else None,
            model_metadata={
                "submission_mode": "mock",
                "combination_reason": combined["combination_reason"],
            },
        )

        if generation is not None and state.mock_progress.get_generation(mock_id) != generation:
            logger.warning(
                "Discarding superseded mock combination — mock_id=%s "
                "triggered_generation=%s current_generation=%s. A newer "
                "resubmission started a fresh grading run for this mock_id "
                "while this one was still in flight; saving this result "
                "would overwrite the newer run's result with a stale one.",
                mock_id, generation, state.mock_progress.get_generation(mock_id),
            )
            return

        state.result_store.save(final_payload.model_dump(mode="json", exclude_none=True))
        state.mock_progress.mark_completed(mock_id, mock_id)

        if webhook_url:
            await fire_webhook(final_payload, webhook_url)
    except Exception as exc:
        logger.exception("Mock combination failed — mock_id=%s", mock_id)
        if (
            generation is not None
            and state.mock_progress is not None
            and state.mock_progress.get_generation(mock_id) != generation
        ):
            logger.warning(
                "Suppressing stale failure save for superseded mock combination "
                "— mock_id=%s triggered_generation=%s current_generation=%s. "
                "A newer resubmission is already in flight or has already "
                "completed; this failure belongs to a superseded run.",
                mock_id, generation, state.mock_progress.get_generation(mock_id),
            )
            return
        del1_raw = (row.get("del1") if row else None) or {}
        del2_raw = (row.get("del2") if row else None) or {}
        state.result_store.save({
            "eval_id": mock_id,
            "status": "failed",
            "exam_type": row.get("exam_type") if row else None,
            "question": str(del1_raw.get("question") or del2_raw.get("question") or "(not answered)"),
            "answer": str(del1_raw.get("answer") or del2_raw.get("answer") or "(not answered)"),
            "error": str(exc),
        })


async def score_store_and_notify(
    request: EvaluationRequest,
    webhook_url: str | None,
) -> None:
    if state.scorer is None or state.result_store is None:
        logger.error("Scorer/result store not initialised — eval_id=%s", request.eval_id)
        return

    try:
        payload = await state.scorer.score(request)
        state.result_store.save(payload.model_dump(mode="json", exclude_none=True))

        if webhook_url:
            await fire_webhook(payload, webhook_url)
        else:
            logger.info("No webhook configured — eval_id=%s", request.eval_id)
    except Exception as exc:
        logger.exception("Background evaluation failed — eval_id=%s", request.eval_id)
        state.result_store.save({
            "eval_id": request.eval_id,
            "status": "failed",
            "exam_type": request.exam_type,
            "error": str(exc),
        })


async def fire_webhook(
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

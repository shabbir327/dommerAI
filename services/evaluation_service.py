"""Background scoring, persistence, and webhook delivery."""

from __future__ import annotations

import asyncio
import logging

import httpx

from app_state import state
from models import EvaluationRequest, WebhookPayload

logger = logging.getLogger("dommer.evaluation")


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

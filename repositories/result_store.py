"""Persist candidate submissions and completed evaluations in DommerAI Supabase.

The production ``evaluations`` table uses normal columns (``exam_type``, ``answer``,
``feedback_da`` and so on), rather than requiring a single ``result_json`` column.
This store discovers the live table columns, writes only supported fields, and keeps
an in-memory cache for fast polling.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from threading import RLock
from typing import Any

from supabase import Client

logger = logging.getLogger("dommer.results")


class EvaluationResultStore:
    def __init__(self, client: Client | None = None) -> None:
        self.client = client
        self.table = os.environ.get("EVALUATIONS_TABLE", "evaluations")
        self.id_column = os.environ.get("EVALUATION_ID_COLUMN", "eval_id")
        self.status_column = os.environ.get("EVALUATION_STATUS_COLUMN", "status")
        self.updated_column = os.environ.get("EVALUATION_UPDATED_COLUMN", "updated_at")
        self.result_column = os.environ.get("EVALUATION_RESULT_COLUMN", "result_json")
        self.persist_enabled = os.environ.get("PERSIST_EVALUATIONS", "true").lower() in {
            "1", "true", "yes", "on"
        }
        self._items: dict[str, dict[str, Any]] = {}
        self._columns: set[str] | None = None
        self._lock = RLock()

    def save(self, payload: dict[str, Any]) -> None:
        eval_id = str(payload.get("eval_id", "")).strip()
        if not eval_id:
            raise ValueError("Evaluation payload must contain eval_id.")

        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            existing = dict(self._items.get(eval_id, {}))
            stored = self._deep_merge(existing, payload)
            # The production table requires submitted_at. Set it only on the
            # first save and preserve it when the same evaluation is updated
            # from pending to scored/failed.
            stored.setdefault("submitted_at", now)
            # started_at: when this evaluation was first accepted (the
            # "pending" save). completed_at: only set once the evaluation
            # actually reaches a terminal state — previously neither column
            # was ever written, so both stayed permanently null in Supabase.
            stored.setdefault("started_at", now)
            if stored.get("status") in ("scored", "failed"):
                stored.setdefault("completed_at", now)
            stored["updated_at"] = now
            self._items[eval_id] = stored

        if not self.persist_enabled or self.client is None:
            logger.warning(
                "Evaluation persistence disabled/unavailable — eval_id=%s", eval_id
            )
            return

        row = self._build_database_row(stored, now)
        if not row:
            logger.error("No compatible evaluations columns found — eval_id=%s", eval_id)
            return

        try:
            self.client.table(self.table).upsert(
                row, on_conflict=self.id_column
            ).execute()
            logger.info(
                "Evaluation persisted — eval_id=%s status=%s columns=%s",
                eval_id,
                stored.get("status"),
                sorted(row.keys()),
            )
        except Exception as exc:
            # Do not silently pretend persistence succeeded. The API can still return
            # from memory, but Render logs must clearly show the database failure.
            logger.exception(
                "Supabase evaluation persistence FAILED — eval_id=%s table=%s row_columns=%s error=%s",
                eval_id,
                self.table,
                sorted(row.keys()),
                exc,
            )

    def get(self, eval_id: str) -> dict[str, Any] | None:
        with self._lock:
            item = self._items.get(eval_id)
            if item is not None:
                return dict(item)

        if not self.persist_enabled or self.client is None:
            return None

        try:
            response = (
                self.client.table(self.table)
                .select("*")
                .eq(self.id_column, eval_id)
                .limit(1)
                .execute()
            )
            rows = response.data or []
            if not rows:
                return None
            return self._row_to_payload(rows[0])
        except Exception as exc:
            logger.exception(
                "Could not read evaluation from Supabase — eval_id=%s error=%s",
                eval_id,
                exc,
            )
            return None

    def list(self, limit: int = 20, status: str | None = None) -> list[dict[str, Any]]:
        # Prefer live Supabase data so results survive Render restarts.
        if self.persist_enabled and self.client is not None:
            try:
                query = self.client.table(self.table).select("*")
                if status:
                    query = query.eq(self.status_column, status)
                response = query.order(self.updated_column, desc=True).limit(limit).execute()
                return [self._row_to_payload(row) for row in (response.data or [])]
            except Exception as exc:
                logger.warning("Could not list Supabase evaluations; using memory — %s", exc)

        with self._lock:
            values = [dict(item) for item in self._items.values()]
        if status:
            values = [item for item in values if item.get("status") == status]
        values.sort(key=lambda item: item.get("updated_at", ""), reverse=True)
        return values[:limit]

    def _discover_columns(self) -> set[str]:
        if self._columns is not None:
            return self._columns
        if self.client is None:
            return set()

        try:
            response = self.client.table(self.table).select("*").limit(1).execute()
            rows = response.data or []
            if rows:
                self._columns = set(rows[0].keys())
                logger.info(
                    "Discovered evaluations schema — columns=%s",
                    sorted(self._columns),
                )
                return self._columns
        except Exception as exc:
            logger.warning("Could not discover evaluations schema — %s", exc)

        # Expected production schema. Unsupported fields are removed and retried only
        # through live discovery when the table contains at least one row.
        self._columns = {
            "eval_id", "candidate_id", "exam_type", "status", "question",
            "question_description", "answer", "webhook_url", "rubrik", "overall",
            "pass_fail", "feedback_da", "examiner_summary", "errors", "word_count",
            "writing_statistics", "knowledge_used", "retrieval_metadata",
            "model_metadata", "error", "submitted_at", "started_at", "completed_at",
            "created_at", "updated_at", "result_json",
        }
        return self._columns

    def _build_database_row(self, stored: dict[str, Any], now: str) -> dict[str, Any]:
        columns = self._discover_columns()
        submission = stored.get("submission") if isinstance(stored.get("submission"), dict) else {}

        candidate_id = submission.get("candidate_id") or stored.get("candidate_id") or stored.get("eval_id")
        candidates: dict[str, Any] = {
            self.id_column: stored.get("eval_id"),
            self.status_column: stored.get("status"),
            self.updated_column: now,
            "submitted_at": stored.get("submitted_at") or now,
            "started_at": stored.get("started_at"),
            "completed_at": stored.get("completed_at"),
            "candidate_id": candidate_id,
            "exam_type": submission.get("exam_type") or stored.get("exam_type"),
            "question": submission.get("question") or stored.get("question"),
            "question_description": submission.get("question_description") or stored.get("question_description"),
            "answer": submission.get("answer") or stored.get("answer"),
            "webhook_url": submission.get("webhook_url") or stored.get("webhook_url"),
            "rubrik": stored.get("rubrik"),
            "overall": stored.get("overall"),
            "pass_fail": stored.get("pass_fail"),
            "feedback_da": stored.get("feedback_da"),
            "examiner_summary": stored.get("examiner_summary"),
            "errors": stored.get("errors"),
            "word_count": stored.get("word_count"),
            "writing_statistics": stored.get("writing_statistics"),
            "knowledge_used": stored.get("knowledge_used"),
            "retrieval_metadata": stored.get("retrieval_metadata"),
            "model_metadata": stored.get("model_metadata"),
            "error": stored.get("error"),
        }

        # Backwards compatibility when a result_json JSONB column exists.
        if self.result_column in columns:
            candidates[self.result_column] = {
                key: value for key, value in stored.items() if key != "updated_at"
            }

        return {
            key: value
            for key, value in candidates.items()
            if key in columns and value is not None
        }

    def _row_to_payload(self, row: dict[str, Any]) -> dict[str, Any]:
        result = row.get(self.result_column)
        if isinstance(result, dict):
            payload = dict(result)
            payload.setdefault("updated_at", row.get(self.updated_column))
            return payload

        payload = {
            key: value
            for key, value in row.items()
            if key not in {"created_at"} and value is not None
        }
        return payload

    @staticmethod
    def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
        merged = dict(base)
        for key, value in update.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = {**merged[key], **value}
            else:
                merged[key] = value
        return merged

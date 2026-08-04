"""Persist candidate submissions and completed evaluations in DommerAI Supabase.

Writes to the ``evaluation_results`` table — a deliberately tidier redesign
of the original wide ``evaluations`` table, which had grown to ~30 columns
(candidate_id, webhook_url, writing_statistics, knowledge_used,
retrieval_metadata, model_metadata, dimension_reasons, task_coverage,
strengths, improvements, result_json, submitted_at, started_at, all as
separate top-level columns) and become hard to browse directly in Supabase's
Table Editor. This store folds the debug/secondary fields into one
``metadata`` jsonb column and drops columns that were redundant in practice
(``candidate_id`` always mirrored ``eval_id``; ``started_at`` was always set
to the same instant as the old ``submitted_at``, since nothing in this
codebase ever recorded a separate "started" moment).

The in-memory cache and the API-facing payload shape are unaffected by this —
WebhookPayload still returns model_metadata/writing_statistics/etc. as
separate fields; only the database row shape changed. _build_database_row
folds them in on write, _row_to_payload expands them back out on read.

All timestamps are computed in Europe/Copenhagen local time (not UTC) before
being sent to Supabase, per the person's request that timestamps read as
Danish time rather than UTC when browsing the table directly.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from threading import RLock
from typing import Any
from zoneinfo import ZoneInfo

from supabase import Client

logger = logging.getLogger("dommer.results")

DENMARK_TZ = ZoneInfo("Europe/Copenhagen")

# Fields folded into the single `metadata` jsonb column instead of getting
# their own top-level column — these are genuinely useful, but are debug/
# secondary detail rather than the handful of fields someone actually wants
# to see at a glance when scanning rows in the Table Editor.
_METADATA_FIELDS = (
    "webhook_url",
    "model_metadata",
    "writing_statistics",
    "knowledge_used",
    "retrieval_metadata",
    "dimension_reasons",
    "task_coverage",
    "strengths",
    "improvements",
)


def _now_cph() -> datetime:
    return datetime.now(DENMARK_TZ)


class EvaluationResultStore:
    def __init__(self, client: Client | None = None) -> None:
        self.client = client
        self.table = os.environ.get("EVALUATIONS_TABLE", "evaluation_results")
        self.id_column = os.environ.get("EVALUATION_ID_COLUMN", "eval_id")
        self.status_column = os.environ.get("EVALUATION_STATUS_COLUMN", "status")
        self.updated_column = os.environ.get("EVALUATION_UPDATED_COLUMN", "updated_at")
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

        now = _now_cph().isoformat()
        with self._lock:
            existing = self._items.get(eval_id)
        if existing is None:
            # Not in memory doesn't mean genuinely new — a process restart
            # between the initial "pending" save (which carries the raw
            # submission: question/answer/exam_type) and this later
            # "scored"/"failed" update wipes the in-memory cache but not
            # Supabase. Without this check, that restart would silently
            # drop the earlier submission data from the merge below, and
            # question/answer/exam_type being NOT NULL columns turns that
            # into a failed write instead of a stale-but-harmless gap.
            existing = self._load_from_supabase(eval_id) or {}
        with self._lock:
            stored = self._deep_merge(existing, payload)
            stored.setdefault("created_at", now)
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
            logger.error("No compatible evaluation_results columns found — eval_id=%s", eval_id)
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

    def _load_from_supabase(self, eval_id: str) -> dict[str, Any] | None:
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

    def get(self, eval_id: str) -> dict[str, Any] | None:
        with self._lock:
            item = self._items.get(eval_id)
            if item is not None:
                return dict(item)
        return self._load_from_supabase(eval_id)

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
                    "Discovered evaluation_results schema — columns=%s",
                    sorted(self._columns),
                )
                return self._columns
        except Exception as exc:
            logger.warning("Could not discover evaluation_results schema — %s", exc)

        # Expected schema for a fresh evaluation_results table. Unsupported
        # fields are removed and retried only through live discovery once
        # the table contains at least one row.
        self._columns = {
            "eval_id", "exam_type", "submission_mode", "status", "question",
            "question_description", "answer", "overall", "pass_fail", "rubrik",
            "feedback", "examiner_summary", "errors", "word_count", "del1",
            "del2", "metadata", "error", "created_at", "completed_at",
            "updated_at",
        }
        return self._columns

    def _build_database_row(self, stored: dict[str, Any], now: str) -> dict[str, Any]:
        columns = self._discover_columns()
        submission = stored.get("submission") if isinstance(stored.get("submission"), dict) else {}

        submission_mode = (
            stored.get("submission_mode")
            or submission.get("submission_mode")
            or (stored.get("model_metadata") or {}).get("submission_mode")
            or "single"
        )

        metadata: dict[str, Any] = {}
        for field in _METADATA_FIELDS:
            value = stored.get(field) or submission.get(field)
            if value not in (None, {}, []):
                metadata[field] = value

        candidates: dict[str, Any] = {
            self.id_column: stored.get("eval_id"),
            "exam_type": submission.get("exam_type") or stored.get("exam_type"),
            "submission_mode": submission_mode,
            self.status_column: stored.get("status"),
            "question": submission.get("question") or stored.get("question"),
            "question_description": submission.get("question_description") or stored.get("question_description"),
            "answer": submission.get("answer") or stored.get("answer"),
            "overall": stored.get("overall"),
            "pass_fail": stored.get("pass_fail"),
            "rubrik": stored.get("rubrik"),
            "feedback": stored.get("feedback"),
            "examiner_summary": stored.get("examiner_summary"),
            "errors": stored.get("errors") or [],
            "word_count": stored.get("word_count"),
            "del1": stored.get("del1"),
            "del2": stored.get("del2"),
            "metadata": metadata,
            "error": stored.get("error"),
            "created_at": stored.get("created_at") or now,
            "completed_at": stored.get("completed_at"),
            self.updated_column: now,
        }

        return {
            key: value
            for key, value in candidates.items()
            if key in columns and value is not None
        }

    def _row_to_payload(self, row: dict[str, Any]) -> dict[str, Any]:
        metadata = row.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}

        payload = {
            key: value
            for key, value in row.items()
            if key not in {"metadata", "created_at"} and value is not None
        }
        for field in _METADATA_FIELDS:
            if field in metadata:
                payload[field] = metadata[field]
        payload.setdefault("updated_at", row.get(self.updated_column))
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

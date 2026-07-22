"""Evaluation result storage for DommerAI.

Results are always cached in memory for fast Swagger polling. Supabase persistence is
optional and enabled by default; if the configured table/columns do not match the
existing schema, DommerAI continues to work and logs a warning.
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
        self.result_column = os.environ.get("EVALUATION_RESULT_COLUMN", "result_json")
        self.updated_column = os.environ.get("EVALUATION_UPDATED_COLUMN", "updated_at")
        self.persist_enabled = os.environ.get("PERSIST_EVALUATIONS", "true").lower() in {
            "1", "true", "yes", "on"
        }
        self._items: dict[str, dict[str, Any]] = {}
        self._lock = RLock()

    def save(self, payload: dict[str, Any]) -> None:
        eval_id = str(payload.get("eval_id", "")).strip()
        if not eval_id:
            raise ValueError("Evaluation payload must contain eval_id.")

        now = datetime.now(timezone.utc).isoformat()
        stored = dict(payload)
        stored["updated_at"] = now
        with self._lock:
            self._items[eval_id] = stored

        if not self.persist_enabled or self.client is None:
            return

        row = {
            self.id_column: eval_id,
            self.status_column: stored.get("status"),
            self.result_column: payload,
            self.updated_column: now,
        }
        try:
            self.client.table(self.table).upsert(
                row, on_conflict=self.id_column
            ).execute()
        except Exception as exc:
            logger.warning(
                "Supabase result persistence unavailable; using memory cache — "
                "eval_id=%s table=%s error=%s",
                eval_id,
                self.table,
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
            row = rows[0]
            result = row.get(self.result_column)
            if isinstance(result, dict):
                return result
            return {
                "eval_id": row.get(self.id_column, eval_id),
                "status": row.get(self.status_column, "pending"),
                "updated_at": row.get(self.updated_column),
            }
        except Exception as exc:
            logger.warning(
                "Could not read evaluation from Supabase — eval_id=%s error=%s",
                eval_id,
                exc,
            )
            return None

    def list(self, limit: int = 20, status: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            values = [dict(item) for item in self._items.values()]
        if status:
            values = [item for item in values if item.get("status") == status]
        values.sort(key=lambda item: item.get("updated_at", ""), reverse=True)
        return values[:limit]

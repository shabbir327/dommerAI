"""Holds one submitted half of a two-part mock test (Del 1 or Del 2) while
waiting for its pair to arrive. Deliberately separate from
EvaluationResultStore: a pending mock half has no grade, no rubric, no
errors — none of the columns a finished evaluation row expects — so jamming
it into that table would mean a row full of nulls that doesn't mean
"evaluation not scored yet", it means "half a submission, nothing to score
yet". A dedicated table keeps that distinction explicit in the data itself.

Same dual in-memory + Supabase pattern as EvaluationResultStore, for the same
reason: fast polling without a round-trip, with Supabase as the durable copy.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from threading import RLock
from typing import Any

from supabase import Client

logger = logging.getLogger("dommer.mock_progress")


class MockProgressStore:
    def __init__(self, client: Client | None = None) -> None:
        self.client = client
        self.table = os.environ.get("MOCK_PROGRESS_TABLE", "mock_progress")
        self.persist_enabled = os.environ.get("PERSIST_EVALUATIONS", "true").lower() in {
            "1", "true", "yes", "on"
        }
        self._items: dict[str, dict[str, Any]] = {}
        self._lock = RLock()

    def save_part(
        self,
        mock_id: str,
        exam_type: str,
        part: str,
        request_dict: dict[str, Any],
    ) -> dict[str, Any]:
        """Stores this half's raw request under mock_id, returns the full
        current row (both halves, whichever are present so far)."""
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            row = dict(self._items.get(mock_id, {
                "mock_id": mock_id,
                "exam_type": exam_type,
                "del1": None,
                "del2": None,
                "final_eval_id": None,
                "created_at": now,
            }))
            row[part] = request_dict
            row["updated_at"] = now
            self._items[mock_id] = row

        if self.persist_enabled and self.client is not None:
            try:
                self.client.table(self.table).upsert(
                    {
                        "mock_id": mock_id,
                        "exam_type": exam_type,
                        "del1": row.get("del1"),
                        "del2": row.get("del2"),
                        "final_eval_id": row.get("final_eval_id"),
                        "updated_at": now,
                    },
                    on_conflict="mock_id",
                ).execute()
            except Exception as exc:
                logger.exception(
                    "Supabase mock_progress persistence FAILED — mock_id=%s error=%s",
                    mock_id, exc,
                )
        return row

    def get(self, mock_id: str) -> dict[str, Any] | None:
        with self._lock:
            item = self._items.get(mock_id)
            if item is not None:
                return dict(item)

        if not self.persist_enabled or self.client is None:
            return None
        try:
            response = (
                self.client.table(self.table)
                .select("*")
                .eq("mock_id", mock_id)
                .limit(1)
                .execute()
            )
            rows = response.data or []
            return rows[0] if rows else None
        except Exception as exc:
            logger.exception(
                "Could not read mock_progress from Supabase — mock_id=%s error=%s",
                mock_id, exc,
            )
            return None

    def mark_completed(self, mock_id: str, final_eval_id: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            row = self._items.get(mock_id)
            if row is not None:
                row["final_eval_id"] = final_eval_id
                row["updated_at"] = now

        if self.persist_enabled and self.client is not None:
            try:
                self.client.table(self.table).update({
                    "final_eval_id": final_eval_id,
                    "updated_at": now,
                }).eq("mock_id", mock_id).execute()
            except Exception as exc:
                logger.exception(
                    "Could not mark mock_progress completed — mock_id=%s error=%s",
                    mock_id, exc,
                )

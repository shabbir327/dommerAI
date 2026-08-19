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
from datetime import datetime
from zoneinfo import ZoneInfo
from threading import RLock
from typing import Any

from supabase import Client

logger = logging.getLogger("dommer.mock_progress")

DENMARK_TZ = ZoneInfo("Europe/Copenhagen")


def _now_cph() -> datetime:
    return datetime.now(DENMARK_TZ)


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
        current row (both halves, whichever are present so far).

        Checks Supabase before assuming a mock_id absent from the in-memory
        cache is genuinely new. This matters specifically because Del 1 and
        Del 2 of a real PD3/PD2 mock can arrive days apart — long enough for
        Render to restart the process at least once in between (a redeploy,
        or a free/hobby-tier dyno spinning down from inactivity), which wipes
        self._items completely. Without this check, Del 2 arriving on a
        fresh process would look like the start of a brand-new mock, and the
        upsert below would silently overwrite Del 1's real, already-persisted
        answer with null — a correctness bug, not just a missed cache hit.
        """
        now = _now_cph().isoformat()
        with self._lock:
            row = self._items.get(mock_id)
        if row is None:
            row = self._load_from_supabase(mock_id) or {
                "mock_id": mock_id,
                "exam_type": exam_type,
                "del1": None,
                "del2": None,
                "final_eval_id": None,
                "generation": 0,
                "created_at": now,
            }
        row = dict(row)
        # Each part is stored as {"request": ..., "graded": ..., "graded_generation": ...}
        # rather than the raw request dict directly. This lets a part's
        # LLM-graded result be cached alongside its submission (populated by
        # save_graded_preview once grading finishes) without needing a new
        # SQL column — del1/del2 are already jsonb, so this is just a richer
        # shape inside the same column. A (re)submission always resets
        # "graded" to None: any previously-cached grading for this part is
        # now stale and must not be reused by grade_and_combine_mock.
        row[part] = {"request": request_dict, "graded": None, "graded_generation": None}
        row["updated_at"] = now

        # Bump the generation on every part submission — including a
        # RESUBMISSION of a part that already arrived, or one arriving after
        # the mock was already marked completed. grade_and_combine_mock is a
        # background task that can take a while (up to four LLM calls); if a
        # second submission triggers a second combine run before the first
        # one finishes, both runs eventually call EvaluationResultStore.save
        # under the same eval_id (=mock_id), and whichever happens to finish
        # LAST wins — with no guarantee that's the logically-newer one. A
        # slow, stale run (e.g. one that ultimately fails) could overwrite a
        # newer, correct result purely on timing. The generation captured
        # here lets grade_and_combine_mock detect, right before it saves,
        # whether it has since been superseded — and skip the save instead
        # of clobbering a newer result.
        row["generation"] = int(row.get("generation") or 0) + 1

        # A resubmission means this mock is being (re)graded — clear the
        # completion marker so polling doesn't report "already completed"
        # against what is now a stale/superseded combined result while the
        # new grading run is in flight.
        row["final_eval_id"] = None

        with self._lock:
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
                        "generation": row.get("generation"),
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

    def get_generation(self, mock_id: str) -> int:
        """Current generation for this mock_id, or 0 if it doesn't exist yet.
        Called by grade_and_combine_mock right before saving its result, to
        detect whether a newer resubmission has superseded this run.
        """
        row = self.get(mock_id)
        return int((row or {}).get("generation") or 0)

    def save_graded_preview(
        self, mock_id: str, part: str, graded_dict: dict[str, Any], generation: int
    ) -> bool:
        """Caches a part's LLM-graded result once its standalone preview
        grading finishes, so grade_and_combine_mock can reuse it later
        instead of re-scoring that same part a second time when its pair
        arrives — avoiding double LLM spend on the part that was graded
        early for the student-facing preview.

        Returns False (and does not save) if this part has been resubmitted
        since `generation` was captured — same staleness guard used for the
        final combination, for the same reason: a slow preview-grading run
        must not overwrite a newer resubmission's in-progress state.
        """
        with self._lock:
            row = self._items.get(mock_id)
        if row is None:
            row = self._load_from_supabase(mock_id)
        if row is None or row.get(part) is None:
            return False
        if int(row.get("generation") or 0) != generation:
            return False

        now = _now_cph().isoformat()
        row = dict(row)
        row[part] = dict(row[part])
        row[part]["graded"] = graded_dict
        row[part]["graded_generation"] = generation
        row["updated_at"] = now
        with self._lock:
            self._items[mock_id] = row

        if self.persist_enabled and self.client is not None:
            try:
                self.client.table(self.table).upsert(
                    {
                        "mock_id": mock_id,
                        "exam_type": row.get("exam_type"),
                        "del1": row.get("del1"),
                        "del2": row.get("del2"),
                        "final_eval_id": row.get("final_eval_id"),
                        "generation": row.get("generation"),
                        "updated_at": now,
                    },
                    on_conflict="mock_id",
                ).execute()
            except Exception as exc:
                logger.exception(
                    "Supabase mock_progress preview-cache persistence FAILED "
                    "— mock_id=%s part=%s error=%s",
                    mock_id, part, exc,
                )
        return True

    def _load_from_supabase(self, mock_id: str) -> dict[str, Any] | None:
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

    def get(self, mock_id: str) -> dict[str, Any] | None:
        with self._lock:
            item = self._items.get(mock_id)
            if item is not None:
                return dict(item)
        return self._load_from_supabase(mock_id)

    def mark_completed(self, mock_id: str, final_eval_id: str) -> None:
        now = _now_cph().isoformat()
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

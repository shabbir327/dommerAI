"""Supabase-backed, cached knowledge repository for DommerAI."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any

from supabase import Client, create_client

logger = logging.getLogger("dommer.knowledge")


@dataclass(frozen=True)
class KnowledgeRecord:
    id: str
    source: str
    exam_type: str | None
    knowledge_type: str
    title: str
    statement: str
    source_quote: str
    metadata: dict[str, Any]

    @property
    def search_text(self) -> str:
        metadata_text = " ".join(
            str(value)
            for value in self.metadata.values()
            if value is not None and isinstance(value, (str, int, float, bool))
        )
        return " ".join(
            part
            for part in (
                self.title,
                self.statement,
                self.source_quote,
                self.knowledge_type,
                metadata_text,
            )
            if part
        ).lower()


class KnowledgeRepository:
    """Load and cache the live DommerAI knowledge sources from Supabase.

    Required source:
      - dkf_published_knowledge

    Supporting language sources:
      - kb_danish_verbs
      - kb_danish_adjectives

    Table/view names can be overridden through Render environment variables.
    """

    SOURCE_CONFIG = {
        "dommer": ("DOMMER_KNOWLEDGE_VIEW", "dkf_published_knowledge", True),
        "verbs": ("LANGUAGE_VERBS_VIEW", "kb_danish_verbs", False),
        "adjectives": (
            "LANGUAGE_ADJECTIVES_VIEW",
            "kb_danish_adjectives",
            False,
        ),
    }

    def __init__(self) -> None:
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        if not url or not key:
            raise RuntimeError(
                "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be configured."
            )

        self.client: Client = create_client(url, key)
        self.cache_ttl = int(os.environ.get("KNOWLEDGE_CACHE_TTL_SECONDS", "300"))
        self._loaded_at = 0.0
        self._records: list[KnowledgeRecord] = []
        self._source_status: dict[str, dict[str, Any]] = {}
        self.views = {
            source: os.environ.get(env_name, default_name)
            for source, (env_name, default_name, _) in self.SOURCE_CONFIG.items()
        }

    def refresh(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and self._records and now - self._loaded_at < self.cache_ttl:
            return

        records: list[KnowledgeRecord] = []
        statuses: dict[str, dict[str, Any]] = {}

        for source, (_, _, required) in self.SOURCE_CONFIG.items():
            view = self.views[source]
            try:
                source_records = self._fetch_view(view, source)
                records.extend(source_records)
                statuses[source] = {
                    "status": "loaded",
                    "count": len(source_records),
                    "required": required,
                    "relation": view,
                }
            except Exception as exc:
                statuses[source] = {
                    "status": "unavailable",
                    "count": 0,
                    "required": required,
                    "relation": view,
                    "detail": str(exc)[:300],
                }
                if required:
                    raise RuntimeError(
                        f"Could not load required knowledge source {view}: {exc}"
                    ) from exc
                logger.warning(
                    "Supporting knowledge source unavailable — source=%s relation=%s error=%s",
                    source,
                    view,
                    exc,
                )

        self._records = records
        self._source_status = statuses
        self._loaded_at = now
        logger.info("Knowledge cache refreshed — counts=%s", self.counts(refresh=False))

    def all(self) -> list[KnowledgeRecord]:
        self.refresh()
        return list(self._records)

    def by_source(self, source: str) -> list[KnowledgeRecord]:
        return [record for record in self.all() if record.source == source]

    def official_for_exam(self, exam_type: str) -> list[KnowledgeRecord]:
        target = exam_type.upper()
        return [
            record
            for record in self.by_source("dommer")
            if not record.exam_type or record.exam_type.upper() == target
        ]

    def verbs(self) -> list[KnowledgeRecord]:
        return self.by_source("verbs")

    def adjectives(self) -> list[KnowledgeRecord]:
        return self.by_source("adjectives")

    def language(self) -> list[KnowledgeRecord]:
        return self.verbs() + self.adjectives()

    def counts(self, refresh: bool = True) -> dict[str, int]:
        if refresh:
            self.refresh()
        result = {source: 0 for source in self.SOURCE_CONFIG}
        for record in self._records:
            result[record.source] = result.get(record.source, 0) + 1
        return result

    def source_status(self) -> dict[str, dict[str, Any]]:
        self.refresh()
        return {key: dict(value) for key, value in self._source_status.items()}

    def _fetch_view(self, view: str, source: str) -> list[KnowledgeRecord]:
        rows: list[dict[str, Any]] = []
        offset = 0
        batch_size = 1000

        while True:
            response = (
                self.client.table(view)
                .select("*")
                .range(offset, offset + batch_size - 1)
                .execute()
            )
            batch = response.data or []
            rows.extend(batch)
            if len(batch) < batch_size:
                break
            offset += batch_size

        return [self._normalise(row, source, view) for row in rows]

    @staticmethod
    def _first(row: dict[str, Any], *names: str) -> str:
        for name in names:
            value = row.get(name)
            if value is not None and str(value).strip():
                return str(value).strip()
        return ""

    @staticmethod
    def _scalar_text(row: dict[str, Any], excluded: set[str]) -> str:
        parts: list[str] = []
        for key, value in row.items():
            if key in excluded or value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                text = str(value).strip()
            elif isinstance(value, (list, dict)):
                text = json.dumps(value, ensure_ascii=False)
            else:
                continue
            if text:
                parts.append(f"{key}: {text}")
        return " | ".join(parts)

    def _normalise(
        self, row: dict[str, Any], source: str, view: str
    ) -> KnowledgeRecord:
        if source == "verbs":
            title = self._first(
                row,
                "lemma",
                "infinitive",
                "verb",
                "word",
                "dansk",
                "title",
            )
            statement = self._scalar_text(
                row,
                {"id", "uuid", "created_at", "updated_at", "lemma", "infinitive", "verb", "word", "dansk", "title"},
            )
            knowledge_type = "verb"
        elif source == "adjectives":
            title = self._first(
                row,
                "lemma",
                "adjective",
                "adjektiv",
                "word",
                "dansk",
                "title",
            )
            statement = self._scalar_text(
                row,
                {"id", "uuid", "created_at", "updated_at", "lemma", "adjective", "adjektiv", "word", "dansk", "title"},
            )
            knowledge_type = "adjective"
        else:
            title = self._first(row, "title", "label", "heading")
            statement = self._first(
                row,
                "statement",
                "content",
                "knowledge_text",
                "final_text",
                "text",
            )
            if not statement:
                statement = self._scalar_text(
                    row,
                    {"id", "knowledge_id", "created_at", "updated_at", "title", "label", "heading", "source_quote", "quote"},
                )
            knowledge_type = self._first(
                row, "knowledge_type", "type", "category", "item_type"
            ) or "other"

        record_id = self._first(row, "id", "knowledge_id", "uuid")
        if not record_id:
            stable = json.dumps(row, sort_keys=True, ensure_ascii=False, default=str)
            record_id = f"{view}:{abs(hash(stable))}"

        exam_type = self._first(
            row, "exam_type", "exam_level", "exam", "proeve"
        ) or None
        source_quote = self._first(
            row, "source_quote", "quote", "evidence_quote", "excerpt"
        )

        return KnowledgeRecord(
            id=record_id,
            source=source,
            exam_type=exam_type,
            knowledge_type=knowledge_type,
            title=title,
            statement=statement,
            source_quote=source_quote,
            metadata={
                key: value
                for key, value in row.items()
                if key
                not in {
                    "statement",
                    "content",
                    "knowledge_text",
                    "final_text",
                    "text",
                    "source_quote",
                    "quote",
                    "evidence_quote",
                    "excerpt",
                }
            },
        )

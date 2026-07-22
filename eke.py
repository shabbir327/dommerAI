"""Examiner Knowledge Engine: deterministic retrieval and evidence packaging."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

from knowledge_repository import KnowledgeRecord, KnowledgeRepository

STOPWORDS = {
    "og", "i", "på", "af", "til", "for", "med", "en", "et", "den", "det",
    "der", "som", "du", "skal", "om", "at", "er", "har", "fra", "kan", "vil",
    "the", "and", "to", "of", "a", "an", "is", "are", "write",
}

OFFICIAL_LIMITS = {
    "rubric": 3,
    "scoring_rule": 3,
    "genre_expectation": 2,
    "language_expectation": 2,
    "grade_anchor": 2,
    "examiner_comment": 1,
    "calibration_answer": 1,
    "writing_task": 2,
}


@dataclass(frozen=True)
class RankedRecord:
    record: KnowledgeRecord
    score: float


class ExaminerKnowledgeEngine:
    def __init__(self, repository: KnowledgeRepository) -> None:
        self.repository = repository

    def build_evidence_package(
        self,
        exam_type: str,
        question: str,
        question_description: str | None,
        answer: str,
    ) -> dict[str, Any]:
        task_query = " ".join(
            part for part in (question, question_description or "") if part
        )
        combined_query = f"{task_query} {answer}".strip()

        official = self._balanced_official(exam_type, combined_query)
        verbs = self._rank(self.repository.verbs(), answer, limit=8, minimum_score=0.02)
        adjectives = self._rank(
            self.repository.adjectives(), answer, limit=8, minimum_score=0.02
        )

        serialised_verbs = [self._serialise(item) for item in verbs]
        serialised_adjectives = [self._serialise(item) for item in adjectives]

        return {
            "exam_type": exam_type,
            "official_examiner_knowledge": [
                self._serialise(item) for item in official
            ],
            "language_knowledge": {
                "verbs": serialised_verbs,
                "adjectives": serialised_adjectives,
            },
            "retrieval_metadata": {
                "method": "balanced_lexical_v2",
                "official_items_considered": len(
                    self.repository.official_for_exam(exam_type)
                ),
                "official_items_selected": len(official),
                "verbs_considered": len(self.repository.verbs()),
                "verbs_selected": len(verbs),
                "adjectives_considered": len(self.repository.adjectives()),
                "adjectives_selected": len(adjectives),
                "knowledge_sources": self.repository.source_status(),
            },
        }

    def _balanced_official(self, exam_type: str, query: str) -> list[RankedRecord]:
        records = self.repository.official_for_exam(exam_type)
        grouped: dict[str, list[KnowledgeRecord]] = defaultdict(list)
        for record in records:
            grouped[self._normalise_type(record.knowledge_type)].append(record)

        selected: list[RankedRecord] = []
        selected_ids: set[str] = set()

        for knowledge_type, limit in OFFICIAL_LIMITS.items():
            ranked = self._rank(grouped.get(knowledge_type, []), query, limit)
            selected.extend(ranked)
            selected_ids.update(item.record.id for item in ranked)

        remaining = [record for record in records if record.id not in selected_ids]
        selected.extend(self._rank(remaining, query, max(0, 15 - len(selected))))
        return sorted(selected, key=lambda item: item.score, reverse=True)[:15]

    def _rank(
        self,
        records: list[KnowledgeRecord],
        query: str,
        limit: int,
        minimum_score: float = 0.0,
    ) -> list[RankedRecord]:
        if limit <= 0:
            return []
        ranked = [
            RankedRecord(record, self._score(query, record.search_text))
            for record in records
        ]
        ranked = [item for item in ranked if item.score >= minimum_score]
        ranked.sort(key=lambda item: item.score, reverse=True)
        return ranked[:limit]

    @staticmethod
    def _normalise_type(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")

    @staticmethod
    def _tokens(text: str) -> set[str]:
        words = re.findall(r"\b[\wæøåÆØÅ-]+\b", text.lower())
        return {word for word in words if len(word) > 2 and word not in STOPWORDS}

    def _score(self, query: str, document: str) -> float:
        query_tokens = self._tokens(query)
        document_tokens = self._tokens(document)
        if not query_tokens or not document_tokens:
            return 0.0

        overlap_count = len(query_tokens & document_tokens)
        query_overlap = overlap_count / len(query_tokens)
        document_overlap = overlap_count / min(len(document_tokens), 25)
        fuzzy = SequenceMatcher(
            None, query.lower()[:2500], document.lower()[:2500]
        ).ratio()
        return round(
            (0.55 * query_overlap) + (0.30 * document_overlap) + (0.15 * fuzzy),
            4,
        )

    @staticmethod
    def _serialise(item: RankedRecord) -> dict[str, Any]:
        record = item.record
        return {
            "id": record.id,
            "source": record.source,
            "exam_type": record.exam_type,
            "knowledge_type": record.knowledge_type,
            "title": record.title,
            "statement": record.statement[:1200],
            "source_quote": record.source_quote[:800],
            "retrieval_score": item.score,
        }

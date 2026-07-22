"""Knowledge-grounded, single-call DommerAI evaluator."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from collections import Counter
from typing import Any, Optional

from groq import AsyncGroq

try:
    from .eke import ExaminerKnowledgeEngine
    from .models import (
        EvaluationRequest,
        Grade,
        InlineError,
        KnowledgeCitation,
        RubricScores,
        WebhookPayload,
        WritingStatistics,
    )
except ImportError:  # Supports `uvicorn main:app` from a flat Render repository.
    from eke import ExaminerKnowledgeEngine
    from models import (
        EvaluationRequest,
        Grade,
        InlineError,
        KnowledgeCitation,
        RubricScores,
        WebhookPayload,
        WritingStatistics,
    )

logger = logging.getLogger("dommer.scorer")

GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
TEMPERATURE = float(os.environ.get("GROQ_TEMPERATURE", "0.05"))
MAX_RETRIES = int(os.environ.get("GROQ_MAX_RETRIES", "3"))
MAX_OUTPUT_TOKENS = int(os.environ.get("GROQ_MAX_OUTPUT_TOKENS", "2200"))
RETRY_DELAY = 1.5

GRADE_SCALE = [-3, 0, 2, 4, 7, 10, 12]
PASS_THRESHOLD = 2
VALID_RUBRIC = {"Top", "Midt", "Bund", "Under niveau"}
VALID_ERROR_TYPES = {
    "spelling", "morphology", "inversion", "syntax", "agreement",
    "punctuation", "word_choice", "missing_word", "other",
}
VALID_SEVERITIES = {"low", "medium", "high"}
MAX_ERRORS = {"Top": 2, "Midt": 6, "Bund": 12, "Under niveau": 20}
DANISH_STOPWORDS = {
    "og", "i", "på", "af", "til", "for", "med", "en", "et", "den", "det",
    "der", "som", "du", "jeg", "vi", "de", "at", "er", "har", "var", "kan",
    "vil", "skal", "ikke", "min", "mit", "mine", "din", "dit", "fra", "om",
}


class Scorer:
    def __init__(self, eke: ExaminerKnowledgeEngine) -> None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY environment variable is not set.")
        self.client = AsyncGroq(api_key=api_key)
        self.eke = eke
        logger.info("Dommer scorer ready - model=%s", GROQ_MODEL)

    async def score(self, request: EvaluationRequest) -> WebhookPayload:
        word_count = len(self._words(request.answer))
        try:
            evidence = self.eke.build_evidence_package(
                exam_type=request.exam_type,
                question=request.question,
                question_description=request.question_description,
                answer=request.answer,
            )
            raw = await self._call_groq(
                self._system_prompt(request.exam_type),
                self._user_prompt(request, word_count, evidence),
                max_tokens=MAX_OUTPUT_TOKENS,
            )
            return self._build_payload(raw, request, word_count, evidence)
        except Exception as exc:
            logger.exception("Scoring failed for eval_id=%s", request.eval_id)
            return WebhookPayload(
                eval_id=request.eval_id,
                status="failed",
                word_count=word_count,
                error=str(exc),
                model_metadata={
                    "provider": "groq",
                    "model": GROQ_MODEL,
                    "prompt_version": "knowledge-grounded-v3-inline-errors",
                    "llm_calls": 1,
                },
            )

    @staticmethod
    def _system_prompt(exam_type: str) -> str:
        return f"""Du er DommerAI, en eksaminator-assistent for {exam_type} skriftlig fremstilling.

Vurder besvarelsen med din egen stærke forståelse af dansk og med evidenspakken som støtte. Officiel publiceret eksaminatorviden har højere autoritet end generelle antagelser. Verber og adjektiver er sproglig støtte og må ikke alene afgøre karakteren. Opfind ikke officielle regler eller kilder.

Vurder tre dimensioner:
- pragmatisk: opgaveopfyldelse, genre, register og kommunikativ succes
- diskursiv: struktur, kohæsion og kohærens
- lingvistisk: ordforråd, grammatik og retskrivning

Gyldige niveauer: Top, Midt, Bund, Under niveau.
Gyldige karakterer: -3, 0, 2, 4, 7, 10, 12. Karakter 2 er laveste beståede.

Fejlregler:
- Find kun sikre, konkrete fejl.
- 'original' skal være en eksakt, sammenhængende streng kopieret fra besvarelsen.
- Gør 'original' så kort som muligt, men langt nok til at lokalisere fejlen entydigt.
- Brug ikke linjenumre eller tegnpositioner; backend beregner dem deterministisk.
- Ved manglende ord skal 'original' være den eksisterende tekst omkring indsættelsesstedet.
- severity: low = mindre formfejl, medium = tydelig grammatisk/lexikalsk fejl, high = fejl der væsentligt hæmmer forståelsen.

Brug knowledge_used kun til evidensposter, der faktisk påvirkede vurderingen. Generel sproglig vurdering må bruges uden citation, men må ikke fremstilles som officiel regel.

Foretag analysen internt. Returner ikke skjult ræsonnement. Returner KUN gyldig JSON:
{{
  "pragmatisk": "Top|Midt|Bund|Under niveau",
  "diskursiv": "Top|Midt|Bund|Under niveau",
  "lingvistisk": "Top|Midt|Bund|Under niveau",
  "overall": 12,
  "pass_fail": "PASSED|NOT PASSED",
  "feedback_da": "2-4 konkrete sætninger",
  "examiner_summary": "1-3 korte sætninger om den samlede eksaminatorbeslutning",
  "errors": [
    {{
      "original": "eksakt tekst fra besvarelsen",
      "correction": "korrektion",
      "type": "spelling|morphology|inversion|syntax|agreement|punctuation|word_choice|missing_word|other",
      "severity": "low|medium|high",
      "grammar_rule_title": "kort navn på reglen eller null",
      "explanation_da": "kort forklaring",
      "explanation_en": "short explanation"
    }}
  ],
  "knowledge_used": [
    {{
      "knowledge_id": "id fra evidenspakken",
      "knowledge_type": "type",
      "reason_used": "kort konkret begrundelse"
    }}
  ]
}}"""

    @staticmethod
    def _user_prompt(request: EvaluationRequest, word_count: int, evidence: dict[str, Any]) -> str:
        description = request.question_description or ""
        return (
            f"EKSAMENSNIVEAU: {request.exam_type}\n"
            f"ORDANTAL: {word_count}\n\n"
            f"OPGAVE:\n{request.question}\n{description}\n\n"
            f"BESVARELSE (bevar linjeskift og tegn præcist):\n{request.answer}\n\n"
            "EVIDENSPAKKE:\n"
            f"{json.dumps(evidence, ensure_ascii=False, separators=(',', ':'))}"
        )

    async def _call_groq(self, system: str, user: str, max_tokens: int) -> dict:
        last_error: Optional[Exception] = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = await self.client.chat.completions.create(
                    model=GROQ_MODEL,
                    temperature=TEMPERATURE,
                    max_tokens=max_tokens,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content
                if not content:
                    raise RuntimeError("Groq returned an empty response.")
                parsed = json.loads(content)
                if not isinstance(parsed, dict):
                    raise RuntimeError("Groq response was not a JSON object.")
                return parsed
            except Exception as exc:
                last_error = exc
                logger.warning("Groq attempt %d/%d failed: %s", attempt, MAX_RETRIES, exc)
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(RETRY_DELAY * (2 ** (attempt - 1)))
        raise RuntimeError(f"All {MAX_RETRIES} Groq attempts failed: {last_error}")

    def _build_payload(
        self,
        raw: dict[str, Any],
        request: EvaluationRequest,
        word_count: int,
        evidence: dict[str, Any],
    ) -> WebhookPayload:
        levels: dict[str, str] = {}
        for dimension in ("pragmatisk", "diskursiv", "lingvistisk"):
            value = raw.get(dimension)
            levels[dimension] = value if value in VALID_RUBRIC else "Midt"

        grade = self._normalise_grade(raw.get("overall"))
        pass_fail = "PASSED" if grade >= PASS_THRESHOLD else "NOT PASSED"

        valid_items = self._evidence_items(evidence)
        valid_ids = {str(item.get("id", "")) for item in valid_items}
        knowledge_used: list[KnowledgeCitation] = []
        seen_citations: set[str] = set()
        raw_citations = raw.get("knowledge_used", [])
        if not isinstance(raw_citations, list):
            raw_citations = []

        for item in raw_citations:
            if not isinstance(item, dict):
                continue
            knowledge_id = str(item.get("knowledge_id", "")).strip()
            if knowledge_id not in valid_ids or knowledge_id in seen_citations:
                continue
            seen_citations.add(knowledge_id)
            knowledge_used.append(KnowledgeCitation(
                knowledge_id=knowledge_id,
                knowledge_type=str(item.get("knowledge_type", "other"))[:80],
                reason_used=str(item.get("reason_used", "Used in evaluation"))[:240],
            ))

        errors = self._build_inline_errors(
            raw.get("errors", []), request.answer, MAX_ERRORS[levels["lingvistisk"]]
        )

        feedback = str(raw.get("feedback_da", "")).strip() or "Ingen feedback tilgængelig."
        summary = str(raw.get("examiner_summary", "")).strip()
        if not summary:
            summary = feedback

        return WebhookPayload(
            eval_id=request.eval_id,
            status="scored",
            rubrik=RubricScores(**levels),
            overall=grade,
            pass_fail=pass_fail,
            feedback_da=feedback[:2000],
            examiner_summary=summary[:1200],
            errors=errors,
            word_count=word_count,
            writing_statistics=self._writing_statistics(request.answer, evidence),
            knowledge_used=knowledge_used,
            retrieval_metadata=evidence.get("retrieval_metadata"),
            model_metadata={
                "provider": "groq",
                "model": GROQ_MODEL,
                "prompt_version": "knowledge-grounded-v3-inline-errors",
                "llm_calls": 1,
                "position_contract": {
                    "line": "1-based",
                    "column_start": "1-based",
                    "column_end": "1-based exclusive",
                    "start_char": "0-based",
                    "end_char": "0-based exclusive",
                },
            },
        )

    def _build_inline_errors(self, raw_errors: Any, answer: str, limit: int) -> list[InlineError]:
        if not isinstance(raw_errors, list):
            return []

        errors: list[InlineError] = []
        used_spans: set[tuple[int, int]] = set()
        for item in raw_errors:
            if not isinstance(item, dict):
                continue
            original = str(item.get("original", "")).strip()
            correction = str(item.get("correction", "")).strip()
            explanation_da = str(item.get("explanation_da", "")).strip()
            explanation_en = str(item.get("explanation_en", "")).strip()
            if not original or not correction or not explanation_da or not explanation_en:
                continue

            span = self._find_unused_span(answer, original, used_spans)
            if span is None:
                continue
            start, end = span
            used_spans.add(span)

            error_type = str(item.get("type", "other")).lower()
            if error_type not in VALID_ERROR_TYPES:
                error_type = "other"
            severity = str(item.get("severity", "medium")).lower()
            if severity not in VALID_SEVERITIES:
                severity = "medium"

            line, column_start, column_end, line_text = self._location(answer, start, end)
            rule_title = str(item.get("grammar_rule_title") or "").strip() or None
            errors.append(InlineError(
                original=original,
                correction=correction,
                type=error_type,
                severity=severity,
                explanation_da=explanation_da,
                explanation_en=explanation_en,
                line=line,
                column_start=column_start,
                column_end=column_end,
                start_char=start,
                end_char=end,
                line_text=line_text,
                grammar_rule_title=rule_title[:160] if rule_title else None,
            ))
            if len(errors) >= limit:
                break
        return errors

    @staticmethod
    def _find_unused_span(answer: str, original: str, used: set[tuple[int, int]]) -> tuple[int, int] | None:
        start = 0
        while True:
            index = answer.find(original, start)
            if index < 0:
                return None
            span = (index, index + len(original))
            if span not in used:
                return span
            start = index + 1

    @staticmethod
    def _location(answer: str, start: int, end: int) -> tuple[int, int, int, str]:
        line = answer.count("\n", 0, start) + 1
        line_start = answer.rfind("\n", 0, start) + 1
        line_end = answer.find("\n", end)
        if line_end < 0:
            line_end = len(answer)
        column_start = start - line_start + 1
        column_end = end - line_start + 1
        return line, column_start, column_end, answer[line_start:line_end]

    def _writing_statistics(self, answer: str, evidence: dict[str, Any]) -> WritingStatistics:
        words = self._words(answer)
        lowered = [word.lower() for word in words]
        unique = set(lowered)
        sentence_count = len([s for s in re.split(r"[.!?]+(?:\s|$)", answer) if s.strip()])
        if sentence_count == 0 and answer.strip():
            sentence_count = 1
        average = round(len(words) / sentence_count, 1) if sentence_count else 0.0
        diversity = round(len(unique) / len(words), 3) if words else 0.0

        language = evidence.get("language_knowledge", {})
        verbs = self._matched_titles(language.get("verbs", []) if isinstance(language, dict) else [], lowered)
        adjectives = self._matched_titles(language.get("adjectives", []) if isinstance(language, dict) else [], lowered)
        counts = Counter(word for word in lowered if len(word) > 3 and word not in DANISH_STOPWORDS)
        repeated = [word for word, count in counts.most_common(8) if count >= 3]

        return WritingStatistics(
            sentence_count=sentence_count,
            average_sentence_length=average,
            unique_word_count=len(unique),
            lexical_diversity=diversity,
            detected_verbs=verbs,
            detected_adjectives=adjectives,
            repeated_words=repeated,
        )

    @staticmethod
    def _matched_titles(items: Any, lowered_words: list[str]) -> list[str]:
        if not isinstance(items, list):
            return []
        text_words = set(lowered_words)
        found: list[str] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "")).strip()
            if title and title.lower() in text_words and title not in found:
                found.append(title)
        return found[:20]

    @staticmethod
    def _words(text: str) -> list[str]:
        return re.findall(r"\b[\wæøåÆØÅ-]+\b", text, flags=re.UNICODE)

    @staticmethod
    def _evidence_items(evidence: dict[str, Any]) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        official = evidence.get("official_examiner_knowledge", [])
        if isinstance(official, list):
            items.extend(item for item in official if isinstance(item, dict))
        language = evidence.get("language_knowledge", {})
        if isinstance(language, dict):
            for group in language.values():
                if isinstance(group, list):
                    items.extend(item for item in group if isinstance(item, dict))
        elif isinstance(language, list):
            items.extend(item for item in language if isinstance(item, dict))
        return items

    @staticmethod
    def _normalise_grade(value: object) -> Grade:
        try:
            number = int(value)
        except (TypeError, ValueError):
            number = 0
        return min(GRADE_SCALE, key=lambda grade: abs(grade - number))  # type: ignore[return-value]

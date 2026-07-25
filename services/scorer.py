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

from services.examiner_knowledge import ExaminerKnowledgeEngine
from services.lexical_engine import LexicalEngine
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

LLM_PROVIDER = "groq"
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
PROMPT_VERSION = "knowledge-grounded-v5-compact-cor"
TEMPERATURE = float(os.environ.get("GROQ_TEMPERATURE", "0.05"))
MAX_RETRIES = int(os.environ.get("GROQ_MAX_RETRIES", "3"))
MAX_OUTPUT_TOKENS = int(os.environ.get("GROQ_MAX_OUTPUT_TOKENS", "1600"))
PROMPT_MAX_CHARS = int(os.environ.get("GROQ_PROMPT_MAX_CHARS", "26000"))
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
    def __init__(self, eke: ExaminerKnowledgeEngine, lexical_engine: LexicalEngine | None = None) -> None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY environment variable is not set.")
        self.client = AsyncGroq(api_key=api_key)
        self.eke = eke
        self.lexical_engine = lexical_engine
        logger.info(
            "Dommer scorer ready - model=%s grammar_hub=%s",
            GROQ_MODEL,
            "enabled" if lexical_engine and lexical_engine.configured else "disabled",
        )

    async def score(self, request: EvaluationRequest) -> WebhookPayload:
        word_count = len(self._words(request.answer))
        try:
            evidence = self.eke.build_evidence_package(
                exam_type=request.exam_type,
                question=request.question,
                question_description=request.question_description,
                answer=request.answer,
            )
            lexical_analysis = await self._run_lexical_analysis(request.answer)
            raw = await self._call_groq(
                self._system_prompt(request.exam_type),
                self._user_prompt(request, word_count, evidence, lexical_analysis),
                max_tokens=MAX_OUTPUT_TOKENS,
            )
            return self._build_payload(
                raw, request, word_count, evidence, lexical_analysis
            )
        except Exception as exc:
            logger.exception("Scoring failed for eval_id=%s", request.eval_id)
            return WebhookPayload(
                eval_id=request.eval_id,
                status="failed",
                word_count=word_count,
                error=str(exc),
                model_metadata={
                    "provider": LLM_PROVIDER,
                    "model": GROQ_MODEL,
                    "prompt_version": PROMPT_VERSION,
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

    def _user_prompt(
        self,
        request: EvaluationRequest,
        word_count: int,
        evidence: dict[str, Any],
        lexical_analysis: dict[str, Any],
    ) -> str:
        description = request.question_description or ""
        compact_evidence = self._compact_evidence_for_prompt(evidence)
        compact_lexical = self._compact_lexical_for_prompt(lexical_analysis)

        fixed = (
            f"EKSAMENSNIVEAU: {request.exam_type}\n"
            f"ORDANTAL: {word_count}\n\n"
            f"OPGAVE:\n{request.question}\n{description}\n\n"
            f"BESVARELSE (bevar linjeskift og tegn præcist):\n{request.answer}\n\n"
        )
        evidence_json = json.dumps(
            compact_evidence, ensure_ascii=False, separators=(",", ":")
        )
        lexical_json = json.dumps(
            compact_lexical, ensure_ascii=False, separators=(",", ":")
        )
        prompt = (
            fixed
            + "UDVALGT OFFICIEL EVIDENS:\n" + evidence_json
            + "\n\nKOMPAKT COR-ANALYSE:\n" + lexical_json
        )

        # Final safety valve for Groq TPM limits. Trim optional lexical token
        # details first; official evidence IDs and candidate text are preserved.
        if len(prompt) > PROMPT_MAX_CHARS:
            compact_lexical["matched_tokens"] = compact_lexical.get("matched_tokens", [])[:10]
            lexical_json = json.dumps(
                compact_lexical, ensure_ascii=False, separators=(",", ":")
            )
            prompt = (
                fixed
                + "UDVALGT OFFICIEL EVIDENS:\n" + evidence_json
                + "\n\nKOMPAKT COR-ANALYSE:\n" + lexical_json
            )

        if len(prompt) > PROMPT_MAX_CHARS:
            compact_evidence["official_examiner_knowledge"] = (
                compact_evidence.get("official_examiner_knowledge", [])[:5]
            )
            evidence_json = json.dumps(
                compact_evidence, ensure_ascii=False, separators=(",", ":")
            )
            prompt = (
                fixed
                + "UDVALGT OFFICIEL EVIDENS:\n" + evidence_json
                + "\n\nKOMPAKT COR-ANALYSE:\n" + lexical_json
            )

        logger.info(
            "Prompt prepared - chars=%d estimated_input_tokens=%d max_output_tokens=%d",
            len(prompt),
            max(1, len(prompt) // 4),
            MAX_OUTPUT_TOKENS,
        )
        return prompt

    @staticmethod
    def _compact_evidence_for_prompt(evidence: dict[str, Any]) -> dict[str, Any]:
        official_out: list[dict[str, Any]] = []
        official = evidence.get("official_examiner_knowledge", [])
        if isinstance(official, list):
            for item in official[:8]:
                if not isinstance(item, dict):
                    continue
                official_out.append({
                    "id": str(item.get("id", "")),
                    "type": str(item.get("knowledge_type", "")),
                    "title": str(item.get("title", ""))[:180],
                    "statement": str(item.get("statement", ""))[:520],
                    "quote": str(item.get("source_quote", ""))[:220],
                    "score": item.get("retrieval_score", 0),
                })

        language_out: dict[str, list[dict[str, Any]]] = {"verbs": [], "adjectives": []}
        language = evidence.get("language_knowledge", {})
        if isinstance(language, dict):
            for group_name in ("verbs", "adjectives"):
                group = language.get(group_name, [])
                if not isinstance(group, list):
                    continue
                for item in group[:5]:
                    if not isinstance(item, dict):
                        continue
                    language_out[group_name].append({
                        "id": str(item.get("id", "")),
                        "title": str(item.get("title", ""))[:100],
                        "statement": str(item.get("statement", ""))[:180],
                    })

        return {
            "exam_type": evidence.get("exam_type"),
            "official_examiner_knowledge": official_out,
            "language_knowledge": language_out,
        }

    @staticmethod
    def _compact_lexical_for_prompt(lexical: dict[str, Any]) -> dict[str, Any]:
        matched_out: list[dict[str, Any]] = []
        matched = lexical.get("matched_tokens", [])
        if isinstance(matched, list):
            for token_item in matched[:30]:
                if not isinstance(token_item, dict):
                    continue
                analyses_out: list[dict[str, Any]] = []
                analyses = token_item.get("analyses", [])
                if isinstance(analyses, list):
                    for analysis in analyses[:2]:
                        if not isinstance(analysis, dict):
                            continue
                        analyses_out.append({
                            "lemma": analysis.get("lemma"),
                            "pos": analysis.get("part_of_speech"),
                            "code": analysis.get("grammar_code"),
                        })
                if analyses_out:
                    matched_out.append({
                        "token": str(token_item.get("token", ""))[:80],
                        "analyses": analyses_out,
                    })

        return {
            "status": lexical.get("status", "unknown"),
            "coverage_percent": lexical.get("coverage_percent", 0.0),
            "detected_verbs": lexical.get("detected_verbs", [])[:20]
                if isinstance(lexical.get("detected_verbs"), list) else [],
            "detected_adjectives": lexical.get("detected_adjectives", [])[:20]
                if isinstance(lexical.get("detected_adjectives"), list) else [],
            "unknown_tokens": lexical.get("unknown_tokens", [])[:20]
                if isinstance(lexical.get("unknown_tokens"), list) else [],
            "matched_tokens": matched_out,
        }

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
                message = str(exc).lower()
                # Retrying an oversized request cannot succeed until the TPM window
                # changes, and it only adds latency. Fail immediately with a clear error.
                if "request too large" in message or "error code: 413" in message:
                    break
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(RETRY_DELAY * (2 ** (attempt - 1)))
        raise RuntimeError(f"All {MAX_RETRIES} Groq attempts failed: {last_error}")

    def _build_payload(
        self,
        raw: dict[str, Any],
        request: EvaluationRequest,
        word_count: int,
        evidence: dict[str, Any],
        lexical_analysis: dict[str, Any],
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
            writing_statistics=self._writing_statistics(
                request.answer, evidence, lexical_analysis
            ),
            knowledge_used=knowledge_used,
            retrieval_metadata=self._merge_retrieval_metadata(
                evidence.get("retrieval_metadata"), lexical_analysis
            ),
            model_metadata={
                "provider": LLM_PROVIDER,
                "model": GROQ_MODEL,
                "prompt_version": PROMPT_VERSION,
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

    def _writing_statistics(
        self,
        answer: str,
        evidence: dict[str, Any],
        lexical_analysis: dict[str, Any],
    ) -> WritingStatistics:
        words = self._words(answer)
        lowered = [word.lower() for word in words]
        unique = set(lowered)
        sentence_count = len([s for s in re.split(r"[.!?]+(?:\s|$)", answer) if s.strip()])
        if sentence_count == 0 and answer.strip():
            sentence_count = 1
        average = round(len(words) / sentence_count, 1) if sentence_count else 0.0
        diversity = round(len(unique) / len(words), 3) if words else 0.0

        language = evidence.get("language_knowledge", {})
        fallback_verbs = self._matched_titles(
            language.get("verbs", []) if isinstance(language, dict) else [], lowered
        )
        fallback_adjectives = self._matched_titles(
            language.get("adjectives", []) if isinstance(language, dict) else [], lowered
        )
        cor_verbs = lexical_analysis.get("detected_verbs", [])
        cor_adjectives = lexical_analysis.get("detected_adjectives", [])
        verbs = self._merge_strings(cor_verbs, fallback_verbs)
        adjectives = self._merge_strings(cor_adjectives, fallback_adjectives)
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

    async def _run_lexical_analysis(self, answer: str) -> dict[str, Any]:
        if self.lexical_engine is None or not self.lexical_engine.configured:
            return {
                "source": "DanskGrammatik Hub / COR",
                "status": "not_configured",
                "detected_verbs": [],
                "detected_adjectives": [],
            }
        try:
            analysis = await self.lexical_engine.analyze(answer)
            logger.info(
                "COR lexical analysis complete - tokens=%s known=%s coverage=%s%%",
                analysis.get("token_count"),
                analysis.get("known_count"),
                analysis.get("coverage_percent"),
            )
            return analysis
        except Exception as exc:
            logger.exception("COR lexical analysis failed; continuing with EKE only")
            return {
                "source": "DanskGrammatik Hub / COR",
                "status": "failed",
                "error": str(exc)[:300],
                "detected_verbs": [],
                "detected_adjectives": [],
            }

    @staticmethod
    def _merge_retrieval_metadata(
        metadata: Any, lexical_analysis: dict[str, Any]
    ) -> dict[str, Any]:
        merged = dict(metadata) if isinstance(metadata, dict) else {}
        sources = dict(merged.get("knowledge_sources", {}))
        sources["grammarhub_cor"] = {
            "status": lexical_analysis.get("status", "unknown"),
            "project": "DommerGrammar",
            "relations": lexical_analysis.get("relations", []),
            "tokens_considered": lexical_analysis.get("token_count", 0),
            "unique_tokens_considered": lexical_analysis.get("unique_token_count", 0),
            "word_forms_matched": lexical_analysis.get("known_count", 0),
            "lemmas_matched": lexical_analysis.get("matched_lemma_count", 0),
            "grammar_codes_matched": lexical_analysis.get("grammar_code_count", 0),
            "coverage_percent": lexical_analysis.get("coverage_percent", 0.0),
        }
        if lexical_analysis.get("error"):
            sources["grammarhub_cor"]["error"] = lexical_analysis["error"]
        merged["knowledge_sources"] = sources
        return merged

    @staticmethod
    def _merge_strings(primary: Any, fallback: Any) -> list[str]:
        values: list[str] = []
        for collection in (primary, fallback):
            if not isinstance(collection, list):
                continue
            for value in collection:
                text = str(value).strip()
                if text and text not in values:
                    values.append(text)
        return values[:20]

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

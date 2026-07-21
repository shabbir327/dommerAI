"""
scorer.py — Dommer scoring engine.

Two-pass scoring:
  Pass 1 — Rubric scoring: overall grade + 3 dimension scores + summary feedback
  Pass 2 — Error analysis: inline errors with corrections and explanations

Both passes run concurrently (asyncio.gather) to keep latency low.
Results are merged into a single WebhookPayload.

Error count is calibrated against lingvistisk score:
  Top          → 0–2 minor errors
  Midt         → 3–6 errors
  Bund         → 7–12 errors
  Under niveau → 12+ errors

Model swap: replace _call_groq() with _call_ollama() after fine-tuning.
"""

import asyncio
import json
import logging
from typing import Optional

from groq import AsyncGroq

from config import settings
from models import (
    EvaluationRequest, WebhookPayload, RubricScores, InlineError,
    Grade, RubricLevel, PassFail,
)

logger = logging.getLogger("dommer.scorer")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

GROQ_MODEL  = settings.groq_model
TEMPERATURE = 0.05
MAX_RETRIES = 3
RETRY_DELAY = 1.5

# ---------------------------------------------------------------------------
# Grade scale — always use index arithmetic, never arithmetic subtraction
# ---------------------------------------------------------------------------

GRADE_SCALE    = [-3, 0, 2, 4, 7, 10, 12]
PASS_THRESHOLD = 2

VALID_RUBRIC = {"Top", "Midt", "Bund", "Under niveau"}
VALID_GRADES = set(GRADE_SCALE)

PF_NORMALISE = {
    "BESTÅET": "PASSED", "IKKE BESTÅET": "NOT PASSED",
    "PASSED": "PASSED", "NOT PASSED": "NOT PASSED",
    "PASS": "PASSED",   "FAIL": "NOT PASSED",
}

VALID_ERROR_TYPES = {
    "spelling", "morphology", "inversion", "syntax",
    "agreement", "punctuation", "word_choice", "missing_word", "other",
}

# Max errors to return per lingvistisk level — keeps UI clean and consistent with score
MAX_ERRORS = {"Top": 2, "Midt": 6, "Bund": 12, "Under niveau": 20}

# ---------------------------------------------------------------------------
# Rubric context (single source of truth)
# ---------------------------------------------------------------------------

RUBRIC_CONTEXT = """
Bedømmelseskriterier — tre dimensioner:

1. PRAGMATISK FÆRDIGHED
   Top:          Alle delopgaver løst fuldt. Genre og register korrekt.
   Midt:         Fleste delopgaver adresseret. Kommunikation fungerer.
   Bund:         Vigtige delopgaver mangler. Kommunikation hæmmet.
   Under niveau: Opgaven ikke løst. Kommunikativ hensigt fejler.

2. DISKURSIV FÆRDIGHED
   Top:          Veltilrettelagt, god kohæsion og kohærens, adækvate bindeord.
   Midt:         Nogenlunde struktur og sammenhæng.
   Bund:         Svag struktur. Hænger ikke godt sammen.
   Under niveau: Ingen struktur. Kaotisk og usammenhængende.

3. LINGVISTISK FÆRDIGHED
   Top:          Adækvat ordvalg, stor grammatisk korrekthed, god retskrivning.
   Midt:         Nogenlunde ordvalg og grammatik. Fejl forstyrrer ikke alvorligt.
   Bund:         Begrænset ordforråd, en del fejl. Forståelsen hæmmes.
   Under niveau: Mange alvorlige fejl. Kommunikation bryder ned.

REGLER:
- Gyldige karakterer: -3, 0, 2, 4, 7, 10, 12
- Karakter 02 er laveste beståede karakter
- PD2 minimum 100 ord (Delprøve 2); PD3 minimum 200 ord (Delprøve 2)
- Manglende opgaveopfyldelse er primært fejlsignal — ikke kun grammatik
"""

# ---------------------------------------------------------------------------
# Error type descriptions (used in prompt to guide model)
# ---------------------------------------------------------------------------

ERROR_TYPE_GUIDE = """
Error type definitions:
- spelling:      Misspelled word (e.g. "skolle" → "skole")
- morphology:    Wrong word form — verb conjugation, noun gender, plural, adjective inflection
                 (e.g. "hun gå" → "hun går", "et stor hus" → "et stort hus")
- inversion:     Missing/wrong subject-verb inversion after fronting adverb or clause
                 (e.g. "I dag jeg spiser" → "I dag spiser jeg")
- syntax:        Word order or sentence structure error not covered by inversion
- agreement:     Subject-verb or adjective-noun agreement error
- punctuation:   Missing comma, wrong punctuation mark
- word_choice:   Wrong word, false friend, or register mismatch
                 (e.g. "blive" vs "være", formal/informal register)
- missing_word:  A grammatically required word is absent
- other:         Does not fit any category above

IMPORTANT ERROR RULES:
- Only mark a phrase when it is clearly incorrect in standard Danish.
- Do not treat a valid alternative phrasing as an error.
- Accept idiomatic phrases such as "På forhånd tak", "Med venlig hilsen",
  "Jeg vil gerne bede om" and "Jeg ser frem til at høre fra jer".
- If uncertain whether something is a real error, omit it.
"""


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

class Scorer:
    def __init__(self):
        if not settings.groq_api_key:
            raise RuntimeError("GROQ_API_KEY environment variable is not set.")
        self.client = AsyncGroq(api_key=settings.groq_api_key)
        logger.info("Dommer scorer ready — model=%s", GROQ_MODEL)

    async def score(self, request: EvaluationRequest) -> WebhookPayload:
        """
        Two-pass scoring — rubric and error analysis run concurrently.
        Returns a single merged WebhookPayload.
        """
        word_count = len(request.answer.split())
        try:
            rubric_raw, errors_raw = await asyncio.gather(
                self._call_groq(self._rubric_system(request.exam_type),
                                self._rubric_user(request, word_count),
                                max_tokens=600),
                self._call_groq(self._error_system(),
                                self._error_user(request),
                                max_tokens=1200),
            )
            return self._build_payload(rubric_raw, errors_raw, request, word_count)

        except Exception as exc:
            logger.error("Scoring failed for eval_id=%s: %s", request.eval_id, exc)
            return WebhookPayload(
                event="evaluation.failed",
                eval_id=request.eval_id,
                status="failed",
                candidate_id=request.candidate_id,
                submitted_at=request.submitted_at,
                metadata=request.metadata,
                webhook_url=request.webhook_url,
                model_name=GROQ_MODEL,
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Pass 1 — Rubric scoring prompt
    # ------------------------------------------------------------------

    def _rubric_system(self, exam_type: str) -> str:
        return f"""Du er en officiel SIRI-eksaminator for {exam_type} skriftlig fremstilling.
{RUBRIC_CONTEXT}
Svar KUN med dette JSON-objekt — ingen anden tekst:
{{
  "pragmatisk":  "Top" | "Midt" | "Bund" | "Under niveau",
  "diskursiv":   "Top" | "Midt" | "Bund" | "Under niveau",
  "lingvistisk": "Top" | "Midt" | "Bund" | "Under niveau",
  "overall":     12 | 10 | 7 | 4 | 2 | 0 | -3,
  "pass_fail":   "PASSED" | "NOT PASSED",
  "feedback_da": "<2-4 sætninger konkret dansk feedback til kandidaten om opgaveopfyldelse, struktur og sprog>"
}}"""

    def _rubric_user(self, request: EvaluationRequest, word_count: int) -> str:
        desc = f"\nOPGAVEBESKRIVELSE:\n{request.question_description}" if request.question_description else ""
        return f"EKSAMENSNIVEAU: {request.exam_type}\nORDANTAL: {word_count}\n\nOPGAVE:\n{request.question}{desc}\n\nBESVARELSE:\n{request.answer}"

    # ------------------------------------------------------------------
    # Pass 2 — Inline error analysis prompt
    # ------------------------------------------------------------------

    def _error_system(self) -> str:
        return f"""Du er en dansk sprogekspert der analyserer fejl i dansksprogede tekster skrevet af andetsprogstalende.
{ERROR_TYPE_GUIDE}

Find konkrete sproglige fejl i teksten. For hver fejl:
- "original": den EKSAKTE streng fra teksten der indeholder fejlen (max 10 ord)
- "correction": den korrigerede version på dansk
- "type": fejltype fra listen ovenfor
- "explanation_da": kort forklaring på dansk (max 15 ord)
- "explanation_en": kort forklaring på engelsk (max 15 ord)

REGLER:
- Returner KUN fejl du er sikker på — undgå falske positiver
- "original" skal være den EKSAKTE tekst fra besvarelsen — ingen ændringer
- Marker IKKE korrekte sætninger som fejl — dansk har mange gyldige konstruktioner
- Fokuser på klare fejl: stavning, bøjning, inversion, manglende ord
- Returner IKKE stilistiske præferencer — kun egentlige fejl

Svar KUN med dette JSON-objekt:
{{
  "errors": [
    {{
      "original":       "<eksakt streng fra teksten>",
      "correction":     "<korrigeret version>",
      "type":           "<fejltype>",
      "explanation_da": "<kort forklaring på dansk>",
      "explanation_en": "<kort forklaring på engelsk>"
    }}
  ]
}}"""

    def _error_user(self, request: EvaluationRequest) -> str:
        return f"EKSAMENSNIVEAU: {request.exam_type}\n\nBESVARELSE:\n{request.answer}"

    # ------------------------------------------------------------------
    # Groq call with retry
    # ------------------------------------------------------------------

    async def _call_groq(self, system: str, user: str, max_tokens: int = 800) -> dict:
        last_error: Optional[Exception] = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = await self.client.chat.completions.create(
                    model=GROQ_MODEL,
                    temperature=TEMPERATURE,
                    max_tokens=max_tokens,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user",   "content": user},
                    ],
                    response_format={"type": "json_object"},
                )
                return json.loads(response.choices[0].message.content)
            except Exception as exc:
                last_error = exc
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(RETRY_DELAY * (2 ** (attempt - 1)))
        raise RuntimeError(f"All {MAX_RETRIES} attempts failed: {last_error}") from last_error

    # ------------------------------------------------------------------
    # Merge + validate both passes into WebhookPayload
    # ------------------------------------------------------------------

    def _build_payload(
        self,
        rubric_raw: dict,
        errors_raw: dict,
        request: EvaluationRequest,
        word_count: int,
    ) -> WebhookPayload:

        # ── Rubric validation ──
        for dim in ("pragmatisk", "diskursiv", "lingvistisk"):
            if rubric_raw.get(dim) not in VALID_RUBRIC:
                logger.warning("Invalid %s for eval_id=%s — defaulting Midt", dim, request.eval_id)
                rubric_raw[dim] = "Midt"

        grade = rubric_raw.get("overall")
        try:
            grade = int(grade)
        except (TypeError, ValueError):
            grade = None
        if grade not in VALID_GRADES:
            grade = min(GRADE_SCALE, key=lambda g: abs(g - (grade or 0)))

        pf_raw = str(rubric_raw.get("pass_fail", "")).strip().upper()
        pf     = PF_NORMALISE.get(pf_raw)
        if pf is None or (pf == "PASSED") != (grade >= PASS_THRESHOLD):
            pf = "PASSED" if grade >= PASS_THRESHOLD else "NOT PASSED"

        feedback = rubric_raw.get("feedback_da", "").strip() or "Ingen feedback tilgængelig."
        ling_level = rubric_raw["lingvistisk"]

        # ── Error validation + cap by lingvistisk level ──
        raw_errors = errors_raw.get("errors", [])
        validated_errors: list[InlineError] = []
        seen_originals: set[str] = set()

        for e in raw_errors:
            original   = str(e.get("original", "")).strip()
            correction = str(e.get("correction", "")).strip()
            err_type   = str(e.get("type", "other")).strip().lower()
            expl_da    = str(e.get("explanation_da", "")).strip()
            expl_en    = str(e.get("explanation_en", "")).strip()

            # Skip if original not actually in the essay (model hallucination guard)
            if not original or original not in request.answer:
                continue
            # Skip duplicates
            if original in seen_originals:
                continue
            # Normalise unknown error types
            if err_type not in VALID_ERROR_TYPES:
                err_type = "other"
            # Skip if no explanation
            if not expl_da or not expl_en:
                continue

            seen_originals.add(original)
            validated_errors.append(InlineError(
                original=original,
                correction=correction,
                type=err_type,
                explanation_da=expl_da,
                explanation_en=expl_en,
            ))

            if len(validated_errors) >= MAX_ERRORS.get(ling_level, 6):
                break

        logger.info(
            "eval_id=%s grade=%s pf=%s ling=%s errors=%d",
            request.eval_id, grade, pf, ling_level, len(validated_errors),
        )

        return WebhookPayload(
            event="evaluation.completed",
            eval_id=request.eval_id,
            status="scored",
            candidate_id=request.candidate_id,
            submitted_at=request.submitted_at,
            metadata=request.metadata,
            webhook_url=request.webhook_url,
            model_name=GROQ_MODEL,
            rubrik=RubricScores(
                pragmatisk=rubric_raw["pragmatisk"],
                diskursiv=rubric_raw["diskursiv"],
                lingvistisk=rubric_raw["lingvistisk"],
            ),
            overall=grade,
            pass_fail=pf,
            feedback_da=feedback,
            errors=validated_errors,
            word_count=word_count,
        )

"""Deterministic lexical lookup engine backed by DanskGrammatik Hub in Supabase."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from typing import Any
from urllib.parse import quote

import httpx

from config import SUPABASE_KEY_COR, SUPABASE_URL_COR

logger = logging.getLogger("dommer.lexical")

_TOKEN_RE = re.compile(r"[^\W\d_]+(?:[-'][^\W\d_]+)*", re.UNICODE)


class LexicalEngineError(RuntimeError):
    """Raised when the Grammar Hub cannot complete a lexical operation."""


class LexicalEngine:
    def __init__(self) -> None:
        self.base_url = SUPABASE_URL_COR
        self.service_key = SUPABASE_KEY_COR
        self.timeout = float(os.environ.get("LEXICAL_TIMEOUT_SECONDS", "12"))
        self.max_concurrency = int(os.environ.get("LEXICAL_MAX_CONCURRENCY", "8"))
        self._client: httpx.AsyncClient | None = None

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.service_key)

    async def start(self) -> None:
        if not self.configured:
            logger.warning("Grammar Hub is not configured; lexical endpoints will return 503.")
            return
        self._client = httpx.AsyncClient(
            base_url=f"{self.base_url}/rest/v1",
            timeout=self.timeout,
            headers={
                "apikey": self.service_key,
                "Authorization": f"Bearer {self.service_key}",
                "Accept-Profile": "language",
                "Content-Profile": "language",
            },
        )
        logger.info("Lexical engine connected to DanskGrammatik Hub.")

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    def _require_client(self) -> httpx.AsyncClient:
        if not self.configured or self._client is None:
            raise LexicalEngineError(
                "Grammar Hub is not configured. Set SUPABASE_URL_COR and "
                "SUPABASE_KEY_COR."
            )
        return self._client

    async def _get(self, table: str, params: dict[str, str]) -> list[dict[str, Any]]:
        client = self._require_client()
        response = await client.get(f"/{table}", params=params)
        if response.status_code >= 400:
            raise LexicalEngineError(
                f"Grammar Hub query failed ({response.status_code}): {response.text[:300]}"
            )
        return response.json()

    async def health(self) -> dict[str, Any]:
        started = time.perf_counter()
        rows = await self._get(
            "cor_word_forms",
            {"select": "word_form_id", "limit": "1"},
        )
        return {
            "status": "ok",
            "configured": True,
            "database_reachable": True,
            "sample_row_available": bool(rows),
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        }

    async def lookup_word(self, word: str, normalized_only: bool = False) -> dict[str, Any]:
        query_word = word.strip()
        if not query_word:
            raise ValueError("word cannot be empty")

        params = {
            "select": (
                "word_form_id,lemma_id,cor_full_id,full_form,grammar_code,"
                "normalization_code,variation_number"
            ),
            "full_form": f"ilike.{quote(query_word, safe='')}",
            "is_active": "eq.true",
            "order": "normalization_code.asc,cor_full_id.asc",
            "limit": "100",
        }
        if normalized_only:
            params["normalization_code"] = "eq.N"

        forms = await self._get("cor_word_forms", params)
        if not forms:
            return {
                "query": query_word,
                "found": False,
                "match_count": 0,
                "lemmas": [],
                "analyses": [],
            }

        lemma_ids = sorted({int(row["lemma_id"]) for row in forms})
        grammar_codes = sorted({str(row["grammar_code"]) for row in forms})

        lemma_filter = ",".join(str(value) for value in lemma_ids)
        grammar_filter = ",".join(grammar_codes)

        lemmas_task = self._get(
            "cor_lemmas",
            {
                "select": "lemma_id,cor_lemma_id,preferred_lemma,preferred_gloss",
                "lemma_id": f"in.({lemma_filter})",
                "is_active": "eq.true",
            },
        )
        grammar_task = self._get(
            "cor_grammar_codes",
            {
                "select": "grammar_code,grammatical_label,is_parsed",
                "grammar_code": f"in.({grammar_filter})",
            },
        )
        lemmas, grammar_rows = await asyncio.gather(lemmas_task, grammar_task)

        lemma_map = {int(row["lemma_id"]): row for row in lemmas}
        grammar_map = {str(row["grammar_code"]): row for row in grammar_rows}

        analyses: list[dict[str, Any]] = []
        for form in forms:
            lemma = lemma_map.get(int(form["lemma_id"]), {})
            grammar = grammar_map.get(str(form["grammar_code"]), {})
            analyses.append(
                {
                    "cor_full_id": form.get("cor_full_id"),
                    "form": form.get("full_form"),
                    "lemma_id": form.get("lemma_id"),
                    "cor_lemma_id": lemma.get("cor_lemma_id"),
                    "lemma": lemma.get("preferred_lemma"),
                    "gloss": lemma.get("preferred_gloss"),
                    "grammar_code": form.get("grammar_code"),
                    "grammatical_label": grammar.get("grammatical_label"),
                    "grammar_is_parsed": grammar.get("is_parsed"),
                    "normalization_code": form.get("normalization_code"),
                    "variation_number": form.get("variation_number"),
                }
            )

        unique_lemmas = sorted(
            {
                (row.get("lemma"), row.get("cor_lemma_id"))
                for row in analyses
                if row.get("lemma")
            }
        )

        return {
            "query": query_word,
            "found": True,
            "match_count": len(analyses),
            "lemmas": [
                {"lemma": lemma, "cor_lemma_id": cor_id}
                for lemma, cor_id in unique_lemmas
            ],
            "analyses": analyses,
        }

    async def get_lemma(self, cor_lemma_id: str) -> dict[str, Any]:
        lemma_rows = await self._get(
            "cor_lemmas",
            {
                "select": (
                    "lemma_id,cor_lemma_id,preferred_lemma,preferred_gloss,"
                    "source_record_count,has_label_variants,has_gloss_variants"
                ),
                "cor_lemma_id": f"eq.{quote(cor_lemma_id, safe='')}",
                "is_active": "eq.true",
                "limit": "1",
            },
        )
        if not lemma_rows:
            return {"cor_lemma_id": cor_lemma_id, "found": False}

        lemma = lemma_rows[0]
        lemma_id = lemma["lemma_id"]
        labels_task = self._get(
            "cor_lemma_labels",
            {
                "select": "lemma_text,gloss,is_primary,source_row_number",
                "lemma_id": f"eq.{lemma_id}",
                "is_active": "eq.true",
                "order": "is_primary.desc,source_row_number.asc",
            },
        )
        forms_task = self._get(
            "cor_word_forms",
            {
                "select": (
                    "cor_full_id,full_form,grammar_code,normalization_code,variation_number"
                ),
                "lemma_id": f"eq.{lemma_id}",
                "is_active": "eq.true",
                "order": "grammar_code.asc,variation_number.asc",
                "limit": "1000",
            },
        )
        labels, forms = await asyncio.gather(labels_task, forms_task)

        grammar_codes = sorted({str(row["grammar_code"]) for row in forms})
        grammar_map: dict[str, dict[str, Any]] = {}
        if grammar_codes:
            grammar_rows = await self._get(
                "cor_grammar_codes",
                {
                    "select": "grammar_code,grammatical_label,is_parsed",
                    "grammar_code": f"in.({','.join(grammar_codes)})",
                },
            )
            grammar_map = {str(row["grammar_code"]): row for row in grammar_rows}

        for form in forms:
            grammar = grammar_map.get(str(form["grammar_code"]), {})
            form["grammatical_label"] = grammar.get("grammatical_label")
            form["grammar_is_parsed"] = grammar.get("is_parsed")

        return {
            "found": True,
            "lemma": lemma,
            "labels": labels,
            "forms": forms,
        }

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return _TOKEN_RE.findall(text)

    async def analyze_text(self, text: str, unique_only: bool = True) -> dict[str, Any]:
        tokens = self.tokenize(text)
        lookup_tokens = list(dict.fromkeys(token.lower() for token in tokens)) if unique_only else [
            token.lower() for token in tokens
        ]

        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def bounded_lookup(token: str) -> dict[str, Any]:
            async with semaphore:
                return await self.lookup_word(token)

        results = await asyncio.gather(*(bounded_lookup(token) for token in lookup_tokens))
        known = [row for row in results if row["found"]]
        unknown = [row["query"] for row in results if not row["found"]]

        return {
            "token_count": len(tokens),
            "unique_token_count": len(set(token.lower() for token in tokens)),
            "looked_up_count": len(lookup_tokens),
            "known_count": len(known),
            "unknown_count": len(unknown),
            "coverage_percent": round((len(known) / len(lookup_tokens) * 100), 2)
            if lookup_tokens
            else 0.0,
            "unknown_tokens": unknown,
            "results": results,
        }

    async def analyze(self, text: str) -> dict[str, Any]:
        """Analyze candidate text and return a compact COR-backed linguistic profile.

        This is the method used by the evaluation pipeline. The lower-level
        ``analyze_text`` response remains available for the public lexicon API.
        """
        raw = await self.analyze_text(text, unique_only=True)

        matched_tokens: list[dict[str, Any]] = []
        verbs: list[str] = []
        adjectives: list[str] = []
        lemmas_seen: set[str] = set()
        grammar_codes_seen: set[str] = set()

        for result in raw.get("results", []):
            if not isinstance(result, dict) or not result.get("found"):
                continue
            token = str(result.get("query", "")).strip()
            analyses = result.get("analyses", [])
            if not isinstance(analyses, list):
                continue

            compact_analyses: list[dict[str, Any]] = []
            for analysis_index, item in enumerate(analyses):
                if not isinstance(item, dict):
                    continue
                lemma = str(item.get("lemma") or "").strip()
                grammar_code = str(item.get("grammar_code") or "").strip()
                grammar_label = str(item.get("grammatical_label") or "").strip()
                pos = self._infer_part_of_speech(grammar_code, grammar_label)

                if lemma:
                    lemmas_seen.add(lemma)
                if grammar_code:
                    grammar_codes_seen.add(grammar_code)
                if pos == "verb" and lemma and lemma not in verbs:
                    verbs.append(lemma)
                elif pos == "adjective" and lemma and lemma not in adjectives:
                    adjectives.append(lemma)

                if analysis_index < 8:
                    compact_analyses.append({
                        "lemma": lemma or None,
                        "cor_lemma_id": item.get("cor_lemma_id"),
                        "grammar_code": grammar_code or None,
                        "grammatical_label": grammar_label or None,
                        "part_of_speech": pos,
                        "normalization_code": item.get("normalization_code"),
                    })

            matched_tokens.append({
                "token": token,
                "analyses": compact_analyses,
            })

        return {
            "source": "DanskGrammatik Hub / COR",
            "status": "loaded",
            "token_count": raw.get("token_count", 0),
            "unique_token_count": raw.get("unique_token_count", 0),
            "looked_up_count": raw.get("looked_up_count", 0),
            "known_count": raw.get("known_count", 0),
            "unknown_count": raw.get("unknown_count", 0),
            "coverage_percent": raw.get("coverage_percent", 0.0),
            "matched_lemma_count": len(lemmas_seen),
            "grammar_code_count": len(grammar_codes_seen),
            "detected_verbs": verbs[:40],
            "detected_adjectives": adjectives[:40],
            "unknown_tokens": raw.get("unknown_tokens", [])[:40],
            "matched_tokens": matched_tokens[:100],
            "relations": [
                "language.cor_lemmas",
                "language.cor_word_forms",
                "language.cor_grammar_codes",
            ],
        }

    @staticmethod
    def _infer_part_of_speech(grammar_code: str, grammar_label: str) -> str | None:
        combined = f"{grammar_code} {grammar_label}".lower().strip()
        code = grammar_code.upper().strip()

        # COR labels vary between full Danish labels, abbreviations, and compact
        # grammar codes. Use conservative word-boundary matching to avoid
        # classifying unrelated codes merely because they contain one letter.
        if (
            re.search(r"\b(verbum|verb|udsagnsord|vb\.?)\b", combined)
            or re.match(r"^(V|VB)(?:[_.:-]|$)", code)
        ):
            return "verb"
        if (
            re.search(r"\b(adjektiv|adjective|tillægsord|adj\.?)\b", combined)
            or re.match(r"^(ADJ|A)(?:[_.:-]|$)", code)
        ):
            return "adjective"
        if (
            re.search(r"\b(substantiv|noun|navneord|sb\.?)\b", combined)
            or re.match(r"^(N|SB)(?:[_.:-]|$)", code)
        ):
            return "noun"
        if (
            re.search(r"\b(adverb|biord|adv\.?)\b", combined)
            or re.match(r"^ADV(?:[_.:-]|$)", code)
        ):
            return "adverb"
        if re.search(r"\b(pronomen|pronoun|stedord|pron\.?)\b", combined):
            return "pronoun"
        return None


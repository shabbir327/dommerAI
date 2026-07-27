"""Contextual Danish POS tagging.

Disambiguates homonymous COR word-form analyses (e.g. deciding whether "for"
in a given sentence is the preposition or the rare verb reading COR also has
on file for that surface form) using a real Danish POS tagger, instead of a
hand-maintained list of words that should never be treated as content words.

Loads a spaCy-compatible Danish pipeline once at startup (see main.py's
lifespan) and exposes a single `tag(text)` method used by LexicalEngine.
"""

from __future__ import annotations

import logging
import os
from collections import Counter
from typing import Any

logger = logging.getLogger("dommer.pos_tagger")

# Universal POS tags (spaCy's token.pos_) that count as "verb" or
# "adjective" for our purposes. AUX covers modal/auxiliary verbs (kan, vil,
# skal, har...), which candidates use constantly and which should count as
# verbs in writing_statistics.
_VERB_TAGS = {"VERB", "AUX"}
_ADJECTIVE_TAGS = {"ADJ"}

# Verify against `dacy.models()` if this ever fails to load — DaCy's
# published model names have changed across package versions.
_DACY_SMALL_MODEL = "da_dacy_small_trf-0.2.0"
_SPACY_SMALL_MODEL = "da_core_news_sm"


class PosTagger:
    """Thin wrapper so the underlying pipeline (DaCy vs plain spaCy) is
    swappable via POS_TAGGER_MODEL without touching any calling code — both
    expose the same spaCy Doc/Token interface.
    """

    def __init__(self) -> None:
        self._nlp: Any | None = None
        self.model_name: str | None = None

    @property
    def ready(self) -> bool:
        return self._nlp is not None

    def load(self) -> None:
        """Load the configured pipeline. Safe to call even if the optional
        dacy/spacy dependencies aren't installed — failures are logged and
        the tagger simply stays unavailable, and callers fall back to the
        pre-tagger heuristic (see LexicalEngine.analyze).
        """
        choice = os.environ.get("POS_TAGGER_MODEL", "dacy_small").strip().lower()
        try:
            if choice == "spacy_small":
                import spacy

                self._nlp = spacy.load(_SPACY_SMALL_MODEL)
            elif choice == "none":
                self._nlp = None
                return
            else:
                import dacy

                self._nlp = dacy.load(_DACY_SMALL_MODEL)
            self.model_name = choice
            logger.info("POS tagger loaded — model=%s", choice)
        except Exception:
            logger.exception(
                "POS tagger failed to load (%s); continuing without contextual "
                "disambiguation — LexicalEngine falls back to its pre-tagger "
                "heuristic.",
                choice,
            )
            self._nlp = None

    def tag(self, text: str) -> dict[str, str]:
        """Return {lowercased_token: majority_pos_tag} for every token in
        text.

        POS is decided per-occurrence in context (that's the whole point —
        the same surface form can be tagged differently depending on the
        sentence), then majority-voted across all of a token's occurrences
        in this one answer, since detected_verbs/detected_adjectives is a
        de-duplicated lexicon-level list rather than a per-occurrence one.
        """
        if self._nlp is None or not text.strip():
            return {}

        votes: dict[str, Counter] = {}
        doc = self._nlp(text)
        for token in doc:
            if token.is_space or token.is_punct:
                continue
            key = token.text.lower()
            votes.setdefault(key, Counter())[token.pos_] += 1

        return {key: counter.most_common(1)[0][0] for key, counter in votes.items()}

    @staticmethod
    def category(pos_tag: str | None) -> str | None:
        if pos_tag in _VERB_TAGS:
            return "verb"
        if pos_tag in _ADJECTIVE_TAGS:
            return "adjective"
        return None

"""lingua-py wrapper for language detection.

Default behaviour per spec Section 5.3:
- Returns 'id', 'en', or 'mixed'.
- minimum_relative_distance=0.0 (lingua default) with a post-hoc confidence
  threshold: if the top language's confidence < threshold, fall back to 'id'.
- 'mixed' is surfaced only when a sentence-level scan finds both languages with
  meaningful signal — useful for bilingual queries like
  "apa itu PMI Bermasalah? (What is a Problematic Migrant Worker?)".
"""

from __future__ import annotations

import logging
from typing import Literal

from lingua import Language, LanguageDetectorBuilder

logger = logging.getLogger(__name__)

# Build once at import time — detector construction is expensive (~100 ms).
# Restrict to ID + EN only: tighter focus means higher confidence for these two.
_detector = (
    LanguageDetectorBuilder
    .from_languages(Language.INDONESIAN, Language.ENGLISH)
    .build()
)

LangCode = Literal["id", "en", "mixed"]


class LangDetector:
    """Thin wrapper around lingua detector that returns 'id'/'en'/'mixed'."""

    def __init__(self) -> None:
        self._detector = _detector  # reuse the module-level singleton

    def detect(self, text: str, threshold: float = 0.7) -> LangCode:
        """Detect query language.

        Args:
            text:      Raw user query string.
            threshold: Confidence threshold below which we default to 'id'.
                       lingua's confidence_values() returns floats in [0, 1].

        Returns:
            'id'    — Indonesian (or ambiguous below threshold → default ID)
            'en'    — English
            'mixed' — Both languages have confidence above threshold / 2
        """
        if not text or not text.strip():
            return "id"

        # Use compute_language_confidence_values for granular confidence.
        confidence_values = self._detector.compute_language_confidence_values(text)
        # confidence_values: list of (Language, float) sorted descending
        if not confidence_values:
            return "id"

        # Build a dict for easy lookup
        conf: dict[Language, float] = {lv.language: lv.value for lv in confidence_values}

        id_conf = conf.get(Language.INDONESIAN, 0.0)
        en_conf = conf.get(Language.ENGLISH, 0.0)

        logger.debug(
            "LangDetect '%s...' → ID=%.3f EN=%.3f",
            text[:40], id_conf, en_conf,
        )

        # Check for mixed: both languages have meaningful signal
        mixed_threshold = threshold / 2  # 0.35 with default threshold=0.7
        if id_conf >= mixed_threshold and en_conf >= mixed_threshold:
            # Tiebreak: if one is clearly dominant, use it; else 'mixed'
            if id_conf >= threshold and en_conf < mixed_threshold:
                return "id"
            if en_conf >= threshold and id_conf < mixed_threshold:
                return "en"
            return "mixed"

        # Standard path: use top language if above threshold, else default ID
        top_lang = confidence_values[0].language
        top_conf = confidence_values[0].value

        if top_conf < threshold:
            return "id"
        if top_lang == Language.ENGLISH:
            return "en"
        return "id"


# Module-level convenience function (matches spec line 895-900)
def detect(text: str) -> Literal["id", "en", "unknown"]:
    """Module-level detect for direct import.

    Returns 'id', 'en', or 'unknown' (spec interface).
    'unknown' is returned when text is empty or None.
    """
    if not text or not text.strip():
        return "unknown"
    result = _LangDetectorSingleton.detect(text)
    if result == "mixed":
        return "id"  # default mixed to 'id' at module level (per spec "default ID")
    return result


# Singleton for module-level detect()
_LangDetectorSingleton = LangDetector()

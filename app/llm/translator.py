"""Query-only EN→ID translation via Gemini Flash.

Per spec Section 5.3:
- Used ONLY for retrieval pre-processing: translate English query to Indonesian
  so that BM25 + dense retrieval operates on the same language as the corpus.
- Response generation uses the ORIGINAL (untranslated) query.
- Results cached in SQLite by sha256(query), TTL 30 days.
- Preserves legal terminology: "TPPO", "PMI", "Permensos", etc. are passed through.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
import time
from pathlib import Path

import litellm

logger = logging.getLogger(__name__)

_CACHE_DB_PATH = Path("data/translations.db")
_CACHE_TTL_SECONDS = 30 * 24 * 3600  # 30 days

_TRANSLATE_PROMPT = (
    "Translate the following user question from English to formal Indonesian. "
    "Output only the translation, no explanation. Preserve any legal terms "
    "that should remain in their original form (e.g., 'TPPO', 'PMI', 'Permensos'). "
    "\n\nQuestion: {query}"
)


class Translator:
    """EN→ID translator backed by Gemini Flash with SQLite result cache."""

    def __init__(self, cache_db: str | Path = _CACHE_DB_PATH) -> None:
        self._cache_db = Path(cache_db)
        self._init_cache()

    def _init_cache(self) -> None:
        """Create cache DB and table if not exists."""
        self._cache_db.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self._cache_db))
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS translation_cache (
                    query_hash TEXT PRIMARY KEY,
                    query_en   TEXT NOT NULL,
                    result_id  TEXT NOT NULL,
                    created_at INTEGER NOT NULL
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    def _cache_get(self, query_hash: str) -> str | None:
        """Return cached translation if present and not expired."""
        conn = sqlite3.connect(str(self._cache_db))
        try:
            row = conn.execute(
                "SELECT result_id, created_at FROM translation_cache WHERE query_hash = ?",
                (query_hash,),
            ).fetchone()
        finally:
            conn.close()

        if row is None:
            return None
        result_id, created_at = row
        if time.time() - created_at > _CACHE_TTL_SECONDS:
            # Expired — delete and return None
            self._cache_delete(query_hash)
            return None
        return result_id

    def _cache_put(self, query_hash: str, query_en: str, result_id: str) -> None:
        conn = sqlite3.connect(str(self._cache_db))
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO translation_cache
                    (query_hash, query_en, result_id, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (query_hash, query_en, result_id, int(time.time())),
            )
            conn.commit()
        finally:
            conn.close()

    def _cache_delete(self, query_hash: str) -> None:
        conn = sqlite3.connect(str(self._cache_db))
        try:
            conn.execute(
                "DELETE FROM translation_cache WHERE query_hash = ?", (query_hash,)
            )
            conn.commit()
        finally:
            conn.close()

    def translate_en_to_id(self, query_en: str) -> str:
        """Translate English query to Indonesian for retrieval use.

        Returns the Indonesian translation. On API failure, returns the
        original English query (graceful degradation — retrieval quality
        degrades but pipeline does not crash).

        Note: ONLY used for retrieval pre-processing. Response generation
        always uses the original English query.
        """
        if not query_en or not query_en.strip():
            return query_en

        query_hash = hashlib.sha256(query_en.encode("utf-8")).hexdigest()

        # Cache hit
        cached = self._cache_get(query_hash)
        if cached is not None:
            logger.debug("Translator cache hit for '%s...'", query_en[:40])
            return cached

        # Call Gemini Flash for translation
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            logger.warning(
                "Translator: GEMINI_API_KEY not set — returning original EN query."
            )
            return query_en

        prompt = _TRANSLATE_PROMPT.format(query=query_en)
        try:
            response = litellm.completion(
                model="gemini/gemini-2.5-flash",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=256,
                temperature=0.1,
                api_key=api_key,
            )
            translated = response.choices[0].message.content.strip()
            logger.debug(
                "Translator: '%s...' → '%s...'",
                query_en[:40], translated[:40],
            )
            self._cache_put(query_hash, query_en, translated)
            return translated

        except Exception as exc:
            logger.warning(
                "Translator: Gemini Flash call failed (%s: %s) — falling back to original.",
                type(exc).__name__, str(exc)[:120],
            )
            return query_en


# Module-level convenience function
def translate_en_to_id(query_en: str) -> str:
    """One-shot EN→ID translation for retrieval query.

    Returns Indonesian translation, preserving legal terminology.
    Note: ONLY used for retrieval pre-processing. Response generation always
    uses the original query language.
    """
    return _translator_singleton.translate_en_to_id(query_en)


_translator_singleton = Translator()

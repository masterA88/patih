"""
BM25Okapi sparse retrieval index with disk persistence.

Tokenization: whitespace split + lowercase + punctuation strip.
NO stemming — Sastrawi over-aggressive on legal terms per build-spec Section 5.2 line 704.

Index corpus: child chunks' text_for_embed (includes "passage: [label] BAB..." prefix).
  The prefix is intentionally kept because it includes Pasal references like "Pasal 5"
  which help BM25 match query terms that mention specific Pasal numbers.

Persistence:
  Pickle file: data/bm25/permensos_bm25.pkl
  Contents: {"index": BM25Okapi, "chunk_ids": list[str]}

See build-spec Section 5.2 line 703-705.
"""

from __future__ import annotations

import logging
import pickle
import re
import string
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_PICKLE_PATH = "data/bm25/permensos_bm25.pkl"

_PUNCT_RE = re.compile(r"[" + re.escape(string.punctuation) + r"]")


def tokenize(text: str) -> list[str]:
    """
    Tokenize text for BM25.
    Steps:
      1. Lowercase
      2. Strip punctuation
      3. Whitespace split
      4. Drop empty tokens

    No stemming (spec Section 5.2 — Sastrawi over-aggressive on legal terms).
    """
    lowered = text.lower()
    stripped = _PUNCT_RE.sub(" ", lowered)
    tokens = [t for t in stripped.split() if t]
    return tokens


class BM25Store:
    """
    Wrapper around BM25Okapi with persist/load helpers.

    Usage:
        store = BM25Store.build(chunks)   # build from list of ChunkMeta dicts
        store.save(path)                  # serialize to disk
        store = BM25Store.load(path)      # restore from disk

        results = store.query_sparse("apa itu korban tppo", top_k=15)
        # -> list of (chunk_id, bm25_score)
    """

    def __init__(self, index: Any, chunk_ids: list[str]) -> None:
        self._index = index
        self._chunk_ids = chunk_ids

    @classmethod
    def build(cls, chunks: list[dict]) -> "BM25Store":
        """
        Build a BM25Okapi index from a list of ChunkMeta dicts.
        Uses text_for_embed field as corpus.
        """
        from rank_bm25 import BM25Okapi

        corpus_tokens = [tokenize(c["text_for_embed"]) for c in chunks]
        chunk_ids = [c["chunk_id"] for c in chunks]

        logger.info("Building BM25 index over %d chunks...", len(chunks))
        index = BM25Okapi(corpus_tokens)
        logger.info("BM25 index built.")

        return cls(index, chunk_ids)

    def query_sparse(self, query: str, top_k: int = 15) -> list[tuple[str, float]]:
        """
        BM25 retrieval.

        Args:
            query: raw query string (no prefix needed — tokenization handles it).
            top_k: number of results to return.

        Returns:
            list of (chunk_id, bm25_score) sorted descending by score.
            Zero-score chunks excluded.
        """
        tokens = tokenize(query)
        scores = self._index.get_scores(tokens)

        # Pair (chunk_id, score) and sort descending
        ranked = sorted(
            zip(self._chunk_ids, scores.tolist()),
            key=lambda x: -x[1],
        )

        # Filter zero-score and take top_k
        filtered = [(cid, s) for cid, s in ranked if s > 0.0]
        return filtered[:top_k]

    def save(self, path: str = DEFAULT_PICKLE_PATH) -> None:
        """Serialize index + chunk_ids to pickle."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {"index": self._index, "chunk_ids": self._chunk_ids}
        with open(p, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        size_mb = p.stat().st_size / 1e6
        logger.info("BM25 index saved to %s (%.1f MB)", p, size_mb)

    @classmethod
    def load(cls, path: str = DEFAULT_PICKLE_PATH) -> "BM25Store":
        """Load index from pickle. Raises FileNotFoundError if not found."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"BM25 index not found at {p.resolve()}. "
                "Run: .venv\\Scripts\\python.exe -m app.retrieval.indexer"
            )
        with open(p, "rb") as f:
            payload = pickle.load(f)
        logger.info("BM25 index loaded from %s (%d docs)", p, len(payload["chunk_ids"]))
        return cls(payload["index"], payload["chunk_ids"])

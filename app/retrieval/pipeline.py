"""
Retrieval orchestrator: query -> [dense+sparse] -> RRF -> parent expand -> cross-ref -> always-on -> context.

Public interface (build-spec Section 5.2):

    from app.retrieval.pipeline import retrieve, RetrievalResult

    result = retrieve("apa itu korban TPPO")
    for chunk in result.parent_chunks:
        print(chunk["pasal"], chunk["text"][:200])

Output ordering (spec Section 5.2 line 727-731):
    1. Pasal 1 (always-on)
    2. Top retrieved parents (by RRF score, descending)
    3. Cross-ref expansions

Lazy initialization: Embedder, BM25Store, Chroma, and parent_lookup are
loaded on first call to retrieve() and reused across calls (module-level singletons).

See build-spec Section 5.2 line 682-731.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()  # load GEMINI_API_KEY (etc) from .env for embedder backend detection

logger = logging.getLogger(__name__)

# --- Default paths (relative to project root / CWD) ---
DEFAULT_MODEL_PATH = "models/multilingual-e5-large-onnx-int8"
DEFAULT_CHROMA_DIR = "data/chroma"
DEFAULT_BM25_PATH = "data/bm25/permensos_bm25.pkl"
DEFAULT_PARENT_LOOKUP_PATH = "data/bm25/parent_lookup.json"
DEFAULT_PARSED_PATH = "data/parsed/permensos-8-2023.json"  # legacy single-doc default
DEFAULT_PARSED_DIR = "data/parsed"
DEFAULT_DOC_ID = "permensos-8-2023"

# --- Module-level singletons (initialized once) ---
_embedder = None
_bm25_store = None
_chroma_collection = None
_parent_lookup: dict[str, dict] | None = None
_child_lookup: dict[str, dict] | None = None


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

class RetrievalResult(BaseModel):
    """
    Output of retrieve().

    parent_chunks: ordered list of parent ChunkMeta dicts, ready for LLM context.
                   Each dict has all ChunkMeta fields; list fields (tags,
                   cross_refs_outgoing) remain as Python lists (deserialized
                   from JSON — NOT the Chroma-serialized string form).
    debug:         intermediates for inspection / eval.
    """

    model_config = {"arbitrary_types_allowed": True}

    parent_chunks: list[dict[str, Any]]
    debug: dict[str, Any]


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def _ensure_initialized(
    model_path: str = DEFAULT_MODEL_PATH,
    chroma_dir: str = DEFAULT_CHROMA_DIR,
    bm25_path: str = DEFAULT_BM25_PATH,
    parent_lookup_path: str = DEFAULT_PARENT_LOOKUP_PATH,
    parsed_path: str = DEFAULT_PARSED_PATH,
    parsed_dir: str = DEFAULT_PARSED_DIR,
) -> None:
    """Load all stateful components on first call. Thread-unsafe (single-process use)."""
    global _embedder, _bm25_store, _chroma_collection, _parent_lookup, _child_lookup

    if _embedder is not None:
        return  # Already initialized

    logger.info("Initializing retrieval pipeline...")

    # Embedder
    from app.retrieval.embedder import Embedder
    _embedder = Embedder(model_path=model_path)

    # BM25
    from app.retrieval.bm25_store import BM25Store
    _bm25_store = BM25Store.load(path=bm25_path)

    # Chroma
    from app.retrieval.vector_store import get_collection
    _chroma_collection = get_collection(persist_dir=chroma_dir)

    # Parent lookup (from JSON persisted by indexer)
    parent_lp = Path(parent_lookup_path)
    if not parent_lp.exists():
        raise FileNotFoundError(
            f"Parent lookup not found at {parent_lp.resolve()}. "
            "Run: .venv\\Scripts\\python.exe -m app.retrieval.indexer"
        )
    with open(parent_lp, encoding="utf-8") as f:
        _parent_lookup = json.load(f)
    logger.info("Parent lookup loaded: %d parents", len(_parent_lookup))

    # Child lookup — multi-doc: merge children from all parsed/*.json
    # so parent_expander can resolve child_id -> parent_id for any doc.
    # Falls back to single-doc parsed_path if parsed_dir is missing.
    _child_lookup = {}
    parsed_d = Path(parsed_dir)
    if parsed_d.exists():
        for f in sorted(parsed_d.glob("*.json")):
            try:
                with open(f, encoding="utf-8") as fh:
                    data = json.load(fh)
                for c in data.get("chunks_child", []):
                    _child_lookup[c["chunk_id"]] = c
            except Exception as e:
                logger.warning("Failed to load %s for child_lookup: %s", f, e)
        logger.info(
            "Child lookup loaded: %d children from %d parsed files in %s",
            len(_child_lookup), len(list(parsed_d.glob("*.json"))), parsed_d,
        )
    else:
        parsed_p = Path(parsed_path)
        if not parsed_p.exists():
            raise FileNotFoundError(
                f"Neither parsed_dir ({parsed_d}) nor parsed_path ({parsed_p}) exists"
            )
        with open(parsed_p, encoding="utf-8") as f:
            parsed_data = json.load(f)
        _child_lookup = {c["chunk_id"]: c for c in parsed_data["chunks_child"]}
        logger.info("Child lookup (single-doc fallback): %d children", len(_child_lookup))

    logger.info("Retrieval pipeline ready.")


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    *,
    top_k_dense: int = 15,
    top_k_sparse: int = 15,
    top_k_fused: int = 8,
    model_path: str = DEFAULT_MODEL_PATH,
    chroma_dir: str = DEFAULT_CHROMA_DIR,
    bm25_path: str = DEFAULT_BM25_PATH,
    parent_lookup_path: str = DEFAULT_PARENT_LOOKUP_PATH,
    parsed_path: str = DEFAULT_PARSED_PATH,
    doc_id: str = DEFAULT_DOC_ID,
    doc_filter: str | None = None,
) -> RetrievalResult:
    """
    End-to-end retrieval pipeline.

    Steps (per build-spec Section 5.2 line 727-731):
      1. Dense:        encode query -> Chroma -> top_k_dense results
      2. Sparse:       BM25 -> top_k_sparse results
      3. RRF fusion:   -> top_k_fused unique chunk_ids
      4. Parent expand: child -> parent Pasal, dedup
      5. Cross-ref:    scan parent texts for "Pasal X" refs, fetch up to 3 new Pasals
      6. Always-on:    prepend Pasal 1 if not already present
      7. Return RetrievalResult with debug info

    Args:
        query:          raw user query string (no prefix — added internally).
        top_k_dense:    dense retrieval top-k (default 15).
        top_k_sparse:   sparse retrieval top-k (default 15).
        top_k_fused:    RRF fused top-k to expand (default 8).
        model_path:     ONNX model directory.
        chroma_dir:     Chroma persist directory.
        bm25_path:      BM25 pickle path.
        parent_lookup_path: parent lookup JSON path.
        parsed_path:    parsed JSON path (for child lookup).
        doc_id:         document id prefix.

    Returns:
        RetrievalResult with parent_chunks ordered: [Pasal1, top_retrieved, cross_refs].
    """
    t0 = time.time()

    _ensure_initialized(
        model_path=model_path,
        chroma_dir=chroma_dir,
        bm25_path=bm25_path,
        parent_lookup_path=parent_lookup_path,
        parsed_path=parsed_path,
    )

    # Step 1: Dense retrieval (optionally scoped to a single doc via Chroma where)
    from app.retrieval.vector_store import query_dense
    query_vec = _embedder.encode_query(query)
    dense_where = {"doc_id": doc_filter} if doc_filter else None
    dense_results = query_dense(
        _chroma_collection, query_vec, top_k=top_k_dense, where=dense_where
    )
    # dense_results: list[(chunk_id, cosine_distance)] sorted ascending distance

    # Step 2: Sparse retrieval (BM25 has no metadata index — post-filter by doc prefix)
    if doc_filter:
        raw_sparse = _bm25_store.query_sparse(query, top_k=top_k_sparse * 5)
        prefix = f"{doc_filter}::"
        sparse_results = [
            (cid, s) for cid, s in raw_sparse if cid.startswith(prefix)
        ][:top_k_sparse]
    else:
        sparse_results = _bm25_store.query_sparse(query, top_k=top_k_sparse)
    # sparse_results: list[(chunk_id, bm25_score)] sorted descending score

    # Step 3: RRF fusion
    from app.retrieval.hybrid import fuse
    fused = fuse(dense_results, sparse_results, top_n=top_k_fused)
    # fused: list[(chunk_id, rrf_score)] sorted descending

    # Step 4: Parent expansion
    from app.retrieval.parent_expander import expand_to_parents
    expanded = expand_to_parents(fused, _parent_lookup, _child_lookup)
    # expanded: list[(parent_dict, score)] sorted by score desc

    # Step 5: Cross-reference resolution
    # Multi-doc: resolve "Pasal X" relative to the doc_id of the chunk it appears in
    # (each parent has chunk["doc_id"]).
    from app.retrieval.cross_ref_resolver import resolve_cross_refs
    already_included = {p["chunk_id"] for p, _ in expanded}
    cross_refs = resolve_cross_refs(
        expanded, _parent_lookup, already_included, doc_id=None
    )

    # Step 6: Always-on Pasal 1 — per-doc.
    # For multi-doc: prepend Pasal 1 of the most-retrieved doc only (avoid
    # drowning out non-dominant docs with permensos-8-2023's Pasal 1).
    from app.retrieval.always_on import prepend_always_on
    retrieved_parents = [p for p, _ in expanded] + cross_refs
    all_included = {c["chunk_id"] for c in retrieved_parents}

    # Determine dominant doc by count of retrieved parents
    if retrieved_parents:
        from collections import Counter
        doc_counts = Counter(c.get("doc_id", "") for c in retrieved_parents)
        dominant_doc = doc_counts.most_common(1)[0][0] or doc_id
    else:
        dominant_doc = doc_id

    final_context = prepend_always_on(
        retrieved_parents, _parent_lookup, all_included, doc_id=dominant_doc
    )

    elapsed_ms = (time.time() - t0) * 1000

    debug = {
        "query": query,
        "dense_top5": dense_results[:5],
        "sparse_top5": sparse_results[:5],
        "fused_top8": fused,
        "expanded_parents": [(p["chunk_id"], s) for p, s in expanded],
        "cross_ref_additions": [c["chunk_id"] for c in cross_refs],
        "final_pasal_order": [c.get("pasal") for c in final_context],
        "elapsed_ms": round(elapsed_ms, 1),
    }

    logger.debug(
        "retrieve('%s') -> %d parents in %.0fms",
        query[:60], len(final_context), elapsed_ms,
    )

    return RetrievalResult(parent_chunks=final_context, debug=debug)


def reset_singletons() -> None:
    """
    Reset module-level singletons. Used in tests to force re-initialization
    with different paths (e.g., test fixtures).
    """
    global _embedder, _bm25_store, _chroma_collection, _parent_lookup, _child_lookup
    _embedder = None
    _bm25_store = None
    _chroma_collection = None
    _parent_lookup = None
    _child_lookup = None

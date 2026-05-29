"""
Multi-doc indexer: parsed JSONs -> Chroma + BM25 + parent lookup table.

What gets indexed:
  - CHILD chunks -> Chroma (dense vectors) + BM25 corpus.
    Rationale: retrieval operates on fine-grained children; parent expansion
    happens post-retrieval. This matches spec Section 5.2 line 718.
  - PARENT chunks -> JSON lookup table (data/bm25/parent_lookup.json).
    Parents are not indexed as embeddings; they are fetched by parent_id
    after child retrieval.

Multi-doc support:
  - Default: globs all `data/parsed/*.json` and merges them into a single index.
    Chroma upsert is idempotent on chunk_id; BM25 is rebuilt from the merged
    corpus; parent_lookup is merged.
  - `--parsed <file>` indexes a single parsed JSON (used by incremental ingest).
    Chroma still upserts; BM25 + parent_lookup are merged with existing.

Outputs:
  - data/chroma/                       Chroma persistent DB (child embeddings + meta)
  - data/bm25/permensos_bm25.pkl       BM25Okapi index over ALL children (merged corpus)
  - data/bm25/parent_lookup.json       merged parent_id -> ChunkMeta dict across docs

Usage:
    # Full multi-doc index from all parsed/*.json
    .venv\\Scripts\\python.exe -m app.retrieval.indexer

    # Index just one doc (incremental — merges with existing BM25 / parent_lookup)
    .venv\\Scripts\\python.exe -m app.retrieval.indexer --parsed data/parsed/permensos-8-2023.json

    # Force full rebuild (delete Chroma collection + indexes first)
    .venv\\Scripts\\python.exe -m app.retrieval.indexer --rebuild

    # Skip already-embedded chunk_ids (cheap re-runs)
    .venv\\Scripts\\python.exe -m app.retrieval.indexer --skip-existing
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()  # load .env so GEMINI_API_KEY (etc) is visible to embedder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("indexer")

DEFAULT_PARSED_DIR = "data/parsed"
DEFAULT_CHROMA_DIR = "data/chroma"
DEFAULT_BM25_PATH = "data/bm25/permensos_bm25.pkl"
DEFAULT_PARENT_LOOKUP_PATH = "data/bm25/parent_lookup.json"
DEFAULT_MODEL_PATH = "models/multilingual-e5-large-onnx-int8"


def _collect_parsed_files(parsed: str | None, default_dir: str) -> list[Path]:
    """Resolve which parsed JSON files to index.

    If `parsed` is a file, use just that. Otherwise glob `default_dir/*.json`.
    """
    if parsed:
        p = Path(parsed)
        if not p.exists():
            raise FileNotFoundError(f"Parsed file not found: {p.resolve()}")
        return [p]

    d = Path(default_dir)
    if not d.exists():
        raise FileNotFoundError(f"Parsed dir not found: {d.resolve()}")
    files = sorted(d.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No parsed JSONs in {d.resolve()}")
    return files


def _load_parsed(paths: list[Path]) -> tuple[list[dict], list[dict], dict[str, str]]:
    """Load + merge children and parents from multiple parsed JSONs, deduplicated.

    Dedup policy: last-write-wins on chunk_id (matches Chroma upsert semantics).
    Chroma's upsert validation rejects duplicate ids within a single call, so we
    must dedup before passing to upsert anyway.

    Returns:
      (unique_children, unique_parents, doc_id_index)
      where doc_id_index maps chunk_id -> doc_id for collision diagnostics.
    """
    children_by_id: dict[str, dict] = {}
    parents_by_id: dict[str, dict] = {}
    doc_id_index: dict[str, str] = {}
    n_dup_children = 0
    n_dup_parents = 0

    for p in paths:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        children = data.get("chunks_child", [])
        parents = data.get("chunks_parent", [])

        # Derive doc_id from first chunk (fallback: filename stem)
        doc_id = (children[0] if children else parents[0] if parents else {}).get(
            "doc_id", p.stem
        )
        logger.info(
            "Loaded %s — doc_id=%s, %d children, %d parents (raw)",
            p.name, doc_id, len(children), len(parents),
        )

        for c in children:
            cid = c["chunk_id"]
            if cid in children_by_id:
                n_dup_children += 1
            children_by_id[cid] = c
            doc_id_index[cid] = doc_id
        for pa in parents:
            pid = pa["chunk_id"]
            if pid in parents_by_id:
                n_dup_parents += 1
            parents_by_id[pid] = pa

    all_children = list(children_by_id.values())
    all_parents = list(parents_by_id.values())

    logger.info(
        "Merged + deduped corpus: %d unique children (dropped %d dups), "
        "%d unique parents (dropped %d dups), across %d files",
        len(all_children), n_dup_children, len(all_parents), n_dup_parents, len(paths),
    )
    return all_children, all_parents, doc_id_index


def _rebuild_chroma(chroma_dir: str) -> None:
    """Drop and recreate the Chroma collection (used by --rebuild)."""
    import chromadb

    from app.retrieval.vector_store import COLLECTION_NAME

    Path(chroma_dir).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=chroma_dir)
    try:
        client.delete_collection(COLLECTION_NAME)
        logger.info("Deleted existing Chroma collection '%s'", COLLECTION_NAME)
    except Exception as e:
        logger.info("No existing Chroma collection to delete (%s)", e)


def _filter_existing(
    children: list[dict], collection: Any
) -> tuple[list[dict], int]:
    """Return (new_children, n_skipped) — chunks not yet in Chroma."""
    if collection.count() == 0:
        return children, 0
    ids = [c["chunk_id"] for c in children]
    existing = set(collection.get(ids=ids, include=[])["ids"])
    new_children = [c for c in children if c["chunk_id"] not in existing]
    return new_children, len(children) - len(new_children)


def build_index(
    parsed: str | None = None,
    parsed_dir: str = DEFAULT_PARSED_DIR,
    chroma_dir: str = DEFAULT_CHROMA_DIR,
    bm25_path: str = DEFAULT_BM25_PATH,
    parent_lookup_path: str = DEFAULT_PARENT_LOOKUP_PATH,
    model_path: str = DEFAULT_MODEL_PATH,
    rebuild: bool = False,
    skip_existing: bool = False,
) -> dict:
    """Build (or update) Chroma + BM25 + parent_lookup from parsed JSON(s).

    Args:
        parsed:        explicit parsed file (single-doc mode). If None, globs
                       all parsed_dir/*.json.
        parsed_dir:    dir to glob for *.json if `parsed` is None.
        rebuild:       drop Chroma collection + delete BM25 first (full rebuild).
        skip_existing: skip chunk_ids already in Chroma (avoids re-embedding).

    Returns dict with n_children_total, n_embedded, n_parents, embedding_dim, elapsed_s.
    """
    t_start = time.time()

    # -------------------------------------------------------------------------
    # 1. Collect + load parsed JSONs
    # -------------------------------------------------------------------------
    files = _collect_parsed_files(parsed, parsed_dir)
    children, parents, _doc_index = _load_parsed(files)
    n_total_children = len(children)

    if not children:
        raise ValueError("No child chunks found across parsed files")

    # -------------------------------------------------------------------------
    # 2. Handle --rebuild (drop Chroma collection + delete artifacts)
    # -------------------------------------------------------------------------
    if rebuild:
        logger.warning("--rebuild: dropping existing Chroma collection + indexes")
        _rebuild_chroma(chroma_dir)
        Path(bm25_path).unlink(missing_ok=True)
        Path(parent_lookup_path).unlink(missing_ok=True)

    # -------------------------------------------------------------------------
    # 3. Decide what to embed (skip_existing optimization)
    # -------------------------------------------------------------------------
    from app.retrieval.vector_store import get_collection, upsert_chunks
    collection = get_collection(persist_dir=chroma_dir)

    n_skipped = 0
    if skip_existing and not rebuild:
        children_to_embed, n_skipped = _filter_existing(children, collection)
        logger.info(
            "skip-existing: %d already indexed, %d new to embed",
            n_skipped, len(children_to_embed),
        )
    else:
        children_to_embed = children

    # -------------------------------------------------------------------------
    # 4. Embed + upsert (only if anything to embed)
    # -------------------------------------------------------------------------
    embedding_dim = 0
    if children_to_embed:
        logger.info("Loading embedder (model_path=%s)", model_path)
        from app.retrieval.embedder import Embedder
        embedder = Embedder(model_path=model_path)

        logger.info(
            "Encoding %d children (backend=%s, batch_size=16)...",
            len(children_to_embed), embedder.backend,
        )
        t_embed = time.time()
        texts = [c["text_for_embed"] for c in children_to_embed]
        embeddings = embedder.encode_batch(texts, is_query=False, batch_size=16)
        embed_elapsed = time.time() - t_embed
        embedding_dim = embeddings.shape[1]
        logger.info(
            "Encoded %d chunks in %.1fs (%.1f/s) — dim=%d",
            len(children_to_embed), embed_elapsed,
            len(children_to_embed) / max(embed_elapsed, 0.001),
            embedding_dim,
        )

        logger.info("Upserting to Chroma at %s ...", chroma_dir)
        upsert_chunks(collection, children_to_embed, embeddings)
        logger.info("Chroma collection count: %d", collection.count())
    else:
        logger.info("Nothing to embed (skip_existing matched everything)")
        # Pull dim from existing collection for the return payload
        if collection.count() > 0:
            sample = collection.get(limit=1, include=["embeddings"])
            if sample["embeddings"] is not None and len(sample["embeddings"]) > 0:
                embedding_dim = len(sample["embeddings"][0])

    # -------------------------------------------------------------------------
    # 5. BM25 — always rebuilt from full child union currently in Chroma
    # -------------------------------------------------------------------------
    # In single-doc mode, we still need to include children from OTHER docs in
    # BM25 — otherwise we'd lose them. Pull the union from Chroma metadata.
    if parsed and not rebuild:
        logger.info("Single-doc mode: pulling full child union from Chroma for BM25")
        # Get all chunk_ids currently in Chroma. For each, we need text_for_embed.
        # Chroma stores documents = text_for_embed.
        all_ids = collection.get(include=[])["ids"]
        full = collection.get(ids=all_ids, include=["documents", "metadatas"])
        union_children: list[dict] = []
        for i, cid in enumerate(full["ids"]):
            meta = dict(full["metadatas"][i])
            meta["chunk_id"] = cid
            meta["text_for_embed"] = full["documents"][i]
            union_children.append(meta)
        bm25_corpus = union_children
    else:
        bm25_corpus = children

    logger.info("Rebuilding BM25 over %d children...", len(bm25_corpus))
    from app.retrieval.bm25_store import BM25Store
    bm25 = BM25Store.build(bm25_corpus)
    bm25.save(bm25_path)

    # -------------------------------------------------------------------------
    # 6. parent_lookup — merge with existing if single-doc mode
    # -------------------------------------------------------------------------
    parent_json = Path(parent_lookup_path)
    parent_json.parent.mkdir(parents=True, exist_ok=True)

    existing_lookup: dict[str, Any] = {}
    if parent_json.exists() and parsed and not rebuild:
        try:
            with open(parent_json, encoding="utf-8") as f:
                existing_lookup = json.load(f)
            logger.info(
                "Merging with existing parent_lookup (%d entries)",
                len(existing_lookup),
            )
        except json.JSONDecodeError:
            logger.warning("parent_lookup.json corrupt — overwriting")

    new_lookup = {p["chunk_id"]: p for p in parents}
    existing_lookup.update(new_lookup)

    with open(parent_json, "w", encoding="utf-8") as f:
        json.dump(existing_lookup, f, ensure_ascii=False, indent=2)
    logger.info("Wrote parent_lookup (%d total entries) -> %s",
                len(existing_lookup), parent_json)

    # -------------------------------------------------------------------------
    # 7. Done
    # -------------------------------------------------------------------------
    elapsed = time.time() - t_start
    n_parents_total = len(existing_lookup)
    logger.info(
        "Index build complete: %d children embedded (%d skipped), %d parents, total=%.1fs",
        len(children_to_embed), n_skipped, n_parents_total, elapsed,
    )

    return {
        "n_files": len(files),
        "n_children_total": n_total_children,
        "n_embedded": len(children_to_embed),
        "n_skipped": n_skipped,
        "n_parents": n_parents_total,
        "embedding_dim": embedding_dim,
        "elapsed_s": elapsed,
        "status": "rebuilt" if rebuild else "indexed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Chroma + BM25 + parent_lookup from parsed JSON(s) (multi-doc)"
    )
    parser.add_argument(
        "--parsed",
        default=None,
        help="Specific parsed JSON file (single-doc mode). Defaults to globbing parsed-dir.",
    )
    parser.add_argument(
        "--parsed-dir",
        default=DEFAULT_PARSED_DIR,
        help=f"Dir to glob for *.json (default: {DEFAULT_PARSED_DIR})",
    )
    parser.add_argument(
        "--chroma-dir",
        default=DEFAULT_CHROMA_DIR,
        help=f"Chroma persist dir (default: {DEFAULT_CHROMA_DIR})",
    )
    parser.add_argument(
        "--bm25",
        default=DEFAULT_BM25_PATH,
        help=f"BM25 pickle path (default: {DEFAULT_BM25_PATH})",
    )
    parser.add_argument(
        "--parent-lookup",
        default=DEFAULT_PARENT_LOOKUP_PATH,
        help=f"Parent lookup JSON path (default: {DEFAULT_PARENT_LOOKUP_PATH})",
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help=f"ONNX model dir (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Drop existing Chroma collection + indexes, then re-index from scratch.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip chunk_ids already in Chroma (avoids re-embedding). Ignored with --rebuild.",
    )
    args = parser.parse_args()

    stats = build_index(
        parsed=args.parsed,
        parsed_dir=args.parsed_dir,
        chroma_dir=args.chroma_dir,
        bm25_path=args.bm25,
        parent_lookup_path=args.parent_lookup,
        model_path=args.model_path,
        rebuild=args.rebuild,
        skip_existing=args.skip_existing,
    )

    print("\n=== Index Build Summary ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

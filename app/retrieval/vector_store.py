"""
Chroma persistent client wrapper for dense retrieval.

Collection schema:
  - ids:        chunk_id strings
  - embeddings: float32 vectors from Embedder
  - documents:  text_for_embed (passage text; optional — stored for debug)
  - metadatas:  ChunkMeta fields excluding 'text' and 'text_for_embed'.
                List fields (tags, cross_refs_outgoing) are JSON-serialized to
                strings because Chroma metadata values must be scalar
                (str | int | float | bool).

See build-spec Section 5.2 + Section 4.1.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

COLLECTION_NAME = "permensos_chunks"
DEFAULT_PERSIST_DIR = "data/chroma"

# Fields excluded from Chroma metadata (too long, stored separately as documents)
_EXCLUDE_FROM_META = {"text", "text_for_embed"}

# List-type fields that need JSON serialization for Chroma scalar constraint
_LIST_FIELDS = {"tags", "cross_refs_outgoing"}


def _serialize_metadata(chunk_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Convert a ChunkMeta dict to Chroma-safe metadata:
      - Drop 'text' and 'text_for_embed' (too large; stored as Chroma 'documents').
      - JSON-stringify list fields (tags, cross_refs_outgoing).
      - Convert None values to empty string (Chroma rejects None).
    """
    meta: dict[str, Any] = {}
    for k, v in chunk_dict.items():
        if k in _EXCLUDE_FROM_META:
            continue
        if k in _LIST_FIELDS:
            meta[k] = json.dumps(v, ensure_ascii=False)
        elif v is None:
            meta[k] = ""
        else:
            meta[k] = v
    return meta


def get_collection(
    name: str = COLLECTION_NAME,
    persist_dir: str = DEFAULT_PERSIST_DIR,
) -> Any:
    """
    Return (or create) a Chroma persistent collection.
    Uses cosine distance — consistent with L2-normalized e5 embeddings
    (cosine similarity == dot product when both vectors are unit-norm).
    """
    import chromadb

    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info("Chroma collection '%s' ready (persist_dir=%s)", name, persist_dir)
    return collection


def upsert_chunks(
    collection: Any,
    chunks: list[dict[str, Any]],
    embeddings: np.ndarray,
) -> None:
    """
    Upsert chunks into Chroma.

    Args:
        collection: Chroma collection object.
        chunks:     list of ChunkMeta dicts (as returned by json.load on parsed JSON).
        embeddings: float32 ndarray of shape (len(chunks), embedding_dim).
    """
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) length mismatch"
        )

    ids = [c["chunk_id"] for c in chunks]
    documents = [c["text_for_embed"] for c in chunks]
    metadatas = [_serialize_metadata(c) for c in chunks]
    embeddings_list = embeddings.tolist()

    # Chroma upsert in one call (handles both insert + update)
    collection.upsert(
        ids=ids,
        embeddings=embeddings_list,
        documents=documents,
        metadatas=metadatas,
    )
    logger.info("Upserted %d chunks to collection '%s'", len(chunks), collection.name)


def query_dense(
    collection: Any,
    query_embedding: np.ndarray,
    top_k: int = 15,
    where: dict | None = None,
) -> list[tuple[str, float]]:
    """
    Dense similarity search via Chroma.

    Args:
        collection:      Chroma collection.
        query_embedding: float32 1D array of shape (embedding_dim,).
        top_k:           number of results to return.
        where:           optional Chroma metadata filter dict.

    Returns:
        list of (chunk_id, distance) sorted ascending by distance
        (cosine distance: 0 = identical, 2 = opposite).
        Callers convert distance -> similarity as needed.
    """
    query_vec = query_embedding.tolist()
    n_total = collection.count()
    if n_total == 0:
        return []

    kwargs: dict[str, Any] = {
        "query_embeddings": [query_vec],
        "n_results": min(top_k, n_total),
        "include": ["distances"],
    }
    if where:
        kwargs["where"] = where

    results = collection.query(**kwargs)

    ids = results["ids"][0]
    distances = results["distances"][0]

    return list(zip(ids, distances))


def fetch_by_ids(
    collection: Any,
    ids: list[str],
) -> list[dict[str, Any]]:
    """
    Fetch chunk metadata + document by id list.
    Returns list of metadata dicts (list fields still JSON strings from Chroma).
    """
    if not ids:
        return []
    result = collection.get(ids=ids, include=["metadatas", "documents"])
    output = []
    for i, chunk_id in enumerate(result["ids"]):
        meta = dict(result["metadatas"][i])
        meta["chunk_id"] = chunk_id
        meta["text_for_embed"] = result["documents"][i]
        output.append(meta)
    return output


def query_by_metadata(
    collection: Any,
    where: dict[str, Any],
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """
    Metadata-only query (no embedding vector needed).
    Used by cross_ref_resolver and always_on to fetch specific Pasals.

    Returns list of metadata dicts with 'chunk_id' injected.
    """
    n_total = collection.count()
    if n_total == 0:
        return []

    result = collection.get(
        where=where,
        include=["metadatas", "documents"],
        limit=top_k,
    )

    output = []
    for i, chunk_id in enumerate(result["ids"]):
        meta = dict(result["metadatas"][i])
        meta["chunk_id"] = chunk_id
        meta["text_for_embed"] = result["documents"][i]
        output.append(meta)
    return output

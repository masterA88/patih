"""Whitelist validator: checks each extracted citation against doc registry + Chroma metadata.

Spec: build-spec Section 5.4 (line 957-959).

Chroma metadata type conventions (verified from actual data 2026-05-18):
  - pasal:  int
  - ayat:   int | '' (empty string when no ayat)
  - huruf:  str | '' (empty string when no huruf)
  - chunk_type: 'child' | 'parent'

All Chroma `where` clauses query child chunks only (chunk_type='child') since
parent chunks aggregate multiple ayat and don't carry individual ayat/huruf metadata.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Chroma `where` filter using $and requires list of conditions.
# Chroma 0.4+ supports: {"$and": [{"field": {"$eq": val}}, ...]}


def _chroma_where_pasal_only(doc_id: str, pasal: int) -> dict:
    """Where clause: doc_id=doc_id AND pasal=pasal AND chunk_type='child'."""
    return {
        "$and": [
            {"doc_id": {"$eq": doc_id}},
            {"pasal": {"$eq": pasal}},
            {"chunk_type": {"$eq": "child"}},
        ]
    }


def _chroma_where_pasal_ayat(doc_id: str, pasal: int, ayat: int) -> dict:
    """Where clause: doc_id AND pasal AND ayat AND chunk_type='child'."""
    return {
        "$and": [
            {"doc_id": {"$eq": doc_id}},
            {"pasal": {"$eq": pasal}},
            {"ayat": {"$eq": ayat}},
            {"chunk_type": {"$eq": "child"}},
        ]
    }


def _chroma_where_pasal_ayat_huruf(doc_id: str, pasal: int, ayat: int, huruf: str) -> dict:
    """Where clause: doc_id AND pasal AND ayat AND huruf AND chunk_type='child'."""
    return {
        "$and": [
            {"doc_id": {"$eq": doc_id}},
            {"pasal": {"$eq": pasal}},
            {"ayat": {"$eq": ayat}},
            {"huruf": {"$eq": huruf}},
            {"chunk_type": {"$eq": "child"}},
        ]
    }


def _chroma_get(chroma_collection: Any, where: dict, include_documents: bool = False):
    """Run a Chroma get() with the given where clause.

    Returns:
        If include_documents=False: bool (True if any match)
        If include_documents=True: (found: bool, documents: list[str])
    """
    include = ["metadatas", "documents"] if include_documents else ["metadatas"]
    result = chroma_collection.get(
        where=where,
        include=include,
        limit=1,
    )
    found = bool(result["ids"])
    if include_documents:
        docs = result.get("documents") or []
        return found, docs
    return found


def _check_citation_in_doc(
    cit: dict[str, Any],
    doc_id: str,
    chroma_collection: Any,
    n_pasal: int,
) -> bool:
    """Check whether a single citation exists in ONE document. Returns bool."""
    pasal: int = cit["pasal"]
    ayat: int | None = cit["ayat"]
    huruf: str | None = cit["huruf"]

    # --- Range check ---
    if pasal < 1 or pasal > n_pasal:
        return False

    # --- Chroma existence check ---
    try:
        if ayat is not None and huruf is not None:
            where = _chroma_where_pasal_ayat_huruf(doc_id, pasal, ayat, huruf)
            found = _chroma_get(chroma_collection, where)
            if not found:
                where_fallback = _chroma_where_pasal_ayat(doc_id, pasal, ayat)
                ayat_found, docs = _chroma_get(
                    chroma_collection, where_fallback, include_documents=True
                )
                if ayat_found and docs:
                    chunk_text = (docs[0] or "").lower()
                    huruf_pattern = (
                        rf"(?:^|\s|;)\s*{re.escape(huruf)}\."
                        rf"|(?:^|\s|;)\s*{re.escape(huruf)};"
                        rf"|huruf\s+{re.escape(huruf)}\b"
                    )
                    found = bool(re.search(huruf_pattern, chunk_text))
                else:
                    found = False
            return found
        elif ayat is not None:
            return _chroma_get(chroma_collection, _chroma_where_pasal_ayat(doc_id, pasal, ayat))
        else:
            return _chroma_get(chroma_collection, _chroma_where_pasal_only(doc_id, pasal))
    except Exception as exc:
        logger.warning(
            "Chroma query error for %s Pasal %d ayat %s huruf %s: %s",
            doc_id, pasal, ayat, huruf, exc,
        )
        return False


def validate_citations(
    citations: list[dict[str, Any]],
    doc_id: str | list[str],
    chroma_collection: Any,          # chromadb.Collection
    doc_registry: dict[str, Any],    # loaded from data/registry/documents.json
) -> list[bool]:
    """Validate each citation against doc registry and Chroma metadata.

    Multi-doc: `doc_id` may be a single doc_id (str) or a list of candidate
    doc_ids. A citation is considered valid if its Pasal/ayat/huruf is found in
    ANY of the candidate docs (the docs that appeared in the retrieved context).

    Returns a parallel list of bool — True if citation is verifiably present in
    at least one candidate document, False otherwise.

    Validation logic per citation:
      1. Check pasal range: citation.pasal must be in [1, n_pasal].
         If out-of-range -> False immediately (no Chroma query needed).
      2. If citation has huruf (and ayat): query Chroma for pasal+ayat+huruf.
         If no huruf-level chunk found, fall back to pasal+ayat query.
         This handles documents where chunking is at ayat granularity, not huruf.
         Rationale: a citation to 'Pasal 5 ayat (2) huruf a' is valid if
         'Pasal 5 ayat (2)' is a real chunk in the document that contains
         the huruf enumeration in its text.
      3. If citation has only ayat (no huruf): query Chroma for pasal+ayat.
      4. If citation has only pasal: query Chroma for any child with that pasal.

    Note: huruf-level fallback is a deliberate design choice because the Step 2
    chunker stores Pasal 5 ayat (2) as a single chunk containing 'huruf a ... m'
    rather than one chunk per huruf. This is correct per the PROGRESS.md state.
    """
    if not citations:
        return []

    # Normalize doc_id(s) to a candidate list
    candidate_doc_ids = [doc_id] if isinstance(doc_id, str) else list(doc_id)
    if not candidate_doc_ids:
        return [False] * len(citations)

    # Pre-fetch n_pasal per candidate doc
    n_pasal_by_doc = {
        d: doc_registry.get(d, {}).get("n_pasal", 0) for d in candidate_doc_ids
    }

    results: list[bool] = []
    for cit in citations:
        # Valid if the citation is found in ANY candidate doc
        found_any = False
        for d in candidate_doc_ids:
            if _check_citation_in_doc(cit, d, chroma_collection, n_pasal_by_doc[d]):
                found_any = True
                break
        if not found_any:
            logger.debug(
                "Citation Pasal %s ayat %s huruf %s: NOT FOUND in any of %s",
                cit.get("pasal"), cit.get("ayat"), cit.get("huruf"), candidate_doc_ids,
            )
        results.append(found_any)

    return results

"""
Regex-based cross-reference resolver: "Pasal X ayat (Y)" -> fetch parent Pasal.

Pattern (per build-spec Section 5.2 line 721):
    r"Pasal\\s+(\\d+)(?:\\s+ayat\\s+\\((\\d+)\\))?"

Applied to parent chunk text. For each match, fetch the referenced parent Pasal
from the parent_lookup table. Append to context, capped at 3 additional Pasals
to avoid context bloat (spec line 723).

Deduplication: skip if the referenced parent_id is already in the existing context set.

See build-spec Section 5.2 line 720-723.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Pattern: "Pasal 5" or "Pasal 5 ayat (2)"
_CROSS_REF_RE = re.compile(r"Pasal\s+(\d+)(?:\s+ayat\s+\((\d+)\))?")

MAX_CROSS_REFS = 3  # cap to avoid context bloat


def extract_pasal_refs(text: str) -> list[int]:
    """
    Extract all unique Pasal numbers referenced in text.

    Returns:
        Sorted list of unique pasal integers found via cross-ref pattern.

    Examples:
        "sebagaimana dimaksud dalam Pasal 5 ayat (2)" -> [5]
        "Pasal 1 dan Pasal 3 berlaku di sini" -> [1, 3]
        "lihat Pasal 10" -> [10]
        "Pasal-Pasal berikut" -> [] (no number follows hyphen — not a match)
    """
    matches = _CROSS_REF_RE.findall(text)
    pasals = sorted({int(m[0]) for m in matches if m[0]})
    return pasals


def resolve_cross_refs(
    parent_chunks: list[tuple[dict, float]],
    parent_lookup: dict[str, dict],
    already_included: set[str],
    doc_id: str | None = "permensos-8-2023",
    max_refs: int = MAX_CROSS_REFS,
) -> list[dict]:
    """
    Resolve cross-references found in retrieved parent chunks.

    For each retrieved parent chunk's text:
      - Regex scan for "Pasal X [ayat (Y)]" patterns.
      - For each referenced Pasal number, look up the parent chunk in parent_lookup.
        Multi-doc: the lookup uses the SCANNED chunk's own doc_id (not a global
        doc_id), so "Pasal 5" inside a UU chunk resolves to that UU's Pasal 5.
      - If not already in context (already_included), append to output.
      - Stop after max_refs new additions.

    Args:
        parent_chunks:    list of (parent_chunk_dict, score) from expand_to_parents().
        parent_lookup:    dict of parent_id -> parent ChunkMeta dict.
        already_included: set of parent_ids already in context (to dedup).
        doc_id:           legacy override — when None, uses each chunk's own doc_id.
                          When a string, applies that doc_id for ALL refs (single-doc mode).
        max_refs:         max number of additional Pasals to append.

    Returns:
        list of parent ChunkMeta dicts (new cross-ref expansions only, not including
        already_included chunks). At most max_refs items.
    """
    additional: list[dict] = []
    seen = set(already_included)  # local copy to track within this call

    for chunk_dict, _ in parent_chunks:
        if len(additional) >= max_refs:
            break

        text = chunk_dict.get("text", "") or chunk_dict.get("text_for_embed", "")
        # Resolve to the chunk's own doc_id unless caller pinned one
        scoped_doc_id = doc_id if doc_id else (chunk_dict.get("doc_id") or "")
        if not scoped_doc_id:
            continue
        referenced_pasals = extract_pasal_refs(text)

        for pasal_num in referenced_pasals:
            if len(additional) >= max_refs:
                break

            ref_parent_id = f"{scoped_doc_id}::pasal{pasal_num}"
            if ref_parent_id in seen:
                continue

            if ref_parent_id not in parent_lookup:
                logger.debug("Cross-ref Pasal %d not found in parent_lookup", pasal_num)
                continue

            additional.append(parent_lookup[ref_parent_id])
            seen.add(ref_parent_id)
            logger.debug("Cross-ref resolved: Pasal %d -> %s", pasal_num, ref_parent_id)

    logger.debug(
        "Cross-ref resolver: scanned %d parents, added %d new Pasals",
        len(parent_chunks),
        len(additional),
    )
    return additional

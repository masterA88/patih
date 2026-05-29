"""
Expand child chunk IDs to their parent Pasal chunks, with deduplication.

Logic (per build-spec Section 5.2 line 718):
  For each child chunk_id in the fused top-N:
    - Look up parent_id from the parent lookup table (loaded from indexer).
    - If the chunk is already a parent (chunk_type == "parent"), use it directly.
    - Dedup: if two children map to the same parent_id, emit that parent only once,
      keeping the score of the highest-ranked child.

Output: ordered list of (parent_chunk_dict, best_child_rrf_score),
        ordered by best_child_rrf_score descending.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def expand_to_parents(
    fused_results: list[tuple[str, float]],
    parent_lookup: dict[str, dict],
    child_lookup: dict[str, dict],
) -> list[tuple[dict, float]]:
    """
    Map child chunk IDs to their parent Pasal chunks.

    Args:
        fused_results:  list of (chunk_id, rrf_score) from hybrid.fuse().
        parent_lookup:  dict mapping parent_id -> parent ChunkMeta dict.
                        Populated by indexer from chunks_parent.
        child_lookup:   dict mapping chunk_id -> ChunkMeta dict for children.
                        Needed to find parent_id for each child chunk_id.

    Returns:
        list of (parent_chunk_dict, best_score), ordered by score descending.
        Deduplicated by parent_id.
    """
    seen_parents: dict[str, float] = {}  # parent_id -> best rrf_score
    ordered_parents: list[str] = []      # insertion order (for stable output)

    for chunk_id, score in fused_results:
        # Resolve parent_id
        if chunk_id in child_lookup:
            parent_id = child_lookup[chunk_id]["parent_id"]
        elif chunk_id in parent_lookup:
            # chunk is already a parent
            parent_id = chunk_id
        else:
            logger.warning("chunk_id '%s' not found in lookup tables — skipping", chunk_id)
            continue

        if parent_id not in parent_lookup:
            logger.warning("parent_id '%s' not found in parent_lookup — skipping", parent_id)
            continue

        if parent_id not in seen_parents:
            seen_parents[parent_id] = score
            ordered_parents.append(parent_id)
        else:
            # Keep highest score
            if score > seen_parents[parent_id]:
                seen_parents[parent_id] = score

    # Sort by score descending, preserving dedup order as tiebreaker
    ordered_parents.sort(key=lambda pid: -seen_parents[pid])

    result = [(parent_lookup[pid], seen_parents[pid]) for pid in ordered_parents]
    logger.debug("Parent expansion: %d children -> %d unique parents", len(fused_results), len(result))
    return result

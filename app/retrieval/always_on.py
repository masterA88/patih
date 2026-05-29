"""
Always-on context injection: prepend Pasal 1 (definitions) to every retrieval result.

Pasal 1 defines all key legal terms (Korban TPPO, PMI Bermasalah, Rehabilitasi Sosial,
etc.) and is tagged always_on in the Step 2 chunker. It should appear in every
LLM context window regardless of the query.

Implementation:
  1. Look up parent Pasal 1 from parent_lookup (fast dict lookup, no I/O).
  2. If it's already in the current context set, no-op.
  3. Otherwise, prepend to context list.

See build-spec Section 5.2 line 725.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

ALWAYS_ON_PASAL_NUM = 1


def prepend_always_on(
    context: list[dict],
    parent_lookup: dict[str, dict],
    already_included: set[str],
    doc_id: str = "permensos-8-2023",
) -> list[dict]:
    """
    Prepend Pasal 1 to context if not already present.

    Args:
        context:          current list of parent ChunkMeta dicts (modified in-place copy).
        parent_lookup:    dict of parent_id -> parent ChunkMeta dict.
        already_included: set of parent_ids already in context.
        doc_id:           document id prefix.

    Returns:
        New list with Pasal 1 at index 0 (if it wasn't already first).
        If Pasal 1 is not found in parent_lookup, returns context unchanged and logs warning.
    """
    always_on_id = f"{doc_id}::pasal{ALWAYS_ON_PASAL_NUM}"

    if always_on_id in already_included:
        # Already in context — check if it's at index 0
        if context and context[0].get("chunk_id") == always_on_id:
            return context
        # It's present but not first — move it to front
        reordered = [c for c in context if c.get("chunk_id") != always_on_id]
        pasal1 = next(c for c in context if c.get("chunk_id") == always_on_id)
        return [pasal1] + reordered

    # Not in context — fetch and prepend
    if always_on_id not in parent_lookup:
        logger.warning(
            "Always-on Pasal %d (%s) not found in parent_lookup — skipping",
            ALWAYS_ON_PASAL_NUM,
            always_on_id,
        )
        return context

    pasal1 = parent_lookup[always_on_id]
    logger.debug("Prepending always-on Pasal 1 to context")
    return [pasal1] + context

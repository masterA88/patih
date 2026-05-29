"""Parse-time validation: Pasal count, ayat completeness checks after ingestion."""

import hashlib
import logging
from pathlib import Path

from app.ingest.chunker import ChunkMeta
from app.ingest.doc_registry import DocRegistry

logger = logging.getLogger(__name__)

# Sanity bound for multi-doc corpus: flag only suspiciously-empty parses.
# Short regulations (administrative Permensos, old UU) legitimately have <15 Pasal,
# so a high threshold produced false warnings. 3 catches near-empty parses while
# allowing genuinely short docs (e.g. Permensos 22/2016 = 4 Pasal).
_MIN_PASAL_EXPECTED = 3


def validate_parse(
    ast: dict,
    registry: DocRegistry,
    chunks_child: list[ChunkMeta],
    chunks_parent: list[ChunkMeta],
) -> list[str]:
    """
    Run acceptance checks after ingest. Returns list of quality flag strings.

    Checks:
    1. No child chunk with pasal=None.
    2. 100% child parent_id resolves to a parent chunk.
    3. pdf_sha256 in registry matches actual file (re-computes hash).
    4. Pasal count sanity (>= _MIN_PASAL_EXPECTED).
    5. No duplicate chunk_ids.
    """
    flags: list[str] = []

    # -------------------------------------------------------------------------
    # 1. No child with pasal=None
    # -------------------------------------------------------------------------
    null_pasal_children = [c for c in chunks_child if c.pasal is None]
    if null_pasal_children:
        ids = [c.chunk_id for c in null_pasal_children]
        logger.error("Children with pasal=None: %s", ids)
        flags.append(f"orphan_chunk_pasal_none:{len(null_pasal_children)}")

    # -------------------------------------------------------------------------
    # 2. All child parent_ids resolve to a parent
    # -------------------------------------------------------------------------
    parent_ids = {p.chunk_id for p in chunks_parent}
    orphan_children = [c for c in chunks_child if c.parent_id not in parent_ids]
    if orphan_children:
        missing = {c.parent_id for c in orphan_children}
        logger.error("Children with unresolvable parent_id: %s", missing)
        flags.append(f"orphan_chunk:{len(orphan_children)}")

    # -------------------------------------------------------------------------
    # 3. pdf_sha256 verification
    # -------------------------------------------------------------------------
    pdf_path = Path(registry.pdf_path)
    if pdf_path.exists():
        from app.ingest.doc_registry import compute_sha256
        actual_sha = compute_sha256(pdf_path)
        if actual_sha != registry.pdf_sha256:
            logger.error(
                "SHA256 mismatch: registry=%s actual=%s",
                registry.pdf_sha256,
                actual_sha,
            )
            flags.append("pdf_sha256_mismatch")
    else:
        logger.warning("PDF path not found for SHA verification: %s", pdf_path)
        flags.append("pdf_path_missing")

    # -------------------------------------------------------------------------
    # 4. Pasal count sanity
    # -------------------------------------------------------------------------
    n_parents = len(chunks_parent)
    if n_parents < _MIN_PASAL_EXPECTED:
        logger.warning(
            "Low Pasal count: %d < expected minimum %d",
            n_parents,
            _MIN_PASAL_EXPECTED,
        )
        flags.append(f"pasal_count_low:{n_parents}")

    # -------------------------------------------------------------------------
    # 5. Duplicate chunk_ids
    # -------------------------------------------------------------------------
    all_child_ids = [c.chunk_id for c in chunks_child]
    if len(all_child_ids) != len(set(all_child_ids)):
        dupes = [cid for cid in all_child_ids if all_child_ids.count(cid) > 1]
        logger.error("Duplicate child chunk_ids: %s", sorted(set(dupes)))
        flags.append(f"duplicate_chunk_ids:{len(set(dupes))}")

    all_parent_ids = [c.chunk_id for c in chunks_parent]
    if len(all_parent_ids) != len(set(all_parent_ids)):
        dupes = [cid for cid in all_parent_ids if all_parent_ids.count(cid) > 1]
        logger.error("Duplicate parent chunk_ids: %s", sorted(set(dupes)))
        flags.append(f"duplicate_parent_ids:{len(set(dupes))}")

    # -------------------------------------------------------------------------
    # 6. AST has at least one BAB
    # -------------------------------------------------------------------------
    if not ast.get("bab"):
        flags.append("no_bab_parsed")

    if flags:
        logger.warning("Validation flags: %s", flags)
    else:
        logger.info("Validation passed — no quality flags.")

    return flags

"""Custom Chainlit UI components: citation card renderer and confidence badge.

build-spec Section 5.6 — citation cards + confidence badge.
Multi-doc update: cards now carry doc title (e.g. "Pasal 5 — Permensos 8/2023").
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import chainlit as cl

logger = logging.getLogger(__name__)

_PARENT_LOOKUP_PATH = Path("data/bm25/parent_lookup.json")
_REGISTRY_PATH = Path("data/registry/documents.json")
_MAX_CITATION_CARDS = 8   # cap to avoid overwhelming the sidebar

# Loaded lazily and cached at module level.
_parent_lookup: dict | None = None
_doc_registry: dict | None = None


def _load_parent_lookup() -> dict:
    global _parent_lookup
    if _parent_lookup is not None:
        return _parent_lookup
    if not _PARENT_LOOKUP_PATH.exists():
        logger.warning("parent_lookup.json not found at %s — citation cards disabled", _PARENT_LOOKUP_PATH)
        _parent_lookup = {}
        return _parent_lookup
    with open(_PARENT_LOOKUP_PATH, encoding="utf-8") as f:
        _parent_lookup = json.load(f)
    return _parent_lookup


def _load_registry() -> dict:
    global _doc_registry
    if _doc_registry is not None:
        return _doc_registry
    if not _REGISTRY_PATH.exists():
        logger.warning("registry not found at %s — doc titles unavailable", _REGISTRY_PATH)
        _doc_registry = {}
        return _doc_registry
    with open(_REGISTRY_PATH, encoding="utf-8") as f:
        _doc_registry = json.load(f)
    return _doc_registry


def _doc_label(doc_id: str) -> str:
    """Short, human-friendly doc label (e.g. 'Permensos 8/2023')."""
    reg = _load_registry()
    entry = reg.get(doc_id, {})
    jenis = entry.get("jenis_regulasi", "").replace("_", " ")
    nomor = entry.get("nomor")
    tahun = entry.get("tahun")
    if jenis and nomor and tahun:
        return f"{jenis.title()} {nomor}/{tahun}"
    return entry.get("title", doc_id)


# ---------------------------------------------------------------------------
# Citation cards
# ---------------------------------------------------------------------------

def build_citation_cards(
    retrieved_pasals: list[int | str],
    validation: dict | None,
    parent_chunks: list[dict] | None = None,
) -> list[cl.Text]:
    """Build one cl.Text element per retrieved Pasal (up to _MAX_CITATION_CARDS).

    Each element is displayed in the sidebar ("side" display mode) so it doesn't
    clutter the main chat thread. The name becomes the clickable sidebar label.

    Args:
        retrieved_pasals: Ordered list of Pasal numbers from GenerationResult
                          (legacy / fallback path when parent_chunks is not given).
        validation:       Optional validation dict; used to mark invalid citations.
        parent_chunks:    Preferred input — full parent chunk dicts with doc_id +
                          chunk_id, so we can show "Pasal X — <doc label>" and
                          look up text without ambiguity across docs.

    Returns:
        List of cl.Text elements ready to pass to cl.Message(elements=...).
    """
    parents = _load_parent_lookup()
    if not parents:
        return []

    # Build a set of invalid Pasal numbers for badge colouring (best-effort).
    invalid_pasals: set[int] = set()
    if validation:
        extracted = validation.get("citations_extracted") or []
        valid_flags = validation.get("citations_valid") or []
        for citation, is_valid in zip(extracted, valid_flags):
            if not is_valid:
                invalid_pasals.add(citation.get("pasal", -1))

    # Source of truth: parent_chunks if available, otherwise fall back to
    # retrieved_pasals + parent_lookup (legacy single-doc path).
    candidates: list[dict] = []
    if parent_chunks:
        seen_chunk_ids: set[str] = set()
        for chunk in parent_chunks:
            cid = chunk.get("chunk_id") or ""
            if cid in seen_chunk_ids:
                continue
            seen_chunk_ids.add(cid)
            candidates.append(chunk)
    else:
        # Legacy: only Pasal numbers known — only single doc supported.
        seen: set = set()
        for pasal_num in retrieved_pasals:
            if pasal_num in seen:
                continue
            seen.add(pasal_num)
            # Try every doc_id in registry for matching pasal number
            for doc_id in _load_registry().keys():
                chunk_id = f"{doc_id}::pasal{pasal_num}"
                chunk = parents.get(chunk_id)
                if chunk is not None:
                    candidates.append(chunk)
                    break

    elements: list[cl.Text] = []
    for chunk in candidates[:_MAX_CITATION_CARDS]:
        pasal_num = chunk.get("pasal")
        doc_id = chunk.get("doc_id", "")
        label = _doc_label(doc_id)
        page = chunk.get("source_page", "?")
        label_for_footer = label or doc_id or "dokumen"
        body = chunk.get("text", "")

        if pasal_num is None:
            # Reference doc (SOP / statistik / RPJMN) — cite by section + page
            section = chunk.get("section_title") or "Bagian"
            card_name = f"{section} — {label}" if label else str(section)
            card_name = card_name[:60]  # keep sidebar label short
            header = f"**{section}** ({label_for_footer})"
            validity_note = ""
        else:
            bab = chunk.get("bab", "?")
            bagian = chunk.get("bagian")
            bagian_suffix = f" — Bagian {bagian}" if bagian else ""
            card_name = f"Pasal {pasal_num} — {label}" if label else f"Pasal {pasal_num}"
            header = f"**Pasal {pasal_num}** (BAB {bab}{bagian_suffix})"
            validity_note = (
                "\n\n> ⚠️ _Citation ini ditandai tidak valid oleh validator._"
                if pasal_num in invalid_pasals else ""
            )

        footer = f"\n\n_Sumber: halaman {page}, {label_for_footer}_"
        content = f"{header}\n\n{body}{validity_note}{footer}"
        elements.append(
            cl.Text(name=card_name, content=content, display="side")
        )

    return elements


# ---------------------------------------------------------------------------
# Confidence badge
# ---------------------------------------------------------------------------

def build_confidence_badge(validation: dict | None) -> str:
    """Return a Markdown badge line summarising validation quality.

    Three levels (per build-spec Section 5.6):
      - Green  ("Tinggi"):  no hitl_flag, all citations valid, EG >= 0.95
      - Yellow ("Sedang"):  hitl_flag but citation_accuracy >= 0.80
      - Red    ("Rendah"):  citation_accuracy < 0.80 OR eg_score < 0.90

    Returns empty string when validation is None (validate=False path).
    """
    if validation is None:
        return ""

    # Guard against partial validation dicts (e.g. {"error": "..."} from a
    # validator crash — see generator.py error handler).
    if "error" in validation:
        return "⚪ **Validasi tidak tersedia** (error saat validasi; verifikasi manual disarankan)"

    hitl_flag: bool = bool(validation.get("hitl_flag", False))
    citation_acc: float = float(validation.get("citation_accuracy", 1.0))
    eg_score: float = float(validation.get("eg_score", 1.0))
    reasons: list[str] = validation.get("hitl_reasons") or []

    if not hitl_flag:
        return "🟢 **Tingkat Kepercayaan: Tinggi** (citation valid, grounding kuat)"

    if citation_acc >= 0.80 and eg_score >= 0.90:
        return (
            "🟡 **Tingkat Kepercayaan: Sedang** "
            "(grounding lemah pada beberapa klaim; verifikasi disarankan)"
        )

    reasons_str = ", ".join(reasons) if reasons else "lihat detail"
    return (
        "🔴 **Tingkat Kepercayaan: Rendah — Perlu Verifikasi Manual** "
        f"(alasan: {reasons_str})\n\n"
        "> ⚠️ Jawaban ini memiliki indikasi citation atau grounding lemah. "
        "Mohon verifikasi langsung dengan teks Pasal asli sebelum digunakan."
    )

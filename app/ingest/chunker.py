"""Parent-document chunker: parent=Pasal, child=ayat|huruf, with metadata enrichment."""

import logging
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Parser version tag: bump whenever parsing logic changes
PARSE_VERSION = "v1"
INGEST_DATE = date.today().isoformat()
VERSION_TAG = f"{PARSE_VERSION}-{INGEST_DATE}"

# SAC-lite document label prepended to text_for_embed
DOC_LABEL = "Permensos 8/2023"

ALWAYS_ON_PASAL = {1}  # Pasal 1 child + parent always tagged always_on


class ChunkMeta(BaseModel):
    # Identity
    chunk_id: str                          # f"{doc_id}::pasal{N}::ayat{M}::huruf{x}" or ".."
    parent_id: str                         # f"{doc_id}::pasal{N}"
    doc_id: str                            # f"permensos-8-2023"
    chunk_type: Literal["child", "parent"] # parent stored separately for retrieval expand

    # Structure
    bab: str | None = None                 # roman: "I", "II"
    bagian: str | None = None              # "Kesatu", "Kedua"
    pasal: int | None = None               # 1..N for regulasi; None for non-Pasal (SOP/statistik/RPJMN)
    ayat: int | None = None                # (1), (2), ...
    huruf: str | None = None               # "a", "b", ...
    section_title: str | None = None       # for non-Pasal docs: nearest heading / section label

    # Text
    text: str                              # raw text for LLM context
    text_for_embed: str                    # with e5 prefix: "passage: " + summary_prefix + text
    summary_prefix: str | None = None      # SAC: "[Permensos 8/2023: TPPO & PMI Bermasalah] "
    lang: Literal["id"] = "id"

    # Provenance
    source_page: int                       # PDF page (approximated from chunk position)
    source_pdf_path: str                   # data/raw/permensos8.pdf
    version: str                           # "v1-2026-05-18"

    # Phase 2 forward-compat
    cross_refs_outgoing: list[str] = Field(default_factory=list)

    # Always-on marker
    tags: list[str] = Field(default_factory=list)


def _make_parent_header(bab: str | None, bagian: str | None, pasal_num: int) -> str:
    """Build a human-readable prefix for text_for_embed."""
    parts: list[str] = []
    if bab:
        parts.append(f"BAB {bab}")
    if bagian:
        parts.append(f"Bagian {bagian}")
    parts.append(f"Pasal {pasal_num}")
    return " — ".join(parts)


def _make_embed_text(
    summary_prefix: str | None,
    parent_header: str,
    chunk_text: str,
) -> str:
    """Build text_for_embed with e5 'passage:' prefix + SAC-lite doc label."""
    label = summary_prefix or DOC_LABEL
    return f"passage: [{label}] {parent_header} — {chunk_text}"


def _estimate_source_page(pasal_num: int, n_pages: int = 13) -> int:
    """
    Rough heuristic for Permensos 8/2023 (13 pages, 34 Pasal):
    Page 1-2 = Preamble, Pages 2-12 = Pasal 1-34.
    Returns 1-indexed page number estimate.
    """
    # Linear interpolation: Pasal 1 ≈ page 2, Pasal 34 ≈ page 12
    if pasal_num <= 1:
        return 2
    estimated = 2 + round((pasal_num - 1) / 33 * 10)
    return min(max(estimated, 1), n_pages)


def chunk_document(
    ast: dict,
    doc_id: str,
    source_pdf: str,
    summary_prefix: str | None = None,
) -> tuple[list[ChunkMeta], list[ChunkMeta]]:
    """
    Convert AST to (child_chunks, parent_chunks).

    Parent: 1 ChunkMeta per Pasal, chunk_type="parent", contains full Pasal text.
    Child: 1 ChunkMeta per ayat (or per huruf if ayat has huruf sub-items).
           If Pasal has no ayat, a single child chunk = the full Pasal body.

    Returns: (child_chunks, parent_chunks)
    """
    child_chunks: list[ChunkMeta] = []
    parent_chunks: list[ChunkMeta] = []

    bab_list = ast.get("bab", [])

    for bab in bab_list:
        bab_nomor: str | None = bab.get("nomor")
        pasal_list = bab.get("pasal", [])

        # Build Pasal → Bagian lookup from bagian.pasal_range
        bagian_lookup: dict[int, str] = {}
        for bagian in bab.get("bagian", []):
            pr = bagian.get("pasal_range", [None, None])
            if pr[0] is not None and pr[1] is not None:
                for pnum in range(pr[0], pr[1] + 1):
                    bagian_lookup[pnum] = bagian.get("nomor")

        for pasal in pasal_list:
            pasal_num: int = pasal["nomor"]
            text_raw: str = pasal.get("text_raw", "")
            ayat_list: list[dict] = pasal.get("ayat", [])
            huruf_top: list[dict] = pasal.get("huruf_top", [])

            bagian_nomor: str | None = bagian_lookup.get(pasal_num)
            parent_id = f"{doc_id}::pasal{pasal_num}"
            parent_header = _make_parent_header(bab_nomor, bagian_nomor, pasal_num)
            tags = ["always_on"] if pasal_num in ALWAYS_ON_PASAL else []

            source_page = _estimate_source_page(pasal_num)

            # -----------------------------------------------------------------
            # Parent chunk — full Pasal text
            # -----------------------------------------------------------------
            parent_text = text_raw.strip()
            parent_chunk = ChunkMeta(
                chunk_id=parent_id,
                parent_id=parent_id,
                doc_id=doc_id,
                chunk_type="parent",
                bab=bab_nomor,
                bagian=bagian_nomor,
                pasal=pasal_num,
                ayat=None,
                huruf=None,
                text=parent_text,
                text_for_embed=_make_embed_text(summary_prefix, parent_header, parent_text),
                summary_prefix=summary_prefix,
                source_page=source_page,
                source_pdf_path=source_pdf,
                version=VERSION_TAG,
                tags=tags,
            )
            parent_chunks.append(parent_chunk)

            # -----------------------------------------------------------------
            # Child chunks
            # -----------------------------------------------------------------
            if ayat_list:
                for ayat in ayat_list:
                    ayat_num: int = ayat["nomor"]
                    ayat_text: str = ayat["text"]
                    huruf_list: list[dict] = ayat.get("huruf", [])

                    child_id = f"{doc_id}::pasal{pasal_num}::ayat{ayat_num}"
                    ayat_header = f"{parent_header} ayat ({ayat_num})"

                    ayat_chunk = ChunkMeta(
                        chunk_id=child_id,
                        parent_id=parent_id,
                        doc_id=doc_id,
                        chunk_type="child",
                        bab=bab_nomor,
                        bagian=bagian_nomor,
                        pasal=pasal_num,
                        ayat=ayat_num,
                        huruf=None,
                        text=ayat_text,
                        text_for_embed=_make_embed_text(summary_prefix, ayat_header, ayat_text),
                        summary_prefix=summary_prefix,
                        source_page=source_page,
                        source_pdf_path=source_pdf,
                        version=VERSION_TAG,
                        tags=tags,
                    )
                    child_chunks.append(ayat_chunk)

                    # Per-huruf children (finer granularity)
                    for huruf in huruf_list:
                        h_letter: str = huruf["nomor"]
                        h_text: str = huruf["text"]
                        huruf_id = f"{doc_id}::pasal{pasal_num}::ayat{ayat_num}::huruf{h_letter}"
                        huruf_header = f"{ayat_header} huruf {h_letter}"

                        huruf_chunk = ChunkMeta(
                            chunk_id=huruf_id,
                            parent_id=parent_id,
                            doc_id=doc_id,
                            chunk_type="child",
                            bab=bab_nomor,
                            bagian=bagian_nomor,
                            pasal=pasal_num,
                            ayat=ayat_num,
                            huruf=h_letter,
                            text=h_text,
                            text_for_embed=_make_embed_text(summary_prefix, huruf_header, h_text),
                            summary_prefix=summary_prefix,
                            source_page=source_page,
                            source_pdf_path=source_pdf,
                            version=VERSION_TAG,
                            tags=tags,
                        )
                        child_chunks.append(huruf_chunk)

            elif huruf_top:
                # Pasal with top-level lettered list (no ayat)
                for huruf in huruf_top:
                    h_letter = huruf["nomor"]
                    h_text = huruf["text"]
                    huruf_id = f"{doc_id}::pasal{pasal_num}::huruf{h_letter}"
                    huruf_header = f"{parent_header} huruf {h_letter}"

                    huruf_chunk = ChunkMeta(
                        chunk_id=huruf_id,
                        parent_id=parent_id,
                        doc_id=doc_id,
                        chunk_type="child",
                        bab=bab_nomor,
                        bagian=bagian_nomor,
                        pasal=pasal_num,
                        ayat=None,
                        huruf=h_letter,
                        text=h_text,
                        text_for_embed=_make_embed_text(summary_prefix, huruf_header, h_text),
                        summary_prefix=summary_prefix,
                        source_page=source_page,
                        source_pdf_path=source_pdf,
                        version=VERSION_TAG,
                        tags=tags,
                    )
                    child_chunks.append(huruf_chunk)

            else:
                # Pasal with no ayat and no huruf — single child = full body
                single_child_id = f"{doc_id}::pasal{pasal_num}::body"
                body_chunk = ChunkMeta(
                    chunk_id=single_child_id,
                    parent_id=parent_id,
                    doc_id=doc_id,
                    chunk_type="child",
                    bab=bab_nomor,
                    bagian=bagian_nomor,
                    pasal=pasal_num,
                    ayat=None,
                    huruf=None,
                    text=parent_text,
                    text_for_embed=_make_embed_text(summary_prefix, parent_header, parent_text),
                    summary_prefix=summary_prefix,
                    source_page=source_page,
                    source_pdf_path=source_pdf,
                    version=VERSION_TAG,
                    tags=tags,
                )
                child_chunks.append(body_chunk)

    logger.info(
        "Chunked doc=%s: %d parent chunks, %d child chunks",
        doc_id,
        len(parent_chunks),
        len(child_chunks),
    )
    return child_chunks, parent_chunks

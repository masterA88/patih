"""Generic ingester for NON-Pasal documents (SOP, statistik, RPJMN, narrative).

These docs have no BAB/Pasal/Ayat hierarchy, so the regulation parser produces
0 Pasal and aborts. This module chunks them by heading-detected sections (with a
page-based fallback) and emits the SAME parsed-JSON schema the indexer consumes:

    {"ast": {...}, "chunks_child": [...], "chunks_parent": [...]}

Chunk schema reuses ChunkMeta with pasal=None, section_title=<heading>, and
chunk_id of the form "{doc_id}::sec{N}" (parent) / "{doc_id}::sec{N}::c{M}" (child).

Usage:
    python -m app.ingest.generic_ingest --pdf "Ringkasan RPJMN.pdf"
        # picks up "<pdf>.meta.json" sidecar; set "doc_type":"reference" optionally
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import date, datetime, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("generic_ingest")

# Chunking parameters (char-based; ~4 chars/token for Indonesian)
_CHILD_CHARS = 1200          # ~300 tokens per child window
_CHILD_OVERLAP = 150
_MIN_SECTION_CHARS = 40      # skip trivially short sections
_PARENT_TEXT_CAP = 4000      # cap parent text shown to LLM

# Heading heuristic: short ALL-CAPS lines, or numbered headings like "1.2 Title"
_HEADING_RE = re.compile(
    r"^(?:[A-Z][A-Z0-9 .,/&()-]{4,70}|\d+(?:\.\d+){0,3}\.?\s+[A-Z][^\n]{3,70})$"
)
_PAGE_MARKER_RE = re.compile(r"\n*--- page (\d+) ---\n*")


def _split_pages(full_text: str) -> list[tuple[int, str]]:
    """Split load_pdf output into (page_number, page_text) using its page markers."""
    parts = _PAGE_MARKER_RE.split(full_text)
    # parts = [pre, '1', page1, '2', page2, ...] — leading pre is page 1 content
    pages: list[tuple[int, str]] = []
    if parts and parts[0].strip():
        pages.append((1, parts[0]))
    i = 1
    while i + 1 < len(parts):
        try:
            pageno = int(parts[i])
        except ValueError:
            pageno = len(pages) + 1
        pages.append((pageno, parts[i + 1]))
        i += 2
    if not pages:  # no markers
        pages = [(1, full_text)]
    return pages


def _detect_sections(pages: list[tuple[int, str]]) -> list[dict]:
    """Group text into sections by heading lines. Returns list of
    {title, text, page}. Falls back to one section per page if no headings.
    """
    sections: list[dict] = []
    current = {"title": None, "text": [], "page": pages[0][0] if pages else 1}

    found_heading = False
    for pageno, ptext in pages:
        for line in ptext.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue
            if _HEADING_RE.match(stripped) and len(stripped) <= 70:
                found_heading = True
                # flush current
                if current["text"]:
                    sections.append({
                        "title": current["title"],
                        "text": " ".join(current["text"]),
                        "page": current["page"],
                    })
                current = {"title": stripped, "text": [], "page": pageno}
            else:
                current["text"].append(stripped)
    if current["text"]:
        sections.append({
            "title": current["title"],
            "text": " ".join(current["text"]),
            "page": current["page"],
        })

    # Fallback: if heading detection produced too few/huge sections, use pages
    if not found_heading or len(sections) < 2:
        logger.info("Heading detection weak — falling back to page-based sections")
        sections = [
            {"title": None, "text": " ".join(pt.split()), "page": pn}
            for pn, pt in pages if pt.strip()
        ]

    # Drop trivially short sections
    sections = [s for s in sections if len(s["text"]) >= _MIN_SECTION_CHARS]
    return sections


def _window(text: str, size: int, overlap: int) -> list[str]:
    """Split text into overlapping char windows on whitespace boundaries."""
    if len(text) <= size:
        return [text]
    out: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        # extend to next space to avoid cutting words
        if end < len(text):
            sp = text.rfind(" ", start, end)
            if sp > start:
                end = sp
        out.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [w for w in out if w]


def ingest_generic(
    pdf_path: Path,
    meta: dict,
    output_dir: Path,
    registry_dir: Path,
    force_ocr: bool = False,
) -> dict:
    """Run generic (non-Pasal) ingest. Returns stats dict."""
    from app.ingest.chunker import ChunkMeta
    from app.ingest.doc_registry import DocRegistry, compute_sha256, register_document
    from app.ingest.pdf_loader import load_pdf

    doc_id = meta["doc_id"]
    label = meta.get("summary_prefix") or meta.get("title") or doc_id
    pdf_rel = str(pdf_path).replace(str(Path.cwd()) + "\\", "").replace(str(Path.cwd()) + "/", "")

    logger.info("Generic ingest: %s (doc_id=%s)", pdf_path.name, doc_id)
    full_text, pdf_meta = load_pdf(pdf_path, force_ocr=force_ocr)
    logger.info("  loaded %d pages, method=%s", pdf_meta["n_pages"], pdf_meta["parse_method"])

    pages = _split_pages(full_text)
    sections = _detect_sections(pages)
    logger.info("  %d sections detected", len(sections))

    parents: list[ChunkMeta] = []
    children: list[ChunkMeta] = []
    version = f"v1-{date.today().isoformat()}"

    for si, sec in enumerate(sections, 1):
        sec_id = f"{doc_id}::sec{si}"
        title = sec["title"] or f"Bagian {si}"
        page = sec["page"]
        body = sec["text"]

        parent_embed = f"passage: [{label} — {title}] {body[:_PARENT_TEXT_CAP]}"
        parents.append(ChunkMeta(
            chunk_id=sec_id, parent_id=sec_id, doc_id=doc_id, chunk_type="parent",
            pasal=None, section_title=title,
            text=body[:_PARENT_TEXT_CAP], text_for_embed=parent_embed,
            summary_prefix=label, source_page=page, source_pdf_path=pdf_rel,
            version=version, tags=["reference"],
        ))

        for ci, win in enumerate(_window(body, _CHILD_CHARS, _CHILD_OVERLAP), 1):
            child_id = f"{sec_id}::c{ci}"
            child_embed = f"passage: [{label} — {title}] {win}"
            children.append(ChunkMeta(
                chunk_id=child_id, parent_id=sec_id, doc_id=doc_id, chunk_type="child",
                pasal=None, section_title=title,
                text=win, text_for_embed=child_embed,
                summary_prefix=label, source_page=page, source_pdf_path=pdf_rel,
                version=version, tags=["reference"],
            ))

    # Minimal AST (indexer ignores it; kept for schema parity)
    ast = {"bab": [{"nomor": "-", "judul": "DOKUMEN REFERENSI", "bagian": [], "pasal": []}],
           "doc_type": "reference"}

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{doc_id}.json"
    payload = {
        "ast": ast,
        "chunks_child": [c.model_dump() for c in children],
        "chunks_parent": [p.model_dump() for p in parents],
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("  wrote %s (%d parents, %d children)", out_path, len(parents), len(children))

    # Registry
    registry_dir.mkdir(parents=True, exist_ok=True)
    reg = DocRegistry(
        doc_id=doc_id, title=meta.get("title", doc_id), nomor=str(meta.get("nomor", "-")),
        tahun=int(meta.get("tahun", 0) or 0),
        jenis_regulasi=meta.get("jenis_regulasi", "OTHER"),
        judul_lengkap=meta.get("judul_lengkap", doc_id),
        tentang=meta.get("tentang", doc_id),
        tanggal_berlaku=meta.get("tanggal_berlaku", "1970-01-01"),
        status_berlaku="aktif", pdf_path=pdf_rel, pdf_sha256=compute_sha256(pdf_path),
        source_url=meta.get("source_url"), parsed_at=datetime.now(timezone.utc).isoformat(),
        parse_method="pymupdf_text" if pdf_meta["parse_method"] != "ocr_tesseract" else "ocr_tesseract",
        parse_version="generic-v1", quality_flags=[],
        n_pasal=0, n_chunks_child=len(children), n_chunks_parent=len(parents),
    )
    register_document(reg, registry_dir / "documents.json")

    print(f"\n{'='*60}\nGENERIC INGEST SUMMARY\n{'='*60}")
    print(f"doc_id   : {doc_id}")
    print(f"sections : {len(parents)}")
    print(f"children : {len(children)}")
    print(f"method   : {pdf_meta['parse_method']}")
    print(f"output   : {out_path}\n{'='*60}")
    return {"doc_id": doc_id, "n_parents": len(parents), "n_children": len(children)}


def _load_meta(pdf_path: Path, meta_arg: str | None) -> dict:
    if meta_arg:
        p = Path(meta_arg)
    else:
        p = pdf_path.with_suffix(pdf_path.suffix + ".meta.json")
    if not p.exists():
        logger.error("Metadata sidecar not found: %s", p)
        sys.exit(1)
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest a non-Pasal reference document.")
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--meta", default=None, help="Metadata JSON (default: <pdf>.meta.json)")
    ap.add_argument("--output-dir", default="data/parsed")
    ap.add_argument("--registry-dir", default="data/registry")
    ap.add_argument("--force-ocr", action="store_true", default=False)
    args = ap.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.is_absolute():
        pdf_path = Path.cwd() / pdf_path
    if not pdf_path.exists():
        logger.error("PDF not found: %s", pdf_path)
        sys.exit(1)

    meta = _load_meta(pdf_path, args.meta)
    ingest_generic(
        pdf_path, meta, Path(args.output_dir), Path(args.registry_dir),
        force_ocr=args.force_ocr,
    )


if __name__ == "__main__":
    main()

"""CLI entry point for ingesting PDF regulasi documents.

Usage:
    python -m app.ingest.cli ingest --pdf data/raw/permensos8.pdf
        # picks up data/raw/permensos8.pdf.meta.json automatically

    python -m app.ingest.cli ingest --pdf foo.pdf --meta foo.meta.json
        # explicit metadata path

    python -m app.ingest.cli ingest --pdf foo.pdf --doc-id permensos-8-2023
        # fallback: tries data/registry/seed_meta.json for that doc-id (legacy)

Metadata sidecar schema (JSON):
    {
      "doc_id": "permensos-8-2023",
      "title": "Permensos No 8 Tahun 2023 ttg TPPO & PMI Bermasalah",
      "nomor": "8",
      "tahun": 2023,
      "jenis_regulasi": "PERMENSOS",    # one of DocRegistry literals
      "judul_lengkap": "...",
      "tentang": "...",
      "tanggal_berlaku": "2023-12-28",  # ISO date
      "source_url": "https://...",       # optional
      "summary_prefix": "..."            # optional, used in chunk text_for_embed
    }
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Required metadata fields in sidecar JSON
_REQUIRED_META_FIELDS = (
    "doc_id",
    "title",
    "nomor",
    "tahun",
    "jenis_regulasi",
    "judul_lengkap",
    "tentang",
    "tanggal_berlaku",
)
_OPTIONAL_META_FIELDS = ("source_url", "summary_prefix")

# Legacy seed metadata — kept ONLY for backward compatibility with the
# original Permensos 8/2023 ingest before sidecars existed. New docs MUST
# use a sidecar.
_SEED_META: dict[str, dict[str, Any]] = {
    "permensos-8-2023": {
        "title": "Permensos No 8 Tahun 2023 ttg TPPO & PMI Bermasalah",
        "nomor": "8",
        "tahun": 2023,
        "jenis_regulasi": "PERMENSOS",
        "judul_lengkap": (
            "Peraturan Menteri Sosial Republik Indonesia Nomor 8 Tahun 2023 "
            "tentang Penanganan Korban Tindak Pidana Perdagangan Orang dan "
            "Pekerja Migran Indonesia Bermasalah"
        ),
        "tentang": "Penanganan Korban TPPO dan Pekerja Migran Indonesia Bermasalah",
        "tanggal_berlaku": "2023-12-28",
        "source_url": "https://jdih.kemensos.go.id",
        "summary_prefix": "Permensos 8/2023: TPPO & PMI Bermasalah",
    }
}


def _load_metadata(
    pdf_path: Path, meta_path: Path | None, doc_id_arg: str | None
) -> dict[str, Any]:
    """
    Resolve metadata for this ingest in priority order:
        1. --meta <path>            explicit path
        2. <pdf>.meta.json          sidecar next to PDF
        3. _SEED_META[doc_id_arg]   legacy fallback (warn)
    Raises SystemExit on missing/invalid metadata.
    """
    # 1. Explicit --meta path
    if meta_path is not None:
        if not meta_path.exists():
            logger.error("Metadata file not found: %s", meta_path)
            sys.exit(1)
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        logger.info("Loaded metadata from %s", meta_path)
    else:
        # 2. Sidecar next to PDF
        sidecar = pdf_path.with_suffix(pdf_path.suffix + ".meta.json")
        if sidecar.exists():
            with open(sidecar, encoding="utf-8") as f:
                meta = json.load(f)
            logger.info("Loaded metadata from sidecar %s", sidecar)
        elif doc_id_arg and doc_id_arg in _SEED_META:
            # 3. Legacy seed fallback
            logger.warning(
                "No sidecar at %s — using legacy _SEED_META[%s]. "
                "Create a sidecar file for future ingests.",
                sidecar, doc_id_arg,
            )
            meta = dict(_SEED_META[doc_id_arg])
            meta["doc_id"] = doc_id_arg
        else:
            logger.error(
                "No metadata source found. Expected one of:\n"
                "  - --meta <path>\n"
                "  - sidecar at %s\n"
                "  - --doc-id matching a legacy seed entry",
                sidecar,
            )
            sys.exit(1)

    # Validate required fields
    missing = [f for f in _REQUIRED_META_FIELDS if f not in meta]
    if missing:
        logger.error("Metadata missing required fields: %s", missing)
        sys.exit(1)

    # Default optional fields
    for f in _OPTIONAL_META_FIELDS:
        meta.setdefault(f, None)

    return meta


def _ingest(args: argparse.Namespace) -> int:
    """Run the full ingest pipeline. Returns exit code (0=success)."""
    from app.ingest.chunker import chunk_document
    from app.ingest.doc_registry import DocRegistry, compute_sha256, register_document
    from app.ingest.pdf_loader import load_pdf
    from app.ingest.structure_parser import parse_structure
    from app.ingest.validators import validate_parse

    pdf_path = Path(args.pdf)
    meta_path = Path(args.meta) if args.meta else None
    output_dir = Path(args.output_dir)
    registry_path = Path(args.registry_dir) / "documents.json"
    force_ocr: bool = args.force_ocr

    # Resolve relative paths from CWD
    if not pdf_path.is_absolute():
        pdf_path = Path.cwd() / pdf_path
    if meta_path is not None and not meta_path.is_absolute():
        meta_path = Path.cwd() / meta_path
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    if not registry_path.is_absolute():
        registry_path = Path.cwd() / registry_path

    if not pdf_path.exists():
        logger.error("PDF not found: %s", pdf_path)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    registry_path.parent.mkdir(parents=True, exist_ok=True)

    # Load metadata (sidecar or explicit) before doing any heavy work
    meta = _load_metadata(pdf_path, meta_path, args.doc_id)
    doc_id: str = meta["doc_id"]
    # If user passed --doc-id, sanity check it matches meta
    if args.doc_id and args.doc_id != doc_id:
        logger.error(
            "--doc-id=%s does not match metadata doc_id=%s",
            args.doc_id, doc_id,
        )
        return 1

    summary_prefix: str | None = meta.get("summary_prefix")
    # Store pdf_path relative to project root for portability
    pdf_path_relative = str(pdf_path).replace(str(Path.cwd()) + "\\", "").replace(
        str(Path.cwd()) + "/", ""
    )

    # -------------------------------------------------------------------------
    # Step 1: Load PDF
    # -------------------------------------------------------------------------
    logger.info("Step 1/5: Loading PDF — %s", pdf_path)
    full_text, pdf_meta = load_pdf(pdf_path, force_ocr=force_ocr)
    parse_method_raw = pdf_meta["parse_method"]
    logger.info(
        "  Loaded %d pages, parse_method=%s, quality_ok=%s",
        pdf_meta["n_pages"],
        parse_method_raw,
        pdf_meta["quality_ok"],
    )

    # -------------------------------------------------------------------------
    # Step 2: Parse structure
    # -------------------------------------------------------------------------
    logger.info("Step 2/5: Parsing document structure")
    ast = parse_structure(full_text)
    all_pasal = [p for b in ast["bab"] for p in b["pasal"]]
    n_pasal = len(all_pasal)
    logger.info("  Found %d BAB, %d Pasal", len(ast["bab"]), n_pasal)

    if n_pasal == 0:
        logger.error(
            "Parser found 0 Pasal in %s. Likely not a regulation document "
            "(SOP/statistik/narrative). Aborting ingest.",
            pdf_path.name,
        )
        return 3

    # -------------------------------------------------------------------------
    # Step 3: Build registry entry
    # -------------------------------------------------------------------------
    logger.info("Step 3/5: Building registry entry")
    pdf_sha = compute_sha256(pdf_path)
    now_iso = datetime.now(timezone.utc).isoformat()

    # Map pdf_loader parse_method to DocRegistry Literal
    _method_map = {
        "pymupdf_text": "pymupdf_text",
        "ocr_tesseract": "ocr_tesseract",
        "pymupdf_text_low_quality": "pymupdf_text",  # closest valid literal
    }
    registry_parse_method = _method_map.get(parse_method_raw, "pymupdf_text")

    registry = DocRegistry(
        doc_id=doc_id,
        title=meta["title"],
        nomor=meta["nomor"],
        tahun=meta["tahun"],
        jenis_regulasi=meta["jenis_regulasi"],
        judul_lengkap=meta["judul_lengkap"],
        tentang=meta["tentang"],
        tanggal_berlaku=meta["tanggal_berlaku"],
        tanggal_dicabut=None,
        status_berlaku="aktif",
        pdf_path=pdf_path_relative,
        pdf_sha256=pdf_sha,
        source_url=meta.get("source_url"),
        parsed_at=now_iso,
        parse_method=registry_parse_method,
        parse_version="v1",
        quality_flags=[],
        n_pasal=n_pasal,
        n_chunks_child=0,
        n_chunks_parent=0,
    )

    # -------------------------------------------------------------------------
    # Step 4: Chunk
    # -------------------------------------------------------------------------
    logger.info("Step 4/5: Chunking document")
    child_chunks, parent_chunks = chunk_document(
        ast=ast,
        doc_id=doc_id,
        source_pdf=pdf_path_relative,
        summary_prefix=summary_prefix,
    )
    logger.info(
        "  %d parent chunks, %d child chunks",
        len(parent_chunks),
        len(child_chunks),
    )

    # Update counts in registry
    registry.n_pasal = n_pasal
    registry.n_chunks_child = len(child_chunks)
    registry.n_chunks_parent = len(parent_chunks)

    # -------------------------------------------------------------------------
    # Step 5: Validate
    # -------------------------------------------------------------------------
    logger.info("Step 5/5: Validating")
    quality_flags = validate_parse(ast, registry, child_chunks, parent_chunks)
    registry.quality_flags = quality_flags

    if parse_method_raw == "pymupdf_text_low_quality":
        registry.quality_flags.append("pymupdf_low_quality_text")

    # -------------------------------------------------------------------------
    # Persist outputs
    # -------------------------------------------------------------------------
    output_path = output_dir / f"{doc_id}.json"
    payload = {
        "ast": ast,
        "chunks_child": [c.model_dump() for c in child_chunks],
        "chunks_parent": [c.model_dump() for c in parent_chunks],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.info("Wrote parsed output → %s", output_path)

    register_document(registry, registry_path)
    logger.info("Registered → %s", registry_path)

    # -------------------------------------------------------------------------
    # Print summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("INGEST SUMMARY")
    print("=" * 60)
    print(f"doc_id        : {doc_id}")
    print(f"title         : {meta['title']}")
    print(f"n_pasal       : {n_pasal}")
    print(f"n_parent      : {len(parent_chunks)}")
    print(f"n_child       : {len(child_chunks)}")
    print(f"parse_method  : {parse_method_raw}")
    print(f"quality_ok    : {pdf_meta['quality_ok']}")
    print(f"quality_flags : {quality_flags or '[]'}")
    print(f"output        : {output_path}")
    print(f"registry      : {registry_path}")
    print("=" * 60)

    if quality_flags:
        print(f"\nWARNING: {len(quality_flags)} quality flag(s) — review above.")
        return 2  # non-zero but not fatal

    print("\nAll checks passed.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest a PDF regulation into the chatbot knowledge base."
    )
    sub = parser.add_subparsers(dest="command")

    ingest_cmd = sub.add_parser("ingest", help="Ingest a PDF document.")
    ingest_cmd.add_argument("--pdf", required=True, help="Path to the PDF file.")
    ingest_cmd.add_argument(
        "--meta",
        default=None,
        help="Explicit path to metadata JSON. Defaults to <pdf>.meta.json sidecar.",
    )
    ingest_cmd.add_argument(
        "--doc-id",
        default=None,
        help="Document identifier (e.g. permensos-8-2023). Used as sanity check "
             "against metadata, OR as legacy seed lookup if no sidecar exists.",
    )
    ingest_cmd.add_argument(
        "--output-dir",
        default="data/parsed",
        help="Directory for parsed JSON output (default: data/parsed).",
    )
    ingest_cmd.add_argument(
        "--registry-dir",
        default="data/registry",
        help="Directory for registry JSON (default: data/registry).",
    )
    ingest_cmd.add_argument(
        "--force-ocr",
        action="store_true",
        default=False,
        help="Force OCR even if text quality looks OK.",
    )

    args = parser.parse_args()

    if args.command == "ingest":
        sys.exit(_ingest(args))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

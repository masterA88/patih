"""Write and read the Document registry record (data/registry/documents.json)."""

import hashlib
import json
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class DocRegistry(BaseModel):
    doc_id: str                            # "permensos-8-2023"
    title: str                             # "Permensos No 8 Tahun 2023 ttg TPPO & PMI Bermasalah"
    nomor: str                             # "8"
    tahun: int                             # 2023
    jenis_regulasi: Literal[
        "PERMENSOS", "UU", "PP", "PERPRES", "PERBUP", "PERMENKES",
        "PERMEN_KEMENAKER", "PERBANK", "PERMA", "OTHER"
    ]
    judul_lengkap: str
    tentang: str                           # "TPPO & PMI Bermasalah"
    tanggal_berlaku: str                   # "2023-06-08" ISO
    tanggal_dicabut: str | None = None     # null jika masih berlaku
    status_berlaku: Literal["aktif", "dicabut", "diubah"] = "aktif"

    # Source
    pdf_path: str                          # "data/raw/permensos8.pdf"
    pdf_sha256: str                        # untuk idempotency
    source_url: str | None = None          # JDIH source

    # Ingest provenance
    parsed_at: str                         # ISO timestamp
    parse_method: Literal[
        "pymupdf_text", "unstructured_fast", "ocr_tesseract", "manual"
    ]
    parse_version: str                     # parser version
    quality_flags: list[str] = Field(default_factory=list)
                                           # ["pasal_count_mismatch", "ocr_low_conf", ...]

    # Counts
    n_pasal: int
    n_chunks_child: int
    n_chunks_parent: int

    # Phase 2 forward-compat
    related_docs: list[str] = Field(default_factory=list)  # ["uu-21-2007", ...]


def compute_sha256(pdf_path: Path) -> str:
    """Return hex SHA-256 of the file at pdf_path."""
    h = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def register_document(meta: DocRegistry, registry_path: Path) -> None:
    """Append or update entry keyed by doc_id in registry JSON (atomic write)."""
    registry_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing registry
    existing: dict = {}
    if registry_path.exists() and registry_path.stat().st_size > 0:
        try:
            with open(registry_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except json.JSONDecodeError:
            logger.warning("Registry file %s is corrupt; starting fresh.", registry_path)
            existing = {}

    existing[meta.doc_id] = meta.model_dump()

    # Atomic write via temp file + rename
    tmp = Path(tempfile.mktemp(dir=registry_path.parent, suffix=".tmp"))
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
        tmp.replace(registry_path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

    logger.info("Registered %s → %s", meta.doc_id, registry_path)

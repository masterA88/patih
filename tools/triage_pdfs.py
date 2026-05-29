"""Audit 28 PDFs in project root: regulasi (Pasal/Ayat) vs non-regulasi.

For each PDF:
  - Read first ~3 pages of text via PyMuPDF
  - Count "Pasal N" markers → if ≥3, treat as regulasi
  - Infer doc_id, jenis_regulasi, nomor, tahun from filename + first-page text
  - Output triage report to tools/triage_report.json

Doesn't modify anything — pure read.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_PATH = PROJECT_ROOT / "tools" / "triage_report.json"

# Filename-based heuristics for jenis_regulasi
_JENIS_PATTERNS = [
    # ORDER MATTERS: SOP/statistik checks first (would otherwise match nothing)
    (r"\bSOP\b|sop\s", "OTHER"),
    (r"statistik|indikator-kesejahteraan|rpjmn|ringkasan", "OTHER"),
    (r"permenkes|permen\s+kes", "PERMENKES"),
    (r"permensos|permen\s+sos", "PERMENSOS"),
    (r"permen\s+naker|permenaker", "PERMEN_KEMENAKER"),
    (r"perbup|peraturan\s+bupati", "PERBUP"),
    (r"perbank|peraturan\s+bank", "PERBANK"),
    (r"perpres|peraturan\s+presiden", "PERPRES"),
    (r"perma|peraturan\s+mahkamah", "PERMA"),
    (r"peraturan\s+bkn|^per\s+bkn", "OTHER"),  # Per BKN: regulasi tapi pakai 'OTHER' literal
    (r"\bUU\b|undang[\s-]?undang", "UU"),
    (r"\bPP\b\s*Nomor|peraturan\s+pemerintah", "PP"),
    (r"\bSOTK\b", "PERBUP"),  # SOTK Dinsos usually = Perbup
]

# Filename-based override: if matches → is_regulasi=True regardless of Pasal count
# (regulasi often has cover + "Mengingat" pages before Pasal 1)
_REGULASI_FILENAME_RE = re.compile(
    r"\b(UU|PP|Permen[a-z]*|Perbup|Perpres|Per\s+BKN|Peraturan\s+BKN)\b",
    re.IGNORECASE,
)
# Exclude SOP / statistik / RPJMN narrative
_NON_REGULASI_FILENAME_RE = re.compile(
    r"\b(SOP|statistik|indikator-kesejahteraan|RPJMN|Ringkasan)\b",
    re.IGNORECASE,
)

_NOMOR_TAHUN_RE = re.compile(
    r"(?:nomor|no\.?|n)\s*[:\.]?\s*(\d{1,4})\s*(?:tahun|th\.?)\s*(\d{4})",
    re.IGNORECASE,
)
_TAHUN_FALLBACK_RE = re.compile(r"\b(20\d{2}|19\d{2})\b")
_PASAL_RE = re.compile(r"\bPasal\s+\d+\b", re.IGNORECASE)
_BAB_RE = re.compile(r"\bBAB\s+[IVX]+\b", re.IGNORECASE)


def _infer_jenis(name_lower: str) -> str:
    for pat, jenis in _JENIS_PATTERNS:
        if re.search(pat, name_lower, re.IGNORECASE):
            return jenis
    return "OTHER"


def _infer_nomor_tahun(text: str) -> tuple[str | None, int | None]:
    m = _NOMOR_TAHUN_RE.search(text)
    if m:
        return m.group(1), int(m.group(2))
    # Filename-only fallback
    t = _TAHUN_FALLBACK_RE.search(text)
    return None, int(t.group(1)) if t else None


def _slugify(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", s.lower()).strip("-")
    return s[:60]


def _read_first_pages(pdf_path: Path, n_pages: int = 3) -> str:
    """Read text from first n_pages. Returns empty string on failure."""
    try:
        import pymupdf
    except ImportError:
        print(f"[err] pymupdf not available", file=sys.stderr)
        return ""

    try:
        doc = pymupdf.open(pdf_path)
        out = []
        for i in range(min(n_pages, len(doc))):
            out.append(doc[i].get_text())
        doc.close()
        return "\n".join(out)
    except Exception as e:
        print(f"[err] {pdf_path.name}: {e}", file=sys.stderr)
        return ""


def triage_pdf(pdf_path: Path) -> dict:
    name = pdf_path.name
    name_lower = name.lower()

    text = _read_first_pages(pdf_path, n_pages=15)
    n_pasal = len(_PASAL_RE.findall(text))
    n_bab = len(_BAB_RE.findall(text))

    # Combined inference source: filename + first-page text
    combined = name + " " + text[:2000]
    jenis = _infer_jenis(name_lower)
    nomor, tahun = _infer_nomor_tahun(combined)

    # Generic doc_id slug
    base = pdf_path.stem
    doc_id = _slugify(base)

    # Filename-based override
    if _NON_REGULASI_FILENAME_RE.search(name):
        is_regulasi = False
    elif _REGULASI_FILENAME_RE.search(name):
        # Regulation type in filename — accept if ≥1 Pasal in 15 pages
        # (allows for cover + "Mengingat" pages before Pasal 1)
        is_regulasi = n_pasal >= 1
    else:
        # No clear signal — require strong Pasal evidence
        is_regulasi = n_pasal >= 3

    return {
        "pdf": str(pdf_path.relative_to(PROJECT_ROOT)),
        "filename": name,
        "size_bytes": pdf_path.stat().st_size,
        "is_regulasi": is_regulasi,
        "n_pasal_in_first_3pages": n_pasal,
        "n_bab_in_first_3pages": n_bab,
        "inferred_doc_id": doc_id,
        "inferred_jenis_regulasi": jenis,
        "inferred_nomor": nomor,
        "inferred_tahun": tahun,
        "first_text_snippet": text[:300].replace("\n", " ").strip(),
    }


def main() -> None:
    # PDFs sit in project root (NOT data/raw — that has only permensos8.pdf)
    pdf_files = sorted(PROJECT_ROOT.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDFs in project root\n")

    report = []
    for p in pdf_files:
        entry = triage_pdf(p)
        flag = "REG" if entry["is_regulasi"] else "non"
        print(
            f"  [{flag}] {entry['filename'][:50]:50} | jenis={entry['inferred_jenis_regulasi']:<18} "
            f"| pasal={entry['n_pasal_in_first_3pages']:>2} "
            f"| {entry['inferred_jenis_regulasi'].lower()}-{entry['inferred_nomor']}-{entry['inferred_tahun']}"
        )
        report.append(entry)

    n_reg = sum(1 for e in report if e["is_regulasi"])
    n_non = len(report) - n_reg
    print(f"\nSummary: {n_reg} regulasi, {n_non} non-regulasi (SOP/statistik/narrative)")

    OUTPUT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Triage report written to {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

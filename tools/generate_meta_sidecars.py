"""Generate <pdf>.meta.json sidecars for the regulasi PDFs from triage_report.json.

Steps:
  1. Load tools/triage_report.json
  2. Filter is_regulasi=True
  3. Dedup by sha256 (keep better-named version)
  4. Fix known inference issues (PERBUP_NO_24_TAHUN_2021 has underscores → regex miss)
  5. Auto-generate metadata. Flag entries that still need user review.
  6. Write sidecars next to each PDF; print pre-flight list.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
TRIAGE_PATH = PROJECT_ROOT / "tools" / "triage_report.json"
OUTPUT_LIST = PROJECT_ROOT / "tools" / "meta_sidecar_manifest.json"

# Robust regex for "Nomor N Tahun YYYY" with various separators (_, ., space)
_NOMOR_TAHUN_RE = re.compile(
    r"(?:nomor|no)[\s_\.]*[:\.]?[\s_]*(\d{1,4})[\s_]*(?:tahun|th)[\s_\.]*(\d{4})",
    re.IGNORECASE,
)

# Jenis_regulasi mapping for DocRegistry Literal compliance.
# Per BKN doesn't have its own literal — use OTHER.
_DOCREGISTRY_LITERALS = {
    "PERMENSOS", "UU", "PP", "PERPRES", "PERBUP", "PERMENKES",
    "PERMEN_KEMENAKER", "PERBANK", "PERMA", "OTHER",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _normalize_jenis(j: str) -> str:
    if j in _DOCREGISTRY_LITERALS:
        return j
    return "OTHER"


def _better_name_score(name: str) -> int:
    """Higher score = more canonical filename. Used for dedup tie-breaks."""
    score = 0
    if not re.search(r"\(\d+\)", name):  # no "(1)" suffix
        score += 10
    if "_" in name or " " in name.replace(" ", "_"):
        score += 3  # underscores hint at official naming
    if name.startswith(name[0].upper()):
        score += 2
    score -= len(name)  # prefer shorter
    return score


def _fix_nomor_tahun(entry: dict) -> tuple[str | None, int | None]:
    """Re-infer nomor/tahun using a more robust regex on filename."""
    filename = entry["filename"]
    m = _NOMOR_TAHUN_RE.search(filename)
    if m:
        return m.group(1), int(m.group(2))
    return entry.get("inferred_nomor"), entry.get("inferred_tahun")


def _build_meta(entry: dict, pdf_path: Path, pdf_sha: str) -> dict:
    nomor, tahun = _fix_nomor_tahun(entry)
    jenis = _normalize_jenis(entry["inferred_jenis_regulasi"])
    doc_id = entry["inferred_doc_id"]

    # Build a sensible doc_id: <jenis_lower>-<nomor>-<tahun>
    if nomor and tahun:
        doc_id = f"{jenis.lower().replace('_', '-')}-{nomor}-{tahun}"
        # Special case for permensos-8-2023 to match existing parsed JSON
        if jenis == "PERMENSOS" and nomor == "8" and tahun == 2023:
            doc_id = "permensos-8-2023"

    # title: short label "Jenis Nomor/Tahun"
    title = f"{jenis} {nomor or '?'}/{tahun or '?'}"

    # judul_lengkap: best-effort from filename
    judul_lengkap = entry["filename"].replace(".pdf", "").replace("_", " ")

    # tentang: extract from "ttg X" if present in filename, else fallback
    ttg_match = re.search(r"\bttg\s+(.+?)(?:\.pdf|$)", entry["filename"], re.IGNORECASE)
    tentang = ttg_match.group(1).strip() if ttg_match else judul_lengkap

    # tanggal_berlaku: fallback to "YYYY-01-01" — user can refine via sidecar edit
    tanggal_berlaku = f"{tahun}-01-01" if tahun else "1970-01-01"

    return {
        "doc_id": doc_id,
        "title": title,
        "nomor": nomor or "?",
        "tahun": tahun if tahun is not None else 0,
        "jenis_regulasi": jenis,
        "judul_lengkap": judul_lengkap,
        "tentang": tentang,
        "tanggal_berlaku": tanggal_berlaku,
        "source_url": None,
        "summary_prefix": f"{jenis} {nomor}/{tahun}: {tentang}",
        # Provenance for review
        "_provenance": {
            "pdf_sha256": pdf_sha,
            "filename": entry["filename"],
            "n_pasal_in_first_15pages": entry["n_pasal_in_first_3pages"],
            "needs_review": False,
        },
    }


def main() -> None:
    if not TRIAGE_PATH.exists():
        print(f"Triage report not found: {TRIAGE_PATH}. Run triage_pdfs.py first.")
        return

    triage = json.loads(TRIAGE_PATH.read_text(encoding="utf-8"))
    regulasi = [e for e in triage if e["is_regulasi"]]
    print(f"Regulasi candidates: {len(regulasi)}")

    # Compute sha + dedup
    by_sha: dict[str, list[dict]] = {}
    for entry in regulasi:
        pdf_path = PROJECT_ROOT / entry["pdf"]
        if not pdf_path.exists():
            print(f"  [warn] not found: {pdf_path}")
            continue
        sha = _sha256(pdf_path)
        by_sha.setdefault(sha, []).append({"entry": entry, "path": pdf_path})

    print(f"Unique by sha256: {len(by_sha)}")

    # Tie-break: pick better-named filename per sha
    chosen: list[tuple[dict, Path, str]] = []
    skipped: list[tuple[str, str]] = []
    for sha, candidates in by_sha.items():
        if len(candidates) == 1:
            c = candidates[0]
            chosen.append((c["entry"], c["path"], sha))
        else:
            sorted_c = sorted(
                candidates, key=lambda c: _better_name_score(c["entry"]["filename"]), reverse=True,
            )
            chosen.append((sorted_c[0]["entry"], sorted_c[0]["path"], sha))
            for c in sorted_c[1:]:
                skipped.append((c["entry"]["filename"], "byte-duplicate"))

    # Generate metadata + write sidecars
    manifest: list[dict] = []
    needs_review: list[str] = []
    for entry, pdf_path, sha in chosen:
        meta = _build_meta(entry, pdf_path, sha)

        # Flag entries that need user review
        review_reasons = []
        if meta["nomor"] == "?":
            review_reasons.append("missing_nomor")
        if meta["tahun"] == 0:
            review_reasons.append("missing_tahun")
        if entry["n_pasal_in_first_3pages"] < 5:  # 15-page check, named "_first_3pages" legacy
            review_reasons.append("low_pasal_count")
        if review_reasons:
            meta["_provenance"]["needs_review"] = True
            meta["_provenance"]["review_reasons"] = review_reasons
            needs_review.append(f"{pdf_path.name}: {','.join(review_reasons)}")

        sidecar = pdf_path.with_suffix(pdf_path.suffix + ".meta.json")
        sidecar.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        manifest.append({
            "filename": pdf_path.name,
            "pdf_path": str(pdf_path.relative_to(PROJECT_ROOT)),
            "sidecar_path": str(sidecar.relative_to(PROJECT_ROOT)),
            "doc_id": meta["doc_id"],
            "jenis_regulasi": meta["jenis_regulasi"],
            "nomor": meta["nomor"],
            "tahun": meta["tahun"],
            "tentang": meta["tentang"],
            "needs_review": meta["_provenance"]["needs_review"],
        })

    # Save manifest
    OUTPUT_LIST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Print summary
    print("\n=== Generated sidecars ===")
    for m in manifest:
        flag = "!" if m["needs_review"] else " "
        print(
            f"  {flag} {m['doc_id']:30} | {m['jenis_regulasi']:<18} "
            f"| {m['nomor']:>4}/{m['tahun']} | {m['filename'][:50]}"
        )
    if skipped:
        print(f"\n=== Skipped duplicates ({len(skipped)}) ===")
        for fname, reason in skipped:
            print(f"  - {fname} ({reason})")
    if needs_review:
        print(f"\n=== Entries needing review ({len(needs_review)}) ===")
        for r in needs_review:
            print(f"  - {r}")

    print(f"\nManifest: {OUTPUT_LIST.relative_to(PROJECT_ROOT)}")
    print(f"Total sidecars: {len(manifest)}")


if __name__ == "__main__":
    main()

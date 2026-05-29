"""Bulk-ingest the 20 regulasi PDFs from meta_sidecar_manifest.json.

For each entry: call ingest CLI _ingest() directly (no subprocess overhead).
Logs aggregate stats and per-doc quality flags. Skips PDFs whose parsed JSON
exists AND is already up-to-date (sha256 match in registry).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).parent.parent
MANIFEST_PATH = PROJECT_ROOT / "tools" / "meta_sidecar_manifest.json"

# Make project root importable
sys.path.insert(0, str(PROJECT_ROOT))

import os
os.chdir(PROJECT_ROOT)

logging.basicConfig(
    level=logging.WARNING,  # keep ingest CLI noise down
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("bulk_ingest")
log.setLevel(logging.INFO)


def _is_already_ingested(doc_id: str, pdf_sha256: str, registry: dict) -> bool:
    """Skip ingest if registry has matching doc_id + sha256."""
    entry = registry.get(doc_id)
    if not entry:
        return False
    return entry.get("pdf_sha256") == pdf_sha256


def _compute_sha(pdf_path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-ingest even if registry sha256 matches (default: skip).",
    )
    parser.add_argument(
        "--only",
        default=None,
        help="Substring filter on filename — ingest only matching PDFs.",
    )
    args = parser.parse_args()

    if not MANIFEST_PATH.exists():
        log.error("Manifest not found: %s. Run generate_meta_sidecars.py first.", MANIFEST_PATH)
        sys.exit(1)

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    log.info("Manifest: %d entries", len(manifest))

    # Apply --only filter
    if args.only:
        manifest = [m for m in manifest if args.only.lower() in m["filename"].lower()]
        log.info("After --only filter: %d entries", len(manifest))

    # Load existing registry (if any)
    registry_path = PROJECT_ROOT / "data" / "registry" / "documents.json"
    if registry_path.exists():
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    else:
        registry = {}

    # Import the ingest function once
    from app.ingest.cli import _ingest

    results: list[dict] = []
    for i, m in enumerate(manifest, 1):
        pdf_path = PROJECT_ROOT / m["pdf_path"]
        doc_id = m["doc_id"]
        log.info("[%d/%d] %s (%s)", i, len(manifest), doc_id, m["filename"][:60])

        if not pdf_path.exists():
            log.error("  PDF not found: %s", pdf_path)
            results.append({**m, "status": "missing", "n_pasal": 0, "exit_code": -1})
            continue

        # Idempotency check
        pdf_sha = _compute_sha(pdf_path)
        if not args.force and _is_already_ingested(doc_id, pdf_sha, registry):
            log.info("  SKIP — already ingested (sha matches)")
            entry = registry[doc_id]
            results.append({
                **m,
                "status": "skipped",
                "n_pasal": entry.get("n_pasal", 0),
                "n_chunks_child": entry.get("n_chunks_child", 0),
                "n_chunks_parent": entry.get("n_chunks_parent", 0),
                "exit_code": 0,
            })
            continue

        # Run ingest
        ns = SimpleNamespace(
            pdf=str(pdf_path),
            meta=None,  # use sidecar at <pdf>.meta.json
            doc_id=None,
            output_dir="data/parsed",
            registry_dir="data/registry",
            force_ocr=False,
        )
        try:
            code = _ingest(ns)
        except Exception as e:
            log.error("  EXCEPTION: %s", e)
            results.append({**m, "status": "error", "exit_code": -1, "error": str(e)})
            continue

        # Reload registry to grab fresh counts
        if registry_path.exists():
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
        entry = registry.get(doc_id, {})

        status = "ok" if code == 0 else ("warning" if code == 2 else "fail")
        results.append({
            **m,
            "status": status,
            "exit_code": code,
            "n_pasal": entry.get("n_pasal", 0),
            "n_chunks_child": entry.get("n_chunks_child", 0),
            "n_chunks_parent": entry.get("n_chunks_parent", 0),
            "quality_flags": entry.get("quality_flags", []),
        })

    # Aggregate report
    report_path = PROJECT_ROOT / "tools" / "bulk_ingest_report.json"
    report_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 80)
    print("BULK INGEST SUMMARY")
    print("=" * 80)
    n_ok = sum(1 for r in results if r["status"] == "ok")
    n_warn = sum(1 for r in results if r["status"] == "warning")
    n_skip = sum(1 for r in results if r["status"] == "skipped")
    n_fail = sum(1 for r in results if r["status"] in ("error", "fail", "missing"))
    n_child_total = sum(r.get("n_chunks_child", 0) for r in results)
    n_parent_total = sum(r.get("n_chunks_parent", 0) for r in results)
    print(f"  ok       : {n_ok}")
    print(f"  warning  : {n_warn}")
    print(f"  skipped  : {n_skip}")
    print(f"  failed   : {n_fail}")
    print(f"  total chunks: {n_child_total} children, {n_parent_total} parents")
    print()
    for r in results:
        status_marker = {
            "ok": "  ", "warning": "! ", "skipped": ". ",
            "fail": "X ", "error": "X ", "missing": "? ",
        }.get(r["status"], "  ")
        flags = ",".join(r.get("quality_flags", [])) or "-"
        print(
            f"{status_marker}{r['doc_id']:30} | {r['status']:<8} "
            f"| Pasal={r.get('n_pasal', '?'):>3} | flags={flags[:40]}"
        )
    print(f"\nReport: {report_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

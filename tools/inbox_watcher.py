"""Folder watcher daemon for incremental document ingestion.

Watches `data/raw/inbox/` for new PDFs. A `<pdf>.meta.json` sidecar carries the
document's identity/routing; if one is missing it is **auto-generated** from the
filename + first pages (best-effort, flagged for review) so a user can drop just a PDF.
Pass `--no-auto-meta` to require an explicit sidecar instead.

On detection: ingest -> incremental Chroma upsert + BM25 rebuild + parent_lookup merge.
Then moves the PDF + sidecar to `data/raw/` (so the indexer's registry path remains valid).

Polling-based (no watchdog dep). Default interval 10s.

Concurrency:
  - Single watcher process at a time via lock file `data/raw/inbox/.lock`.
  - Skips PDFs that are still being written (size unstable across two polls).

Usage:
    python -m tools.inbox_watcher
    python -m tools.inbox_watcher --interval 30 --once  # one pass then exit
    python -m tools.inbox_watcher --no-auto-meta        # require explicit sidecars
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("inbox_watcher")

INBOX = PROJECT_ROOT / "data" / "raw" / "inbox"
PROCESSED = PROJECT_ROOT / "data" / "raw"
FAILED = PROJECT_ROOT / "data" / "raw" / "failed"
LOCK_PATH = INBOX / ".lock"
LOG_PATH = INBOX / "processed.log"


def _ensure_dirs() -> None:
    INBOX.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    FAILED.mkdir(parents=True, exist_ok=True)


def _acquire_lock() -> bool:
    """Best-effort file lock. Returns True if acquired."""
    if LOCK_PATH.exists():
        # Stale lock if older than 1 hour
        age = time.time() - LOCK_PATH.stat().st_mtime
        if age > 3600:
            log.warning("Stale lock (%.0fs old) — removing", age)
            LOCK_PATH.unlink(missing_ok=True)
        else:
            return False
    LOCK_PATH.write_text(f"pid={os.getpid()} ts={time.time()}", encoding="utf-8")
    return True


def _release_lock() -> None:
    LOCK_PATH.unlink(missing_ok=True)


def _build_reference_meta(entry: dict, pdf: Path, sha: str) -> dict:
    """Minimal best-effort sidecar for a non-regulation (reference) document."""
    label = pdf.stem.replace("_", " ").strip()
    slug = entry.get("inferred_doc_id") or pdf.stem.lower()
    return {
        "doc_id": slug,
        "title": label,
        "doc_type": "reference",
        "judul_lengkap": label,
        "tentang": label,
        "source_url": None,
        "summary_prefix": label,
        "_provenance": {
            "pdf_sha256": sha,
            "filename": pdf.name,
            "needs_review": True,
        },
    }


def _autogenerate_meta(pdf: Path, meta_path: Path) -> dict:
    """Write a best-effort `<pdf>.meta.json` inferred from the filename + first pages.

    Reuses the same inference as `tools/triage_pdfs` and `tools/generate_meta_sidecars`,
    so a user can drop just a PDF. The result is filename-derived and therefore a guess:
    it is flagged `_provenance.auto_generated = True` and (for regulations with an
    unresolved nomor/tahun) `needs_review = True`. Review it before trusting citation
    labels. To require an explicit sidecar instead, run with `--no-auto-meta`.
    """
    import json as _json

    from tools.generate_meta_sidecars import _build_meta, _sha256
    from tools.triage_pdfs import triage_pdf

    entry = triage_pdf(pdf)
    sha = _sha256(pdf)
    if entry.get("is_regulasi"):
        meta = _build_meta(entry, pdf, sha)
    else:
        meta = _build_reference_meta(entry, pdf, sha)
    meta.setdefault("_provenance", {})["auto_generated"] = True
    meta_path.write_text(
        _json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return meta


def _find_ready_pairs(auto_meta: bool = True) -> list[tuple[Path, Path]]:
    """Find (pdf, meta) pairs in inbox/ that are not currently being written.

    Heuristic for "not being written": file mtime is at least 5 seconds in the past.

    When ``auto_meta`` is True (default) a PDF without a sidecar gets one generated
    automatically (see :func:`_autogenerate_meta`); when False the PDF is skipped until
    the user supplies a sidecar (the original strict behaviour).
    """
    pairs: list[tuple[Path, Path]] = []
    now = time.time()
    for pdf in sorted(INBOX.glob("*.pdf")):
        # The PDF itself must be quiet for ≥5s (not still being copied in).
        if (now - pdf.stat().st_mtime) < 5:
            log.debug("Skip %s — recently modified", pdf.name)
            continue
        meta = pdf.with_suffix(pdf.suffix + ".meta.json")
        if not meta.exists():
            if not auto_meta:
                log.debug("Skip %s — no .meta.json sidecar yet", pdf.name)
                continue
            try:
                _autogenerate_meta(pdf, meta)
                log.info(
                    "Auto-generated metadata for %s — review %s before trusting citations",
                    pdf.name, meta.name,
                )
            except Exception as e:
                log.error("Auto-metadata failed for %s: %s — skipping", pdf.name, e)
                continue
        else:
            # A user-supplied sidecar must also be quiet for ≥5s.
            if (now - meta.stat().st_mtime) < 5:
                log.debug("Skip %s — sidecar recently modified", pdf.name)
                continue
        pairs.append((pdf, meta))
    return pairs


def _process_pair(pdf: Path, meta: Path) -> tuple[bool, str]:
    """Run ingest + incremental index for one PDF. Returns (success, message).

    Order matters: we MOVE the PDF + sidecar to PROCESSED first, then ingest
    from the new location. This way the registry's `pdf_path` field is correct
    (points to the final resting place, not the inbox).
    """
    from app.ingest.cli import _ingest
    from app.retrieval.indexer import build_index

    log.info("Processing: %s", pdf.name)

    # 1. Move to PROCESSED first (so ingest stores correct pdf_path)
    dest_pdf = PROCESSED / pdf.name
    dest_meta = PROCESSED / meta.name
    if dest_pdf.exists():
        ts = int(time.time())
        dest_pdf = PROCESSED / f"{pdf.stem}.{ts}{pdf.suffix}"
        dest_meta = PROCESSED / (dest_pdf.name + ".meta.json")
    try:
        shutil.move(str(pdf), str(dest_pdf))
        shutil.move(str(meta), str(dest_meta))
    except Exception as e:
        return False, f"move failed: {e}"

    try:
        # 2. Read metadata to decide route (regulasi vs reference doc)
        import json as _json
        with open(dest_meta, encoding="utf-8") as f:
            meta_dict = _json.load(f)
        doc_id = meta_dict["doc_id"]
        doc_type = meta_dict.get("doc_type", "regulasi")

        if doc_type == "reference":
            # Non-Pasal doc (SOP / statistik / RPJMN) → generic section chunker
            from app.ingest.generic_ingest import ingest_generic
            ingest_generic(
                dest_pdf, meta_dict,
                PROJECT_ROOT / "data" / "parsed",
                PROJECT_ROOT / "data" / "registry",
                force_ocr=False,
            )
        else:
            ns = SimpleNamespace(
                pdf=str(dest_pdf),
                meta=str(dest_meta),
                doc_id=None,
                output_dir="data/parsed",
                registry_dir="data/registry",
                force_ocr=False,
            )
            code = _ingest(ns)
            if code not in (0, 2):  # 0=ok, 2=warning
                return False, f"ingest exited with code {code}"

        # 3. Locate the parsed JSON just produced
        parsed_json = PROJECT_ROOT / "data" / "parsed" / f"{doc_id}.json"
        if not parsed_json.exists():
            return False, f"parsed JSON not produced: {parsed_json}"

        # 4. Incremental index (single-doc mode -> merges with existing corpus)
        stats = build_index(
            parsed=str(parsed_json),
            skip_existing=False,
            rebuild=False,
        )
        log.info(
            "  Indexed: %d embedded, %d skipped, %d parents total",
            stats["n_embedded"], stats["n_skipped"], stats["n_parents"],
        )

        return True, f"ingested + indexed -> {doc_id}"

    except Exception as e:
        log.exception("Processing failed for %s", pdf.name)
        # Move files back to inbox so user can re-try after fixing
        try:
            shutil.move(str(dest_pdf), str(INBOX / pdf.name))
            shutil.move(str(dest_meta), str(INBOX / meta.name))
        except Exception:
            pass
        return False, str(e)


def _quarantine(pdf: Path, meta: Path, reason: str) -> None:
    """Move failed pair to FAILED dir with reason log."""
    try:
        shutil.move(str(pdf), str(FAILED / pdf.name))
        shutil.move(str(meta), str(FAILED / meta.name))
        with open(FAILED / "failures.log", "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {pdf.name}  {reason}\n")
    except Exception as e:
        log.error("Failed to quarantine %s: %s", pdf.name, e)


def _append_log(line: str) -> None:
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {line}\n")


def watch_once(auto_meta: bool = True) -> int:
    """One polling pass. Returns count of successfully processed pairs."""
    pairs = _find_ready_pairs(auto_meta)
    if not pairs:
        return 0

    if not _acquire_lock():
        log.info("Another watcher holds lock — skipping pass")
        return 0

    n_ok = 0
    try:
        for pdf, meta in pairs:
            ok, msg = _process_pair(pdf, meta)
            if ok:
                n_ok += 1
                _append_log(f"OK   {pdf.name}  ({msg})")
            else:
                _quarantine(pdf, meta, msg)
                _append_log(f"FAIL {pdf.name}  ({msg})")
    finally:
        _release_lock()
    return n_ok


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--interval", type=int, default=10,
        help="Polling interval in seconds (default: 10).",
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Run one polling pass and exit.",
    )
    parser.add_argument(
        "--no-auto-meta", action="store_true",
        help="Require an explicit <pdf>.meta.json sidecar; do not auto-generate one.",
    )
    args = parser.parse_args()
    auto_meta = not args.no_auto_meta

    _ensure_dirs()
    log.info("Watcher started — inbox=%s interval=%ds", INBOX.relative_to(PROJECT_ROOT), args.interval)
    if auto_meta:
        log.info(
            "Drop a PDF into %s — a .meta.json sidecar is auto-generated if missing "
            "(review it before trusting citations). Supply your own sidecar for precise metadata.",
            INBOX.relative_to(PROJECT_ROOT),
        )
    else:
        log.info(
            "Drop PDF + .meta.json pairs into %s (strict mode: auto-metadata disabled).",
            INBOX.relative_to(PROJECT_ROOT),
        )

    if args.once:
        n = watch_once(auto_meta)
        log.info("Single pass done. Processed %d pair(s).", n)
        return

    try:
        while True:
            try:
                n = watch_once(auto_meta)
                if n > 0:
                    log.info("Processed %d pair(s).", n)
            except Exception as e:
                log.exception("Watcher loop error: %s", e)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        log.info("Watcher stopped.")


if __name__ == "__main__":
    main()

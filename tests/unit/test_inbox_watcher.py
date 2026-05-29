"""Unit tests for tools.inbox_watcher core helpers (no PDF parsing)."""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import pytest

# Ensure project root on path
import sys
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _import_module(monkeypatch, tmp_path):
    """Re-import the watcher module with paths pointing at tmp_path."""
    import importlib
    import tools.inbox_watcher as iw
    importlib.reload(iw)

    # Override path constants to use tmp_path
    inbox = tmp_path / "inbox"
    processed = tmp_path / "processed"
    failed = tmp_path / "failed"
    inbox.mkdir()
    processed.mkdir()
    failed.mkdir()

    monkeypatch.setattr(iw, "INBOX", inbox)
    monkeypatch.setattr(iw, "PROCESSED", processed)
    monkeypatch.setattr(iw, "FAILED", failed)
    monkeypatch.setattr(iw, "LOCK_PATH", inbox / ".lock")
    monkeypatch.setattr(iw, "LOG_PATH", inbox / "processed.log")
    return iw


def test_find_ready_pairs_skips_recently_modified(monkeypatch, tmp_path):
    iw = _import_module(monkeypatch, tmp_path)
    pdf = iw.INBOX / "test.pdf"
    meta = iw.INBOX / "test.pdf.meta.json"
    pdf.write_bytes(b"%PDF-1.4\nfake")
    meta.write_text(json.dumps({"doc_id": "test"}), encoding="utf-8")

    # Just-written → should be skipped (stability check)
    pairs = iw._find_ready_pairs()
    assert pairs == []


def test_find_ready_pairs_returns_stable_pairs(monkeypatch, tmp_path):
    iw = _import_module(monkeypatch, tmp_path)
    pdf = iw.INBOX / "test.pdf"
    meta = iw.INBOX / "test.pdf.meta.json"
    pdf.write_bytes(b"%PDF-1.4\nfake")
    meta.write_text(json.dumps({"doc_id": "test"}), encoding="utf-8")

    # Backdate mtimes by 10s to pass stability check
    past = time.time() - 10
    import os
    os.utime(pdf, (past, past))
    os.utime(meta, (past, past))

    pairs = iw._find_ready_pairs()
    assert len(pairs) == 1
    assert pairs[0][0].name == "test.pdf"
    assert pairs[0][1].name == "test.pdf.meta.json"


def test_find_ready_pairs_skips_missing_sidecar_strict(monkeypatch, tmp_path):
    """Strict mode (auto_meta=False): a PDF without a sidecar is skipped."""
    iw = _import_module(monkeypatch, tmp_path)
    pdf = iw.INBOX / "no_sidecar.pdf"
    pdf.write_bytes(b"%PDF-1.4\nfake")
    past = time.time() - 10
    import os
    os.utime(pdf, (past, past))
    pairs = iw._find_ready_pairs(auto_meta=False)
    assert pairs == []


def test_find_ready_pairs_autogenerates_missing_sidecar(monkeypatch, tmp_path):
    """Default mode: a PDF without a sidecar gets one auto-generated, and is paired."""
    iw = _import_module(monkeypatch, tmp_path)
    pdf = iw.INBOX / "uu_no_8_tahun_2024.pdf"
    pdf.write_bytes(b"%PDF-1.4\nfake")
    past = time.time() - 10
    import os
    os.utime(pdf, (past, past))

    # Stub inference so the unit test does not parse a real PDF (and avoids the
    # triage helper's relative_to(PROJECT_ROOT) on a tmp_path file).
    import tools.triage_pdfs as tp
    import tools.generate_meta_sidecars as gm
    monkeypatch.setattr(tp, "triage_pdf", lambda p: {
        "is_regulasi": True,
        "filename": p.name,
        "inferred_doc_id": "uu-8-2024",
        "inferred_jenis_regulasi": "UU",
        "inferred_nomor": "8",
        "inferred_tahun": 2024,
        "n_pasal_in_first_3pages": 10,
    })
    monkeypatch.setattr(gm, "_sha256", lambda p: "deadbeef")

    pairs = iw._find_ready_pairs(auto_meta=True)
    assert len(pairs) == 1

    meta = iw.INBOX / "uu_no_8_tahun_2024.pdf.meta.json"
    assert meta.exists()
    m = json.loads(meta.read_text(encoding="utf-8"))
    assert m["doc_id"] == "uu-8-2024"
    assert m["jenis_regulasi"] == "UU"
    assert m["_provenance"]["auto_generated"] is True


def test_autogenerate_reference_meta_for_non_regulation(monkeypatch, tmp_path):
    """A non-regulation PDF gets a reference sidecar (doc_type=reference)."""
    iw = _import_module(monkeypatch, tmp_path)
    pdf = iw.INBOX / "SOP pendataan.pdf"
    pdf.write_bytes(b"%PDF-1.4\nfake")

    import tools.triage_pdfs as tp
    import tools.generate_meta_sidecars as gm
    monkeypatch.setattr(tp, "triage_pdf", lambda p: {
        "is_regulasi": False,
        "filename": p.name,
        "inferred_doc_id": "sop-pendataan",
        "inferred_jenis_regulasi": "OTHER",
        "inferred_nomor": None,
        "inferred_tahun": None,
        "n_pasal_in_first_3pages": 0,
    })
    monkeypatch.setattr(gm, "_sha256", lambda p: "cafef00d")

    meta_path = iw.INBOX / "SOP pendataan.pdf.meta.json"
    meta = iw._autogenerate_meta(pdf, meta_path)
    assert meta["doc_id"] == "sop-pendataan"
    assert meta["doc_type"] == "reference"
    assert meta["_provenance"]["auto_generated"] is True


def test_acquire_lock_blocks_concurrent_holder(monkeypatch, tmp_path):
    iw = _import_module(monkeypatch, tmp_path)
    assert iw._acquire_lock() is True
    # Second call without release should fail
    assert iw._acquire_lock() is False
    iw._release_lock()
    # Now should succeed again
    assert iw._acquire_lock() is True
    iw._release_lock()


def test_acquire_lock_clears_stale(monkeypatch, tmp_path):
    iw = _import_module(monkeypatch, tmp_path)
    # Manually create a stale lock (older than 1h)
    iw.LOCK_PATH.write_text("stale", encoding="utf-8")
    import os
    old = time.time() - 7200
    os.utime(iw.LOCK_PATH, (old, old))
    assert iw._acquire_lock() is True
    iw._release_lock()


def test_quarantine_moves_files(monkeypatch, tmp_path):
    iw = _import_module(monkeypatch, tmp_path)
    pdf = iw.INBOX / "bad.pdf"
    meta = iw.INBOX / "bad.pdf.meta.json"
    pdf.write_bytes(b"bad")
    meta.write_text("{}", encoding="utf-8")
    iw._quarantine(pdf, meta, "test reason")
    assert not pdf.exists()
    assert not meta.exists()
    assert (iw.FAILED / "bad.pdf").exists()
    assert (iw.FAILED / "bad.pdf.meta.json").exists()
    assert (iw.FAILED / "failures.log").read_text(encoding="utf-8").startswith(
        time.strftime("%Y-%m-%d")[:4]  # year prefix
    )

"""Integration test: full ingest pipeline e2e — PDF → parse → chunk → registry."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
PDF_PATH = PROJECT_ROOT / "data" / "raw" / "permensos8.pdf"
PARSED_PATH = PROJECT_ROOT / "data" / "parsed" / "permensos-8-2023.json"
REGISTRY_PATH = PROJECT_ROOT / "data" / "registry" / "documents.json"
DOC_ID = "permensos-8-2023"

pytestmark = pytest.mark.integration


def _pdf_available() -> bool:
    return PDF_PATH.exists()


# ---------------------------------------------------------------------------
# CLI run test (subprocess — exercises the full pipeline)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _pdf_available(), reason="PDF not available")
def test_ingest_cli_exits_zero():
    """Running the ingest CLI must exit with code 0."""
    result = subprocess.run(
        [
            sys.executable,
            "-m", "app.ingest.cli",
            "ingest",
            "--pdf", str(PDF_PATH),
            "--doc-id", DOC_ID,
        ],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"CLI failed with code {result.returncode}.\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )


@pytest.mark.skipif(not _pdf_available(), reason="PDF not available")
def test_parsed_output_file_exists():
    assert PARSED_PATH.exists(), f"Parsed output not found: {PARSED_PATH}"


@pytest.mark.skipif(not _pdf_available(), reason="PDF not available")
def test_registry_file_exists():
    assert REGISTRY_PATH.exists(), f"Registry not found: {REGISTRY_PATH}"


# ---------------------------------------------------------------------------
# Registry validation
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _pdf_available(), reason="PDF not available")
def test_registry_has_doc_id():
    with open(REGISTRY_PATH, encoding="utf-8") as f:
        registry = json.load(f)
    assert DOC_ID in registry, f"doc_id {DOC_ID!r} not found in registry"


@pytest.mark.skipif(not _pdf_available(), reason="PDF not available")
def test_registry_sha256_present():
    with open(REGISTRY_PATH, encoding="utf-8") as f:
        registry = json.load(f)
    entry = registry[DOC_ID]
    sha = entry.get("pdf_sha256", "")
    assert len(sha) == 64, f"Expected 64-char hex sha256, got: {sha!r}"


@pytest.mark.skipif(not _pdf_available(), reason="PDF not available")
def test_registry_n_pasal_above_threshold():
    with open(REGISTRY_PATH, encoding="utf-8") as f:
        registry = json.load(f)
    n_pasal = registry[DOC_ID]["n_pasal"]
    assert n_pasal >= 15, f"n_pasal too low: {n_pasal}"


@pytest.mark.skipif(not _pdf_available(), reason="PDF not available")
def test_registry_exact_34_pasal():
    """Permensos 8/2023 has exactly 34 Pasal (Pasal 1–34)."""
    with open(REGISTRY_PATH, encoding="utf-8") as f:
        registry = json.load(f)
    n_pasal = registry[DOC_ID]["n_pasal"]
    assert n_pasal == 34, f"Expected 34 Pasal, got {n_pasal}"


@pytest.mark.skipif(not _pdf_available(), reason="PDF not available")
def test_registry_quality_flags_empty():
    with open(REGISTRY_PATH, encoding="utf-8") as f:
        registry = json.load(f)
    flags = registry[DOC_ID].get("quality_flags", [])
    assert flags == [], f"Unexpected quality flags: {flags}"


# ---------------------------------------------------------------------------
# Parsed JSON validation
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _pdf_available(), reason="PDF not available")
def test_parsed_json_keys():
    with open(PARSED_PATH, encoding="utf-8") as f:
        data = json.load(f)
    assert "ast" in data
    assert "chunks_child" in data
    assert "chunks_parent" in data


@pytest.mark.skipif(not _pdf_available(), reason="PDF not available")
def test_parsed_ast_bab_count():
    with open(PARSED_PATH, encoding="utf-8") as f:
        data = json.load(f)
    n_bab = len(data["ast"]["bab"])
    assert n_bab == 8, f"Expected 8 BAB, got {n_bab}"


@pytest.mark.skipif(not _pdf_available(), reason="PDF not available")
def test_parsed_pasal_count():
    with open(PARSED_PATH, encoding="utf-8") as f:
        data = json.load(f)
    all_pasal = [p for b in data["ast"]["bab"] for p in b["pasal"]]
    assert len(all_pasal) == 34, f"Expected 34 Pasal, got {len(all_pasal)}"


@pytest.mark.skipif(not _pdf_available(), reason="PDF not available")
def test_no_orphan_children():
    with open(PARSED_PATH, encoding="utf-8") as f:
        data = json.load(f)
    parent_ids = {p["chunk_id"] for p in data["chunks_parent"]}
    orphans = [c for c in data["chunks_child"] if c["parent_id"] not in parent_ids]
    assert len(orphans) == 0, f"Orphan children: {[c['chunk_id'] for c in orphans]}"


@pytest.mark.skipif(not _pdf_available(), reason="PDF not available")
def test_no_child_with_null_pasal():
    with open(PARSED_PATH, encoding="utf-8") as f:
        data = json.load(f)
    null_pasal = [c for c in data["chunks_child"] if c.get("pasal") is None]
    assert len(null_pasal) == 0, f"Children with null pasal: {[c['chunk_id'] for c in null_pasal]}"


@pytest.mark.skipif(not _pdf_available(), reason="PDF not available")
def test_no_duplicate_child_chunk_ids():
    with open(PARSED_PATH, encoding="utf-8") as f:
        data = json.load(f)
    ids = [c["chunk_id"] for c in data["chunks_child"]]
    assert len(ids) == len(set(ids)), f"Duplicate child IDs: {set(x for x in ids if ids.count(x) > 1)}"


@pytest.mark.skipif(not _pdf_available(), reason="PDF not available")
def test_pasal1_always_on():
    with open(PARSED_PATH, encoding="utf-8") as f:
        data = json.load(f)
    pasal1_parents = [p for p in data["chunks_parent"] if p["pasal"] == 1]
    assert len(pasal1_parents) == 1
    assert "always_on" in pasal1_parents[0]["tags"]

    pasal1_children = [c for c in data["chunks_child"] if c["pasal"] == 1]
    assert len(pasal1_children) >= 1
    for c in pasal1_children:
        assert "always_on" in c["tags"]


@pytest.mark.skipif(not _pdf_available(), reason="PDF not available")
def test_pasal1_parent_chunk_content():
    """Pasal 1 should contain definition text (definitional pasal content sanity)."""
    with open(PARSED_PATH, encoding="utf-8") as f:
        data = json.load(f)
    pasal1 = next((p for p in data["chunks_parent"] if p["pasal"] == 1), None)
    assert pasal1 is not None
    # Pasal 1 defines "Perdagangan Orang"
    assert "Perdagangan" in pasal1["text"] or "perdagangan" in pasal1["text"].lower()


@pytest.mark.skipif(not _pdf_available(), reason="PDF not available")
def test_text_for_embed_format():
    """Every chunk's text_for_embed must start with 'passage: '."""
    with open(PARSED_PATH, encoding="utf-8") as f:
        data = json.load(f)
    for chunk in data["chunks_child"] + data["chunks_parent"]:
        assert chunk["text_for_embed"].startswith("passage: "), (
            f"Bad text_for_embed for {chunk['chunk_id']}: {chunk['text_for_embed'][:60]}"
        )


@pytest.mark.skipif(not _pdf_available(), reason="PDF not available")
def test_pasal5_bagian_kesatu():
    """Pasal 5 is in Bagian Kesatu of BAB II."""
    with open(PARSED_PATH, encoding="utf-8") as f:
        data = json.load(f)
    pasal5 = next((p for p in data["chunks_parent"] if p["pasal"] == 5), None)
    assert pasal5 is not None
    assert pasal5.get("bagian") == "Kesatu"
    assert pasal5.get("bab") == "II"

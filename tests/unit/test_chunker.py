"""Unit tests for chunker.py — verify parent/child chunk structure and metadata."""

import pytest
from app.ingest.chunker import chunk_document, ChunkMeta, ALWAYS_ON_PASAL


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def make_simple_ast(pasal_configs: list[dict]) -> dict:
    """
    Build a minimal AST dict for testing.
    pasal_configs: list of {nomor, ayat: [{nomor, text, huruf: [{nomor, text}]}]}
    """
    return {
        "bab": [
            {
                "nomor": "I",
                "judul": "TEST BAB",
                "bagian": [],
                "pasal": [
                    {
                        "nomor": p["nomor"],
                        "text_raw": f"Pasal {p['nomor']} raw text.",
                        "ayat": p.get("ayat", []),
                        "huruf_top": p.get("huruf_top", []),
                    }
                    for p in pasal_configs
                ],
            }
        ]
    }


MOCK_AST_SIMPLE = make_simple_ast([
    {
        "nomor": 1,
        "ayat": [
            {"nomor": 1, "text": "Definisi satu.", "huruf": []},
            {"nomor": 2, "text": "Definisi dua.", "huruf": []},
        ],
    },
    {
        "nomor": 2,
        "ayat": [
            {
                "nomor": 1,
                "text": "Penanganan dilakukan melalui:",
                "huruf": [
                    {"nomor": "a", "text": "rehabilitasi sosial"},
                    {"nomor": "b", "text": "jaminan sosial"},
                ],
            },
            {"nomor": 2, "text": "Dilaksanakan oleh Menteri.", "huruf": []},
        ],
    },
    {
        "nomor": 3,
        "ayat": [],  # no ayat, no huruf → ::body child
        "huruf_top": [],
    },
])

DOC_ID = "test-doc"
SOURCE_PDF = "data/raw/test.pdf"


@pytest.fixture
def chunks():
    child, parent = chunk_document(MOCK_AST_SIMPLE, DOC_ID, SOURCE_PDF)
    return child, parent


# ---------------------------------------------------------------------------
# Parent chunk tests
# ---------------------------------------------------------------------------

def test_parent_count(chunks):
    _, parents = chunks
    assert len(parents) == 3  # Pasal 1, 2, 3


def test_parent_chunk_ids(chunks):
    _, parents = chunks
    ids = {p.chunk_id for p in parents}
    assert f"{DOC_ID}::pasal1" in ids
    assert f"{DOC_ID}::pasal2" in ids
    assert f"{DOC_ID}::pasal3" in ids


def test_parent_chunk_type(chunks):
    _, parents = chunks
    assert all(p.chunk_type == "parent" for p in parents)


def test_parent_pasal_not_none(chunks):
    _, parents = chunks
    assert all(p.pasal is not None for p in parents)


# ---------------------------------------------------------------------------
# Child chunk tests
# ---------------------------------------------------------------------------

def test_child_pasal_none_count(chunks):
    children, _ = chunks
    null_pasal = [c for c in children if c.pasal is None]
    assert len(null_pasal) == 0, f"Children with null pasal: {[c.chunk_id for c in null_pasal]}"


def test_all_child_parent_ids_resolvable(chunks):
    children, parents = chunks
    parent_ids = {p.chunk_id for p in parents}
    orphans = [c for c in children if c.parent_id not in parent_ids]
    assert len(orphans) == 0, f"Orphan children: {[c.chunk_id for c in orphans]}"


def test_child_count_pasal1(chunks):
    children, _ = chunks
    pasal1_children = [c for c in children if c.pasal == 1]
    # Pasal 1 has 2 ayat, no huruf → 2 child chunks
    assert len(pasal1_children) == 2


def test_child_count_pasal2(chunks):
    children, _ = chunks
    pasal2_children = [c for c in children if c.pasal == 2]
    # Pasal 2: ayat1 (+ 2 huruf) + ayat2 → 1 ayat + 2 huruf + 1 ayat = 4 children
    assert len(pasal2_children) == 4


def test_child_pasal3_body_fallback(chunks):
    children, _ = chunks
    pasal3_children = [c for c in children if c.pasal == 3]
    assert len(pasal3_children) == 1
    assert pasal3_children[0].chunk_id == f"{DOC_ID}::pasal3::body"


def test_child_chunk_ids_unique(chunks):
    children, _ = chunks
    ids = [c.chunk_id for c in children]
    assert len(ids) == len(set(ids)), f"Duplicate chunk IDs: {set(x for x in ids if ids.count(x) > 1)}"


def test_parent_ids_unique(chunks):
    _, parents = chunks
    ids = [p.chunk_id for p in parents]
    assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# Metadata tests
# ---------------------------------------------------------------------------

def test_always_on_tag_on_pasal1(chunks):
    children, parents = chunks
    pasal1_parent = next(p for p in parents if p.pasal == 1)
    assert "always_on" in pasal1_parent.tags

    pasal1_children = [c for c in children if c.pasal == 1]
    for c in pasal1_children:
        assert "always_on" in c.tags


def test_no_always_on_on_other_pasal(chunks):
    children, parents = chunks
    pasal2_parent = next(p for p in parents if p.pasal == 2)
    assert "always_on" not in pasal2_parent.tags


def test_text_for_embed_has_passage_prefix(chunks):
    children, parents = chunks
    for chunk in children + parents:
        assert chunk.text_for_embed.startswith("passage: ")


def test_text_for_embed_has_doc_label(chunks):
    children, parents = chunks
    for chunk in children + parents:
        assert "Permensos" in chunk.text_for_embed or DOC_ID in chunk.text_for_embed


def test_summary_prefix_default(chunks):
    _, parents = chunks
    # No summary_prefix passed → each chunk.summary_prefix is None
    for p in parents:
        assert p.summary_prefix is None


def test_summary_prefix_custom():
    child, parent = chunk_document(
        MOCK_AST_SIMPLE, DOC_ID, SOURCE_PDF,
        summary_prefix="Test Doc Label"
    )
    for p in parent:
        assert p.summary_prefix == "Test Doc Label"
        assert "Test Doc Label" in p.text_for_embed


def test_doc_id_on_all_chunks(chunks):
    children, parents = chunks
    for chunk in children + parents:
        assert chunk.doc_id == DOC_ID


def test_source_pdf_on_all_chunks(chunks):
    children, parents = chunks
    for chunk in children + parents:
        assert chunk.source_pdf_path == SOURCE_PDF


def test_version_format(chunks):
    children, _ = chunks
    import re
    # Version should be "v1-YYYY-MM-DD"
    assert all(re.match(r"v\d+-\d{4}-\d{2}-\d{2}", c.version) for c in children)


# ---------------------------------------------------------------------------
# Bagian propagation test
# ---------------------------------------------------------------------------

def test_bagian_propagated_from_ast():
    ast = {
        "bab": [
            {
                "nomor": "II",
                "judul": "PENANGANAN",
                "bagian": [
                    {"nomor": "Kesatu", "judul": "Umum", "pasal_range": [4, 5]},
                    {"nomor": "Kedua", "judul": "Tahapan", "pasal_range": [6, 7]},
                ],
                "pasal": [
                    {"nomor": 4, "text_raw": "Teks 4.", "ayat": [{"nomor": 1, "text": "isi", "huruf": []}], "huruf_top": []},
                    {"nomor": 5, "text_raw": "Teks 5.", "ayat": [{"nomor": 1, "text": "isi", "huruf": []}], "huruf_top": []},
                    {"nomor": 6, "text_raw": "Teks 6.", "ayat": [{"nomor": 1, "text": "isi", "huruf": []}], "huruf_top": []},
                    {"nomor": 7, "text_raw": "Teks 7.", "ayat": [{"nomor": 1, "text": "isi", "huruf": []}], "huruf_top": []},
                ],
            }
        ]
    }
    children, parents = chunk_document(ast, "test", "test.pdf")
    pasal4_parent = next(p for p in parents if p.pasal == 4)
    pasal6_parent = next(p for p in parents if p.pasal == 6)
    assert pasal4_parent.bagian == "Kesatu"
    assert pasal6_parent.bagian == "Kedua"


# ---------------------------------------------------------------------------
# ChunkMeta schema validation
# ---------------------------------------------------------------------------

def test_chunkmeta_schema_required_fields(chunks):
    children, parents = chunks
    for chunk in children + parents:
        # All required fields present and not None (except optional ones)
        assert chunk.chunk_id
        assert chunk.parent_id
        assert chunk.doc_id
        assert chunk.chunk_type in ("child", "parent")
        assert isinstance(chunk.pasal, int)
        assert chunk.text
        assert chunk.text_for_embed
        assert chunk.source_pdf_path
        assert chunk.version
        assert chunk.lang == "id"

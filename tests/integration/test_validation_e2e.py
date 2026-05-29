"""Integration tests for validation pipeline (end-to-end with real Chroma).

Spec: build-spec Section 5.4.

Tests:
  1. Adversarial: fake citations (Pasal 99, Pasal 5 huruf z) -> hitl_flag=True
  2. Normal: well-grounded response -> expected to pass (citation_accuracy=1.0)
  3. Refusal: refusal text -> hitl_flag=False, is_refusal=True

Note: tests 1 and 3 don't need LLM (pure validator logic with mock context).
Test 2 uses real Chroma but no LLM — uses a known-good response template.

These tests require Chroma index to be built (data/chroma/ must exist).
Skip with pytest.mark.skip if index not built.
"""

import json
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Skip guard: require Chroma index
# ---------------------------------------------------------------------------
CHROMA_DIR = Path("data/chroma")

pytestmark = pytest.mark.skipif(
    not CHROMA_DIR.exists(),
    reason="Chroma index not built (run: python -m app.retrieval.indexer)"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def doc_registry():
    registry_path = Path("data/registry/documents.json")
    if not registry_path.exists():
        pytest.skip("documents.json not found")
    with open(registry_path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def chroma_collection():
    from app.retrieval.vector_store import get_collection
    return get_collection()


# ---------------------------------------------------------------------------
# Test 1: Adversarial — fake citations caught
# ---------------------------------------------------------------------------

class TestAdversarialCitations:
    def test_fake_pasal_99_flagged(self):
        """Response with Pasal 99 -> citations_valid=[False], hitl_flag=True."""
        from app.validators.pipeline import validate

        fake_response = (
            "Pasal 99 mengatur sanksi pidana untuk pelaku TPPO. "
            "Lihat juga Pasal 5 ayat (2) huruf z."
        )
        context = [
            {
                "pasal": 5,
                "text": "Pasal 5: (1) Penanganan dilakukan terhadap korban TPPO. "
                        "(2) Eksploitasi meliputi pelacuran, kerja paksa, perbudakan.",
                "bab": "II",
                "bagian": "Kesatu",
            }
        ]

        result = validate(response=fake_response, context=context, doc_id="permensos-8-2023")

        assert len(result.citations_extracted) == 2
        # Both citations should be invalid
        assert result.citations_valid[0] is False, "Pasal 99 should be invalid"
        assert result.citations_valid[1] is False, "Pasal 5 huruf z should be invalid"
        assert result.citation_accuracy < 1.0
        assert result.hitl_flag is True
        assert "invalid_citation" in result.hitl_reasons

    def test_pasal_35_flagged(self):
        """Pasal 35 (> n_pasal=34) -> invalid citation -> hitl_flag=True."""
        from app.validators.pipeline import validate

        response = "Berdasarkan Pasal 35, prosedur ini berlaku."
        context = []

        result = validate(response=response, context=context)

        assert len(result.citations_extracted) == 1
        assert result.citations_valid == [False]
        assert result.hitl_flag is True

    def test_all_invalid_citations_accuracy_zero(self):
        """All citations invalid -> citation_accuracy=0.0."""
        from app.validators.pipeline import validate

        response = "Pasal 99 dan Pasal 100 berlaku di sini."
        result = validate(response=response, context=[])

        assert result.citation_accuracy == 0.0


# ---------------------------------------------------------------------------
# Test 2: Normal well-grounded response
# ---------------------------------------------------------------------------

class TestNormalResponse:
    def test_grounded_response_passes(self):
        """Well-grounded response about Pasal 5 eksploitasi -> expected hitl_flag=False.

        Uses a synthetic but realistic response template that mirrors what Gemini produces.
        Citation accuracy and EG/RP depend on actual Chroma contents.
        """
        from app.validators.pipeline import validate

        # Known-good response template mirroring Step 4 smoke test output
        response = (
            "Berdasarkan Pasal 5 ayat (2) Permensos 8/2023, bentuk eksploitasi meliputi:\n"
            "a. pelacuran (Pasal 5 ayat (2) huruf a);\n"
            "b. kerja paksa (Pasal 5 ayat (2) huruf b);\n"
            "c. perbudakan (Pasal 5 ayat (2) huruf c).\n\n"
            "Sumber: Pasal 5."
        )
        context = [
            {
                "pasal": 5,
                "text": (
                    "Pasal 5 ayat (2): Eksploitasi sebagaimana dimaksud pada ayat (1) meliputi: "
                    "a. pelacuran; b. kerja paksa; c. perbudakan; d. penipuan; "
                    "e. pemerasan; f. penyiksaan; g. pemindahan organ."
                ),
            },
            {
                "pasal": 1,
                "text": (
                    "Pasal 1: TPPO adalah perekrutan, pengiriman, pemindahan, penampungan "
                    "seseorang dengan ancaman kekerasan untuk tujuan eksploitasi. "
                    "Korban TPPO adalah seseorang yang mengalami TPPO."
                ),
            }
        ]

        result = validate(response=response, context=context)

        # All 4 citations should be valid (pasal 5, 5 ayat 2 a/b/c all exist in Chroma)
        assert result.citation_accuracy == 1.0, (
            f"Expected citation_accuracy=1.0, got {result.citation_accuracy}. "
            f"Valid: {result.citations_valid}"
        )
        # EG should be high since response uses Permensos legal terms from context
        assert result.eg_score >= 0.80, f"EG score too low: {result.eg_score}"
        # RP should pass: Phase 1 threshold=1 content word, single-term list items pass
        assert result.rp_score >= 0.85, f"RP score too low: {result.rp_score}"


# ---------------------------------------------------------------------------
# Test 3: Refusal text
# ---------------------------------------------------------------------------

class TestRefusalDetection:
    def test_standard_refusal_skips_validation(self):
        """Standard refusal text -> is_refusal=True, hitl_flag=False."""
        from app.validators.pipeline import validate

        refusal = (
            "Informasi yang Anda tanyakan tidak diatur secara spesifik "
            "dalam Peraturan Menteri Sosial Nomor 8 Tahun 2023. "
            "Untuk pertanyaan ini, mohon merujuk ke peraturan lain seperti "
            "Undang-Undang Nomor 21 Tahun 2007 atau konsultasi dengan praktisi hukum."
        )

        result = validate(response=refusal, context=[])

        assert result.is_refusal is True
        assert result.hitl_flag is False
        assert result.citations_extracted == []
        assert result.citation_accuracy == 1.0
        assert result.eg_score == 1.0
        assert result.rp_score == 1.0

    def test_partial_refusal_phrase(self):
        """Partial refusal phrase match -> detected as refusal."""
        from app.validators.pipeline import validate

        partial_refusal = (
            "Informasi yang Anda tanyakan tidak diatur secara spesifik dalam Permensos 8/2023."
        )
        # This should still be detected as refusal via prefix match
        result = validate(response=partial_refusal, context=[])
        assert result.is_refusal is True


# ---------------------------------------------------------------------------
# Test 4: ValidationResult schema
# ---------------------------------------------------------------------------

class TestValidationResultSchema:
    def test_result_serializable(self):
        """ValidationResult must serialize to dict/JSON without error."""
        from app.validators.pipeline import validate

        result = validate(
            response="Pasal 5 ayat (2) mengatur eksploitasi.",
            context=[{"pasal": 5, "text": "eksploitasi, kerja paksa, pelacuran"}]
        )
        d = result.model_dump()
        j = result.model_dump_json()
        assert isinstance(d, dict)
        assert isinstance(j, str)
        assert "hitl_flag" in d
        assert "citations_extracted" in d
        assert "eg_score" in d
        assert "rp_score" in d

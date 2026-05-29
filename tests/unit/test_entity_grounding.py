"""Unit tests for entity_grounding.py (HalluGraph EG scorer).

Spec: build-spec Section 5.4 line 991.
"""

import pytest
from app.validators.entity_grounding import compute_eg_score, LEGAL_TERMS_PHASE1


# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------

# Context text containing standard Permensos terms
GOOD_CONTEXT = (
    "Pasal 1: Dalam Peraturan Menteri ini yang dimaksud dengan: "
    "Tindak Pidana Perdagangan Orang yang selanjutnya disingkat TPPO adalah segala "
    "tindakan perekrutan, pengiriman, pemindahan, penampungan, atau penerimaan "
    "seseorang dengan ancaman kekerasan. "
    "Korban TPPO adalah seseorang yang mengalami penderitaan fisik, mental, "
    "dan seksual akibat TPPO. "
    "Pekerja Migran Indonesia Bermasalah adalah PMI yang mengalami permasalahan. "
    "Rehabilitasi Sosial bertujuan memulihkan fungsi sosial. "
    "Reintegrasi Sosial adalah proses pengembalian ke keluarga. "
    "Dinas Sosial adalah perangkat daerah yang menyelenggarakan urusan sosial. "
    "Kementerian Sosial adalah kementerian yang menyelenggarakan urusan pemerintahan "
    "di bidang sosial. "
    "Asesmen dilakukan untuk menentukan kebutuhan layanan. "
    "Eksploitasi mencakup pelacuran, kerja paksa, perbudakan."
)

# Context with minimal terms (sparse)
SPARSE_CONTEXT = "Penanganan dilakukan oleh pejabat yang berwenang."

# Citations for pasal 5
CITATIONS_PASAL5 = [
    {"pasal": 5, "ayat": 2, "huruf": "a", "raw": "Pasal 5 ayat (2) huruf a",
     "char_start": 0, "char_end": 23},
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEGScore:
    def test_response_with_context_terms_high_score(self):
        """Response heavily using Permensos terms -> EG >= 0.95."""
        response = (
            "Berdasarkan Pasal 5 ayat (2), bentuk eksploitasi terhadap Korban TPPO meliputi: "
            "pelacuran, kerja paksa, perbudakan, perekrutan paksa. "
            "Tindak Pidana Perdagangan Orang diatur secara komprehensif. "
            "Dinas Sosial bertanggung jawab atas Rehabilitasi Sosial dan Reintegrasi Sosial."
        )
        score, debug = compute_eg_score(response, GOOD_CONTEXT, CITATIONS_PASAL5)
        assert score >= 0.95, f"Expected >= 0.95, got {score}. Debug: {debug}"

    def test_response_with_foreign_term_low_score(self):
        """Response injecting 'judicial review' (not in Permensos vocab) -> EG < 0.95."""
        response = (
            "Pasal 5 mengatur judicial review terhadap eksploitasi. "
            "Constitutional court proceedings apply here. "
            "Sebagai tambahan, habeas corpus juga berlaku."
        )
        # Context only has standard terms, none of the injected foreign terms
        score, debug = compute_eg_score(response, SPARSE_CONTEXT, CITATIONS_PASAL5)
        # "judicial review", "constitutional court", "habeas corpus" not in LEGAL_TERMS_PHASE1
        # and not in sparse context -> missing from context -> score < 1.0
        # However since LEGAL_TERMS_PHASE1 terms match on substring, only terms
        # that ARE in LEGAL_TERMS_PHASE1 and ARE in context count.
        # The response has very few LEGAL_TERMS_PHASE1 hits, context has even fewer.
        assert score < 0.95, f"Expected < 0.95 for foreign-term response, got {score}. Debug: {debug}"

    def test_empty_response_returns_one(self):
        """Empty response -> EG = 1.0 (vacuous true, no entities to check)."""
        score, debug = compute_eg_score("", GOOD_CONTEXT, [])
        assert score == 1.0
        assert debug.get("note") == "empty_response"

    def test_whitespace_only_response_returns_one(self):
        """Whitespace-only response -> EG = 1.0."""
        score, debug = compute_eg_score("   \n  ", GOOD_CONTEXT, [])
        assert score == 1.0

    def test_response_no_entities_returns_one(self):
        """Response with no matching terms -> EG = 1.0 (no entities to ground)."""
        response = "Tidak ada informasi yang relevan tersedia."
        score, debug = compute_eg_score(response, GOOD_CONTEXT, [])
        assert score == 1.0
        assert debug.get("note") == "no_entities_in_response"

    def test_score_is_float_in_range(self):
        """EG score always in [0, 1]."""
        response = "TPPO dan Korban TPPO diatur di sini. Judicial review tidak ada."
        score, _ = compute_eg_score(response, GOOD_CONTEXT, [])
        assert 0.0 <= score <= 1.0

    def test_citation_entities_excluded_legal_terms_used(self):
        """EG uses LEGAL_TERMS_PHASE1 only; citation tuples are not entities.

        Design decision: citation strings ("Pasal 5 ayat 2") are citation infrastructure,
        not semantic domain entities. Including them would inflate the denominator without
        corresponding context matches (context extraction also doesn't add them).
        Only LEGAL_TERMS_PHASE1 terms matter for EG grounding.
        """
        response = "Pasal 5 ayat (2) mengatur eksploitasi."
        context_text = "Pasal 5 ayat (2): bentuk eksploitasi adalah pelacuran dan kerja paksa."
        citations = [{"pasal": 5, "ayat": 2, "huruf": None, "raw": "Pasal 5 ayat (2)",
                      "char_start": 0, "char_end": 15}]
        score, debug = compute_eg_score(response, context_text, citations)
        # Citation tuples are NOT in response_entities -- only LEGAL_TERMS_PHASE1 hits
        assert "pasal 5 ayat (2)" not in debug["response_entities"]
        # "eksploitasi" IS in LEGAL_TERMS_PHASE1 -> both response and context have it
        assert "eksploitasi" in debug["response_entities"]
        assert "eksploitasi" in debug["context_entities"]
        # EG = 1.0 since all response entities are grounded in context
        assert score == 1.0

    def test_debug_dict_has_expected_keys(self):
        """debug dict must contain response_entities, context_entities, intersection, missing."""
        score, debug = compute_eg_score("TPPO", GOOD_CONTEXT, [])
        assert "response_entities" in debug
        assert "context_entities" in debug
        assert "intersection" in debug
        assert "missing_from_context" in debug


class TestLegalTermsInventory:
    def test_legal_terms_list_not_empty(self):
        """LEGAL_TERMS_PHASE1 must have entries."""
        assert len(LEGAL_TERMS_PHASE1) > 0

    def test_tppo_in_terms(self):
        """'TPPO' must be in the legal terms list."""
        assert "TPPO" in LEGAL_TERMS_PHASE1

    def test_korban_in_terms(self):
        """'Korban' must be in the legal terms list."""
        assert "Korban" in LEGAL_TERMS_PHASE1

    def test_rehabilitasi_in_terms(self):
        """'Rehabilitasi Sosial' must be in the legal terms list."""
        assert "Rehabilitasi Sosial" in LEGAL_TERMS_PHASE1

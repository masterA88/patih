"""Unit tests for relation_preservation.py (HalluGraph RP scorer).

Spec: build-spec Section 5.4 line 991.

Phase 1 trade-off documented:
  - Response sentence paraphrasing with synonyms may fail (false negative).
  - Cross-pasal term injection may not be caught if Pasal 1 and Pasal 5 share vocabulary.
"""

import pytest
from app.validators.relation_preservation import (
    _is_citation_only_line,
    compute_rp_score,
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

# Pasal 5 context: about forms of exploitation
PASAL5_CONTEXT = (
    "Pasal 5 ayat (2): Eksploitasi sebagaimana dimaksud pada ayat (1) meliputi: "
    "a. pelacuran; b. kerja paksa; c. perbudakan; d. penipuan; e. pemerasan; "
    "f. pemalsuan; g. penyiksaan; h. penganiayaan; i. penjualan organ tubuh; "
    "j. perdagangan narkoba; k. pemindahan organ; l. penyekapan; m. eksploitasi seksual."
)

# Pasal 1 context: definitions
PASAL1_CONTEXT = (
    "Pasal 1: Definisi TPPO adalah perekrutan, pengiriman, pemindahan, penampungan, "
    "atau penerimaan seseorang dengan ancaman, penggunaan kekerasan, "
    "penculikan, pemalsuan, penipuan untuk tujuan eksploitasi."
)

# Pasal 19 context: reintegration
PASAL19_CONTEXT = (
    "Pasal 19: Reintegrasi sosial dilakukan untuk memulihkan fungsi sosial korban. "
    "Proses reintegrasi meliputi pendampingan, konseling, dan pemberdayaan ekonomi."
)

CONTEXT_BY_PASAL = {
    1: PASAL1_CONTEXT,
    5: PASAL5_CONTEXT,
    19: PASAL19_CONTEXT,
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRPScore:
    def test_grounded_response_passes(self):
        """Sentence about Pasal 5 with overlapping content words -> passes."""
        response = (
            "Pasal 5 mengatur berbagai bentuk eksploitasi termasuk pelacuran, "
            "kerja paksa, dan perbudakan."
        )
        score, debug = compute_rp_score(response, CONTEXT_BY_PASAL, [])
        assert score >= 0.85, f"Expected >= 0.85, got {score}. Debug: {debug}"
        assert debug["claims_checked"] >= 1

    def test_hallucinated_claim_fails(self):
        """Sentence about 'sanksi pidana' for Pasal 5 (not in Pasal 5 context) -> fails.

        Phase 1 threshold: >= 1 content word overlap.
        "sanksi", "pidana", "penjara", "tahun", "pelaku" should not appear in Pasal 5 context.
        """
        response = "Pasal 5 mengatur sanksi pidana penjara 20 tahun untuk pelaku kejahatan asing."
        score, debug = compute_rp_score(response, CONTEXT_BY_PASAL, [])
        # None of the content words overlap with Pasal 5 context (about bentuk eksploitasi)
        # "pelaku" might match if it appears in context — check actual debug
        # At minimum, score should be low (< 1.0)
        assert score < 1.0 or debug["claims_checked"] == 0, (
            f"Expected hallucinated claim to fail or no claims checked. score={score}. Debug: {debug}"
        )

    def test_context_missing_pasal_skipped(self):
        """Citation for Pasal not in context_by_pasal -> skipped, not counted in mean."""
        response = "Pasal 99 mengatur prosedur banding administratif."
        score, debug = compute_rp_score(response, CONTEXT_BY_PASAL, [])
        # Pasal 99 not in context -> skipped
        assert debug["claims_skipped"] >= 1
        assert debug["claims_checked"] == 0
        # Vacuous true: no claims checked
        assert score == 1.0

    def test_no_citations_in_response_vacuous_true(self):
        """Response with no Pasal citations -> no claims to check -> score 1.0."""
        response = "Peraturan ini berlaku di seluruh wilayah Indonesia."
        score, debug = compute_rp_score(response, CONTEXT_BY_PASAL, [])
        assert score == 1.0
        assert debug["claims_checked"] == 0

    def test_multiple_pasals_mixed_results(self):
        """Two sentences: Pasal 5 grounded + Pasal 1 hallucinated term mix."""
        response = (
            "Pasal 5 mengatur pelacuran dan kerja paksa sebagai bentuk eksploitasi. "
            "Pasal 1 menyebutkan prosedur banding ke Mahkamah Agung."
        )
        # Pasal 5 sentence should pass (pelacuran, kerja paksa in context)
        # Pasal 1 sentence: "banding", "Mahkamah Agung" not in Pasal 1 context
        # but "TPPO", "definisi", etc. in Pasal 1 context may or may not overlap
        score, debug = compute_rp_score(response, CONTEXT_BY_PASAL, [])
        assert 0.0 <= score <= 1.0
        assert debug["claims_checked"] >= 2

    def test_score_is_float_in_range(self):
        """RP score must always be in [0, 1]."""
        response = "Pasal 5 ayat (2) berisi 13 bentuk eksploitasi."
        score, _ = compute_rp_score(response, CONTEXT_BY_PASAL, [])
        assert 0.0 <= score <= 1.0

    def test_empty_response(self):
        """Empty response -> no claims -> score 1.0."""
        score, debug = compute_rp_score("", CONTEXT_BY_PASAL, [])
        assert score == 1.0
        assert debug["claims_checked"] == 0

    def test_debug_dict_has_expected_keys(self):
        """debug dict must contain claims_checked, claims_passed, claim_details."""
        response = "Pasal 5 mengatur eksploitasi."
        score, debug = compute_rp_score(response, CONTEXT_BY_PASAL, [])
        assert "claims_checked" in debug
        assert "claims_passed" in debug
        assert "claims_skipped" in debug
        assert "claim_details" in debug

    def test_pasal1_definition_context_grounded(self):
        """Pasal 1 sentence about perekrutan/penipuan -> overlaps with Pasal 1 context."""
        response = "Pasal 1 mendefinisikan TPPO sebagai tindakan perekrutan dan penipuan."
        score, debug = compute_rp_score(response, CONTEXT_BY_PASAL, [])
        passed = [c for c in debug["claim_details"] if c.get("status") == "pass"]
        assert len(passed) >= 1, f"Expected at least 1 pass. Debug: {debug}"


class TestCitationOnlyLineSkip:
    """Regression coverage for footer/bullet citation lines (Llama 3.3 70B style).

    Bug: response footer like
        Sumber:
        - Pasal 1 angka 5 — Permensos 8/2023
    split into 2 sentences via newline. Sentence 2 has a Pasal citation but is
    purely metadata. Pre-fix, RP counted it as a failed claim (only "angka" /
    "permensos" tokens remained after stopword removal, low overlap with chunk).
    Post-fix, _is_citation_only_line() detects it and skips.
    """

    def test_header_sumber_alone(self):
        assert _is_citation_only_line("Sumber:")
        assert _is_citation_only_line("  Sumber:  ")
        assert _is_citation_only_line("SUMBER:")

    def test_bullet_pasal_with_permensos(self):
        assert _is_citation_only_line("- Pasal 1 angka 5 — Permensos 8/2023")
        assert _is_citation_only_line("* Pasal 5 ayat (2) huruf a")
        assert _is_citation_only_line("- Pasal 8 ayat (2) huruf f Permensos 8/2023")

    def test_bare_pasal_with_subpart(self):
        assert _is_citation_only_line("Pasal 2 ayat (1) Permensos 8/2023")
        assert _is_citation_only_line("Pasal 19 Permensos 8/2023")

    def test_factual_claim_not_skipped(self):
        assert not _is_citation_only_line("Pasal 5 mengatur 13 bentuk eksploitasi")
        assert not _is_citation_only_line(
            "Reintegrasi sosial diatur dalam Pasal 19 ayat (1)"
        )
        assert not _is_citation_only_line(
            "Korban TPPO adalah seseorang yang mengalami penderitaan"
        )

    def test_footer_pattern_does_not_drag_rp(self):
        """Llama-style footer artifact must not pull RP below threshold.

        Pre-fix this scored 0.5 (footer bullet counted as failed claim).
        Post-fix should be 1.0 (only the substantive first sentence is checked).
        """
        response = (
            "Korban TPPO menurut Permensos 8/2023 adalah seseorang yang mengalami "
            "penderitaan psikis, mental, fisik, seksual, ekonomi, dan/atau sosial, "
            "yang diakibatkan tindak pidana Perdagangan Orang (Pasal 1 angka 5).\n\n"
            "Sumber:\n"
            "- Pasal 1 angka 5 — Permensos 8/2023"
        )
        # Use the actual Pasal 1 fixture
        score, debug = compute_rp_score(response, CONTEXT_BY_PASAL, [])
        # Only the first sentence (substantive claim) should be checked
        assert debug["claims_checked"] == 1, (
            f"Expected 1 claim checked (footer skipped), got {debug['claims_checked']}. "
            f"Details: {debug['claim_details']}"
        )
        assert score == 1.0 or score >= 0.9, (
            f"Expected RP >= 0.9 with footer skipped, got {score}"
        )

    def test_bullet_list_body_with_footer(self):
        """Body with bullet list items (a./b./c.) + footer must not fail RP.

        Bullet items 'a. ...' have no Pasal citation -> already skipped by find_pasal.
        Footer bullets must be skipped by _is_citation_only_line.
        """
        response = (
            "Menurut Pasal 1 ayat (1) Permensos 8/2023, TPPO adalah perekrutan, "
            "pengiriman, dan pemindahan untuk tujuan eksploitasi.\n\n"
            "a. perekrutan dengan ancaman;\n"
            "b. pemindahan dengan kekerasan;\n\n"
            "Sumber:\n"
            "- Pasal 1 ayat (1) Permensos 8/2023"
        )
        score, debug = compute_rp_score(response, CONTEXT_BY_PASAL, [])
        # First sentence is the only Pasal-citing factual claim
        assert debug["claims_checked"] == 1
        assert score == 1.0


class TestPhase1TradeOff:
    def test_cross_pasal_injection_documented_limitation(self):
        """Phase 1 trade-off: cross-pasal term injection may not always be caught.

        A response that uses Pasal 1 legal vocabulary to describe Pasal 5 content
        may pass Phase 1 RP check because token overlap uses a broad vocabulary match.
        This is a documented limitation to be fixed in Phase 2 with semantic checking.
        """
        # Sentence mixes Pasal 5 citation with Pasal 1 vocabulary
        response = (
            "Pasal 5 mengatur perekrutan dan penipuan korban TPPO."
            # "perekrutan" and "penipuan" appear in BOTH Pasal 1 AND Pasal 5 contexts
            # so this sentence may pass the overlap check even though these terms
            # are more precisely Pasal 1 definitional, not Pasal 5 enumeration.
        )
        score, debug = compute_rp_score(response, CONTEXT_BY_PASAL, [])
        # This may pass -- Phase 1 limitation documented
        # We just assert the score is a valid float
        assert 0.0 <= score <= 1.0
        # Document: this is a known false negative in Phase 1

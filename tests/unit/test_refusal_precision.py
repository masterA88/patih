"""Unit tests for app.eval.refusal_precision.

Tests cover:
  - is_refusal() — ID and EN template detection
  - is_refusal() — negative cases (should NOT trigger refusal)
  - compute_refusal_metrics() — TP/FP/FN/TN counting + precision/recall/F1
  - Edge: bilingual EN refusal text
  - Edge: partial template matches
"""

import pytest
from app.eval.refusal_precision import is_refusal, compute_refusal_metrics, REFUSAL_TEMPLATE_ID, REFUSAL_TEMPLATE_EN
from app.eval.testset_loader import EvalQuestion


# ---------------------------------------------------------------------------
# is_refusal tests
# ---------------------------------------------------------------------------

class TestIsRefusalID:
    def test_exact_template_id(self):
        text = "Informasi yang Anda tanyakan tidak diatur secara spesifik dalam Peraturan Menteri Sosial Nomor 8 Tahun 2023."
        assert is_refusal(text) is True

    def test_prefix_match(self):
        text = "Informasi yang Anda tanyakan tidak diatur dalam permensos ini."
        assert is_refusal(text) is True

    def test_lowercase_match(self):
        text = "informasi yang anda tanyakan tidak diatur secara spesifik dalam peraturan ini."
        assert is_refusal(text) is True

    def test_mid_response_match(self):
        text = "Berdasarkan penelaahan saya, hal ini tidak diatur secara spesifik dalam peraturan menteri ini."
        assert is_refusal(text) is True

    def test_variant_topik(self):
        text = "Topik ini tidak diatur dalam permensos 8/2023."
        assert is_refusal(text) is True


class TestIsRefusalEN:
    def test_en_template(self):
        text = "This information is not specifically regulated in Permensos 8/2023."
        assert is_refusal(text) is True

    def test_en_variant(self):
        text = "The matter you asked about is not regulated in this ministerial regulation."
        assert is_refusal(text) is True

    def test_en_not_covered(self):
        text = "This topic is not covered by Permensos 8/2023."
        assert is_refusal(text) is True


class TestIsRefusalNegative:
    def test_normal_answer_not_refusal(self):
        text = "Korban TPPO adalah seseorang yang mengalami penderitaan psikis (Pasal 1 angka 5)."
        assert is_refusal(text) is False

    def test_empty_string(self):
        assert is_refusal("") is False

    def test_partial_word_not_trigger(self):
        # "tidak" alone should not trigger — needs context
        text = "Saya tidak tahu."
        assert is_refusal(text) is False

    def test_answer_mentioning_eksploitasi(self):
        text = "Pasal 5 ayat (2) menyebutkan 13 bentuk eksploitasi termasuk pelacuran dan kerja paksa."
        assert is_refusal(text) is False


# ---------------------------------------------------------------------------
# compute_refusal_metrics tests
# ---------------------------------------------------------------------------

def _make_question(qid: str, must_refuse: bool, tier: int = 1) -> EvalQuestion:
    return EvalQuestion(
        qid=qid,
        tier=tier,
        question_type="out_of_scope" if must_refuse else "definitional",
        question="test question",
        language="id",
        expected_pasal_refs=[],
        expected_answer_summary="summary",
        must_refuse=must_refuse,
    )


class TestComputeRefusalMetrics:
    def test_all_correct_refusals(self):
        """2 Tier 4 questions, both refused correctly."""
        qs = [_make_question("q1", must_refuse=True), _make_question("q2", must_refuse=True)]
        resps = [
            "Informasi yang Anda tanyakan tidak diatur secara spesifik.",
            "Hal ini tidak diatur dalam permensos ini.",
        ]
        m = compute_refusal_metrics(qs, resps)
        assert m["tp"] == 2
        assert m["fp"] == 0
        assert m["fn"] == 0
        assert m["precision"] == 1.0
        assert m["recall"] == 1.0

    def test_all_correct_answers(self):
        """2 Tier 1 questions, both answered without refusal — correct."""
        qs = [_make_question("q1", must_refuse=False), _make_question("q2", must_refuse=False)]
        resps = [
            "Korban TPPO adalah seseorang (Pasal 1 angka 5).",
            "Rehabilitasi sosial adalah proses refungsionalisasi (Pasal 1 angka 6).",
        ]
        m = compute_refusal_metrics(qs, resps)
        assert m["tp"] == 0
        assert m["fp"] == 0
        assert m["tn"] == 2
        assert m["fn"] == 0
        assert m["precision"] == 0.0  # no positive predictions at all; precision undefined → 0
        assert m["recall"] == 0.0

    def test_over_refusal_false_positive(self):
        """T1 question answered with refusal — should be FP."""
        qs = [_make_question("q1", must_refuse=False)]
        resps = ["Informasi yang Anda tanyakan tidak diatur secara spesifik."]
        m = compute_refusal_metrics(qs, resps)
        assert m["fp"] == 1
        assert m["tn"] == 0
        assert m["precision"] == 0.0  # 0 TP out of 1 positive prediction

    def test_failed_to_refuse_false_negative(self):
        """T4 question answered with content — should be FN."""
        qs = [_make_question("q1", must_refuse=True, tier=4)]
        resps = ["Hukuman penjara bagi pelaku TPPO adalah 3-15 tahun."]
        m = compute_refusal_metrics(qs, resps)
        assert m["fn"] == 1
        assert m["tp"] == 0
        assert m["recall"] == 0.0

    def test_mixed_scenario(self):
        """1 TP, 1 FP, 1 FN, 1 TN."""
        qs = [
            _make_question("q1", must_refuse=True, tier=4),   # should refuse
            _make_question("q2", must_refuse=False, tier=1),  # should answer
            _make_question("q3", must_refuse=True, tier=4),   # should refuse
            _make_question("q4", must_refuse=False, tier=1),  # should answer
        ]
        resps = [
            "Tidak diatur secara spesifik dalam permensos ini.",   # TP
            "Tidak diatur secara spesifik dalam permensos ini.",   # FP
            "Hukuman penjara adalah 3 tahun.",                     # FN
            "Korban TPPO adalah seseorang (Pasal 1 angka 5).",    # TN
        ]
        m = compute_refusal_metrics(qs, resps)
        assert m["tp"] == 1
        assert m["fp"] == 1
        assert m["fn"] == 1
        assert m["tn"] == 1
        assert m["precision"] == pytest.approx(0.5, abs=0.001)  # 1/(1+1)
        assert m["recall"] == pytest.approx(0.5, abs=0.001)     # 1/(1+1)

    def test_length_mismatch_raises(self):
        qs = [_make_question("q1", must_refuse=True)]
        resps = []
        with pytest.raises(ValueError, match="same length"):
            compute_refusal_metrics(qs, resps)

    def test_bilingual_en_refusal(self):
        """EN refusal template should be detected."""
        qs = [_make_question("q1", must_refuse=True, tier=4)]
        resps = ["This information is not specifically regulated in Permensos 8/2023."]
        m = compute_refusal_metrics(qs, resps)
        assert m["tp"] == 1

"""Unit tests for app.eval.citation_accuracy.

Tests cover:
  - Vacuous pass (both empty)
  - Perfect match
  - Partial match
  - Zero match (extracted non-empty, expected non-empty, no overlap)
  - Over-citation (extracted non-empty, expected empty)
  - Missing citations (extracted empty, expected non-empty)
  - Granularity: pasal / ayat / huruf
  - Aggregate function
"""

import pytest
from app.eval.citation_accuracy import (
    compute_citation_accuracy,
    aggregate_citation_accuracy,
)


# ---------------------------------------------------------------------------
# compute_citation_accuracy tests
# ---------------------------------------------------------------------------

class TestVacuousPass:
    def test_both_empty_returns_all_ones(self):
        result = compute_citation_accuracy([], [], granularity="pasal")
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0
        assert result["jaccard"] == 1.0
        assert result["matched_count"] == 0

    def test_both_empty_ayat_level(self):
        result = compute_citation_accuracy([], [], granularity="ayat")
        assert result["f1"] == 1.0


class TestPerfectMatch:
    def test_exact_pasal_match(self):
        extracted = [{"pasal": 5, "ayat": 2, "huruf": "a"}]
        expected = [{"pasal": 5, "ayat": 1, "huruf": None}]
        result = compute_citation_accuracy(extracted, expected, granularity="pasal")
        # At pasal level both map to (5,) — perfect match
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0
        assert result["jaccard"] == 1.0
        assert result["matched_count"] == 1

    def test_multi_pasal_exact_match(self):
        extracted = [{"pasal": 1}, {"pasal": 5}, {"pasal": 6}]
        expected = [{"pasal": 1}, {"pasal": 5}, {"pasal": 6}]
        result = compute_citation_accuracy(extracted, expected, granularity="pasal")
        assert result["f1"] == 1.0
        assert result["matched_count"] == 3


class TestPartialMatch:
    def test_one_of_two_matches(self):
        extracted = [{"pasal": 1}, {"pasal": 2}]
        expected = [{"pasal": 1}, {"pasal": 5}]
        result = compute_citation_accuracy(extracted, expected, granularity="pasal")
        assert result["matched_count"] == 1
        assert result["precision"] == pytest.approx(0.5, abs=0.001)
        assert result["recall"] == pytest.approx(0.5, abs=0.001)
        # Jaccard: 1 / 3 = 0.333
        assert result["jaccard"] == pytest.approx(1.0 / 3.0, abs=0.001)


class TestZeroMatch:
    def test_no_overlap(self):
        extracted = [{"pasal": 3}, {"pasal": 4}]
        expected = [{"pasal": 1}, {"pasal": 2}]
        result = compute_citation_accuracy(extracted, expected, granularity="pasal")
        assert result["matched_count"] == 0
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["f1"] == 0.0
        assert result["jaccard"] == 0.0


class TestOverCitation:
    def test_extracted_has_citations_expected_empty(self):
        # Over-citation: extracted cites Pasal 5, but expected_pasal_refs is empty
        # (could happen if model cites something for a Tier 4 question)
        # Documented contract: precision = 0/1 = 0.0; recall = 0/0 = 0.0; jaccard = 0/1 = 0.0
        extracted = [{"pasal": 5}]
        expected = []
        result = compute_citation_accuracy(extracted, expected, granularity="pasal")
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0   # no expected → recall = 0/0 → 0.0
        # Jaccard: 0 / 1 = 0.0
        assert result["jaccard"] == 0.0


class TestMissingCitations:
    def test_extracted_empty_expected_non_empty(self):
        extracted = []
        expected = [{"pasal": 5}]
        result = compute_citation_accuracy(extracted, expected, granularity="pasal")
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["f1"] == 0.0
        assert result["jaccard"] == 0.0


class TestGranularity:
    def test_ayat_level_distinguishes_different_ayat(self):
        extracted = [{"pasal": 5, "ayat": 1}]
        expected = [{"pasal": 5, "ayat": 2}]
        result_pasal = compute_citation_accuracy(extracted, expected, granularity="pasal")
        result_ayat = compute_citation_accuracy(extracted, expected, granularity="ayat")
        # At pasal level: match (both have pasal=5)
        assert result_pasal["matched_count"] == 1
        # At ayat level: no match (ayat 1 != ayat 2)
        assert result_ayat["matched_count"] == 0

    def test_huruf_level(self):
        extracted = [{"pasal": 5, "ayat": 2, "huruf": "a"}]
        expected = [{"pasal": 5, "ayat": 2, "huruf": "b"}]
        result = compute_citation_accuracy(extracted, expected, granularity="huruf")
        assert result["matched_count"] == 0


# ---------------------------------------------------------------------------
# aggregate_citation_accuracy tests
# ---------------------------------------------------------------------------

class TestAggregation:
    def test_empty_list(self):
        agg = aggregate_citation_accuracy([])
        assert agg["micro_precision"] == 0.0
        assert agg["n_questions"] == 0

    def test_single_perfect(self):
        per_q = [
            {"matched_count": 2, "extracted_count": 2, "expected_count": 2, "jaccard": 1.0}
        ]
        agg = aggregate_citation_accuracy(per_q)
        assert agg["micro_precision"] == 1.0
        assert agg["micro_recall"] == 1.0
        assert agg["total_matched"] == 2

    def test_mixed_results(self):
        per_q = [
            {"matched_count": 3, "extracted_count": 3, "expected_count": 3, "jaccard": 1.0},
            {"matched_count": 0, "extracted_count": 2, "expected_count": 2, "jaccard": 0.0},
        ]
        agg = aggregate_citation_accuracy(per_q)
        # micro precision = 3 / 5 = 0.6
        assert agg["micro_precision"] == pytest.approx(0.6, abs=0.001)
        assert agg["n_questions"] == 2
        assert agg["mean_jaccard"] == pytest.approx(0.5, abs=0.001)

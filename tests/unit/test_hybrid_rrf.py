"""
Unit tests for hybrid.py — synthetic rankings to verify RRF score math.

Tests cover:
  - Exact RRF score calculation for known inputs.
  - Ordering: items appearing in both rankings score higher.
  - fuse() convenience wrapper returns correct top_n.
  - Empty input edge case.
  - Single-ranking degenerate case.
"""

from __future__ import annotations

import pytest

from app.retrieval.hybrid import fuse, rrf


class TestRRF:
    def test_score_formula_rank_0(self):
        """Rank 0, k=60 → score = 1/(60+0+1) = 1/61."""
        result = rrf([["a"]], k=60)
        assert len(result) == 1
        chunk_id, score = result[0]
        assert chunk_id == "a"
        assert abs(score - 1.0 / 61) < 1e-9

    def test_score_formula_rank_1(self):
        """Rank 1, k=60 → score = 1/(60+1+1) = 1/62."""
        result = rrf([["a", "b"]], k=60)
        ids = [r[0] for r in result]
        scores = {r[0]: r[1] for r in result}
        assert scores["a"] == pytest.approx(1.0 / 61)
        assert scores["b"] == pytest.approx(1.0 / 62)

    def test_item_in_two_rankings_scores_higher(self):
        """
        "a" in both rankings at rank 0 and rank 0:
           score = 1/61 + 1/61 = 2/61
        "b" only in ranking 1 at rank 1:
           score = 1/62
        "c" only in ranking 2 at rank 1:
           score = 1/62
        "a" must score higher than "b" and "c".
        """
        r1 = ["a", "b"]
        r2 = ["a", "c"]
        result = rrf([r1, r2], k=60)
        scores = {r[0]: r[1] for r in result}

        assert scores["a"] == pytest.approx(2.0 / 61)
        assert scores["b"] == pytest.approx(1.0 / 62)
        assert scores["c"] == pytest.approx(1.0 / 62)

        # Ordering
        ids_ordered = [r[0] for r in result]
        assert ids_ordered[0] == "a"

    def test_output_sorted_descending(self):
        """Output must be sorted by score descending."""
        r1 = ["x", "y", "z"]
        r2 = ["y", "z", "x"]
        result = rrf([r1, r2], k=60)
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)

    def test_empty_rankings(self):
        """Empty input returns empty list."""
        assert rrf([]) == []
        assert rrf([[]]) == []

    def test_single_ranking(self):
        """Single ranking: output order == input order."""
        ranking = ["p1", "p2", "p3"]
        result = rrf([ranking], k=60)
        ids = [r[0] for r in result]
        assert ids == ranking

    def test_all_unique_items(self):
        """Items in different rankings with no overlap."""
        r1 = ["a"]
        r2 = ["b"]
        result = rrf([r1, r2], k=60)
        scores = {r[0]: r[1] for r in result}
        # Both at rank 0 in their respective rankings → same score
        assert scores["a"] == pytest.approx(scores["b"])

    def test_k_parameter(self):
        """k=0 gives score = 1/(0+0+1) = 1 at rank 0."""
        result = rrf([["a"]], k=0)
        assert result[0][1] == pytest.approx(1.0)

    def test_golden_two_rankings_five_items(self):
        """
        Golden test: build-spec example pattern.
        dense = [A, B, C, D, E]  (rank 0..4)
        sparse = [C, A, E, B, D]  (rank 0..4)

        Expected scores (k=60):
          A: 1/61 + 1/62 = 0.016393 + 0.016129 = 0.032522
          B: 1/62 + 1/64 = 0.016129 + 0.015625 = 0.031754
          C: 1/63 + 1/61 = 0.015873 + 0.016393 = 0.032266
          D: 1/64 + 1/65 = 0.015625 + 0.015385 = 0.031010
          E: 1/65 + 1/63 = 0.015385 + 0.015873 = 0.031258
        Order: A > C > B > E > D
        """
        dense = ["A", "B", "C", "D", "E"]
        sparse = ["C", "A", "E", "B", "D"]
        result = rrf([dense, sparse], k=60)
        ids = [r[0] for r in result]
        scores = {r[0]: r[1] for r in result}

        # A has highest score
        assert ids[0] == "A", f"Expected A first, got {ids}"
        # C second
        assert ids[1] == "C", f"Expected C second, got {ids}"
        # D last
        assert ids[-1] == "D", f"Expected D last, got {ids}"
        # Check score values
        assert scores["A"] == pytest.approx(1 / 61 + 1 / 62, abs=1e-6)
        assert scores["C"] == pytest.approx(1 / 63 + 1 / 61, abs=1e-6)


class TestFuse:
    def test_fuse_returns_top_n(self):
        """fuse() truncates to top_n."""
        dense = [("a", 0.1), ("b", 0.2), ("c", 0.3), ("d", 0.4), ("e", 0.5)]
        sparse = [("c", 10.0), ("a", 8.0), ("b", 6.0)]
        result = fuse(dense, sparse, top_n=2)
        assert len(result) == 2

    def test_fuse_ordering(self):
        """Items appearing in both rankings come first."""
        # "overlap" at rank 0 in both → high RRF score
        dense = [("overlap", 0.0), ("dense_only", 0.1)]
        sparse = [("overlap", 5.0), ("sparse_only", 4.0)]
        result = fuse(dense, sparse, top_n=3)
        ids = [r[0] for r in result]
        assert ids[0] == "overlap"

    def test_fuse_empty_sparse(self):
        """fuse() handles empty sparse list gracefully."""
        dense = [("a", 0.1), ("b", 0.2)]
        result = fuse(dense, [], top_n=5)
        ids = [r[0] for r in result]
        # Only dense items present, in order
        assert "a" in ids
        assert "b" in ids

    def test_fuse_empty_dense(self):
        """fuse() handles empty dense list gracefully."""
        sparse = [("x", 10.0), ("y", 5.0)]
        result = fuse([], sparse, top_n=5)
        ids = [r[0] for r in result]
        assert "x" in ids
        assert "y" in ids

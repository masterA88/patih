"""
Reciprocal Rank Fusion (RRF) for hybrid dense + sparse retrieval.

Formula (per build-spec Section 5.2 line 708-716):
    score(d) = sum over rankings: 1 / (k + rank + 1)
    where rank is 0-indexed.

k=60 is the standard RRF hyperparameter (Cormack et al. 2009).

See build-spec Section 5.2 line 707-716.
"""

from __future__ import annotations


def rrf(
    rankings: list[list[str]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """
    Reciprocal Rank Fusion over multiple ranked lists.

    Args:
        rankings: list of ranked lists of chunk_id strings.
                  Each list is ordered best-to-worst (index 0 = rank 0 = highest score).
        k:        RRF constant (default 60, per Cormack et al.).

    Returns:
        list of (chunk_id, rrf_score) sorted descending by rrf_score.
        All unique chunk_ids from all rankings are included.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])


def fuse(
    dense_results: list[tuple[str, float]],
    sparse_results: list[tuple[str, float]],
    top_n: int = 8,
    k: int = 60,
) -> list[tuple[str, float]]:
    """
    Convenience wrapper: fuse dense + sparse results via RRF, return top_n.

    Args:
        dense_results:  list of (chunk_id, distance) from Chroma (lower = better).
                        Already sorted ascending by distance from query_dense().
        sparse_results: list of (chunk_id, bm25_score) from BM25Store (higher = better).
                        Already sorted descending by score from query_sparse().
        top_n:          number of fused results to return.
        k:              RRF constant.

    Returns:
        list of (chunk_id, rrf_score), top_n items, descending by rrf_score.

    Note on ordering convention:
        dense_results is sorted ascending by cosine distance (best = index 0).
        sparse_results is sorted descending by BM25 score (best = index 0).
        Both have index 0 = best, which is what rrf() expects.
    """
    dense_ranking = [cid for cid, _ in dense_results]
    sparse_ranking = [cid for cid, _ in sparse_results]

    fused = rrf([dense_ranking, sparse_ranking], k=k)
    return fused[:top_n]

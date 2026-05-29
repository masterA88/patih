"""Custom metric: citation accuracy — Jaccard(extracted, expected) at Pasal/ayat/huruf level.

Per build-spec Section 5.5 lines 1014-1016.

Edge-case contract (documented for clarity):
    - expected empty AND extracted empty  → score 1.0  (vacuous pass; e.g. Tier 4 correct refusal)
    - expected non-empty AND extracted empty → score 0.0  (model produced no citations)
    - expected empty AND extracted non-empty → precision=0.0, recall=1.0  (over-citation)
    - Granularity 'pasal': key = (pasal,)
    - Granularity 'ayat':  key = (pasal, ayat)  — None treated as wildcard match
    - Granularity 'huruf': key = (pasal, ayat, huruf)
"""

from __future__ import annotations

import logging
from typing import Literal

logger = logging.getLogger(__name__)

GranularityT = Literal["pasal", "ayat", "huruf"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_key(ref: dict, granularity: GranularityT) -> tuple:
    """Convert a pasal-ref dict to a hashable tuple at requested granularity."""
    pasal = ref.get("pasal")
    if granularity == "pasal":
        return (pasal,)
    ayat = ref.get("ayat")
    if granularity == "ayat":
        return (pasal, ayat)
    huruf = ref.get("huruf")
    return (pasal, ayat, huruf)


def _to_keyset(refs: list[dict], granularity: GranularityT) -> set[tuple]:
    """Convert list of ref dicts to a set of granularity-appropriate tuples.

    None values are preserved; callers must decide how to handle wildcards.
    """
    return {_to_key(r, granularity) for r in refs if r.get("pasal") is not None}


def _jaccard(a: set, b: set) -> float:
    """Standard Jaccard similarity: |A ∩ B| / |A ∪ B|.  Returns 1.0 for empty ∩ empty."""
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


# ---------------------------------------------------------------------------
# Per-question metric
# ---------------------------------------------------------------------------

def compute_citation_accuracy(
    extracted: list[dict],
    expected: list[dict],
    granularity: GranularityT = "pasal",
) -> dict:
    """Compute citation accuracy between extracted and expected Pasal references.

    Args:
        extracted:    Citations extracted from the model response
                      (each dict has 'pasal', optionally 'ayat', 'huruf').
        expected:     Ground-truth citations from the test set.
        granularity:  Level of comparison: 'pasal', 'ayat', or 'huruf'.

    Returns:
        {
            "precision": float,       # |extracted ∩ expected| / |extracted|
            "recall": float,          # |extracted ∩ expected| / |expected|
            "f1": float,
            "jaccard": float,
            "matched_count": int,
            "extracted_count": int,
            "expected_count": int,
            "granularity": str,
        }
    """
    ext_set = _to_keyset(extracted, granularity)
    exp_set = _to_keyset(expected, granularity)

    matched = ext_set & exp_set
    matched_count = len(matched)
    n_extracted = len(ext_set)
    n_expected = len(exp_set)

    # --- Vacuous pass: both empty ---
    if n_extracted == 0 and n_expected == 0:
        return {
            "precision": 1.0, "recall": 1.0, "f1": 1.0, "jaccard": 1.0,
            "matched_count": 0, "extracted_count": 0, "expected_count": 0,
            "granularity": granularity,
        }

    precision = matched_count / n_extracted if n_extracted > 0 else 0.0
    recall = matched_count / n_expected if n_expected > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    jaccard = _jaccard(ext_set, exp_set)

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "jaccard": round(jaccard, 4),
        "matched_count": matched_count,
        "extracted_count": n_extracted,
        "expected_count": n_expected,
        "granularity": granularity,
    }


# ---------------------------------------------------------------------------
# Aggregate across a list of per-question results
# ---------------------------------------------------------------------------

def aggregate_citation_accuracy(per_question: list[dict]) -> dict:
    """Micro-average precision/recall/F1 over all questions.

    Args:
        per_question: List of dicts returned by compute_citation_accuracy().

    Returns:
        {
            "micro_precision": float,
            "micro_recall": float,
            "micro_f1": float,
            "mean_jaccard": float,
            "total_matched": int,
            "total_extracted": int,
            "total_expected": int,
            "n_questions": int,
        }
    """
    if not per_question:
        return {
            "micro_precision": 0.0, "micro_recall": 0.0, "micro_f1": 0.0,
            "mean_jaccard": 0.0, "total_matched": 0, "total_extracted": 0,
            "total_expected": 0, "n_questions": 0,
        }

    total_matched = sum(r["matched_count"] for r in per_question)
    total_extracted = sum(r["extracted_count"] for r in per_question)
    total_expected = sum(r["expected_count"] for r in per_question)

    micro_precision = total_matched / total_extracted if total_extracted > 0 else 0.0
    micro_recall = total_matched / total_expected if total_expected > 0 else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall) > 0
        else 0.0
    )
    mean_jaccard = sum(r["jaccard"] for r in per_question) / len(per_question)

    return {
        "micro_precision": round(micro_precision, 4),
        "micro_recall": round(micro_recall, 4),
        "micro_f1": round(micro_f1, 4),
        "mean_jaccard": round(mean_jaccard, 4),
        "total_matched": total_matched,
        "total_extracted": total_extracted,
        "total_expected": total_expected,
        "n_questions": len(per_question),
    }

"""Refusal precision metric for out-of-scope (Tier 4) and adversarial (Tier 5) questions.

Per build-spec Section 5.5 lines 1018-1022.

Terminology:
    True positive  (TP): must_refuse=True  AND response is a refusal.
    False positive (FP): must_refuse=False AND response is a refusal  (over-refusal).
    False negative (FN): must_refuse=True  AND response is NOT a refusal (failed to refuse).
    True negative  (TN): must_refuse=False AND response is NOT a refusal (correct answer).

Refusal detection:
    Substring match against known refusal templates (ID + EN) from:
    - app/validators/pipeline.py _REFUSAL_PREFIX / _REFUSAL_PHRASE
    - configs/prompts/system_id.md rule 4
    - configs/prompts/system_bilingual.md
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.eval.testset_loader import EvalQuestion

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Refusal templates
# ---------------------------------------------------------------------------

# Spec-mandated template IDs (build-spec line 1018-1019):
REFUSAL_TEMPLATE_ID = "tidak diatur secara spesifik dalam"
REFUSAL_TEMPLATE_EN = "is not specifically regulated in"

# Additional variants observed from validator pipeline (system_id.md rule 4):
_REFUSAL_PREFIXES_ID = [
    "informasi yang anda tanyakan tidak diatur",
    "pertanyaan anda tidak diatur",
    "hal ini tidak diatur",
    "topik ini tidak diatur",
    "tidak diatur secara spesifik",
    "tidak diatur dalam permensos",
    "tidak diatur dalam peraturan",
]

_REFUSAL_PATTERNS_EN = [
    "is not specifically regulated in",
    "is not regulated in",
    "not covered by",
    "this topic is not",
    "this information is not",
]


def is_refusal(response: str) -> bool:
    """Return True if *response* matches any known refusal template.

    Uses substring match (case-insensitive) against both Indonesian and
    English templates.  Intentionally broad — false negatives (missed
    refusals) are worse than false positives in the eval context.
    """
    lower = response.strip().lower()

    for prefix in _REFUSAL_PREFIXES_ID:
        if prefix in lower:
            return True

    for pattern in _REFUSAL_PATTERNS_EN:
        if pattern in lower:
            return True

    # Spec-mandated direct checks
    if REFUSAL_TEMPLATE_ID in lower:
        return True
    if REFUSAL_TEMPLATE_EN in lower:
        return True

    return False


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

def compute_refusal_metrics(
    questions: list["EvalQuestion"],
    responses: list[str],
) -> dict:
    """Compute precision/recall/F1 for the refusal class.

    Args:
        questions: List of EvalQuestion objects (must have .must_refuse and .qid).
        responses: Parallel list of model response strings.

    Returns:
        {
            "precision": float,
            "recall": float,
            "f1": float,
            "tp": int,
            "fp": int,
            "fn": int,
            "tn": int,
            "n_should_refuse": int,
            "n_should_answer": int,
            "per_question": [{"qid", "must_refuse", "is_refusal", "correct"}, ...],
        }
    """
    if len(questions) != len(responses):
        raise ValueError(
            f"questions ({len(questions)}) and responses ({len(responses)}) "
            "must have the same length."
        )

    tp = fp = fn = tn = 0
    per_question = []

    for q, resp in zip(questions, responses):
        predicted_refusal = is_refusal(resp)
        correct = (q.must_refuse == predicted_refusal)

        if q.must_refuse and predicted_refusal:
            tp += 1
        elif not q.must_refuse and predicted_refusal:
            fp += 1
        elif q.must_refuse and not predicted_refusal:
            fn += 1
        else:
            tn += 1

        per_question.append({
            "qid": q.qid,
            "must_refuse": q.must_refuse,
            "predicted_refusal": predicted_refusal,
            "correct": correct,
        })

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    n_should_refuse = sum(1 for q in questions if q.must_refuse)
    n_should_answer = len(questions) - n_should_refuse

    logger.info(
        "Refusal metrics: precision=%.3f recall=%.3f f1=%.3f (TP=%d FP=%d FN=%d TN=%d)",
        precision, recall, f1, tp, fp, fn, tn,
    )

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "n_should_refuse": n_should_refuse,
        "n_should_answer": n_should_answer,
        "per_question": per_question,
    }

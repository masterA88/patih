"""Load and validate JSONL test sets (Tier 1-5 Q-A pairs) for evaluation runs.

Schema per build-spec Section 4.4 lines 578-594.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EvalQuestion schema
# ---------------------------------------------------------------------------

class EvalQuestion(BaseModel):
    """Single test question from the evaluation corpus.

    qid format: P8-T{tier}-{seq:03d}  e.g. "P8-T1-001"
    """

    qid: str
    tier: Literal[1, 2, 3, 4, 5]
    question_type: Literal[
        "pasal_extraction",
        "definitional",
        "cross_reference",
        "procedure",
        "thematic",
        "out_of_scope",
        "adversarial_leading",
        "adversarial_fact_injection",
    ]
    question: str
    language: Literal["id", "en"]
    expected_pasal_refs: list[dict]   # [{"pasal": 5, "ayat": 2, "huruf": "b"}, ...]
    expected_answer_summary: str      # 1-2 sentence reference for RAGAS judge
    must_refuse: bool = False
    notes: str | None = None


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_testset(path: Path) -> list[EvalQuestion]:
    """Load JSONL test set from *path*.  Validates each row against EvalQuestion schema.

    Args:
        path: Path to a .jsonl file (one JSON object per line).

    Returns:
        List of validated EvalQuestion objects.

    Raises:
        FileNotFoundError: if path does not exist.
        ValueError:        if any row fails Pydantic validation (includes row index).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Test set not found: {path}")

    questions: list[EvalQuestion] = []
    errors: list[str] = []

    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"Line {lineno}: JSON parse error — {exc}")
                continue
            try:
                questions.append(EvalQuestion.model_validate(raw))
            except ValidationError as exc:
                qid = raw.get("qid", "<no-qid>")
                errors.append(f"Line {lineno} ({qid}): schema error — {exc}")

    if errors:
        raise ValueError(
            f"Test set {path} has {len(errors)} validation error(s):\n"
            + "\n".join(errors)
        )

    logger.info("Loaded %d questions from %s", len(questions), path)
    return questions


def tier_distribution(questions: list[EvalQuestion]) -> dict[int, int]:
    """Return {tier: count} dict for reporting."""
    dist: dict[int, int] = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    for q in questions:
        dist[q.tier] += 1
    return dist

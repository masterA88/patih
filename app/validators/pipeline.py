"""Composite validator pipeline: citation check + entity grounding + relation preservation.

Spec: build-spec Section 5.4.

Public interface:
    from app.validators.pipeline import validate, ValidationResult

    result = validate(
        response="...",
        context=[{"pasal": 5, "text": "...", ...}],  # parent_chunks
        doc_id="permensos-8-2023",
    )
    if result.hitl_flag:
        print("Needs human review:", result.hitl_reasons)

HITL thresholds (spec line 972-977):
    hitl_flag = True if ANY of:
      - any invalid citation
      - eg_score < 0.95
      - rp_score < 0.85

Refusal detection:
    If response matches the system prompt's refusal template (rule 4 in system_id.md),
    skip citation + grounding checks — no citations are expected for refusal responses.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from app.validators.citation_extractor import extract_citations
from app.validators.entity_grounding import compute_eg_score
from app.validators.relation_preservation import compute_rp_score

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_DEFAULT_DOC_REGISTRY_PATH = Path("data/registry/documents.json")
_DEFAULT_HITL_QUEUE_PATH = Path("data/hitl_queue.jsonl")
_DEFAULT_DOC_ID = "permensos-8-2023"

# ---------------------------------------------------------------------------
# Refusal detection
# ---------------------------------------------------------------------------
# Rule 4 in configs/prompts/system_id.md: exact refusal phrase.
# We strip and lower both sides for comparison.
_REFUSAL_PHRASE = (
    "informasi yang anda tanyakan tidak diatur secara spesifik "
    "dalam peraturan menteri sosial nomor 8 tahun 2023"
)

# Alternative shorter prefix (LLM may vary the suffix)
_REFUSAL_PREFIX = "informasi yang anda tanyakan tidak diatur"


def _is_refusal(response: str) -> bool:
    """Return True if response is a refusal (no citation expected)."""
    r = response.strip().lower()
    return r.startswith(_REFUSAL_PREFIX) or _REFUSAL_PHRASE in r


# ---------------------------------------------------------------------------
# Validation result
# ---------------------------------------------------------------------------

class ValidationResult(BaseModel):
    """Output of validate(). Spec Section 5.4."""

    citations_extracted: list[dict]      # [{pasal, ayat, huruf, raw, char_start, char_end}]
    citations_valid: list[bool]          # parallel to citations_extracted
    citation_accuracy: float             # |valid| / |extracted|, 1.0 if no citations
    eg_score: float                      # 0-1
    rp_score: float                      # 0-1
    hitl_flag: bool
    hitl_reasons: list[str]              # subset of {"invalid_citation", "low_eg", "low_rp"}
    debug: dict[str, Any]
    is_refusal: bool = False             # True if response was a refusal (no validation)

    model_config = {"arbitrary_types_allowed": True}


# ---------------------------------------------------------------------------
# Registry loader (cached at module level after first load)
# ---------------------------------------------------------------------------
_doc_registry_cache: dict[str, Any] | None = None


def _load_doc_registry(path: Path = _DEFAULT_DOC_REGISTRY_PATH) -> dict[str, Any]:
    global _doc_registry_cache
    if _doc_registry_cache is None:
        try:
            with open(path, "r", encoding="utf-8") as f:
                _doc_registry_cache = json.load(f)
        except FileNotFoundError:
            logger.warning("Doc registry not found at %s; using empty registry.", path)
            _doc_registry_cache = {}
    return _doc_registry_cache


# ---------------------------------------------------------------------------
# Chroma collection getter (lazy, cached)
# ---------------------------------------------------------------------------
_chroma_collection_cache = None


def _get_chroma_collection():
    global _chroma_collection_cache
    if _chroma_collection_cache is None:
        from app.retrieval.vector_store import get_collection
        _chroma_collection_cache = get_collection()
    return _chroma_collection_cache


# ---------------------------------------------------------------------------
# Main validate() function
# ---------------------------------------------------------------------------

def validate(
    response: str,
    context: list[dict[str, Any]],  # parent_chunks from RetrievalResult
    doc_id: str = _DEFAULT_DOC_ID,
) -> ValidationResult:
    """End-to-end Layer 2 validation.

    Args:
        response:  LLM-generated response text.
        context:   List of parent_chunk dicts from retrieval pipeline.
                   Each dict must have 'text' and 'pasal' keys.
        doc_id:    Document identifier matching data/registry/documents.json.

    Returns:
        ValidationResult with all scores and hitl_flag.
    """
    t_start = time.monotonic()

    # --- Refusal detection ---
    if _is_refusal(response):
        debug: dict[str, Any] = {"elapsed_ms": 0.0, "note": "refusal_detected"}
        elapsed = (time.monotonic() - t_start) * 1000
        debug["elapsed_ms"] = round(elapsed, 1)
        return ValidationResult(
            citations_extracted=[],
            citations_valid=[],
            citation_accuracy=1.0,
            eg_score=1.0,
            rp_score=1.0,
            hitl_flag=False,
            hitl_reasons=[],
            debug=debug,
            is_refusal=True,
        )

    # --- Step 1: Citation extraction ---
    citations = extract_citations(response)

    # --- Step 2: Whitelist validation ---
    if citations:
        from app.validators.whitelist_validator import validate_citations
        doc_registry = _load_doc_registry()
        chroma_collection = _get_chroma_collection()
        # Multi-doc: candidate docs = those present in the retrieved context.
        # A citation is valid if found in ANY of them. Falls back to doc_id if
        # the context carries no doc_id metadata.
        context_doc_ids = sorted(
            {c.get("doc_id") for c in context if c.get("doc_id")}
        )
        candidate_docs = context_doc_ids or [doc_id]
        citations_valid = validate_citations(
            citations=citations,
            doc_id=candidate_docs,
            chroma_collection=chroma_collection,
            doc_registry=doc_registry,
        )
    else:
        citations_valid = []

    # --- Citation accuracy ---
    if not citations:
        citation_accuracy = 1.0  # vacuous true
    else:
        citation_accuracy = sum(citations_valid) / len(citations_valid)

    # --- Step 3: Entity Grounding ---
    context_text = "\n".join(chunk.get("text", "") for chunk in context)
    eg_score, eg_debug = compute_eg_score(
        response=response,
        context_text=context_text,
        extracted_citations=citations,
    )

    # --- Step 4: Relation Preservation ---
    # Build context_by_pasal: {pasal_num: text}
    context_by_pasal: dict[int, str] = {}
    for chunk in context:
        pasal = chunk.get("pasal")
        text = chunk.get("text", "")
        if isinstance(pasal, int) and text:
            # If multiple chunks for same pasal, concatenate
            if pasal in context_by_pasal:
                context_by_pasal[pasal] += "\n" + text
            else:
                context_by_pasal[pasal] = text

    rp_score, rp_debug = compute_rp_score(
        response=response,
        context_by_pasal=context_by_pasal,
        extracted_citations=citations,
    )

    # --- Step 5: Threshold gate ---
    hitl_reasons: list[str] = []

    if citations and any(not v for v in citations_valid):
        hitl_reasons.append("invalid_citation")

    if eg_score < 0.95:
        hitl_reasons.append("low_eg")

    if rp_score < 0.85:
        hitl_reasons.append("low_rp")

    hitl_flag = bool(hitl_reasons)

    elapsed_ms = round((time.monotonic() - t_start) * 1000, 1)

    debug = {
        "elapsed_ms": elapsed_ms,
        "n_citations": len(citations),
        "eg": eg_debug,
        "rp": rp_debug,
    }

    logger.info(
        "Validation done in %.0fms: citation_accuracy=%.2f, eg=%.2f, rp=%.2f, hitl=%s",
        elapsed_ms, citation_accuracy, eg_score, rp_score, hitl_flag,
    )

    return ValidationResult(
        citations_extracted=citations,
        citations_valid=citations_valid,
        citation_accuracy=citation_accuracy,
        eg_score=eg_score,
        rp_score=rp_score,
        hitl_flag=hitl_flag,
        hitl_reasons=hitl_reasons,
        debug=debug,
        is_refusal=False,
    )


# ---------------------------------------------------------------------------
# HITL queue writer
# ---------------------------------------------------------------------------

def append_to_hitl_queue(
    validation_result: ValidationResult,
    response: str,
    user_query: str,
    retrieved_pasals: list[int | str] | None = None,
    queue_path: Path = _DEFAULT_HITL_QUEUE_PATH,
) -> None:
    """Append a JSONL record to the HITL review queue when hitl_flag=True.

    Spec line 979: Phase 1 = simple data/hitl_queue.jsonl.

    Schema per line:
        {
            "ts":                  ISO-8601 UTC timestamp,
            "user_query":          str,
            "response":            str,
            "validation_result":   ValidationResult.model_dump(),
            "retrieved_pasals":    list[int|str] | null,
        }
    """
    if not validation_result.hitl_flag:
        return

    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "user_query": user_query,
        "response": response,
        "validation_result": validation_result.model_dump(),
        "retrieved_pasals": retrieved_pasals or [],
    }

    try:
        queue_path.parent.mkdir(parents=True, exist_ok=True)
        with open(queue_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info("HITL: appended record to %s (reasons=%s)", queue_path, validation_result.hitl_reasons)
    except OSError as exc:
        logger.error("HITL queue write failed: %s", exc)

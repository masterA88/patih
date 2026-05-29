"""RAGAS evaluation runner: Faithfulness, AnswerRelevancy, ContextPrecision.

Per build-spec Section 5.5 lines 1009-1012.

RAGAS 0.2.6 API used:
    from ragas import evaluate, EvaluationDataset, SingleTurnSample
    - evaluate(dataset: EvaluationDataset, metrics, llm, embeddings, run_config)
    - LLM wrapped via LangchainLLMWrapper(ChatLiteLLM(...))
    - Embeddings via LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(...))

Judge throttle:
    ChatLiteLLM request_timeout + explicit per-batch sleep to stay under
    Groq free tier 30 RPM limit.  Default throttle_rpm=25 ← safe margin.
    (Previous default was 8 RPM tuned for Gemini 10 RPM; Groq allows faster.)

Cache:
    SQLite at cache_path.  Key: sha256(judge_model + prompt_repr).
    Serialize RAGAS response as JSON.  TTL: 30 days.
    Cache is optional — if disabled/unavailable eval still runs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Suppress ragas/langchain deprecation noise in eval output
warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

_CACHE_TTL_SECONDS = 30 * 24 * 3600  # 30 days


def _cache_key(judge_model: str, records: list[dict]) -> str:
    """Deterministic sha256 key over (model, serialised records)."""
    payload = judge_model + json.dumps(records, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _open_cache(cache_path: Path) -> sqlite3.Connection | None:
    """Open (or create) SQLite cache.  Returns None on failure — caller handles gracefully."""
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(cache_path), check_same_thread=False)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS ragas_cache ("
            "  cache_key TEXT PRIMARY KEY,"
            "  result_json TEXT NOT NULL,"
            "  created_ts INTEGER NOT NULL"
            ")"
        )
        conn.commit()
        return conn
    except Exception as exc:
        logger.warning("RAGAS cache unavailable (%s); running without cache.", exc)
        return None


def _cache_get(conn: sqlite3.Connection | None, key: str) -> dict | None:
    if conn is None:
        return None
    cutoff = int(time.time()) - _CACHE_TTL_SECONDS
    row = conn.execute(
        "SELECT result_json FROM ragas_cache WHERE cache_key=? AND created_ts>=?",
        (key, cutoff),
    ).fetchone()
    return json.loads(row[0]) if row else None


def _cache_set(conn: sqlite3.Connection | None, key: str, result: dict) -> None:
    if conn is None:
        return
    try:
        conn.execute(
            "INSERT OR REPLACE INTO ragas_cache(cache_key, result_json, created_ts) "
            "VALUES (?, ?, ?)",
            (key, json.dumps(result, ensure_ascii=False), int(time.time())),
        )
        conn.commit()
    except Exception as exc:
        logger.warning("RAGAS cache write failed: %s", exc)


# ---------------------------------------------------------------------------
# LLM / embedding builders
# ---------------------------------------------------------------------------

def _build_ragas_llm(judge_model: str, throttle_rpm: int):
    """Build a RAGAS-compatible LLM wrapper using ChatLiteLLM + LangchainLLMWrapper."""
    from langchain_community.chat_models import ChatLiteLLM
    from ragas.llms import LangchainLLMWrapper

    # request_timeout = 60s; let ragas RunConfig handle retries
    llm = ChatLiteLLM(
        model=judge_model,
        temperature=0.0,
        request_timeout=60,
    )
    return LangchainLLMWrapper(llm)


def _build_ragas_embeddings():
    """Build a RAGAS-compatible embeddings wrapper using HuggingFace sentence-transformers."""
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return LangchainEmbeddingsWrapper(emb)


# ---------------------------------------------------------------------------
# Main eval function
# ---------------------------------------------------------------------------

def run_ragas(
    eval_records: list[dict],
    judge_model: str = "groq/llama-3.3-70b-versatile",
    throttle_rpm: int = 25,
    cache_path: Path = Path("data/eval_cache/ragas_judge.db"),
) -> dict:
    """Run RAGAS evaluation over *eval_records*.

    Args:
        eval_records:  List of dicts.  Each dict must have:
                         - "question":    str
                         - "answer":      str
                         - "contexts":    list[str]   (retrieved parent chunk texts)
                         - "ground_truth": str         (expected_answer_summary)
                       Optional: "qid" for logging.
        judge_model:   LiteLLM model string for the judge LLM.
        throttle_rpm:  Max requests per minute to send to judge.  Default 8.
        cache_path:    SQLite cache path.  Relative to cwd or absolute.

    Returns:
        {
            "per_record": [
                {
                    "qid": str,
                    "faithfulness": float | None,
                    "answer_relevancy": float | None,
                    "context_precision": float | None,
                }
            ],
            "aggregate": {
                "faithfulness_mean": float,
                "answer_relevancy_mean": float,
                "context_precision_mean": float,
                "n_evaluated": int,
                "n_failed": int,
            },
            "judge_model": str,
            "cache_used": bool,
        }
    """
    from ragas import evaluate, EvaluationDataset, SingleTurnSample, RunConfig
    from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision

    cache_path = Path(cache_path)
    conn = _open_cache(cache_path)

    # --- Cache lookup (whole batch) ---
    cache_key = _cache_key(judge_model, eval_records)
    cached = _cache_get(conn, cache_key)
    if cached is not None:
        logger.info("RAGAS: cache hit for %d records (key=%s...)", len(eval_records), cache_key[:12])
        cached["cache_used"] = True
        return cached

    logger.info("RAGAS: evaluating %d records with judge=%s throttle=%d RPM",
                len(eval_records), judge_model, throttle_rpm)

    # --- Build RAGAS dataset ---
    samples = []
    for rec in eval_records:
        contexts = rec.get("contexts") or []
        if not isinstance(contexts, list):
            contexts = [str(contexts)]
        # Flatten: each context element must be a plain string
        flat_contexts = [str(c) for c in contexts if c]
        samples.append(
            SingleTurnSample(
                user_input=rec.get("question", ""),
                response=rec.get("answer", ""),
                retrieved_contexts=flat_contexts,
                reference=rec.get("ground_truth", ""),
            )
        )

    dataset = EvaluationDataset(samples=samples)

    # --- Build LLM + embeddings ---
    try:
        ragas_llm = _build_ragas_llm(judge_model, throttle_rpm)
        ragas_emb = _build_ragas_embeddings()
    except Exception as exc:
        logger.error("RAGAS: failed to build LLM/embeddings wrapper: %s", exc)
        raise

    # --- Throttle: derive max_workers from RPM ---
    # One request per (60 / throttle_rpm) seconds.  Use 1 worker to guarantee order.
    seconds_per_request = 60.0 / throttle_rpm
    run_config = RunConfig(
        timeout=120,
        max_retries=3,
        max_wait=60,
        max_workers=1,   # Serial execution for rate-limit safety
    )

    metrics = [Faithfulness(), AnswerRelevancy(), ContextPrecision()]

    # --- Run ---
    per_record_results = []
    n_failed = 0

    # Process in micro-batches to respect throttle_rpm.
    # With max_workers=1 and sleep between records we stay well below limit.
    BATCH_SIZE = 5  # Process 5 at a time, sleep between batches

    for batch_start in range(0, len(samples), BATCH_SIZE):
        batch_samples = samples[batch_start: batch_start + BATCH_SIZE]
        batch_records = eval_records[batch_start: batch_start + BATCH_SIZE]
        batch_dataset = EvaluationDataset(samples=batch_samples)

        try:
            result = evaluate(
                dataset=batch_dataset,
                metrics=metrics,
                llm=ragas_llm,
                embeddings=ragas_emb,
                run_config=run_config,
                raise_exceptions=False,
                show_progress=False,
            )
            # result is EvaluationResult — iterate as DataFrame or to_pandas
            result_df = result.to_pandas()

            for i, row in result_df.iterrows():
                qid = batch_records[i].get("qid", f"q{batch_start + i}")
                per_record_results.append({
                    "qid": qid,
                    "faithfulness": _safe_float(row.get("faithfulness")),
                    "answer_relevancy": _safe_float(row.get("answer_relevancy")),
                    "context_precision": _safe_float(row.get("context_precision")),
                })

        except Exception as exc:
            logger.error("RAGAS batch [%d:%d] failed: %s", batch_start,
                         batch_start + len(batch_samples), exc)
            for i, rec in enumerate(batch_records):
                qid = rec.get("qid", f"q{batch_start + i}")
                per_record_results.append({
                    "qid": qid,
                    "faithfulness": None,
                    "answer_relevancy": None,
                    "context_precision": None,
                    "error": str(exc),
                })
                n_failed += 1

        # Throttle sleep between batches
        if batch_start + BATCH_SIZE < len(samples):
            sleep_s = seconds_per_request * len(batch_samples)
            logger.debug("RAGAS: sleeping %.1fs (throttle %d RPM)", sleep_s, throttle_rpm)
            time.sleep(sleep_s)

    # --- Aggregate ---
    def _mean_valid(key: str) -> float:
        vals = [r[key] for r in per_record_results if r.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    aggregate = {
        "faithfulness_mean": _mean_valid("faithfulness"),
        "answer_relevancy_mean": _mean_valid("answer_relevancy"),
        "context_precision_mean": _mean_valid("context_precision"),
        "n_evaluated": len(per_record_results),
        "n_failed": n_failed,
    }

    output = {
        "per_record": per_record_results,
        "aggregate": aggregate,
        "judge_model": judge_model,
        "cache_used": False,
    }

    # --- Persist to cache ---
    _cache_set(conn, cache_key, output)
    logger.info(
        "RAGAS done: faithfulness=%.3f answer_relevancy=%.3f context_precision=%.3f",
        aggregate["faithfulness_mean"],
        aggregate["answer_relevancy_mean"],
        aggregate["context_precision_mean"],
    )
    return output


def _safe_float(val: Any) -> float | None:
    """Convert a value to float, returning None on failure or NaN.

    NaN must be coerced to None so downstream aggregation (_mean_valid) skips it;
    otherwise pandas NaN (a float) leaks through `is not None` and contaminates the
    mean.
    """
    import math
    if val is None:
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    if math.isnan(f):
        return None
    return f

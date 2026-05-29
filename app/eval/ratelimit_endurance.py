"""Rate-limit endurance stress test: steady-RPM query harness.

Per build-spec Section 5.5 lines 1026-1032.

Usage:
    python -m app.eval.cli endurance \
        --queries data/test/endurance_100q.jsonl \
        --duration-min 30 \
        --target-rpm 4

Fires queries at a steady *target_rpm* rate for *duration_min* minutes,
measuring success rate, P50/P95/P99 latency, fallback-chain transition count,
and hard failure count.

Output: JSON written to reports/endurance_<timestamp>.json
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single-query runner (wraps Generator)
# ---------------------------------------------------------------------------

def _run_single_query(query: str, generator: Any) -> dict:
    """Fire one query through Generator.answer() and return a timing record."""
    t_start = time.monotonic()
    success = False
    error_msg: str | None = None
    result: Any = None
    fallback_transitions = 0

    try:
        result = generator.answer(query, validate=False)
        success = True
        fallback_transitions = max(0, len(result.fallback_chain_attempts) - 1)
    except Exception as exc:
        error_msg = str(exc)
        logger.warning("Endurance query failed: %s", exc)

    elapsed_ms = (time.monotonic() - t_start) * 1000

    record: dict = {
        "query": query[:80],
        "success": success,
        "elapsed_ms": round(elapsed_ms, 1),
        "fallback_transitions": fallback_transitions,
        "error": error_msg,
    }
    if result is not None:
        record["provider_used"] = result.llm_provider_used
        record["latency_ms"] = result.latency_ms
    return record


# ---------------------------------------------------------------------------
# Main endurance runner
# ---------------------------------------------------------------------------

def run_endurance(
    queries: list[str],
    duration_min: int = 30,
    target_rpm: int = 4,
    out_dir: Path = Path("reports"),
) -> dict:
    """Fire *queries* at steady *target_rpm* for up to *duration_min* minutes.

    If the query list is exhausted before *duration_min*, the list wraps
    (repeating from the start) to fill the window.

    Args:
        queries:      List of query strings to fire.
        duration_min: Maximum wall-clock duration in minutes.
        target_rpm:   Steady request-per-minute rate.
        out_dir:      Directory to write the output JSON report.

    Returns:
        {
            "summary": {
                "total_fired": int,
                "success_count": int,
                "hard_failure_count": int,
                "success_rate": float,
                "fallback_transitions_total": int,
                "duration_actual_s": float,
                "target_rpm": int,
                "actual_rpm": float,
            },
            "latency": { "p50": float, "p95": float, "p99": float, "mean": float },
            "records": [ ... ],
            "report_path": str,
        }
    """
    if not queries:
        raise ValueError("queries list is empty")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Lazy init Generator
    from app.llm.generator import Generator
    gen = Generator()

    interval_s = 60.0 / target_rpm
    deadline_s = duration_min * 60.0
    t_run_start = time.monotonic()

    records: list[dict] = []
    query_idx = 0
    total_fired = 0

    logger.info(
        "Endurance: target=%d RPM, duration=%d min, %d unique queries",
        target_rpm, duration_min, len(queries),
    )

    while (time.monotonic() - t_run_start) < deadline_s:
        query = queries[query_idx % len(queries)]
        query_idx += 1

        rec = _run_single_query(query, gen)
        rec["fired_at_s"] = round(time.monotonic() - t_run_start, 2)
        records.append(rec)
        total_fired += 1

        logger.debug(
            "Endurance [%d/%s] success=%s elapsed=%.0fms",
            total_fired, "?" , rec["success"], rec["elapsed_ms"],
        )

        # Sleep to maintain target RPM (account for query execution time)
        elapsed_in_cycle = rec["elapsed_ms"] / 1000.0
        sleep_needed = max(0.0, interval_s - elapsed_in_cycle)
        if sleep_needed > 0 and (time.monotonic() - t_run_start + sleep_needed) < deadline_s:
            time.sleep(sleep_needed)

    duration_actual_s = time.monotonic() - t_run_start

    # --- Aggregate ---
    success_count = sum(1 for r in records if r["success"])
    hard_failure_count = total_fired - success_count
    fallback_transitions_total = sum(r.get("fallback_transitions", 0) for r in records)
    success_rate = success_count / total_fired if total_fired > 0 else 0.0
    actual_rpm = total_fired / (duration_actual_s / 60.0) if duration_actual_s > 0 else 0.0

    latencies = [r["elapsed_ms"] for r in records if r["success"]]
    latency_stats = _percentile_stats(latencies)

    summary = {
        "total_fired": total_fired,
        "success_count": success_count,
        "hard_failure_count": hard_failure_count,
        "success_rate": round(success_rate, 4),
        "fallback_transitions_total": fallback_transitions_total,
        "duration_actual_s": round(duration_actual_s, 1),
        "target_rpm": target_rpm,
        "actual_rpm": round(actual_rpm, 2),
    }

    output = {
        "run_ts": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "latency": latency_stats,
        "records": records,
    }

    # Write report
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_path = out_dir / f"endurance_{ts}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info(
        "Endurance done: %d fired, %.1f%% success, P95=%.0fms → %s",
        total_fired, success_rate * 100, latency_stats.get("p95", 0), report_path,
    )

    output["report_path"] = str(report_path)
    return output


# ---------------------------------------------------------------------------
# Internal stats
# ---------------------------------------------------------------------------

def _percentile_stats(values: list[float]) -> dict:
    if not values:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0, "n": 0}
    s = sorted(values)
    n = len(s)

    def _p(pct: float) -> float:
        idx = max(0, int(n * pct / 100) - 1)
        return round(s[idx], 1)

    return {
        "p50": _p(50),
        "p95": _p(95),
        "p99": _p(99),
        "mean": round(sum(s) / n, 1),
        "n": n,
    }

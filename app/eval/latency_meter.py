"""Per-stage latency measurement: P50/P95/P99 from eval run records.

Per build-spec Section 5.5 line 1024.

Input format: each record is a dict with a 'latency_ms' key that itself is a dict:
    {
        "retrieval_ms": float,
        "llm_ms":       float,
        "total_ms":     float,
        # optional validation stage
        "validation_ms": float,
    }

Also accepts a flat latency dict at the top level (for legacy / simpler callers).
"""

from __future__ import annotations

import logging
import statistics
from typing import Any

logger = logging.getLogger(__name__)

# Stages to measure.  'total' is always derived from the overall timing.
_STAGES = ["retrieval_ms", "llm_ms", "validation_ms", "total_ms"]


def _percentile(values: list[float], pct: float) -> float:
    """Return the *pct*-th percentile (0-100) of *values* using nearest-rank."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    # nearest-rank formula
    index = max(0, int(len(sorted_vals) * pct / 100) - 1)
    # linear interpolation for smoother estimates
    rank = len(sorted_vals) * pct / 100
    lower_idx = max(0, int(rank) - 1)
    upper_idx = min(len(sorted_vals) - 1, lower_idx + 1)
    frac = rank - int(rank)
    return sorted_vals[lower_idx] * (1 - frac) + sorted_vals[upper_idx] * frac


def compute_latency_stats(records: list[dict]) -> dict:
    """Compute P50/P95/P99 per stage from a list of run records.

    Args:
        records: Each record must contain a 'latency_ms' sub-dict
                 (as returned by GenerationResult.latency_ms).
                 Records missing a stage key are skipped for that stage only.

    Returns:
        {
            "per_stage": {
                "<stage_name>": {
                    "p50": float, "p95": float, "p99": float,
                    "mean": float, "min": float, "max": float,
                    "n": int,
                }
            },
            "n_records": int,
        }
    """
    # Collect per-stage sample lists
    samples: dict[str, list[float]] = {s: [] for s in _STAGES}

    for rec in records:
        lat = rec.get("latency_ms") or {}
        if not isinstance(lat, dict):
            # Flat layout fallback
            lat = {k: rec.get(k, 0) for k in _STAGES}

        for stage in _STAGES:
            val = lat.get(stage)
            if val is not None and isinstance(val, (int, float)) and val >= 0:
                samples[stage].append(float(val))

    per_stage: dict[str, dict] = {}
    for stage, vals in samples.items():
        if not vals:
            per_stage[stage] = {"p50": 0.0, "p95": 0.0, "p99": 0.0,
                                 "mean": 0.0, "min": 0.0, "max": 0.0, "n": 0}
            continue
        per_stage[stage] = {
            "p50": round(_percentile(vals, 50), 1),
            "p95": round(_percentile(vals, 95), 1),
            "p99": round(_percentile(vals, 99), 1),
            "mean": round(statistics.mean(vals), 1),
            "min": round(min(vals), 1),
            "max": round(max(vals), 1),
            "n": len(vals),
        }

    logger.info(
        "Latency stats from %d records: total P95=%.0fms",
        len(records),
        per_stage.get("total_ms", {}).get("p95", 0.0),
    )
    return {"per_stage": per_stage, "n_records": len(records)}

"""Markdown evaluation report generator.

Per build-spec Section 5.5 lines 1034-1037 and Section 8.2 lines 1356-1383.

Report shape:
    # Eval Report — <date>
    ## Summary  (metric table vs thresholds)
    ## Per-tier breakdown
    ## Worst-performing questions (top-10)
    ## Langfuse traces (if langfuse_tag provided)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Acceptance thresholds (per build-spec Section 8.1)
# ---------------------------------------------------------------------------

THRESHOLDS = {
    "faithfulness_mean": 0.85,
    "answer_relevancy_mean": 0.85,
    "citation_accuracy": 0.90,    # micro_precision at Pasal-level
    "refusal_precision": 0.80,
    "p95_latency_ms": 30_000,     # 30 seconds
    "eg_mean": 0.90,
    "rp_mean": 0.80,
}


def _pass_fail(value: float | None, threshold: float, higher_is_better: bool = True) -> str:
    if value is None:
        return "N/A"
    if higher_is_better:
        return "PASS" if value >= threshold else "FAIL"
    return "PASS" if value <= threshold else "FAIL"


def _fmt(value: float | None, pct: bool = False) -> str:
    if value is None:
        return "N/A"
    if pct:
        return f"{value * 100:.1f}%"
    return f"{value:.4f}"


def _fmt_ms(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value / 1000:.1f}s" if value >= 1000 else f"{value:.0f}ms"


# ---------------------------------------------------------------------------
# Per-tier breakdown helpers
# ---------------------------------------------------------------------------

def _tier_breakdown(
    test_results: list[dict],
    ragas_results: dict,
    citation_results_per_q: list[dict],
) -> str:
    """Build per-tier metrics table.

    *test_results* is the flat list from the main eval loop, each record has:
        qid, tier, must_refuse, response, latency_ms, eg_score, rp_score,
        citation_accuracy (from custom metric), is_refusal, correct_refusal
    """
    tiers = {1: [], 2: [], 3: [], 4: [], 5: []}
    for rec in test_results:
        t = rec.get("tier")
        if t in tiers:
            tiers[t].append(rec)

    # Build ragas lookup by qid
    ragas_by_qid = {}
    for r in ragas_results.get("per_record", []):
        ragas_by_qid[r["qid"]] = r

    # Build citation lookup by qid
    cite_by_qid = {}
    for c in (citation_results_per_q or []):
        cite_by_qid[c.get("qid", "")] = c

    lines = ["## Per-tier Breakdown", ""]
    lines.append("| Tier | N | Faithfulness | Cite Acc | EG mean | RP mean | Refusal P | Notes |")
    lines.append("|------|---|-------------|----------|---------|---------|-----------|-------|")

    tier_labels = {
        1: "T1 Factual",
        2: "T2 Cross-ref",
        3: "T3 Thematic",
        4: "T4 OOS",
        5: "T5 Adversarial",
    }

    for tier_num in sorted(tiers):
        recs = tiers[tier_num]
        n = len(recs)
        if n == 0:
            lines.append(f"| {tier_labels[tier_num]} | 0 | - | - | - | - | - | no data |")
            continue

        # Faithfulness
        faith_vals = [ragas_by_qid[r["qid"]]["faithfulness"]
                      for r in recs if r["qid"] in ragas_by_qid
                      and ragas_by_qid[r["qid"]]["faithfulness"] is not None]
        faith_mean = sum(faith_vals) / len(faith_vals) if faith_vals else None

        # Citation accuracy
        cite_vals = [cite_by_qid[r["qid"]]["precision"]
                     for r in recs if r["qid"] in cite_by_qid]
        cite_mean = sum(cite_vals) / len(cite_vals) if cite_vals else None

        # EG / RP
        eg_vals = [r["eg_score"] for r in recs if r.get("eg_score") is not None]
        rp_vals = [r["rp_score"] for r in recs if r.get("rp_score") is not None]
        eg_mean = sum(eg_vals) / len(eg_vals) if eg_vals else None
        rp_mean = sum(rp_vals) / len(rp_vals) if rp_vals else None

        # Refusal precision (only meaningful for Tier 4/5)
        should_refuse = [r for r in recs if r.get("must_refuse")]
        correct_refuse = [r for r in should_refuse if r.get("correct_refusal")]
        ref_prec = len(correct_refuse) / len(should_refuse) if should_refuse else None

        notes = ""
        if tier_num in (4, 5) and ref_prec is not None:
            notes = f"{len(correct_refuse)}/{len(should_refuse)} refused correctly"

        lines.append(
            f"| {tier_labels[tier_num]} | {n} "
            f"| {_fmt(faith_mean)} "
            f"| {_fmt(cite_mean, pct=True)} "
            f"| {_fmt(eg_mean)} "
            f"| {_fmt(rp_mean)} "
            f"| {_fmt(ref_prec, pct=True)} "
            f"| {notes} |"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Worst-performing questions
# ---------------------------------------------------------------------------

def _worst_questions(
    test_results: list[dict],
    ragas_results: dict,
    cite_per_q: list[dict],
    n: int = 10,
) -> str:
    ragas_by_qid = {r["qid"]: r for r in ragas_results.get("per_record", [])}
    cite_by_qid = {c.get("qid", ""): c for c in (cite_per_q or [])}

    scored = []
    for rec in test_results:
        qid = rec["qid"]
        faith = (ragas_by_qid.get(qid) or {}).get("faithfulness") or 1.0
        cite_acc = (cite_by_qid.get(qid) or {}).get("precision") or 1.0
        eg = rec.get("eg_score") or 1.0
        # Composite worst-case score (lower = worse)
        combined = (faith + cite_acc + eg) / 3.0
        scored.append((combined, rec, qid, faith, cite_acc, eg))

    scored.sort(key=lambda x: x[0])  # ascending = worst first

    lines = ["## Worst-performing Questions", ""]
    for rank, (combined, rec, qid, faith, cite_acc, eg) in enumerate(scored[:n], start=1):
        question_text = rec.get("question", "")[:100]
        tier = rec.get("tier", "?")
        diagnosis = []
        if faith < 0.7:
            diagnosis.append("low faithfulness (generation miss)")
        if cite_acc < 0.7:
            diagnosis.append("low citation accuracy (retrieval or extraction miss)")
        if eg < 0.7:
            diagnosis.append("low EG (entity hallucination risk)")
        if rec.get("must_refuse") and not rec.get("correct_refusal"):
            diagnosis.append("failed to refuse out-of-scope query")
        diag_str = "; ".join(diagnosis) if diagnosis else "marginal on all metrics"
        lines.append(
            f"{rank}. **[{qid}]** (Tier {tier}) \"{question_text}...\"\n"
            f"   Faith={_fmt(faith)}, CiteAcc={_fmt(cite_acc, pct=True)}, "
            f"EG={_fmt(eg)} — {diag_str}\n"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main report generator
# ---------------------------------------------------------------------------

def generate_report(
    test_results: list[dict],
    ragas_results: dict,
    citation_results: dict,
    refusal_results: dict,
    latency_results: dict,
    eg_rp_results: dict,
    out_path: Path,
    langfuse_tag: str | None = None,
) -> None:
    """Write Markdown evaluation report to *out_path*.

    Args:
        test_results:     List of per-question result dicts from eval loop.
        ragas_results:    Output of ragas_runner.run_ragas().
        citation_results: Output of citation_accuracy.aggregate_citation_accuracy().
        refusal_results:  Output of refusal_precision.compute_refusal_metrics().
        latency_results:  Output of latency_meter.compute_latency_stats().
        eg_rp_results:    {"eg_mean": float, "rp_mean": float} from test_results.
        out_path:         Output file path (Markdown).
        langfuse_tag:     Optional Langfuse tag for trace link rendering.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y-%m-%d %H:%M UTC")

    # --- Extract key metrics ---
    faith_mean = ragas_results.get("aggregate", {}).get("faithfulness_mean")
    ar_mean = ragas_results.get("aggregate", {}).get("answer_relevancy_mean")
    cp_mean = ragas_results.get("aggregate", {}).get("context_precision_mean")
    cite_acc = citation_results.get("micro_precision")
    ref_prec = refusal_results.get("precision")
    p95_ms = latency_results.get("per_stage", {}).get("total_ms", {}).get("p95")
    eg_mean = eg_rp_results.get("eg_mean")
    rp_mean = eg_rp_results.get("rp_mean")

    n_questions = len(test_results)
    n_errored = eg_rp_results.get("n_errored", 0)
    n_subset = ragas_results.get("aggregate", {}).get("n_evaluated", 0)
    is_subset = n_subset < n_questions and n_subset > 0

    subset_marker = " *(smoke subset only — not full DoD)*" if is_subset else ""
    errored_marker = f" | **{n_errored} hard-failure(s) excluded from EG/RP mean**" if n_errored else ""

    # --- Summary section ---
    lines = [
        f"# Eval Report — {date_str}",
        "",
        f"**Test set**: {n_questions} questions{subset_marker}{errored_marker}",
        f"**Judge model**: {ragas_results.get('judge_model', 'N/A')}",
        f"**Cache used**: {ragas_results.get('cache_used', False)}",
        "",
        "## Summary",
        "",
        "| Metric | Threshold | Actual | Pass? |",
        "|--------|-----------|--------|-------|",
        f"| Faithfulness mean | ≥ 0.85 | {_fmt(faith_mean)} | {_pass_fail(faith_mean, 0.85)} |",
        f"| Answer relevancy mean | ≥ 0.85 | {_fmt(ar_mean)} | {_pass_fail(ar_mean, 0.85)} |",
        f"| Context precision mean | — | {_fmt(cp_mean)} | — |",
        f"| Citation accuracy (Pasal-level) | ≥ 90% | {_fmt(cite_acc, pct=True)} | {_pass_fail(cite_acc, 0.90)} |",
        f"| Refusal precision (Tier 4+5) | ≥ 80% | {_fmt(ref_prec, pct=True)} | {_pass_fail(ref_prec, 0.80)} |",
        f"| P95 latency end-to-end | ≤ 30s | {_fmt_ms(p95_ms)} | {_pass_fail(p95_ms, 30_000, higher_is_better=False)} |",
        f"| EG score mean | ≥ 0.90 | {_fmt(eg_mean)} | {_pass_fail(eg_mean, 0.90)} |",
        f"| RP score mean | ≥ 0.80 | {_fmt(rp_mean)} | {_pass_fail(rp_mean, 0.80)} |",
        "",
    ]

    # --- Detailed refusal breakdown ---
    tp = refusal_results.get("tp", 0)
    fp = refusal_results.get("fp", 0)
    fn = refusal_results.get("fn", 0)
    tn = refusal_results.get("tn", 0)
    lines += [
        "### Refusal detail",
        f"- TP (correctly refused): {tp}",
        f"- FP (over-refusal): {fp}",
        f"- FN (failed to refuse): {fn}",
        f"- TN (correctly answered): {tn}",
        "",
    ]

    # --- Latency detail ---
    lat_per_stage = latency_results.get("per_stage", {})
    lines += [
        "### Latency breakdown",
        "",
        "| Stage | P50 | P95 | P99 | Mean |",
        "|-------|-----|-----|-----|------|",
    ]
    for stage_key, label in [
        ("retrieval_ms", "Retrieval"),
        ("llm_ms", "LLM"),
        ("validation_ms", "Validation"),
        ("total_ms", "Total"),
    ]:
        st = lat_per_stage.get(stage_key, {})
        lines.append(
            f"| {label} | {_fmt_ms(st.get('p50'))} "
            f"| {_fmt_ms(st.get('p95'))} "
            f"| {_fmt_ms(st.get('p99'))} "
            f"| {_fmt_ms(st.get('mean'))} |"
        )
    lines.append("")

    # --- Citation detail ---
    lines += [
        "### Citation accuracy detail",
        f"- Total matched: {citation_results.get('total_matched', 0)}",
        f"- Total extracted: {citation_results.get('total_extracted', 0)}",
        f"- Total expected: {citation_results.get('total_expected', 0)}",
        f"- Mean Jaccard: {_fmt(citation_results.get('mean_jaccard'))}",
        f"- Micro recall: {_fmt(citation_results.get('micro_recall'), pct=True)}",
        "",
    ]

    # --- Per-tier breakdown ---
    cite_per_q = _extract_cite_per_q(test_results)
    lines.append(_tier_breakdown(test_results, ragas_results, cite_per_q))
    lines.append("")

    # --- Worst questions ---
    lines.append(_worst_questions(test_results, ragas_results, cite_per_q))
    lines.append("")

    # --- Langfuse traces ---
    if langfuse_tag:
        lines += [
            "## Langfuse Traces",
            "",
            f"- Tag: `{langfuse_tag}`",
            "- Dashboard: (check your Langfuse self-hosted instance)",
            "",
        ]

    # --- RAGAS failure notes ---
    n_failed = ragas_results.get("aggregate", {}).get("n_failed", 0)
    if n_failed > 0:
        lines += [
            f"> **Note**: {n_failed} RAGAS evaluations failed (see per_record for details).",
            "",
        ]

    # --- Recommendation ---
    all_pass = all([
        faith_mean is not None and faith_mean >= 0.85,
        ar_mean is not None and ar_mean >= 0.85,
        cite_acc is not None and cite_acc >= 0.90,
        ref_prec is not None and ref_prec >= 0.80,
        p95_ms is not None and p95_ms <= 30_000,
        eg_mean is not None and eg_mean >= 0.90,
        rp_mean is not None and rp_mean >= 0.80,
    ])

    lines += [
        "## Recommendation",
        "",
    ]
    if is_subset:
        lines.append(
            "**SMOKE SUBSET ONLY** — This run covered a subset of questions. "
            "Full DoD acceptance requires a complete 50-question run."
        )
    elif all_pass:
        lines.append(
            "**READY FOR DEPLOY (Step 8)** — All acceptance thresholds met. "
            "Proceed to Oracle VM deployment."
        )
    else:
        lines.append(
            "**NOT READY** — One or more metrics below threshold. "
            "Review worst-performing questions above; consider prompt tuning "
            "or chunk policy adjustment before Step 8."
        )
    lines.append("")

    content = "\n".join(lines)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)

    logger.info("Report written to %s", out_path)


def _extract_cite_per_q(test_results: list[dict]) -> list[dict]:
    """Pull citation_accuracy_per_q from test_results if embedded."""
    result = []
    for r in test_results:
        ca = r.get("citation_accuracy_detail")
        if ca and isinstance(ca, dict):
            ca["qid"] = r.get("qid", "")
            result.append(ca)
    return result

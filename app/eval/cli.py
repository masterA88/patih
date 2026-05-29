"""CLI entry point for the evaluation pipeline.

Usage:
    python -m app.eval.cli run \\
        --test-set data/test/permensos8_50q.jsonl \\
        --out reports/eval_20260518_1400.md \\
        --provider groq/llama-3.3-70b-versatile \\
        --judge-throttle 25 \\
        --langfuse-tag eval-baseline-v3

    python -m app.eval.cli endurance \\
        --queries data/test/endurance_100q.jsonl \\
        --duration-min 30 \\
        --target-rpm 4

    python -m app.eval.cli subset \\
        --test-set data/test/permensos8_50q.jsonl \\
        --tier 1 \\
        --out reports/eval_subset_t1.md

Per build-spec Section 5.5 lines 999-1004.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("app.eval.cli")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_out_path() -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    Path("reports").mkdir(parents=True, exist_ok=True)
    return Path(f"reports/eval_{ts}.md")


def _filter_questions(questions, tier: int | None, qids: list[str] | None):
    if tier is not None:
        questions = [q for q in questions if q.tier == tier]
    if qids:
        qid_set = set(qids)
        questions = [q for q in questions if q.qid in qid_set]
    return questions


# ---------------------------------------------------------------------------
# Core eval loop (shared by run + subset)
# ---------------------------------------------------------------------------

def _run_eval_loop(
    questions,
    provider: str,
    judge_throttle: int,
    langfuse_tag: str | None,
    out_path: Path,
    is_subset: bool = False,
    min_gap_s: float = 12.0,
    judge_provider: str | None = None,
) -> None:
    """Full evaluation over *questions*.  Writes markdown report to *out_path*."""

    from app.llm.generator import Generator
    from app.eval.citation_accuracy import compute_citation_accuracy, aggregate_citation_accuracy
    from app.eval.refusal_precision import compute_refusal_metrics
    from app.eval.latency_meter import compute_latency_stats
    from app.eval.ragas_runner import run_ragas
    from app.eval.report import generate_report

    gen = Generator()

    logger.info(
        "Starting eval: %d questions, provider=%s, min_gap_s=%.1f",
        len(questions), provider, min_gap_s,
    )

    test_results: list[dict] = []
    ragas_records: list[dict] = []
    citation_per_q: list[dict] = []
    refusal_questions = []
    refusal_responses = []

    last_question_start: float = 0.0

    for i, q in enumerate(questions, start=1):
        # Rate-limit pacing: enforce minimum gap between question starts
        if last_question_start and min_gap_s > 0:
            elapsed = time.monotonic() - last_question_start
            if elapsed < min_gap_s:
                sleep_for = min_gap_s - elapsed
                logger.info("Rate-limit pacing: sleeping %.1fs before question %d", sleep_for, i)
                time.sleep(sleep_for)

        last_question_start = time.monotonic()
        logger.info("[%d/%d] %s: %s", i, len(questions), q.qid, q.question[:80])

        t_start = time.monotonic()
        try:
            result = gen.answer(q.question, validate=True)
        except Exception as exc:
            logger.error("[%s] Generator failed: %s", q.qid, exc)
            # Record failure
            test_results.append({
                "qid": q.qid, "tier": q.tier, "question": q.question,
                "must_refuse": q.must_refuse, "response": f"ERROR: {exc}",
                "eg_score": 0.0, "rp_score": 0.0, "is_refusal": False,
                "correct_refusal": False, "latency_ms": {"total_ms": 0},
                "error": str(exc),
            })
            continue

        elapsed_ms = (time.monotonic() - t_start) * 1000
        response = result.response
        val = result.validation or {}

        # --- Citation accuracy ---
        extracted_citations = val.get("citations_extracted", [])
        cite_acc = compute_citation_accuracy(
            extracted=extracted_citations,
            expected=q.expected_pasal_refs,
            granularity="pasal",
        )
        cite_acc["qid"] = q.qid
        citation_per_q.append(cite_acc)

        # --- Refusal tracking ---
        from app.eval.refusal_precision import is_refusal
        predicted_refusal = is_refusal(response) or val.get("is_refusal", False)
        correct_refusal = (q.must_refuse == predicted_refusal)

        refusal_questions.append(q)
        refusal_responses.append(response)

        # --- EG / RP from validation ---
        eg_score = val.get("eg_score", 1.0)
        rp_score = val.get("rp_score", 1.0)

        rec = {
            "qid": q.qid,
            "tier": q.tier,
            "question": q.question,
            "must_refuse": q.must_refuse,
            "response": response,
            "eg_score": eg_score,
            "rp_score": rp_score,
            "is_refusal": predicted_refusal,
            "correct_refusal": correct_refusal,
            "latency_ms": result.latency_ms,
            "provider_used": result.llm_provider_used,
            "citation_accuracy_detail": cite_acc,
        }
        test_results.append(rec)

        # --- Build RAGAS record ---
        # Use parent_chunks (list[dict] with 'text' key) — NOT retrieved_pasals (list[int])
        contexts_text = [
            chunk.get("text", "") for chunk in result.parent_chunks
            if chunk.get("text")
        ]
        ragas_records.append({
            "qid": q.qid,
            "question": q.question,
            "answer": response,
            "contexts": contexts_text,
            "ground_truth": q.expected_answer_summary,
        })

        logger.info(
            "[%s] done: EG=%.2f RP=%.2f cite_P=%.2f refusal=%s/%s lat=%.0fms",
            q.qid, eg_score, rp_score, cite_acc["precision"],
            predicted_refusal, q.must_refuse, result.latency_ms.get("total_ms", 0),
        )

    # --- Save eval run artifact ---
    run_artifact_dir = Path("data/eval_runs")
    run_artifact_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    artifact_path = run_artifact_dir / f"eval_records_{ts}.jsonl"
    with open(artifact_path, "w", encoding="utf-8") as f:
        for rec in test_results:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    logger.info("Eval records saved to %s", artifact_path)

    # --- RAGAS eval ---
    # Judge model defaults to generator provider but can be split (e.g. Groq generator
    # + Gemini judge) to avoid double-burning TPD quota on free-tier providers.
    effective_judge = judge_provider or provider
    judge_model = effective_judge if "/" in effective_judge else f"gemini/{effective_judge}"
    logger.info(
        "Running RAGAS judge over %d records (judge=%s) ...",
        len(ragas_records), judge_model,
    )
    ragas_results = run_ragas(
        eval_records=ragas_records,
        judge_model=judge_model,
        throttle_rpm=judge_throttle,
        cache_path=Path("data/eval_cache/ragas_judge.db"),
    )

    # --- Aggregate citation accuracy ---
    citation_agg = aggregate_citation_accuracy(citation_per_q)

    # --- Refusal metrics ---
    refusal_metrics = compute_refusal_metrics(refusal_questions, refusal_responses)

    # --- Latency stats ---
    latency_stats = compute_latency_stats(test_results)

    # --- EG / RP means (exclude hard-failure / error records) ---
    # Error records have eg_score=0.0 from the error path — exclude so failures
    # from rate-limits / network errors don't drag the mean down unfairly.
    n_errored = sum(1 for r in test_results if r.get("error"))
    successful_results = [r for r in test_results if not r.get("error")]
    eg_vals = [r.get("eg_score") for r in successful_results if r.get("eg_score") is not None]
    rp_vals = [r.get("rp_score") for r in successful_results if r.get("rp_score") is not None]
    eg_rp = {
        "eg_mean": round(sum(eg_vals) / len(eg_vals), 4) if eg_vals else 0.0,
        "rp_mean": round(sum(rp_vals) / len(rp_vals), 4) if rp_vals else 0.0,
        "n_errored": n_errored,
    }
    if n_errored:
        logger.warning(
            "%d hard-failure records excluded from EG/RP mean (provider all-failed)", n_errored
        )

    # --- Generate report ---
    generate_report(
        test_results=test_results,
        ragas_results=ragas_results,
        citation_results=citation_agg,
        refusal_results=refusal_metrics,
        latency_results=latency_stats,
        eg_rp_results=eg_rp,
        out_path=out_path,
        langfuse_tag=langfuse_tag,
    )

    # --- Print summary to stdout ---
    print("\n" + "=" * 60)
    print(f"EVAL COMPLETE  ({len(test_results)} questions)")
    print("=" * 60)
    agg = ragas_results.get("aggregate", {})
    print(f"Faithfulness:       {agg.get('faithfulness_mean', 'N/A'):.4f}  (threshold >=0.85)")
    print(f"Answer relevancy:   {agg.get('answer_relevancy_mean', 'N/A'):.4f}  (threshold >=0.85)")
    print(f"Citation accuracy:  {citation_agg.get('micro_precision', 0):.1%}  (threshold >=90%)")
    print(f"Refusal precision:  {refusal_metrics.get('precision', 0):.1%}  (threshold >=80%)")
    p95 = latency_stats.get("per_stage", {}).get("total_ms", {}).get("p95", 0)
    print(f"P95 latency:        {p95/1000:.1f}s  (threshold <=30s)")
    print(f"EG mean:            {eg_rp['eg_mean']:.4f}  (threshold >=0.90)")
    print(f"RP mean:            {eg_rp['rp_mean']:.4f}  (threshold >=0.80)")
    print(f"\nReport: {out_path}")
    if is_subset:
        print("\n[SMOKE SUBSET -- not full DoD run]")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Subcommand: run
# ---------------------------------------------------------------------------

def cmd_run(args: argparse.Namespace) -> None:
    from app.eval.testset_loader import load_testset

    test_set_path = Path(args.test_set)
    questions = load_testset(test_set_path)
    logger.info("Loaded %d questions from %s", len(questions), test_set_path)

    out_path = Path(args.out) if args.out else _default_out_path()

    _run_eval_loop(
        questions=questions,
        provider=args.provider,
        judge_throttle=args.judge_throttle,
        langfuse_tag=args.langfuse_tag,
        out_path=out_path,
        is_subset=False,
        min_gap_s=args.min_gap_s,
        judge_provider=getattr(args, "judge_provider", None),
    )


# ---------------------------------------------------------------------------
# Subcommand: subset
# ---------------------------------------------------------------------------

def cmd_subset(args: argparse.Namespace) -> None:
    from app.eval.testset_loader import load_testset

    questions = load_testset(Path(args.test_set))

    tier: int | None = int(args.tier) if args.tier else None
    qids: list[str] | None = (
        [q.strip() for q in args.qid.split(",") if q.strip()]
        if args.qid else None
    )

    filtered = _filter_questions(questions, tier, qids)
    if not filtered:
        logger.error("No questions match the filter (tier=%s, qid=%s)", args.tier, args.qid)
        sys.exit(1)

    logger.info("Subset: %d questions selected (tier=%s, qid=%s)", len(filtered), tier, qids)

    out_path = Path(args.out) if args.out else _default_out_path()

    _run_eval_loop(
        questions=filtered,
        provider=args.provider,
        judge_throttle=args.judge_throttle,
        langfuse_tag=args.langfuse_tag,
        out_path=out_path,
        is_subset=True,
        min_gap_s=args.min_gap_s,
        judge_provider=getattr(args, "judge_provider", None),
    )


# ---------------------------------------------------------------------------
# Subcommand: judge-only
# ---------------------------------------------------------------------------

def cmd_judge_only(args: argparse.Namespace) -> None:
    """Re-run RAGAS judge over an existing eval_records JSONL.

    Re-retrieves contexts via the (LLM-free) retrieval pipeline so RAGAS
    has the parent-chunk texts it needs. Reuses the saved test_results for
    everything else (response, citation detail, latency, EG/RP).

    Use when the original run's judge phase was quota-blocked but the gen
    phase succeeded — avoids re-burning generator quota.
    """
    from app.eval.testset_loader import load_testset
    from app.eval.citation_accuracy import aggregate_citation_accuracy
    from app.eval.refusal_precision import compute_refusal_metrics
    from app.eval.latency_meter import compute_latency_stats
    from app.eval.ragas_runner import run_ragas
    from app.eval.report import generate_report
    from app.retrieval.pipeline import retrieve

    records_path = Path(args.records)
    if not records_path.exists():
        logger.error("Records file not found: %s", records_path)
        sys.exit(1)

    test_results: list[dict] = []
    with open(records_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                test_results.append(json.loads(line))
    logger.info("Loaded %d records from %s", len(test_results), records_path)

    test_set = load_testset(Path(args.test_set))
    qid_to_q = {q.qid: q for q in test_set}
    logger.info("Loaded %d test-set questions for ground_truth lookup", len(test_set))

    ragas_records: list[dict] = []
    for i, rec in enumerate(test_results, start=1):
        if rec.get("error"):
            continue
        qid = rec.get("qid")
        q = qid_to_q.get(qid)
        if q is None:
            logger.warning("[%s] not in test set — skipping", qid)
            continue
        logger.info("[%d/%d] re-retrieve %s", i, len(test_results), qid)
        ret_result = retrieve(rec["question"])
        contexts = [
            c.get("text", "") for c in ret_result.parent_chunks if c.get("text")
        ]
        ragas_records.append({
            "qid": qid,
            "question": rec["question"],
            "answer": rec["response"],
            "contexts": contexts,
            "ground_truth": q.expected_answer_summary,
        })

    judge_provider = args.judge_provider
    judge_model = judge_provider if "/" in judge_provider else f"gemini/{judge_provider}"
    logger.info("Running RAGAS judge over %d records (judge=%s) ...",
                len(ragas_records), judge_model)
    ragas_results = run_ragas(
        eval_records=ragas_records,
        judge_model=judge_model,
        throttle_rpm=args.judge_throttle,
        cache_path=Path("data/eval_cache/ragas_judge.db"),
    )

    citation_per_q = [
        r["citation_accuracy_detail"] for r in test_results
        if r.get("citation_accuracy_detail")
    ]
    citation_agg = aggregate_citation_accuracy(citation_per_q)

    refusal_questions = [qid_to_q[r["qid"]] for r in test_results if r.get("qid") in qid_to_q]
    refusal_responses = [r["response"] for r in test_results if r.get("qid") in qid_to_q]
    refusal_metrics = compute_refusal_metrics(refusal_questions, refusal_responses)

    latency_stats = compute_latency_stats(test_results)

    n_errored = sum(1 for r in test_results if r.get("error"))
    successful = [r for r in test_results if not r.get("error")]
    eg_vals = [r.get("eg_score") for r in successful if r.get("eg_score") is not None]
    rp_vals = [r.get("rp_score") for r in successful if r.get("rp_score") is not None]
    eg_rp = {
        "eg_mean": round(sum(eg_vals) / len(eg_vals), 4) if eg_vals else 0.0,
        "rp_mean": round(sum(rp_vals) / len(rp_vals), 4) if rp_vals else 0.0,
        "n_errored": n_errored,
    }

    out_path = Path(args.out) if args.out else _default_out_path()
    generate_report(
        test_results=test_results,
        ragas_results=ragas_results,
        citation_results=citation_agg,
        refusal_results=refusal_metrics,
        latency_results=latency_stats,
        eg_rp_results=eg_rp,
        out_path=out_path,
        langfuse_tag=args.langfuse_tag,
    )

    print("\n" + "=" * 60)
    print(f"JUDGE-ONLY RERUN COMPLETE  ({len(test_results)} records, {len(ragas_records)} judged)")
    print("=" * 60)
    agg = ragas_results.get("aggregate", {})
    faith = agg.get("faithfulness_mean", float("nan"))
    relev = agg.get("answer_relevancy_mean", float("nan"))
    print(f"Faithfulness:       {faith:.4f}  (threshold >=0.85)")
    print(f"Answer relevancy:   {relev:.4f}  (threshold >=0.85)")
    print(f"Citation accuracy:  {citation_agg.get('micro_precision', 0):.1%}  (threshold >=90%)")
    print(f"Refusal precision:  {refusal_metrics.get('precision', 0):.1%}  (threshold >=80%)")
    p95 = latency_stats.get("per_stage", {}).get("total_ms", {}).get("p95", 0)
    print(f"P95 latency:        {p95/1000:.1f}s  (threshold <=30s)")
    print(f"EG mean:            {eg_rp['eg_mean']:.4f}  (threshold >=0.90)")
    print(f"RP mean:            {eg_rp['rp_mean']:.4f}  (threshold >=0.80)")
    print(f"\nReport: {out_path}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Subcommand: endurance
# ---------------------------------------------------------------------------

def cmd_endurance(args: argparse.Namespace) -> None:
    from app.eval.ratelimit_endurance import run_endurance

    queries_path = Path(args.queries)
    if not queries_path.exists():
        logger.error("Queries file not found: %s", queries_path)
        sys.exit(1)

    queries: list[str] = []
    with open(queries_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                obj = json.loads(line)
                q = obj.get("question") or obj.get("query") or str(obj)
            except json.JSONDecodeError:
                q = line
            queries.append(q)

    if not queries:
        logger.error("No queries loaded from %s", queries_path)
        sys.exit(1)

    result = run_endurance(
        queries=queries,
        duration_min=args.duration_min,
        target_rpm=args.target_rpm,
        out_dir=Path("reports"),
    )

    summary = result.get("summary", {})
    print("\n" + "=" * 60)
    print("ENDURANCE TEST COMPLETE")
    print("=" * 60)
    print(f"Total fired:    {summary.get('total_fired')}")
    print(f"Success rate:   {summary.get('success_rate', 0):.1%}")
    print(f"Hard failures:  {summary.get('hard_failure_count')}")
    print(f"Fallback trans: {summary.get('fallback_transitions_total')}")
    lat = result.get("latency", {})
    print(f"P50 latency:    {lat.get('p50', 0)/1000:.1f}s")
    print(f"P95 latency:    {lat.get('p95', 0)/1000:.1f}s")
    print(f"Report:         {result.get('report_path')}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.eval.cli",
        description="Evaluation pipeline for Chatbot Permensos 8/2023.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- run --
    p_run = sub.add_parser("run", help="Full eval over 50-question test set.")
    p_run.add_argument("--test-set", required=True, help="Path to .jsonl test set.")
    p_run.add_argument("--out", default=None, help="Output markdown report path.")
    p_run.add_argument("--provider", default="groq/llama-3.3-70b-versatile",
                       help="LiteLLM model string for generation and judge.")
    p_run.add_argument("--judge-throttle", type=int, default=25,
                       help="RAGAS judge max RPM (default 25 = safe margin under Groq 30 RPM).")
    p_run.add_argument("--judge-provider", default=None,
                       help="LiteLLM model for RAGAS judge. Defaults to --provider "
                            "(use a separate provider to avoid TPD double-burn, "
                            "e.g. --provider groq/... --judge-provider gemini/gemini-2.5-flash).")
    p_run.add_argument("--langfuse-tag", default=None, help="Langfuse run tag.")
    p_run.add_argument(
        "--min-gap-s", type=float, default=2.0,
        help="Minimum seconds between questions (default 2 = 30 RPM, "
             "Groq Llama 3.3 70B free tier limit).",
    )

    # -- subset --
    p_sub = sub.add_parser("subset", help="Run eval on a subset of questions.")
    p_sub.add_argument("--test-set", required=True)
    p_sub.add_argument("--tier", default=None, help="Filter to tier N (1-5).")
    p_sub.add_argument("--qid", default=None, help="Comma-separated qid list.")
    p_sub.add_argument("--out", default=None)
    p_sub.add_argument("--provider", default="groq/llama-3.3-70b-versatile")
    p_sub.add_argument("--judge-throttle", type=int, default=25)
    p_sub.add_argument("--judge-provider", default=None,
                       help="LiteLLM model for RAGAS judge (default = --provider).")
    p_sub.add_argument("--langfuse-tag", default=None)
    p_sub.add_argument(
        "--min-gap-s", type=float, default=2.0,
        help="Minimum seconds between questions (default 2 = 30 RPM, "
             "Groq Llama 3.3 70B free tier limit).",
    )

    # -- judge-only --
    p_judge = sub.add_parser(
        "judge-only",
        help="Re-run RAGAS judge over an existing eval_records JSONL (no LLM gen).",
    )
    p_judge.add_argument("--records", required=True,
                         help="Path to eval_records_*.jsonl from a prior run.")
    p_judge.add_argument("--test-set", required=True,
                         help="Path to the test-set .jsonl (for ground_truth lookup).")
    p_judge.add_argument("--out", default=None, help="Output markdown report path.")
    p_judge.add_argument("--judge-provider", default="gemini/gemini-2.5-flash",
                         help="LiteLLM model for RAGAS judge (default gemini/gemini-2.5-flash).")
    p_judge.add_argument("--judge-throttle", type=int, default=4,
                         help="RAGAS judge max RPM (default 4 = safe under Gemini 5 RPM).")
    p_judge.add_argument("--langfuse-tag", default=None, help="Langfuse run tag.")

    # -- endurance --
    p_end = sub.add_parser("endurance", help="Rate-limit endurance stress test.")
    p_end.add_argument("--queries", required=True, help="Path to queries .jsonl file.")
    p_end.add_argument("--duration-min", type=int, default=30)
    p_end.add_argument("--target-rpm", type=int, default=4)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run":
        cmd_run(args)
    elif args.command == "subset":
        cmd_subset(args)
    elif args.command == "judge-only":
        cmd_judge_only(args)
    elif args.command == "endurance":
        cmd_endurance(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

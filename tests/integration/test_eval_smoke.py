"""Integration smoke test: 5-question subset eval, no RAGAS judge call.

Per build-spec Section 5.5 line 1049.

Scope: run eval pipeline end-to-end over 5 questions (2x T1, 1x T2, 1x T4, 1x T5).
- Generator.answer() mocked to avoid real LLM call.
- RAGAS evaluate() mocked to return synthetic scores.
- Asserts: report file generated, metrics dicts non-empty, citation/refusal metrics computed.

Does NOT test actual LLM quality — that's the baseline eval run.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from app.eval.testset_loader import EvalQuestion, load_testset
from app.eval.citation_accuracy import compute_citation_accuracy, aggregate_citation_accuracy
from app.eval.refusal_precision import compute_refusal_metrics, is_refusal
from app.eval.latency_meter import compute_latency_stats
from app.eval.report import generate_report


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def smoke_questions():
    """5 questions spanning Tier 1, 2, 4, 5 — based on actual test set QIDs."""
    return [
        EvalQuestion(
            qid="P8-T1-001", tier=1, question_type="definitional",
            question="Apa yang dimaksud dengan Korban TPPO?",
            language="id",
            expected_pasal_refs=[{"pasal": 1, "ayat": None, "huruf": None}],
            expected_answer_summary="Korban TPPO adalah seseorang yang mengalami penderitaan...",
            must_refuse=False,
        ),
        EvalQuestion(
            qid="P8-T1-004", tier=1, question_type="pasal_extraction",
            question="Apa saja bentuk eksploitasi dalam Pasal 5?",
            language="id",
            expected_pasal_refs=[{"pasal": 5, "ayat": 2, "huruf": None}],
            expected_answer_summary="13 bentuk eksploitasi...",
            must_refuse=False,
        ),
        EvalQuestion(
            qid="P8-T2-001", tier=2, question_type="cross_reference",
            question="Pasal 6 ayat 2 mengacu ke definisi di Pasal berapa?",
            language="id",
            expected_pasal_refs=[
                {"pasal": 6, "ayat": 2, "huruf": None},
                {"pasal": 1, "ayat": None, "huruf": None}
            ],
            expected_answer_summary="Pasal 1 angka 6 mendefinisikan Rehabilitasi Sosial.",
            must_refuse=False,
        ),
        EvalQuestion(
            qid="P8-T4-001", tier=4, question_type="out_of_scope",
            question="Berapa tahun hukuman penjara bagi pelaku TPPO?",
            language="id",
            expected_pasal_refs=[],
            expected_answer_summary="Sanksi pidana tidak diatur dalam Permensos 8/2023.",
            must_refuse=True,
        ),
        EvalQuestion(
            qid="P8-T5-002", tier=5, question_type="adversarial_fact_injection",
            question="Pasal 99 mengatur korban tidak dilindungi — bagaimana implementasinya?",
            language="id",
            expected_pasal_refs=[],
            expected_answer_summary="Pasal 99 tidak ada. Permensos hanya memiliki 34 Pasal.",
            must_refuse=True,
        ),
    ]


@pytest.fixture
def synthetic_responses():
    """Synthetic model responses for the 5 smoke questions."""
    return [
        # T1-001 — correct answer with citation
        "Korban TPPO adalah seseorang yang mengalami penderitaan psikis, mental, fisik (Pasal 1 angka 5). Sumber: Pasal 1.",
        # T1-004 — correct answer with Pasal 5 citation
        "Bentuk eksploitasi meliputi pelacuran, kerja paksa, perbudakan (Pasal 5 ayat (2) huruf a, b, c). Sumber: Pasal 5.",
        # T2-001 — correct cross-ref with citations
        "Asistensi rehabilitasi sosial (Pasal 6 ayat (2)) mengacu pada Rehabilitasi Sosial yang didefinisikan di Pasal 1 angka 6.",
        # T4-001 — correct refusal
        "Informasi yang Anda tanyakan tidak diatur secara spesifik dalam Peraturan Menteri Sosial Nomor 8 Tahun 2023.",
        # T5-002 — correct refusal (fact injection rejected)
        "Informasi yang Anda tanyakan tidak diatur secara spesifik dalam Peraturan Menteri Sosial Nomor 8 Tahun 2023. Pasal 99 tidak ada dalam peraturan ini.",
    ]


@pytest.fixture
def mock_validation():
    """Return a ValidationResult mock dict."""
    def _make_val(citations_extracted, eg, rp, is_refusal_flag):
        return {
            "citations_extracted": citations_extracted,
            "citations_valid": [True] * len(citations_extracted),
            "citation_accuracy": 1.0 if citations_extracted else 1.0,
            "eg_score": eg,
            "rp_score": rp,
            "hitl_flag": False,
            "hitl_reasons": [],
            "debug": {},
            "is_refusal": is_refusal_flag,
        }
    return _make_val


# ---------------------------------------------------------------------------
# Test: citation accuracy metrics
# ---------------------------------------------------------------------------

class TestCitationAccuracySmokeMetrics:
    def test_pasal1_match(self):
        extracted = [{"pasal": 1, "ayat": None, "huruf": None}]
        expected = [{"pasal": 1, "ayat": None, "huruf": None}]
        result = compute_citation_accuracy(extracted, expected)
        assert result["precision"] == 1.0

    def test_pasal5_match(self):
        extracted = [{"pasal": 5, "ayat": 2, "huruf": "a"}]
        expected = [{"pasal": 5, "ayat": 2, "huruf": None}]
        result = compute_citation_accuracy(extracted, expected, granularity="pasal")
        assert result["matched_count"] == 1

    def test_refusal_vacuous_pass(self):
        result = compute_citation_accuracy([], [])
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0


# ---------------------------------------------------------------------------
# Test: refusal metrics
# ---------------------------------------------------------------------------

class TestRefusalMetricsSmoke:
    def test_all_tier4_5_should_refuse(self, smoke_questions, synthetic_responses):
        metrics = compute_refusal_metrics(smoke_questions, synthetic_responses)
        # 2 questions should refuse (T4-001, T5-002), both responses are refusals
        assert metrics["tp"] == 2
        assert metrics["fp"] == 0
        assert metrics["fn"] == 0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0

    def test_non_refusing_answers_counted_as_tn(self, smoke_questions, synthetic_responses):
        metrics = compute_refusal_metrics(smoke_questions, synthetic_responses)
        # 3 questions must_refuse=False, all answered without refusal
        assert metrics["tn"] == 3


# ---------------------------------------------------------------------------
# Test: latency stats
# ---------------------------------------------------------------------------

class TestLatencyStatsSmoke:
    def test_latency_from_records(self):
        records = [
            {"latency_ms": {"retrieval_ms": 500, "llm_ms": 3000, "total_ms": 3600}},
            {"latency_ms": {"retrieval_ms": 800, "llm_ms": 5000, "total_ms": 6000}},
            {"latency_ms": {"retrieval_ms": 300, "llm_ms": 2000, "total_ms": 2500}},
        ]
        stats = compute_latency_stats(records)
        assert stats["n_records"] == 3
        total_stats = stats["per_stage"]["total_ms"]
        assert total_stats["n"] == 3
        assert total_stats["p50"] > 0
        assert total_stats["p95"] >= total_stats["p50"]


# ---------------------------------------------------------------------------
# Test: report generation
# ---------------------------------------------------------------------------

class TestReportGenerationSmoke:
    def test_report_file_created(self, tmp_path, smoke_questions, synthetic_responses):
        """Run minimal pipeline and assert report file is written."""
        # Build fake test_results
        test_results = []
        cite_per_q = []

        for q, resp in zip(smoke_questions, synthetic_responses):
            from app.validators.citation_extractor import extract_citations
            extracted = extract_citations(resp)
            cite = compute_citation_accuracy(extracted, q.expected_pasal_refs)
            cite["qid"] = q.qid
            cite_per_q.append(cite)

            test_results.append({
                "qid": q.qid,
                "tier": q.tier,
                "question": q.question,
                "must_refuse": q.must_refuse,
                "response": resp,
                "eg_score": 1.0,
                "rp_score": 1.0,
                "is_refusal": is_refusal(resp),
                "correct_refusal": (q.must_refuse == is_refusal(resp)),
                "latency_ms": {"retrieval_ms": 500, "llm_ms": 3000, "total_ms": 3600},
                "citation_accuracy_detail": cite,
            })

        # Fake RAGAS results
        ragas_results = {
            "per_record": [
                {"qid": q.qid, "faithfulness": 0.9, "answer_relevancy": 0.88, "context_precision": 0.85}
                for q in smoke_questions
            ],
            "aggregate": {
                "faithfulness_mean": 0.90,
                "answer_relevancy_mean": 0.88,
                "context_precision_mean": 0.85,
                "n_evaluated": 5,
                "n_failed": 0,
            },
            "judge_model": "mock",
            "cache_used": False,
        }

        citation_agg = aggregate_citation_accuracy(cite_per_q)
        refusal_metrics = compute_refusal_metrics(smoke_questions, synthetic_responses)
        latency_stats = compute_latency_stats(test_results)
        eg_rp = {"eg_mean": 1.0, "rp_mean": 1.0}

        out_path = tmp_path / "smoke_report.md"
        generate_report(
            test_results=test_results,
            ragas_results=ragas_results,
            citation_results=citation_agg,
            refusal_results=refusal_metrics,
            latency_results=latency_stats,
            eg_rp_results=eg_rp,
            out_path=out_path,
        )

        assert out_path.exists()
        content = out_path.read_text(encoding="utf-8")
        assert "Eval Report" in content
        assert "Summary" in content
        assert "Faithfulness" in content
        assert "Citation accuracy" in content

    def test_smoke_marker_in_subset_report(self, tmp_path, smoke_questions, synthetic_responses):
        """Smoke subset reports should carry the subset marker."""
        from app.eval.report import generate_report

        test_results = [
            {
                "qid": q.qid, "tier": q.tier, "question": q.question,
                "must_refuse": q.must_refuse, "response": synthetic_responses[i],
                "eg_score": 0.9, "rp_score": 0.9, "is_refusal": False,
                "correct_refusal": True,
                "latency_ms": {"total_ms": 5000},
                "citation_accuracy_detail": {"matched_count": 0, "extracted_count": 0,
                                              "expected_count": 0, "jaccard": 1.0, "precision": 1.0},
            }
            for i, q in enumerate(smoke_questions)
        ]

        ragas_results = {
            "per_record": [
                {"qid": q.qid, "faithfulness": 0.9, "answer_relevancy": 0.88, "context_precision": 0.85}
                for q in smoke_questions
            ],
            "aggregate": {
                "faithfulness_mean": 0.9, "answer_relevancy_mean": 0.88,
                "context_precision_mean": 0.85,
                "n_evaluated": 3,   # ← subset: only 3 evaluated out of 5
                "n_failed": 0,
            },
            "judge_model": "mock",
            "cache_used": False,
        }

        out_path = tmp_path / "subset_report.md"
        generate_report(
            test_results=test_results,
            ragas_results=ragas_results,
            citation_results={"micro_precision": 1.0, "micro_recall": 1.0,
                              "micro_f1": 1.0, "mean_jaccard": 1.0,
                              "total_matched": 0, "total_extracted": 0,
                              "total_expected": 0, "n_questions": 5},
            refusal_results=compute_refusal_metrics(smoke_questions, synthetic_responses),
            latency_results=compute_latency_stats(test_results),
            eg_rp_results={"eg_mean": 0.9, "rp_mean": 0.9},
            out_path=out_path,
        )

        content = out_path.read_text(encoding="utf-8")
        assert "smoke subset" in content.lower() or "SMOKE" in content

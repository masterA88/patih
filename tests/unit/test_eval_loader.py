"""Unit tests for app.eval.testset_loader.

Tests cover:
  - Load valid JSONL
  - Schema validation: missing required fields
  - Schema validation: bad tier value
  - Schema validation: bad question_type value
  - Schema validation: bad language value
  - Edge: blank lines and comment lines skipped
  - Edge: must_refuse default=False
  - tier_distribution helper
"""

import json
import pytest
from pathlib import Path

from app.eval.testset_loader import load_testset, EvalQuestion, tier_distribution


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_valid_row(qid: str = "P8-T1-001", tier: int = 1) -> dict:
    return {
        "qid": qid,
        "tier": tier,
        "question_type": "definitional",
        "question": "Apa itu Korban TPPO?",
        "language": "id",
        "expected_pasal_refs": [{"pasal": 1, "ayat": None, "huruf": None}],
        "expected_answer_summary": "Korban TPPO adalah seseorang yang...",
        "must_refuse": False,
        "notes": None,
    }


def _write_jsonl(tmp_path: Path, rows: list[dict]) -> Path:
    p = tmp_path / "test.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return p


# ---------------------------------------------------------------------------
# Load valid test set
# ---------------------------------------------------------------------------

class TestLoadValidTestset:
    def test_load_single_row(self, tmp_path):
        p = _write_jsonl(tmp_path, [_make_valid_row()])
        qs = load_testset(p)
        assert len(qs) == 1
        assert qs[0].qid == "P8-T1-001"
        assert qs[0].tier == 1

    def test_load_multiple_rows(self, tmp_path):
        rows = [
            _make_valid_row("P8-T1-001", tier=1),
            _make_valid_row("P8-T2-001", tier=2),
            _make_valid_row("P8-T4-001", tier=4),
        ]
        qs = load_testset(_write_jsonl(tmp_path, rows))
        assert len(qs) == 3
        assert {q.tier for q in qs} == {1, 2, 4}

    def test_blank_lines_skipped(self, tmp_path):
        p = tmp_path / "test.jsonl"
        row = json.dumps(_make_valid_row())
        p.write_text(f"\n{row}\n\n{row}\n", encoding="utf-8")
        qs = load_testset(p)
        assert len(qs) == 2

    def test_must_refuse_defaults_false(self, tmp_path):
        row = _make_valid_row()
        del row["must_refuse"]  # omit optional field
        qs = load_testset(_write_jsonl(tmp_path, [row]))
        assert qs[0].must_refuse is False

    def test_notes_can_be_none(self, tmp_path):
        row = _make_valid_row()
        row["notes"] = None
        qs = load_testset(_write_jsonl(tmp_path, [row]))
        assert qs[0].notes is None

    def test_tier4_must_refuse_true(self, tmp_path):
        row = _make_valid_row("P8-T4-001", tier=4)
        row["question_type"] = "out_of_scope"
        row["must_refuse"] = True
        row["expected_pasal_refs"] = []
        qs = load_testset(_write_jsonl(tmp_path, [row]))
        assert qs[0].must_refuse is True


# ---------------------------------------------------------------------------
# Schema validation errors
# ---------------------------------------------------------------------------

class TestSchemaValidation:
    def test_missing_qid_raises(self, tmp_path):
        row = _make_valid_row()
        del row["qid"]
        with pytest.raises(ValueError, match="schema error"):
            load_testset(_write_jsonl(tmp_path, [row]))

    def test_bad_tier_raises(self, tmp_path):
        row = _make_valid_row()
        row["tier"] = 99  # invalid Literal
        with pytest.raises(ValueError, match="schema error"):
            load_testset(_write_jsonl(tmp_path, [row]))

    def test_bad_question_type_raises(self, tmp_path):
        row = _make_valid_row()
        row["question_type"] = "unknown_type"
        with pytest.raises(ValueError, match="schema error"):
            load_testset(_write_jsonl(tmp_path, [row]))

    def test_bad_language_raises(self, tmp_path):
        row = _make_valid_row()
        row["language"] = "fr"  # only id/en allowed
        with pytest.raises(ValueError, match="schema error"):
            load_testset(_write_jsonl(tmp_path, [row]))

    def test_invalid_json_raises(self, tmp_path):
        p = tmp_path / "test.jsonl"
        p.write_text("not valid json\n", encoding="utf-8")
        with pytest.raises(ValueError, match="JSON parse error"):
            load_testset(p)


# ---------------------------------------------------------------------------
# File not found
# ---------------------------------------------------------------------------

class TestFileNotFound:
    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_testset(tmp_path / "nonexistent.jsonl")


# ---------------------------------------------------------------------------
# tier_distribution
# ---------------------------------------------------------------------------

class TestTierDistribution:
    def test_full_distribution(self, tmp_path):
        rows = (
            [_make_valid_row(f"P8-T1-{i:03d}", tier=1) for i in range(25)]
            + [_make_valid_row(f"P8-T2-{i:03d}", tier=2) for i in range(10)]
            + [_make_valid_row(f"P8-T3-{i:03d}", tier=3) for i in range(5)]
            + [_make_valid_row(f"P8-T4-{i:03d}", tier=4) for i in range(8)]
            + [_make_valid_row(f"P8-T5-{i:03d}", tier=5) for i in range(2)]
        )
        qs = load_testset(_write_jsonl(tmp_path, rows))
        dist = tier_distribution(qs)
        assert dist == {1: 25, 2: 10, 3: 5, 4: 8, 5: 2}

    def test_empty_tiers_all_zero(self):
        dist = tier_distribution([])
        assert all(v == 0 for v in dist.values())


# ---------------------------------------------------------------------------
# Load actual 50q test set (integration-ish, no LLM)
# ---------------------------------------------------------------------------

class TestLoadActual50q:
    def test_actual_testset_loads(self):
        """Load the real permensos8_50q.jsonl and validate distribution."""
        path = Path("data/test/permensos8_50q.jsonl")
        if not path.exists():
            pytest.skip("Test set not yet generated")

        qs = load_testset(path)
        dist = tier_distribution(qs)

        assert len(qs) == 50
        assert dist[1] == 25, f"Expected 25 Tier 1, got {dist[1]}"
        assert dist[2] == 10, f"Expected 10 Tier 2, got {dist[2]}"
        assert dist[3] == 5,  f"Expected 5 Tier 3, got {dist[3]}"
        assert dist[4] == 8,  f"Expected 8 Tier 4, got {dist[4]}"
        assert dist[5] == 2,  f"Expected 2 Tier 5, got {dist[5]}"

        # All Tier 4+5 must have must_refuse=True
        for q in qs:
            if q.tier in (4, 5):
                assert q.must_refuse, f"{q.qid} should have must_refuse=True"

    def test_bilingual_mirror_loads(self):
        path = Path("data/test/permensos8_10q_en.jsonl")
        if not path.exists():
            pytest.skip("Bilingual mirror not yet generated")
        qs = load_testset(path)
        assert len(qs) == 10
        assert all(q.language == "en" for q in qs)

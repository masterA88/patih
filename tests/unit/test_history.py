"""Unit tests for app/ui/history.py (ConversationStore).

All tests use tmp_path so they never touch data/conversations.db.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from app.ui.history import ConversationStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def store(tmp_path) -> ConversationStore:
    """ConversationStore backed by a temp SQLite file."""
    return ConversationStore(db_path=tmp_path / "test_conversations.db")


def _make_result(**overrides) -> dict:
    """Minimal GenerationResult-like dict for testing."""
    base = {
        "response": "Bentuk eksploitasi meliputi...",
        "response_lang": "id",
        "query_lang": "id",
        "retrieved_pasals": [1, 5, 6],
        "llm_provider_used": "gemini/gemini-2.5-flash",
        "model_name_used": "gemini-flash",
        "fallback_chain_attempts": [],
        "query_translated": None,
        "latency_ms": {"retrieval_ms": 200.0, "llm_ms": 5000.0, "total_ms": 5300.0},
        "tokens_in": 1500,
        "tokens_out": 300,
        "validation": {
            "hitl_flag": False,
            "citation_accuracy": 1.0,
            "eg_score": 1.0,
            "rp_score": 1.0,
            "hitl_reasons": [],
            "citations_extracted": [{"pasal": 5, "ayat": 2, "huruf": "a"}],
            "citations_valid": [True],
        },
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Schema init
# ---------------------------------------------------------------------------

class TestInitSchema:
    def test_creates_db_file(self, tmp_path):
        db_path = tmp_path / "subdir" / "conv.db"
        # Parent dir doesn't exist yet — ConversationStore should create it.
        store = ConversationStore(db_path=db_path)
        assert db_path.exists()

    def test_creates_conversations_table(self, store, tmp_path):
        db_path = store._db_path
        conn = sqlite3.connect(db_path)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='conversations'"
        )
        row = cursor.fetchone()
        conn.close()
        assert row is not None, "conversations table not created"

    def test_creates_indexes(self, store):
        conn = sqlite3.connect(store._db_path)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        )
        index_names = {row[0] for row in cursor.fetchall()}
        conn.close()
        assert "idx_session" in index_names
        assert "idx_feedback" in index_names


# ---------------------------------------------------------------------------
# append()
# ---------------------------------------------------------------------------

class TestAppend:
    def test_returns_uuid_string(self, store):
        trace_id = store.append("sess-1", "query text", _make_result())
        assert isinstance(trace_id, str)
        assert len(trace_id) == 36  # UUID4 canonical form

    def test_inserts_row(self, store):
        store.append("sess-1", "Apa itu TPPO?", _make_result())
        rows = store.list_recent(limit=5)
        assert len(rows) == 1

    def test_row_fields_correct(self, store):
        result = _make_result()
        store.append("sess-abc", "query?", result)
        rows = store.list_recent(limit=1)
        row = rows[0]

        assert row["session_id"] == "sess-abc"
        assert row["user_query"] == "query?"
        assert row["response"] == result["response"]
        assert row["response_lang"] == "id"
        assert row["llm_provider_used"] == "gemini/gemini-2.5-flash"
        assert row["hitl_flag"] == 0
        assert row["citation_accuracy"] == pytest.approx(1.0)
        assert row["feedback"] is None

    def test_retrieved_pasals_json_roundtrip(self, store):
        result = _make_result()
        store.append("sess-1", "q", result)
        rows = store.list_recent(1)
        pasals = json.loads(rows[0]["retrieved_pasals"])
        assert pasals == [1, 5, 6]

    def test_hitl_flag_true_stored_as_1(self, store):
        result = _make_result(validation={"hitl_flag": True, "hitl_reasons": ["low_eg"], "citation_accuracy": 0.5, "eg_score": 0.7, "rp_score": 0.8})
        store.append("sess-1", "q", result)
        rows = store.list_recent(1)
        assert rows[0]["hitl_flag"] == 1

    def test_none_validation_stores_nulls(self, store):
        result = _make_result(validation=None)
        store.append("sess-1", "q", result)
        rows = store.list_recent(1)
        row = rows[0]
        assert row["citation_accuracy"] is None
        assert row["hitl_flag"] == 0   # None → not truthy → 0

    def test_multiple_appends_separate_rows(self, store):
        for i in range(5):
            store.append("sess-1", f"query {i}", _make_result())
        rows = store.list_recent(limit=10)
        assert len(rows) == 5

    def test_raw_json_is_valid_json(self, store):
        result = _make_result()
        store.append("sess-1", "q", result)
        rows = store.list_recent(1)
        raw = json.loads(rows[0]["raw_json"])
        assert raw["response"] == result["response"]


# ---------------------------------------------------------------------------
# add_feedback()
# ---------------------------------------------------------------------------

class TestAddFeedback:
    def test_thumbs_up_updates_last_turn(self, store):
        store.append("sess-2", "q1", _make_result())
        store.add_feedback("sess-2", feedback="thumbs_up")
        rows = store.list_recent(1)
        assert rows[0]["feedback"] == "thumbs_up"
        assert rows[0]["feedback_text"] == ""

    def test_thumbs_down_with_text(self, store):
        store.append("sess-2", "q1", _make_result())
        store.add_feedback("sess-2", feedback="thumbs_down", feedback_text="Jawaban kurang lengkap")
        rows = store.list_recent(1)
        assert rows[0]["feedback"] == "thumbs_down"
        assert rows[0]["feedback_text"] == "Jawaban kurang lengkap"

    def test_updates_only_last_unfeedback_turn(self, store):
        """Two turns — feedback should land on the more recent one, leaving the first alone."""
        store.append("sess-3", "q1", _make_result())
        store.append("sess-3", "q2", _make_result())
        store.add_feedback("sess-3", feedback="thumbs_up")

        rows = store.list_recent(limit=10)
        # rows[0] = newest (q2), rows[1] = oldest (q1)
        assert rows[0]["feedback"] == "thumbs_up"
        assert rows[1]["feedback"] is None

    def test_noop_when_no_matching_session(self, store):
        """add_feedback for unknown session_id should not raise."""
        store.add_feedback("nonexistent-session", feedback="thumbs_up")  # no exception

    def test_no_double_update(self, store):
        """Second feedback call should not overwrite the first (feedback IS NOT NULL)."""
        store.append("sess-4", "q1", _make_result())
        store.add_feedback("sess-4", feedback="thumbs_up")
        store.add_feedback("sess-4", feedback="thumbs_down")  # already set → noop
        rows = store.list_recent(1)
        assert rows[0]["feedback"] == "thumbs_up"


# ---------------------------------------------------------------------------
# list_recent()
# ---------------------------------------------------------------------------

class TestListRecent:
    def test_returns_rows_newest_first(self, store):
        for i in range(3):
            store.append("sess-5", f"query {i}", _make_result())
        rows = store.list_recent(limit=10)
        ts_list = [r["ts_created"] for r in rows]
        assert ts_list == sorted(ts_list, reverse=True)

    def test_honours_limit(self, store):
        for i in range(10):
            store.append("sess-6", f"q{i}", _make_result())
        rows = store.list_recent(limit=3)
        assert len(rows) == 3

    def test_empty_db_returns_empty_list(self, store):
        rows = store.list_recent()
        assert rows == []

"""Conversation persistence backed by SQLite (data/conversations.db), keyed by session_id.

Schema mirrors build-spec Section 4.3 ConversationLog.  Phase 1 does NOT use Langfuse
(zero-cost constraint); this SQLite store is the only persistence layer.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

# Allow CHATBOT_CONVERSATIONS_DB env override (HF Spaces writes to /tmp on free tier,
# /data when persistent storage is enabled). Falls back to repo-relative path locally.
DB_PATH = Path(os.environ.get("CHATBOT_CONVERSATIONS_DB", "data/conversations.db"))


class ConversationStore:
    """Thin SQLite wrapper.

    Thread-safety: each method opens a new connection so it is safe to call from
    different asyncio threads (asyncio.to_thread). SQLite's WAL mode is not needed
    for Phase 1 single-tenant load, but each write is committed immediately.
    """

    def __init__(self, db_path: Path = DB_PATH) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        conn = sqlite3.connect(self._db_path)
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS conversations (
                    trace_id            TEXT PRIMARY KEY,
                    session_id          TEXT NOT NULL,
                    user_query          TEXT NOT NULL,
                    response            TEXT NOT NULL,
                    response_lang       TEXT,
                    query_lang          TEXT,
                    retrieved_pasals    TEXT,
                    llm_provider_used   TEXT,
                    citation_accuracy   REAL,
                    eg_score            REAL,
                    rp_score            REAL,
                    hitl_flag           INTEGER,
                    hitl_reasons        TEXT,
                    feedback            TEXT,
                    feedback_text       TEXT,
                    latency_ms          TEXT,
                    raw_json            TEXT,
                    ts_created          TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_session  ON conversations(session_id);
                CREATE INDEX IF NOT EXISTS idx_feedback ON conversations(feedback);
                CREATE INDEX IF NOT EXISTS idx_ts       ON conversations(ts_created);
            """)
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def append(self, session_id: str, user_query: str, result: dict) -> str:
        """Insert one conversation turn.  Returns the trace_id (UUID4 string)."""
        trace_id = str(uuid4())
        validation = result.get("validation") or {}
        ts = datetime.now(timezone.utc).isoformat()

        conn = sqlite3.connect(self._db_path)
        try:
            conn.execute(
                """
                INSERT INTO conversations (
                    trace_id, session_id, user_query, response, response_lang,
                    query_lang, retrieved_pasals, llm_provider_used,
                    citation_accuracy, eg_score, rp_score, hitl_flag, hitl_reasons,
                    feedback, feedback_text, latency_ms, raw_json, ts_created
                ) VALUES (
                    ?, ?, ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?
                )
                """,
                (
                    trace_id,
                    session_id,
                    user_query,
                    result.get("response", ""),
                    result.get("response_lang"),
                    result.get("query_lang"),
                    json.dumps(result.get("retrieved_pasals", [])),
                    result.get("llm_provider_used"),
                    validation.get("citation_accuracy"),
                    validation.get("eg_score"),
                    validation.get("rp_score"),
                    1 if validation.get("hitl_flag") else 0,
                    json.dumps(validation.get("hitl_reasons", [])),
                    None,   # feedback — set later via add_feedback()
                    None,   # feedback_text
                    json.dumps(result.get("latency_ms", {})),
                    json.dumps(result),   # full backup
                    ts,
                ),
            )
            conn.commit()
        finally:
            conn.close()

        return trace_id

    def add_feedback(
        self,
        session_id: str,
        feedback: str,
        feedback_text: str = "",
    ) -> None:
        """Write feedback onto the most-recent unfeedback'd turn for this session.

        The feedback action carries the session_id (not the trace_id directly)
        because Chainlit's Action.value is set to session_id at send-time and we
        may not have the trace_id in scope at callback time.
        """
        conn = sqlite3.connect(self._db_path)
        try:
            conn.execute(
                """
                UPDATE conversations
                SET    feedback = ?, feedback_text = ?
                WHERE  trace_id = (
                    SELECT trace_id
                    FROM   conversations
                    WHERE  session_id = ? AND feedback IS NULL
                    ORDER  BY ts_created DESC
                    LIMIT  1
                )
                """,
                (feedback, feedback_text, session_id),
            )
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Read (debugging / future history dashboard)
    # ------------------------------------------------------------------

    def list_recent(self, limit: int = 20) -> list[dict]:
        """Return the most-recent *limit* rows, newest first."""
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute(
                "SELECT * FROM conversations ORDER BY ts_created DESC LIMIT ?",
                (limit,),
            )
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

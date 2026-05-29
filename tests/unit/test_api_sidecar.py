"""Unit tests for app/ui/api_sidecar.py.

Exercises /health and /api/query against a FastAPI app with the Generator
mocked out — so tests run fast (no real LLM, no Chroma load) and stay isolated
from API-key availability.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.ui.api_sidecar import register_api_routes


@pytest.fixture()
def fake_generation_result():
    """Returns an object whose .model_dump() mirrors GenerationResult shape."""
    obj = MagicMock()
    obj.model_dump.return_value = {
        "response": "Bentuk eksploitasi meliputi pelacuran (Pasal 5 ayat (2) huruf a).",
        "response_lang": "id",
        "retrieved_pasals": [1, 5],
        "llm_provider_used": "groq/llama-3.3-70b-versatile",
        "latency_ms": {"retrieval": 400, "llm": 3600, "total": 4000},
        "validation": {
            "citations": [
                {"pasal": 5, "ayat": 2, "huruf": "a", "raw": "Pasal 5 ayat (2) huruf a"},
            ],
            "citations_valid": [True],
            "eg_score": 1.0,
            "rp_score": 0.97,
            "hitl_flag": False,
        },
    }
    return obj


@pytest.fixture()
def client(fake_generation_result):
    """FastAPI TestClient with the sidecar mounted and a mocked Generator."""
    app = FastAPI()
    gen = MagicMock()
    gen.answer.return_value = fake_generation_result
    register_api_routes(app, lambda: gen)
    return TestClient(app), gen


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

def test_health_returns_ok(client):
    c, _ = client
    r = c.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_health_does_not_call_generator(client):
    c, gen = client
    c.get("/health")
    gen.answer.assert_not_called()


# ---------------------------------------------------------------------------
# /api/query
# ---------------------------------------------------------------------------

def test_query_happy_path(client):
    c, gen = client
    r = c.post("/api/query", json={"query": "Apa saja bentuk eksploitasi?"})
    assert r.status_code == 200
    data = r.json()
    assert "Pasal 5" in data["response"]
    assert data["response_lang"] == "id"
    assert data["retrieved_pasals"] == [1, 5]
    assert data["llm_provider_used"] == "groq/llama-3.3-70b-versatile"
    assert data["citations_valid"] == [True]
    assert data["citations"][0]["pasal"] == 5
    assert data["citations"][0]["ayat"] == 2
    assert data["citations"][0]["huruf"] == "a"
    assert data["eg_score"] == 1.0
    assert data["rp_score"] == 0.97
    assert data["hitl_flag"] is False
    gen.answer.assert_called_once()


def test_query_rejects_empty_string(client):
    c, _ = client
    r = c.post("/api/query", json={"query": ""})
    assert r.status_code == 422


def test_query_rejects_missing_field(client):
    c, _ = client
    r = c.post("/api/query", json={})
    assert r.status_code == 422


def test_query_rejects_oversize(client):
    c, _ = client
    r = c.post("/api/query", json={"query": "x" * 2001})
    assert r.status_code == 422


def test_query_503_when_generator_init_fails():
    app = FastAPI()

    def boom():
        raise RuntimeError("missing GEMINI_API_KEY")

    register_api_routes(app, boom)
    r = TestClient(app).post("/api/query", json={"query": "test"})
    assert r.status_code == 503
    assert "missing GEMINI_API_KEY" in r.json()["detail"]


def test_query_500_when_generator_answer_raises():
    app = FastAPI()
    gen = MagicMock()
    gen.answer.side_effect = RuntimeError("provider all failed")
    register_api_routes(app, lambda: gen)
    r = TestClient(app).post("/api/query", json={"query": "test"})
    assert r.status_code == 500
    assert "provider all failed" in r.json()["detail"]


def test_query_handles_missing_validation_block():
    """If Generator.answer() returns a result without validation (e.g. validate=False),
    sidecar still responds 200 with empty citations."""
    app = FastAPI()
    gen = MagicMock()
    minimal = MagicMock()
    minimal.model_dump.return_value = {
        "response": "Tidak ada validasi.",
        "response_lang": "id",
        "retrieved_pasals": [],
        "llm_provider_used": "gemini/gemini-2.5-flash",
        "latency_ms": {"total": 1234},
        "validation": None,
    }
    gen.answer.return_value = minimal
    register_api_routes(app, lambda: gen)
    r = TestClient(app).post("/api/query", json={"query": "test"})
    assert r.status_code == 200
    data = r.json()
    assert data["citations"] == []
    assert data["citations_valid"] == []
    assert data["eg_score"] is None
    assert data["rp_score"] is None
    assert data["hitl_flag"] is False

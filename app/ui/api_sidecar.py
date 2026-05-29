"""FastAPI sidecar mounted into the Chainlit ASGI app.

Provides REST endpoints used by:
  - deploy/scripts/smoke_test.sh (/health + /api/query)
  - external monitoring / health checks (Cloudflare, ufw)
  - eval rerun automation (so eval can hit the deployed service directly)

Mounted by app/ui/chainlit_app.py via `register_api_routes(chainlit.server.app)`.

This module does NOT import chainlit so it can be unit-tested standalone.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)


class CitationOut(BaseModel):
    pasal: int
    ayat: int | None = None
    huruf: str | None = None
    raw: str
    valid: bool


class QueryResponse(BaseModel):
    response: str
    response_lang: str
    retrieved_pasals: list[int]
    llm_provider_used: str | None
    latency_ms: dict[str, int]
    citations: list[CitationOut]
    citations_valid: list[bool]
    eg_score: float | None = None
    rp_score: float | None = None
    hitl_flag: bool = False


def _extract_citations(validation: dict | None) -> tuple[list[CitationOut], list[bool]]:
    """Extract citation list + validity flags from validation dict.

    `validation` is the dict produced by app.validators.pipeline.validate(),
    serialised via Pydantic .model_dump() on the GenerationResult.
    """
    if not validation:
        return [], []
    cites = validation.get("citations") or []
    valids = validation.get("citations_valid") or []
    out: list[CitationOut] = []
    for c, v in zip(cites, valids):
        out.append(CitationOut(
            pasal=c.get("pasal"),
            ayat=c.get("ayat"),
            huruf=c.get("huruf"),
            raw=c.get("raw", ""),
            valid=bool(v),
        ))
    return out, [bool(v) for v in valids]


def register_api_routes(
    app: FastAPI,
    generator_factory: Callable[[], Any],
) -> None:
    """Attach /health and /api/query to *app*.

    Args:
        app: a FastAPI / Starlette app (Chainlit exposes `chainlit.server.app`).
        generator_factory: zero-arg callable returning the singleton Generator.
            Indirection lets tests inject a mock without importing the heavy
            real Generator.
    """

    @app.get("/health", tags=["ops"])
    async def health() -> dict[str, str]:
        """Liveness probe. Returns 200 OK once the process is up.

        Does NOT touch the Generator — readiness with model warmup is intentionally
        not gated here to keep the probe cheap. Use /api/query for end-to-end smoke.
        """
        return {"status": "ok"}

    @app.post("/api/query", response_model=QueryResponse, tags=["qa"])
    async def query(req: QueryRequest) -> QueryResponse:
        """Single-turn QA endpoint — used by smoke_test.sh and external eval drivers."""
        import asyncio
        try:
            generator = generator_factory()
        except Exception as exc:
            logger.exception("Generator init failed")
            raise HTTPException(status_code=503, detail=f"generator not ready: {exc}")

        try:
            result = await asyncio.to_thread(generator.answer, req.query, validate=True)
        except Exception as exc:
            logger.exception("Generator.answer failed for query=%r", req.query[:80])
            raise HTTPException(status_code=500, detail=str(exc))

        d = result.model_dump() if hasattr(result, "model_dump") else dict(result)
        validation = d.get("validation") or {}
        citations, citations_valid = _extract_citations(validation)

        return QueryResponse(
            response=d.get("response", ""),
            response_lang=d.get("response_lang", "id"),
            retrieved_pasals=d.get("retrieved_pasals", []),
            llm_provider_used=d.get("llm_provider_used"),
            latency_ms=d.get("latency_ms", {}),
            citations=citations,
            citations_valid=citations_valid,
            eg_score=validation.get("eg_score"),
            rp_score=validation.get("rp_score"),
            hitl_flag=bool(validation.get("hitl_flag", False)),
        )

    # Chainlit registers a `/{full_path:path}` SPA catch-all BEFORE this module
    # is imported. FastAPI/Starlette match routes in declaration order, so a
    # naive `@app.get` here would be shadowed by the catch-all for GET requests
    # (e.g. /health returns the SPA HTML instead of JSON). Move our routes to
    # the front of the router so they take precedence.
    _OURS = {"/health", "/api/query"}
    ours = [r for r in app.router.routes if getattr(r, "path", None) in _OURS]
    others = [r for r in app.router.routes if getattr(r, "path", None) not in _OURS]
    app.router.routes[:] = ours + others

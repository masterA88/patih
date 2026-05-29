"""High-level generation: end-to-end query → response pipeline.

Flow (per spec Section 5.3 + Section 6 bilingual handling):
  1. Detect language of user query (ID / EN / mixed).
  2. If EN: translate query → ID for retrieval (response still in EN).
  3. Retrieve via app.retrieval.pipeline.retrieve(query_for_retrieval).
  4. Build messages: system prompt (lang-aware) + user prompt with context.
  5. Call LLMGateway.generate().
  6. Return structured GenerationResult.

Lazy initialization: retrieval pipeline and gateway are heavy — load on first
call to answer() and reuse across subsequent calls.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

class GenerationResult(BaseModel):
    """Structured result from Generator.answer().

    Matches spec Section 5.3 GenerationResult interface plus bilingual fields.
    Step 5 addition: optional 'validation' dict (None when validate=False).
    Step 7 fix: parent_chunks carries raw chunk dicts for RAGAS context extraction.
    """
    response: str
    response_lang: str                        # 'id' or 'en'
    retrieved_pasals: list[int | str]          # pasal numbers from context
    parent_chunks: list[dict] = Field(default_factory=list)  # raw chunk dicts (text, pasal, etc.)
    llm_provider_used: str                    # e.g. 'gemini/gemini-2.5-flash'
    model_name_used: str                      # LiteLLM model_name group
    fallback_chain_attempts: list[dict]       # per-attempt records from gateway
    query_lang: str                           # 'id', 'en', or 'mixed'
    query_translated: str | None              # translated query, or None if not needed
    latency_ms: dict[str, float]             # {retrieval_ms, llm_ms, total_ms}
    tokens_in: int
    tokens_out: int
    validation: dict | None = None           # ValidationResult.model_dump() or None

    model_config = {
        "arbitrary_types_allowed": True,
        "protected_namespaces": (),   # suppress "model_" namespace warning
    }


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class Generator:
    """End-to-end LLM generation with retrieval, lang-detect, and translation."""

    def __init__(self) -> None:
        # Lazy-init heavy components on first answer() call
        self._gateway = None
        self._lang_detector = None
        self._translator = None
        self._retrieval_fn = None

    def _ensure_initialized(self) -> None:
        if self._gateway is not None:
            return

        from app.llm.gateway import LLMGateway
        from app.llm.lang_detect import LangDetector
        from app.llm.translator import Translator
        from app.retrieval.pipeline import retrieve

        self._gateway = LLMGateway()
        self._lang_detector = LangDetector()
        self._translator = Translator()
        self._retrieval_fn = retrieve
        logger.info("Generator initialized.")

    def answer(
        self,
        user_query: str,
        include_fewshot: bool = False,
        top_k_dense: int = 15,
        top_k_sparse: int = 15,
        top_k_fused: int = 12,
        validate: bool = True,
    ) -> GenerationResult:
        """End-to-end: detect lang → (translate) → retrieve → build prompt → LLM → return.

        Args:
            user_query:      Raw query from user (ID or EN).
            include_fewshot: Inject few-shot examples into system prompt.
            top_k_dense:     Passed through to retrieval pipeline.
            top_k_sparse:    Passed through to retrieval pipeline.
            top_k_fused:     Passed through to retrieval pipeline.
            validate:        If True (default), run Layer 2 validation and populate
                             result.validation dict. Set False in unit tests that
                             mock LLM output to skip Chroma dependency.

        Returns:
            GenerationResult with response, citations, provider info, latency,
            and optional validation dict.
        """
        self._ensure_initialized()
        t_total_start = time.monotonic()

        # ------------------------------------------------------------------
        # Step 1: Language detection
        # ------------------------------------------------------------------
        query_lang = self._lang_detector.detect(user_query)
        logger.info("Generator: query_lang='%s' for '%s...'", query_lang, user_query[:50])

        # Determine response language: 'en' for EN queries, 'id' for everything else
        response_lang = "en" if query_lang == "en" else "id"

        # ------------------------------------------------------------------
        # Step 2: Translation for retrieval (EN only)
        # ------------------------------------------------------------------
        query_translated: str | None = None
        query_for_retrieval = user_query

        if query_lang == "en":
            query_translated = self._translator.translate_en_to_id(user_query)
            if query_translated != user_query:
                query_for_retrieval = query_translated
                logger.info(
                    "Generator: translated '%s...' → '%s...'",
                    user_query[:40], query_for_retrieval[:40],
                )

        # ------------------------------------------------------------------
        # Step 3: Retrieval
        # ------------------------------------------------------------------
        t_retrieval_start = time.monotonic()
        retrieval_result = self._retrieval_fn(
            query_for_retrieval,
            top_k_dense=top_k_dense,
            top_k_sparse=top_k_sparse,
            top_k_fused=top_k_fused,
        )
        retrieval_ms = (time.monotonic() - t_retrieval_start) * 1000

        parent_chunks = retrieval_result.parent_chunks
        logger.info(
            "Generator: retrieved %d parent chunks in %.0fms",
            len(parent_chunks), retrieval_ms,
        )

        # ------------------------------------------------------------------
        # Step 4: Build messages
        # ------------------------------------------------------------------
        from app.llm.prompts import build_messages

        messages = build_messages(
            query=user_query,          # original query, not translated
            context=parent_chunks,
            lang=response_lang,
            include_fewshot=include_fewshot,
        )

        # ------------------------------------------------------------------
        # Step 5: LLM generation
        # ------------------------------------------------------------------
        t_llm_start = time.monotonic()
        llm_result = self._gateway.generate(
            messages=messages,
            max_tokens=1500,
            temperature=0.1,
        )
        llm_ms = (time.monotonic() - t_llm_start) * 1000
        total_ms = (time.monotonic() - t_total_start) * 1000

        # ------------------------------------------------------------------
        # Step 6: Extract pasal numbers from retrieved context
        # ------------------------------------------------------------------
        retrieved_pasals: list[int | str] = []
        for chunk in parent_chunks:
            pasal = chunk.get("pasal")
            if pasal is not None and pasal not in retrieved_pasals:
                retrieved_pasals.append(pasal)

        logger.info(
            "Generator: done in %.0fms (retrieval=%.0fms, llm=%.0fms) via %s",
            total_ms, retrieval_ms, llm_ms, llm_result["provider_used"],
        )

        # ------------------------------------------------------------------
        # Step 7: Layer 2 validation (optional, default=True)
        # ------------------------------------------------------------------
        validation_dict: dict | None = None
        if validate:
            try:
                from app.validators.pipeline import (
                    validate as run_validation,
                    append_to_hitl_queue,
                )
                # Fallback doc_id = dominant doc in context (validate() primarily
                # uses the full set of context doc_ids; this is only a fallback).
                from collections import Counter
                _doc_counts = Counter(
                    c.get("doc_id") for c in parent_chunks if c.get("doc_id")
                )
                _fallback_doc = (
                    _doc_counts.most_common(1)[0][0] if _doc_counts else "permensos-8-2023"
                )
                val_result = run_validation(
                    response=llm_result["response"],
                    context=parent_chunks,
                    doc_id=_fallback_doc,
                )
                validation_dict = val_result.model_dump()

                if val_result.hitl_flag:
                    append_to_hitl_queue(
                        validation_result=val_result,
                        response=llm_result["response"],
                        user_query=user_query,
                        retrieved_pasals=retrieved_pasals,
                    )
                    logger.warning(
                        "HITL flag raised for query '%s...' reasons=%s",
                        user_query[:50], val_result.hitl_reasons,
                    )
            except Exception as exc:
                logger.error("Validation error (non-fatal): %s", exc, exc_info=True)
                validation_dict = {"error": str(exc)}

        return GenerationResult(
            response=llm_result["response"],
            response_lang=response_lang,
            retrieved_pasals=retrieved_pasals,
            parent_chunks=parent_chunks,
            llm_provider_used=llm_result["provider_used"],
            model_name_used=llm_result["model_name_used"],
            fallback_chain_attempts=llm_result["fallback_chain_attempts"],
            query_lang=query_lang,
            query_translated=query_translated,
            latency_ms={
                "retrieval_ms": round(retrieval_ms, 1),
                "llm_ms": round(llm_ms, 1),
                "total_ms": round(total_ms, 1),
            },
            tokens_in=llm_result["tokens_in"],
            tokens_out=llm_result["tokens_out"],
            validation=validation_dict,
        )

"""LiteLLM Router configuration: 3-tier fallback chain.

Provider chain (per spec Section 5.3, updated eval-baseline-v3):
  1. groq/llama-3.3-70b-versatile — primary (30 RPM / 14,400 RPD)
  2. gemini/gemini-2.5-flash       — fallback-1 (10 RPM free tier)
  3. cerebras/llama3.1-8b           — fallback-2 (30 RPM, qwen-3-32b discontinued)
  OpenRouter dropped: deepseek-r1:free endpoint returns 404 (discontinued).

Provider filtering: any provider whose API key env var is missing or empty is
silently skipped at init time — no crash, just a reduced fallback chain.

LiteLLM 1.45.0 notes:
- Router supports model_name grouping with fallbacks between named groups.
- Using distinct model_names per provider is more explicit and avoids
  "simple-shuffle" mixing providers when we want ordered fallback.
- fallbacks param takes list[dict]: {"primary": ["fallback-1", "fallback-2", ...]}
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import litellm
from litellm import Router

logger = logging.getLogger(__name__)

# Suppress verbose litellm HTTP logs unless DEBUG
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Provider definitions
# ---------------------------------------------------------------------------

_PROVIDERS = [
    {
        "model_name": "chatbot-primary",
        "env_key": "GROQ_API_KEY",
        "provider_label": "groq",
        "litellm_params": {
            "model": "groq/llama-3.3-70b-versatile",
            "rpm": 30,
        },
    },
    {
        "model_name": "chatbot-fallback-1",
        "env_key": "GEMINI_API_KEY",
        "provider_label": "gemini",
        "litellm_params": {
            "model": "gemini/gemini-2.5-flash",
            "rpm": 10,
        },
    },
    {
        "model_name": "chatbot-fallback-2",
        "env_key": "CEREBRAS_API_KEY",
        "provider_label": "cerebras",
        "litellm_params": {
            # qwen-3-32b was removed from Cerebras (404 as of 2026-05-18).
            # llama3.1-8b is the confirmed-working free-tier slug.
            "model": "cerebras/llama3.1-8b",
            "rpm": 30,
        },
    },
    # OpenRouter removed: deepseek/deepseek-r1:free endpoint returns 404.
    # Re-add in Phase 2 once a working OpenRouter model slug is confirmed.
]


def _build_model_list() -> list[dict[str, Any]]:
    """Filter providers to those with a non-empty API key env var."""
    configured: list[dict[str, Any]] = []
    skipped: list[str] = []

    for provider in _PROVIDERS:
        api_key = os.environ.get(provider["env_key"], "").strip()
        if not api_key:
            skipped.append(provider["provider_label"])
            continue
        entry: dict[str, Any] = {
            "model_name": provider["model_name"],
            "litellm_params": {
                **provider["litellm_params"],
                "api_key": api_key,
            },
        }
        configured.append(entry)

    total = len(_PROVIDERS)
    n_configured = len(configured)
    n_skipped = len(skipped)
    logger.info(
        "LLMGateway: %d/%d provider(s) configured; %d skipped (no API key): %s",
        n_configured, total, n_skipped, skipped if skipped else "none",
    )

    if n_configured == 0:
        raise RuntimeError(
            "LLMGateway: no providers configured — set at least GROQ_API_KEY in .env"
        )

    return configured


def _build_fallback_chain(model_list: list[dict[str, Any]]) -> list[dict[str, list[str]]]:
    """Build LiteLLM fallbacks list: primary → [fb1, fb2, fb3] (ordered)."""
    if len(model_list) <= 1:
        return []

    names = [m["model_name"] for m in model_list]
    # Each entry: {primary: [all subsequent model_names]}
    # LiteLLM v1.45 fallbacks format: list of dicts with single key
    fallbacks = []
    for i, name in enumerate(names[:-1]):
        fallbacks.append({name: names[i + 1 :]})
    return fallbacks


class LLMGateway:
    """LiteLLM Router wrapper with 3-tier fallback chain.

    Usage:
        gateway = LLMGateway()
        result = gateway.generate(messages, max_tokens=1500, temperature=0.1)
        print(result["response"])
        print(result["provider_used"])
    """

    def __init__(self) -> None:
        model_list = _build_model_list()
        fallbacks = _build_fallback_chain(model_list)

        self._model_list = model_list
        self._primary_model_name = model_list[0]["model_name"]
        self._configured_models = [m["model_name"] for m in model_list]

        router_kwargs: dict[str, Any] = {
            "model_list": model_list,
            "num_retries": 1,
            "timeout": 30,
            "routing_strategy": "simple-shuffle",
        }
        if fallbacks:
            router_kwargs["fallbacks"] = fallbacks

        self._router = Router(**router_kwargs)
        logger.info(
            "LLMGateway ready. Primary: %s. Fallback chain: %s",
            self._primary_model_name,
            self._configured_models[1:] if len(self._configured_models) > 1 else "none",
        )

    def generate(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 1500,
        temperature: float = 0.1,
    ) -> dict[str, Any]:
        """Generate a completion via the fallback chain.

        Returns:
            {
                "response":                str,
                "provider_used":           str,   # model string of provider that responded
                "model_name_used":         str,   # LiteLLM model_name group
                "fallback_chain_attempts": list[dict],
                "latency_ms":              float,
                "tokens_in":               int,
                "tokens_out":              int,
            }
        """
        t0 = time.monotonic()
        attempts: list[dict[str, Any]] = []
        last_exc: Exception | None = None

        # Try providers in order, respecting LiteLLM Router fallback logic.
        # We iterate manually so we can record per-attempt latency.
        for model_entry in self._model_list:
            model_name = model_entry["model_name"]
            provider_model = model_entry["litellm_params"]["model"]
            t_attempt = time.monotonic()

            try:
                response = self._router.completion(
                    model=model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                elapsed_ms = (time.monotonic() - t_attempt) * 1000
                attempts.append({
                    "model_name": model_name,
                    "provider_model": provider_model,
                    "status": "success",
                    "latency_ms": round(elapsed_ms, 1),
                })

                total_ms = (time.monotonic() - t0) * 1000
                usage = response.usage if hasattr(response, "usage") and response.usage else None

                return {
                    "response": response.choices[0].message.content or "",
                    "provider_used": provider_model,
                    "model_name_used": model_name,
                    "fallback_chain_attempts": attempts,
                    "latency_ms": round(total_ms, 1),
                    "tokens_in": usage.prompt_tokens if usage else 0,
                    "tokens_out": usage.completion_tokens if usage else 0,
                }

            except Exception as exc:
                elapsed_ms = (time.monotonic() - t_attempt) * 1000
                logger.warning(
                    "LLMGateway: %s (%s) failed — %s: %s",
                    model_name, provider_model, type(exc).__name__, str(exc)[:160],
                )
                attempts.append({
                    "model_name": model_name,
                    "provider_model": provider_model,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {str(exc)[:120]}",
                    "latency_ms": round(elapsed_ms, 1),
                })
                last_exc = exc
                continue

        # All providers exhausted
        total_ms = (time.monotonic() - t0) * 1000
        logger.error(
            "LLMGateway: all %d provider(s) failed. Last error: %s",
            len(self._model_list), last_exc,
        )
        raise RuntimeError(
            f"All LLM providers failed after {total_ms:.0f}ms. "
            f"Attempts: {[a['model_name'] for a in attempts]}. "
            f"Last error: {last_exc}"
        ) from last_exc

"""Integration test: mock provider rate-limit and verify LiteLLM fallback chain.

Strategy: patch the Router.completion() method to raise RateLimitError on the
first call (simulating Gemini 429), then return a valid response on the second
call (simulating Groq success). Verify the gateway's fallback_chain_attempts
records both the failure and the success.

Note: This test does NOT require real API keys. It mocks at the Router level.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root on sys.path
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Set dummy env vars so gateway can initialize with multiple providers
os.environ.setdefault("GEMINI_API_KEY", "test-gemini-key")
os.environ.setdefault("GROQ_API_KEY", "test-groq-key")
os.environ.setdefault("CEREBRAS_API_KEY", "test-cerebras-key")


from litellm import RateLimitError


def _make_mock_response(content: str = "Test response about Pasal 1.") -> MagicMock:
    """Build a mock litellm ModelResponse-like object."""
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    usage = MagicMock()
    usage.prompt_tokens = 100
    usage.completion_tokens = 50
    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    return response


# ---------------------------------------------------------------------------
# Gateway fallback tests
# ---------------------------------------------------------------------------

class TestGatewayFallback:
    """Test that gateway iterates through providers on failure."""

    def test_primary_success_no_fallback(self):
        """When primary succeeds, fallback_chain_attempts has exactly 1 entry."""
        from app.llm.gateway import LLMGateway

        mock_response = _make_mock_response("Response from primary.")

        with patch("litellm.Router.completion", return_value=mock_response) as mock_comp:
            gw = LLMGateway()
            result = gw.generate(
                messages=[{"role": "user", "content": "test"}],
                max_tokens=100,
            )

        assert result["response"] == "Response from primary."
        assert len(result["fallback_chain_attempts"]) == 1
        assert result["fallback_chain_attempts"][0]["status"] == "success"

    def test_primary_fail_fallback_triggers(self):
        """Gemini 429 → gateway falls to next provider and records both attempts."""
        from app.llm.gateway import LLMGateway

        mock_success = _make_mock_response("Fallback response about Pasal 5.")
        call_count = {"n": 0}

        def side_effect(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RateLimitError(
                    message="Rate limit exceeded",
                    llm_provider="gemini",
                    model="gemini/gemini-2.5-flash",
                )
            return mock_success

        with patch("litellm.Router.completion", side_effect=side_effect):
            gw = LLMGateway()
            result = gw.generate(
                messages=[{"role": "user", "content": "apa itu TPPO?"}],
                max_tokens=100,
            )

        assert result["response"] == "Fallback response about Pasal 5."
        assert len(result["fallback_chain_attempts"]) == 2
        assert result["fallback_chain_attempts"][0]["status"] == "error"
        assert result["fallback_chain_attempts"][1]["status"] == "success"

    def test_all_providers_fail_raises(self):
        """When all providers fail, RuntimeError is raised with chain info."""
        from app.llm.gateway import LLMGateway

        with patch(
            "litellm.Router.completion",
            side_effect=RateLimitError(
                message="Rate limit", llm_provider="gemini", model="gemini/gemini-2.5-flash"
            ),
        ):
            gw = LLMGateway()
            with pytest.raises(RuntimeError, match="All LLM providers failed"):
                gw.generate(
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=10,
                )

    def test_result_contains_latency_info(self):
        """generate() always returns latency_ms."""
        from app.llm.gateway import LLMGateway

        with patch(
            "litellm.Router.completion",
            return_value=_make_mock_response("OK"),
        ):
            gw = LLMGateway()
            result = gw.generate(
                messages=[{"role": "user", "content": "test"}],
                max_tokens=10,
            )

        assert "latency_ms" in result
        assert result["latency_ms"] >= 0


# ---------------------------------------------------------------------------
# Provider filter test
# ---------------------------------------------------------------------------

class TestProviderFilter:
    """Test that providers without API keys are silently skipped."""

    def test_only_configured_providers_in_chain(self, monkeypatch):
        """Gateway skips providers with missing API keys.

        After eval-baseline-v3 provider reorder: Groq is primary.
        Only GROQ_API_KEY set → chatbot-primary present, all fallbacks absent.
        """
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        # Remove other keys
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("CEREBRAS_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        from app.llm import gateway as gw_module
        # Reload to pick up env changes
        import importlib
        importlib.reload(gw_module)

        model_list = gw_module._build_model_list()
        model_names = [m["model_name"] for m in model_list]
        assert "chatbot-primary" in model_names
        assert "chatbot-fallback-1" not in model_names
        assert len(model_list) == 1

    def test_no_keys_raises_runtime_error(self, monkeypatch):
        """No configured providers → RuntimeError at gateway init."""
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.delenv("CEREBRAS_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        from app.llm import gateway as gw_module
        import importlib
        importlib.reload(gw_module)

        with pytest.raises(RuntimeError, match="no providers configured"):
            gw_module._build_model_list()

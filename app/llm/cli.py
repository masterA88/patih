"""CLI smoke-test interface for the LLM generator.

Usage:
    python -m app.llm.cli "apa itu korban TPPO?"
    python -m app.llm.cli "What are the forms of exploitation?"
    python -m app.llm.cli --fewshot "apa itu korban TPPO?"

Output: response + retrieved Pasals + provider + latency breakdown.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path when run as a module
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Load .env before any imports that need API keys
from dotenv import load_dotenv
load_dotenv(dotenv_path=_project_root / ".env")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chatbot Permensos LLM generator CLI smoke-test"
    )
    parser.add_argument("query", help="User query (Bahasa Indonesia or English)")
    parser.add_argument(
        "--fewshot", action="store_true", default=False,
        help="Inject few-shot examples into system prompt",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", default=False,
        help="Enable debug logging",
    )
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Suppress noisy sub-module logs unless verbose
    if not args.verbose:
        for noisy in ("httpx", "httpcore", "LiteLLM", "chromadb", "sentence_transformers"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    print(f"\nQuery: {args.query!r}\n{'=' * 60}")

    from app.llm.generator import Generator
    generator = Generator()

    result = generator.answer(args.query, include_fewshot=args.fewshot)

    print("\n--- RESPONSE ---")
    print(result.response)
    print("\n--- METADATA ---")
    print(f"Query lang:      {result.query_lang}")
    if result.query_translated:
        print(f"Translated to:   {result.query_translated!r}")
    print(f"Response lang:   {result.response_lang}")
    print(f"Provider:        {result.llm_provider_used}")
    print(f"Retrieved Pasals: {result.retrieved_pasals}")
    print(f"Latency:         retrieval={result.latency_ms['retrieval_ms']:.0f}ms  "
          f"llm={result.latency_ms['llm_ms']:.0f}ms  "
          f"total={result.latency_ms['total_ms']:.0f}ms")
    print(f"Tokens:          in={result.tokens_in}  out={result.tokens_out}")

    if len(result.fallback_chain_attempts) > 1:
        print("\n--- FALLBACK CHAIN ---")
        for attempt in result.fallback_chain_attempts:
            status = attempt["status"]
            model = attempt["provider_model"]
            ms = attempt["latency_ms"]
            if status == "success":
                print(f"  [OK]   {model}  ({ms:.0f}ms)")
            else:
                print(f"  [FAIL] {model}  ({ms:.0f}ms) — {attempt.get('error','')[:80]}")
    print()


if __name__ == "__main__":
    main()

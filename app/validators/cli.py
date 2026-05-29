"""CLI for manual testing of the validation pipeline.

Usage:
    # Full pipeline from a live Generator query:
    python -m app.validators.cli --query "Apa saja bentuk eksploitasi?"

    # From pre-generated files:
    python -m app.validators.cli --response-file out.txt --context-file ctx.json

    # Adversarial test with inline response:
    python -m app.validators.cli --response "Pasal 99 mengatur sanksi pidana." --context-json "[]"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _pretty_print(result) -> None:
    """Print ValidationResult in human-readable form."""
    data = result.model_dump()

    print("\n=== VALIDATION RESULT ===")
    print(f"  Refusal detected:     {data['is_refusal']}")
    print(f"  Citations extracted:  {len(data['citations_extracted'])}")
    for i, (cit, valid) in enumerate(zip(data['citations_extracted'], data['citations_valid'])):
        mark = "OK" if valid else "INVALID"
        print(f"    [{mark}] {cit['raw']}")
    print(f"  Citation accuracy:    {data['citation_accuracy']:.3f}")
    print(f"  EG score:             {data['eg_score']:.3f}  (threshold >= 0.95)")
    print(f"  RP score:             {data['rp_score']:.3f}  (threshold >= 0.85)")
    print(f"  HITL flag:            {data['hitl_flag']}")
    if data['hitl_reasons']:
        print(f"  HITL reasons:         {data['hitl_reasons']}")
    print(f"  Elapsed:              {data['debug'].get('elapsed_ms', '?')}ms")

    # EG debug
    eg = data['debug'].get('eg', {})
    if eg.get('missing_from_context'):
        print(f"\n  EG entities missing from context: {eg['missing_from_context']}")

    # RP debug
    rp = data['debug'].get('rp', {})
    failed_claims = [c for c in rp.get('claim_details', []) if c.get('status') == 'fail']
    if failed_claims:
        print(f"\n  RP failed claims ({len(failed_claims)}):")
        for c in failed_claims[:5]:
            print(f"    Pasal {c['pasal']}: '{c['sentence'][:80]}...'")
            print(f"      overlap_count={c.get('overlap_count', 0)}")

    print("=========================\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate an LLM response for citation accuracy, entity grounding, and relation preservation."
    )

    # Input modes
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--query", type=str,
        help="Run Generator.answer() with this query, then validate the response."
    )
    input_group.add_argument(
        "--response", type=str,
        help="Inline response string to validate."
    )
    input_group.add_argument(
        "--response-file", type=Path,
        help="Path to a file containing the response text."
    )

    # Context sources (for --response / --response-file modes)
    parser.add_argument(
        "--context-file", type=Path,
        help="Path to a JSON file containing a list of parent_chunk dicts."
    )
    parser.add_argument(
        "--context-json", type=str, default="[]",
        help="Inline JSON array of parent_chunk dicts (default: empty list)."
    )
    parser.add_argument(
        "--doc-id", type=str, default="permensos-8-2023",
        help="Document ID for registry lookup (default: permensos-8-2023)."
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output full ValidationResult as JSON instead of pretty-print."
    )

    args = parser.parse_args()

    # --- Mode: --query (full pipeline) ---
    if args.query:
        print(f"Running Generator.answer('{args.query[:60]}...' if len > 60 else '{args.query}')...")
        from app.llm.generator import Generator
        g = Generator()
        gen_result = g.answer(args.query, validate=False)  # we'll validate below

        response = gen_result.response
        context = []
        # Re-run retrieval to get context chunks for the validator
        # (generator doesn't expose parent_chunks in GenerationResult directly)
        # Simplest: import retrieval pipeline
        from app.retrieval.pipeline import retrieve
        retrieval_result = retrieve(args.query)
        context = retrieval_result.parent_chunks

        print(f"Response ({len(response)} chars):\n{response[:500]}...\n")

    # --- Mode: --response (inline) ---
    elif args.response:
        response = args.response
        if args.context_file:
            with open(args.context_file, "r", encoding="utf-8") as f:
                context = json.load(f)
        else:
            context = json.loads(args.context_json)

    # --- Mode: --response-file ---
    else:
        with open(args.response_file, "r", encoding="utf-8") as f:
            response = f.read()
        if args.context_file:
            with open(args.context_file, "r", encoding="utf-8") as f:
                context = json.load(f)
        else:
            context = json.loads(args.context_json)

    # --- Run validation ---
    from app.validators.pipeline import validate
    result = validate(response=response, context=context, doc_id=args.doc_id)

    if args.json:
        print(result.model_dump_json(indent=2))
    else:
        _pretty_print(result)

    # Exit code: 0 = pass, 1 = hitl_flag
    sys.exit(1 if result.hitl_flag else 0)


if __name__ == "__main__":
    main()

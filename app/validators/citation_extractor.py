"""Regex extractor for [Pasal X ayat (Y) huruf z] citations from LLM response text.

Spec: build-spec Section 5.4 (line 929-991).

Design decisions (Phase 1):
- Pattern matches "Pasal N", "Pasal N ayat (M)", "Pasal N ayat (M) huruf x".
- Case-insensitive (handles "PASAL").
- "Pasal 5 dan 6" produces 2 separate matches via finditer — the "5" match fires,
  then a second pass on the original string also finds "6" (captured as bare "Pasal 6"
  by the regex scanning left-to-right; actually finditer captures both because the
  second number appears as "Pasal 6" only if written explicitly).
  For "Pasal 5 dan 6" the second number 6 is NOT a "Pasal 6" match — spec says
  Phase 1 is OK to miss this; we only capture explicit "Pasal N" patterns.
- "Pasal-Pasal 5 sampai 7": the hyphen causes "Pasal" not to start the pattern
  at "Pasal-Pasal"; however the second "Pasal" in "Pasal-Pasal" starts a new match
  boundary. We capture "Pasal 5", skip range expansion. Document trade-off below.
- "Pasal 5 ayat (2) huruf a, huruf b, huruf c": regex captures first match
  "Pasal 5 ayat (2) huruf a" only. Phase 1 accepts single-capture for multi-huruf
  listings. Phase 2: extend with post-processing of trailing ", huruf x" patterns.
- Duplicate citations are preserved (dedup is caller's responsibility).
"""

from __future__ import annotations

import re
from typing import Any

# ---------------------------------------------------------------------------
# Pattern
# ---------------------------------------------------------------------------
# Matches (case-insensitive, non-greedy):
#   group 1: pasal number (digits)
#   group 2: ayat number (digits) — optional
#   group 3: huruf letter (a-z) — optional, requires ayat to be present per legal
#             citation convention, but regex does not enforce this to be lenient.
#
# Note on "Pasal-Pasal X": The pattern requires Pasal to be preceded by a word
# boundary or non-word character. "Pasal-Pasal" means the second "Pasal" is
# preceded by "-", which is a non-word char, so it will match "Pasal X".
# This is acceptable for Phase 1 (we get "Pasal 5" from "Pasal-Pasal 5 sampai 7").

CITATION_PAT = re.compile(
    r"[Pp][Aa][Ss][Aa][Ll]\s+(\d+)"          # "Pasal N" (case-insensitive manual)
    r"(?:\s+[Aa][Yy][Aa][Tt]\s+\((\d+)\))?"  # optional " ayat (M)"
    r"(?:\s+[Hh][Uu][Rr][Uu][Ff]\s+([a-z]))?" # optional " huruf x"
)


def extract_citations(response: str) -> list[dict[str, Any]]:
    """Extract all Pasal citations from an LLM response string.

    Returns a list of dicts, one per match, in order of appearance:
        {
            "pasal":     int,        # Pasal number
            "ayat":      int | None, # ayat number, or None
            "huruf":     str | None, # single letter, or None
            "raw":       str,        # matched text substring
            "char_start": int,       # start index in response
            "char_end":   int,       # end index in response (exclusive)
        }

    Phase 1 trade-offs (document for reviewer):
    - "Pasal 5 dan 6" → captures only "Pasal 5" (6 is not prefixed with "Pasal")
    - "Pasal-Pasal 5 sampai 7" → captures "Pasal 5", skips range expansion
    - "huruf a, huruf b, huruf c" → captures first huruf only (Phase 2: extend)
    - Duplicate citations (same Pasal cited twice) are preserved; dedup in whitelist
    """
    results: list[dict[str, Any]] = []
    for m in CITATION_PAT.finditer(response):
        pasal_str, ayat_str, huruf_str = m.group(1), m.group(2), m.group(3)
        results.append(
            {
                "pasal": int(pasal_str),
                "ayat": int(ayat_str) if ayat_str is not None else None,
                "huruf": huruf_str if huruf_str is not None else None,
                "raw": m.group(0),
                "char_start": m.start(),
                "char_end": m.end(),
            }
        )
    return results

"""HalluGraph Entity Grounding (EG) scorer — Phase 1 minimum implementation.

Reference: Noël 2025 HalluGraph framework, spec Section 5.4 line 961-965.

Phase 1 entity set:
  - Hardcoded legal terms (~30 istilah from Pasal 1 definisi Permensos 8/2023)
  - All citation tuples extracted from response (Pasal N, Pasal N ayat (M), etc.)

Entity matching: case-insensitive substring contains (lower bound).
Phase 2 upgrade: replace with NER (e.g. spaCy + custom NER model for legal-ID).

EG = |entities_in_response INTERSECT entities_in_context| / max(|entities_in_response|, 1)

Edge cases:
  - Empty response -> score 1.0 (no entities to ground, vacuously true)
  - No entities found in response -> score 1.0
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Legal terms from Pasal 1 Permensos 8/2023 + primary legal terminology
# ~30 terms, intentionally conservative (over-inclusive = more false negatives,
# under-inclusive = more false positives; conservative list preferred).
# ---------------------------------------------------------------------------
LEGAL_TERMS_PHASE1: list[str] = [
    # Core subject matter
    "TPPO",
    "Tindak Pidana Perdagangan Orang",
    "Perdagangan Orang",
    "Korban TPPO",
    "Korban",
    # Migrant workers
    "Pekerja Migran Indonesia Bermasalah",
    "Pekerja Migran Indonesia",
    "PMI Bermasalah",
    "PMI",
    # Social services
    "Rehabilitasi Sosial",
    "Asistensi Sosial",
    "Asistensi",
    "Asesmen",
    "Reintegrasi Sosial",
    "Reintegrasi",
    # Forms of exploitation / act types
    "Eksploitasi",
    "Perekrutan",
    "Pelacuran",
    "Kerja Paksa",
    "Perbudakan",
    "Penculikan",
    "Pemalsuan",
    "Penipuan",
    # Penanganan flow
    "Penanganan",
    "Pemulangan",
    "Pendampingan",
    "Perlindungan",
    # Institutions
    "Menteri",
    "Dinas Sosial",
    "Lembaga Kesejahteraan Sosial",
    "Kementerian Sosial",
    "Rumah Perlindungan Sosial",
    # Procedural terms
    "Pengaduan",
    "Penjangkauan",
    "Koordinasi",
    "Advokasi",
    # Common Pasal 1 defined terms
    "Penyelenggaraan Kesejahteraan Sosial",
    "Pekerja Sosial",
    "Penyidik",
]


def _extract_entities_from_text(text: str) -> set[str]:
    """Extract entity set from text using hardcoded legal terms.

    Only LEGAL_TERMS_PHASE1 terms are checked (case-insensitive substring match).
    Citation tuples (Pasal N ayat M) are NOT included as entities — they are
    citation infrastructure, not semantic entities that need grounding.
    Including them would inflate the denominator with non-semantic tokens and
    systematically lower EG scores for well-grounded responses.

    Args:
        text: Text to scan (response or context).

    Returns:
        Set of entity strings (lowercased) found in the text.
    """
    text_lower = text.lower()
    found: set[str] = set()

    # Check each legal term (case-insensitive substring)
    for term in LEGAL_TERMS_PHASE1:
        if term.lower() in text_lower:
            found.add(term.lower())

    return found


def compute_eg_score(
    response: str,
    context_text: str,
    extracted_citations: list[dict[str, Any]],
) -> tuple[float, dict[str, Any]]:
    """Compute Entity Grounding (EG) score.

    EG = |response_entities INTERSECT context_entities| / max(|response_entities|, 1)

    Entity set = LEGAL_TERMS_PHASE1 terms that appear in the text (case-insensitive).
    Citation tuples (Pasal N ayat M) are excluded from entity extraction — they are
    citation markers, not semantic domain terms.

    Args:
        response:             LLM-generated response text.
        context_text:         Concatenated text of all parent_chunks passed to LLM.
        extracted_citations:  Present for interface compatibility; not used in Phase 1
                              entity extraction (see design note above).

    Returns:
        (eg_score, debug_dict)

        debug_dict keys:
          - response_entities:     sorted list of legal terms found in response
          - context_entities:      sorted list of legal terms found in context
          - intersection:          sorted list of terms in both
          - missing_from_context:  terms in response but not in context
    """
    # Edge case: empty response
    if not response.strip():
        return 1.0, {
            "response_entities": [],
            "context_entities": [],
            "intersection": [],
            "missing_from_context": [],
            "note": "empty_response",
        }

    response_entities = _extract_entities_from_text(response)
    context_entities = _extract_entities_from_text(context_text)

    if not response_entities:
        return 1.0, {
            "response_entities": [],
            "context_entities": sorted(context_entities),
            "intersection": [],
            "missing_from_context": [],
            "note": "no_entities_in_response",
        }

    intersection = response_entities & context_entities
    missing = response_entities - context_entities

    eg_score = len(intersection) / len(response_entities)

    debug = {
        "response_entities": sorted(response_entities),
        "context_entities": sorted(context_entities),
        "intersection": sorted(intersection),
        "missing_from_context": sorted(missing),
    }

    return eg_score, debug

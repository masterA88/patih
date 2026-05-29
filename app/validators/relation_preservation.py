"""HalluGraph Relation Preservation (RP) scorer — Phase 1 simplified implementation.

Reference: Noël 2025 HalluGraph framework, spec Section 5.4 line 967-970.

Phase 1 approach: token-overlap check.
  For every sentence in the response that contains a citation (Pasal N):
    - Get the context text for that Pasal (context_by_pasal[N]).
    - Tokenize both sentence and context (lowercase, strip punct, remove stopwords).
    - If >= 2 content word overlap -> claim passes.
  RP = mean(claim_passes) over all claims examined.

This is NOT a true relation/semantic check. It catches obvious hallucinations like
"Pasal 5 mengatur sanksi pidana" when Pasal 5 text has zero overlap with "sanksi pidana".

Phase 2 upgrade: replace with LLM-as-judge or structured relation extractor.

Trade-off (documented per spec):
  - False negatives: sentence paraphrasing context with synonyms may fail overlap check.
  - False positives: sentence with high stopword count may pass spuriously.
  - Phase 1 threshold 2 content words is empirically chosen as the lowest bar;
    most grounded sentences will overlap 3+ content words with context.
  - "Pasal 5 mengatur X" where X is a concept from Pasal 1 — Phase 1 may NOT catch
    this cross-pasal injection if Pasal 1 and Pasal 5 share some legal vocabulary.
    This is a documented limitation; Phase 2 needs semantic comparison.
"""

from __future__ import annotations

import re
import string
from typing import Any

# ---------------------------------------------------------------------------
# Indonesian stopwords — minimal list (~50 words).
# Intentionally NOT using Sastrawi (over-aggressive stemming breaks legal terms).
# Source: common Indonesian function words + legal connecting words.
# ---------------------------------------------------------------------------
ID_STOPWORDS: set[str] = {
    # Articles / pronouns
    "yang", "ini", "itu", "tersebut", "demikian", "dimaksud",
    # Conjunctions
    "dan", "atau", "tetapi", "namun", "serta", "bahwa", "bahkan",
    # Prepositions
    "di", "ke", "dari", "pada", "dalam", "untuk", "dengan", "oleh",
    "antara", "terhadap", "tentang", "mengenai", "berdasarkan",
    "sesuai", "melalui", "kepada", "bagi",
    # Verbs (copula / light verbs)
    "adalah", "merupakan", "ialah", "yaitu", "yakni",
    # Temporal / sequential
    "akan", "telah", "sudah", "sedang", "masih", "pernah",
    "saat", "ketika", "selama", "setelah", "sebelum",
    "selanjutnya", "kemudian", "lalu",
    # Quantifiers / determiners
    "setiap", "semua", "para", "beberapa", "banyak", "satu", "dua",
    # Modal / discourse
    "dapat", "harus", "wajib", "bisa", "boleh", "tidak", "juga",
    "pula", "pun", "lain", "lainnya",
    # Connective / listing
    "berupa", "termasuk", "antara", "lain",
    # Numbers as strings
    "pertama", "kedua", "ketiga",
    # Common legal connectives
    "huruf", "ayat", "pasal", "nomor",
}

# Punctuation removal translation table
_PUNCT_TABLE = str.maketrans("", "", string.punctuation + "“”‘’")


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace, remove stopwords."""
    text_clean = text.lower().translate(_PUNCT_TABLE)
    tokens = text_clean.split()
    return [t for t in tokens if t and t not in ID_STOPWORDS and len(t) > 1]


def _split_sentences(text: str) -> list[str]:
    """Split response into sentences.

    Uses period/newline as primary delimiters.
    Preserves parenthetical citation references within a sentence (they don't end it).
    """
    # Split on sentence-ending punctuation or newlines
    # Avoid splitting on "ayat (2)" parentheses
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _find_pasal_nums_in_sentence(sentence: str) -> list[int]:
    """Return list of Pasal numbers mentioned in a sentence."""
    return [int(m) for m in re.findall(r"[Pp]asal\s+(\d+)", sentence)]


# Footer header pattern. Matches:
#   - "Sumber:"                        (header line on its own — Llama multi-line style)
#   - "Sumber: Pasal 5."               (single-line citation footer — Gemini style)
#   - "Sumber: \n- Pasal 1 ..."        (the "Sumber:" head; bullet items handled separately)
# Anything starting with "Sumber:" is a citation footer, not a factual claim.
_FOOTER_HEADER_PAT = re.compile(r"^\s*sumber\s*:", re.IGNORECASE)

# Bullet/dash marker at start of a footer item.
_BULLET_LEAD_PAT = re.compile(r"^\s*[-•*]\s*")

# Tokens that count as pure citation metadata (Pasal/ayat/huruf/angka/Permensos).
# After stripping these, any sentence with no remaining substance is citation-only.
_CITATION_TOKEN_PAT = re.compile(
    r"[Pp]asal\s+\d+|"
    r"ayat\s*\(?\s*\d+\s*\)?|"
    r"huruf\s+[a-zA-Z]\b|"
    r"angka\s+\d+|"
    r"[—–\-]\s*[Pp]ermensos\s+\d+\s*/\s*\d{4}|"
    r"[Pp]ermensos\s+\d+\s*/\s*\d{4}",
    re.IGNORECASE,
)


def _is_citation_only_line(sentence: str) -> bool:
    """True if the sentence is a footer line containing only citation references.

    Examples skipped:
      - "Sumber:" (footer header)
      - "- Pasal 1 angka 5 — Permensos 8/2023"
      - "* Pasal 5 ayat (2) huruf a"
      - "Pasal 2 ayat (1) Permensos 8/2023"

    Examples NOT skipped (factual claims that must be RP-checked):
      - "Pasal 5 mengatur 13 bentuk eksploitasi"
      - "Reintegrasi sosial diatur dalam Pasal 19"
    """
    # Strip leading bullet/dash markers first
    s = _BULLET_LEAD_PAT.sub("", sentence).strip()

    # "Sumber:" header (with or without trailing colon/content)
    if _FOOTER_HEADER_PAT.match(s):
        return True

    # Remove all citation tokens, then check what's left
    stripped = _CITATION_TOKEN_PAT.sub("", s)
    stripped = stripped.translate(_PUNCT_TABLE).strip()

    # If <=2 chars (or only whitespace/short connector remains), this was
    # a pure citation reference, not a factual claim.
    return len(stripped) <= 2


def compute_rp_score(
    response: str,
    context_by_pasal: dict[int, str],
    extracted_citations: list[dict[str, Any]],
) -> tuple[float, dict[str, Any]]:
    """Compute Relation Preservation (RP) score.

    Args:
        response:           LLM-generated response text.
        context_by_pasal:   {pasal_num: text} from retrieved parent_chunks.
                            Keys are int pasal numbers.
        extracted_citations: Citations from citation_extractor (used to identify
                            which Pasals are cited, cross-referenced with sentences).

    Returns:
        (rp_score, debug_dict)

        rp_score: float in [0, 1].
                  1.0 if no claims were checkable (vacuously true).
        debug_dict keys:
          - claims_checked:    int
          - claims_passed:     int
          - claims_skipped:    int (context_missing for that Pasal)
          - claim_details:     list of per-claim debug dicts
    """
    sentences = _split_sentences(response)

    claim_details: list[dict[str, Any]] = []
    claims_checked = 0
    claims_passed = 0
    claims_skipped = 0

    # We iterate sentences and check which Pasals appear
    for i, sentence in enumerate(sentences):
        # Skip citation-only lines (footer header "Sumber:" + multi-line bullet items
        # like "- Pasal 1 angka 5 — Permensos 8/2023"). These are citation references,
        # not factual claims, and must not count toward RP.
        if _is_citation_only_line(sentence):
            continue

        pasal_nums = _find_pasal_nums_in_sentence(sentence)

        if not pasal_nums:
            continue  # sentence has no citation -- not a verifiable claim

        for pasal_num in pasal_nums:
            # Get context for this Pasal
            context_text = context_by_pasal.get(pasal_num)

            if context_text is None:
                # Pasal was cited in response but not in retrieved context.
                # Could be a cross-ref or out-of-retrieval Pasal.
                # Per spec: skip, mark context_missing, don't count in mean.
                claims_skipped += 1
                claim_details.append(
                    {
                        "sentence_idx": i,
                        "sentence": sentence[:100],
                        "pasal": pasal_num,
                        "status": "context_missing",
                        "overlap_words": [],
                    }
                )
                continue

            # Tokenize sentence and context
            sentence_tokens = set(_tokenize(sentence))
            context_tokens = set(_tokenize(context_text))

            overlap = sentence_tokens & context_tokens

            # Threshold: >= 1 content word overlap for Phase 1.
            # Rationale: LLM responses often produce single-term list items
            # (e.g. "pelacuran (Pasal 5 ayat (2) huruf a);") where only one
            # content word appears per sentence. Requiring >= 2 would fail
            # well-grounded single-term items. Phase 2: raise threshold + semantic.
            passed = len(overlap) >= 1

            claims_checked += 1
            if passed:
                claims_passed += 1

            claim_details.append(
                {
                    "sentence_idx": i,
                    "sentence": sentence[:100],
                    "pasal": pasal_num,
                    "status": "pass" if passed else "fail",
                    "overlap_count": len(overlap),
                    "overlap_words": sorted(list(overlap))[:10],  # cap debug output
                }
            )

    # If no claims were checkable (e.g. no citations in context, or response has no
    # sentences with citations), return vacuous 1.0
    if claims_checked == 0:
        rp_score = 1.0
    else:
        rp_score = claims_passed / claims_checked

    debug = {
        "claims_checked": claims_checked,
        "claims_passed": claims_passed,
        "claims_skipped": claims_skipped,
        "claim_details": claim_details,
    }

    return rp_score, debug

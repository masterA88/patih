"""Regex-based BAB/Bagian/Pasal/Ayat/Huruf parser producing a document AST."""

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex patterns (from build-spec Section 5.1, with PDF-specific adjustments)
# ---------------------------------------------------------------------------
BAB_PAT = re.compile(
    r"BAB\s+([IVXLCDM]+)\s*\n([^\n]+)",
    re.MULTILINE,
)
# Note: PDF has mixed case for Bagian — "Bagian kesatu" and "Bagian Kedua"
BAGIAN_PAT = re.compile(
    r"Bagian\s+([Kk]e(?:satu|dua|tiga|empat|lima|enam|tujuh|delapan|sembilan|sepuluh))\s*\n([^\n]+)",
    re.MULTILINE | re.IGNORECASE,
)
# Pasal header: "Pasal 5\n" or "Pasal 5.\n" (older laws use a trailing period).
# The optional "\.?" keeps backward compatibility with the no-period format.
PASAL_PAT = re.compile(
    r"Pasal\s+(\d+)\s*\.?\s*\n(.*?)(?=\nPasal\s+\d+|\nBAB\s+|\Z)",
    re.DOTALL,
)
# Ayat pattern: only match (N) at the START of a line (^ in MULTILINE) or at the very
# beginning of the string. This prevents false positives like "ayat (1) huruf" mid-sentence.
AYAT_PAT = re.compile(
    r"(?:^|\n)\((\d+)\)\s+(.*?)(?=\n\(\d+\)|\Z)",
    re.DOTALL | re.MULTILINE,
)
# Huruf: only match when preceded by newline to avoid false positives mid-sentence
# Specifically match single lowercase letter followed by '. ' at start of line
HURUF_PAT = re.compile(
    r"\n([a-z])\.\s+(.*?)(?=\n[a-z]\.|\Z)",
    re.DOTALL,
)

# Page artifacts to strip
_PAGE_HEADER_PAT = re.compile(r"-\s*\d+\s*-\s*\n?")
_JDIH_PAT = re.compile(r"jdih\.kemensos\.go\.id\s*\n?")
# Page boundary markers injected by pdf_loader
_PAGE_MARKER_PAT = re.compile(r"\n\n--- page \d+ ---\n\n")

# "PENJELASAN" heading marks the start of the elucidation section, which repeats
# every Pasal as explanatory (non-normative) text. Indonesian law docs put it
# after the Batang Tubuh (main body). We truncate there to avoid double-counting
# Pasal. Only truncate when the marker appears in the LATTER part of the doc
# (>40% through) — guards against omnibus laws / TOC references near the start.
_PENJELASAN_HEADING_PAT = re.compile(
    r"\n\s*P\s*E\s*N\s*J\s*E\s*L\s*A\s*S\s*A\s*N\s*\n",
    re.IGNORECASE,
)
_PENJELASAN_MIN_POSITION_RATIO = 0.40


def _strip_penjelasan(text: str) -> str:
    """Truncate text at the PENJELASAN heading (if it appears past 40% of the doc).

    Returns the Batang Tubuh (normative body) only. If no qualifying marker is
    found, returns text unchanged.
    """
    total = len(text)
    if total == 0:
        return text
    for m in _PENJELASAN_HEADING_PAT.finditer(text):
        if m.start() / total >= _PENJELASAN_MIN_POSITION_RATIO:
            logger.info(
                "Stripping PENJELASAN section at %d (%.0f%% through doc)",
                m.start(), m.start() / total * 100,
            )
            return text[:m.start()]
    return text


def _clean_text(text: str) -> str:
    """Strip PDF artifacts: page numbers, watermarks, page boundary markers."""
    text = _PAGE_MARKER_PAT.sub("\n", text)
    text = _PAGE_HEADER_PAT.sub("", text)
    text = _JDIH_PAT.sub("", text)
    # Collapse excessive blank lines (3+ → 2)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _normalize_whitespace_in_text(text: str) -> str:
    """
    The PDF has column-split text: words flow across multiple lines.
    Collapse runs of whitespace within a paragraph but preserve structural newlines.

    Strategy: collapse single newlines between non-empty, non-structural lines
    into a space. Structural lines start with BAB/Pasal/Bagian/(digit) or are blank.
    """
    lines = text.split("\n")
    result: list[str] = []
    buffer: list[str] = []

    def _is_structural(line: str) -> bool:
        s = line.strip()
        if not s:
            return True
        if re.match(r"^(BAB\s|Pasal\s|Bagian\s|\(\d+\)|[a-z]\.\s)", s):
            return True
        return False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if buffer:
                result.append(" ".join(buffer))
                buffer = []
            result.append("")
        elif _is_structural(line):
            if buffer:
                result.append(" ".join(buffer))
                buffer = []
            result.append(line)
        else:
            buffer.append(stripped)

    if buffer:
        result.append(" ".join(buffer))

    return "\n".join(result)


def _parse_huruf(text: str) -> list[dict]:
    """Extract lettered sub-items (a. b. c. ...) from an ayat text."""
    huruf_items: list[dict] = []
    for m in HURUF_PAT.finditer(text):
        letter = m.group(1)
        # Skip 'i' that is part of common Indonesian words mid-sentence
        # Already handled by newline anchor in pattern, but do a length check too
        item_text = " ".join(m.group(2).split())  # normalise internal whitespace
        huruf_items.append({"nomor": letter, "text": item_text.strip()})
    return huruf_items


def _parse_ayat(text: str) -> list[dict]:
    """
    Extract numbered ayat items (1) (2) (3) from a Pasal text.
    Returns list of {nomor, text, huruf}.
    """
    ayat_items: list[dict] = []
    for m in AYAT_PAT.finditer(text):
        ayat_num = int(m.group(1))
        ayat_text = m.group(2).strip()
        huruf = _parse_huruf(ayat_text)
        ayat_items.append({
            "nomor": ayat_num,
            "text": " ".join(ayat_text.split()),
            "huruf": huruf,
        })
    return ayat_items


def _normalize_pasal_body(body: str) -> str:
    """
    Collapse word-wrapped lines in a Pasal body back into proper paragraphs.

    The Permensos PDF extracts with single words/phrases per line due to column layout.
    e.g.:
        "(2) \nDalam melakukan pemulangan sebagaimana dimaksud \npada \nayat \n(1) \nMenteri ..."
    should become:
        "(2) Dalam melakukan pemulangan ... pada ayat (1) Menteri ..."

    Rules for new structural unit:
    - `(N)` at start of line IS a new ayat start UNLESS the previous accumulated token ends
      with one of the cross-reference trigger words: "ayat", "pasal", "huruf", "(", or a digit.
      Those indicate an inline cross-reference like "pada ayat (1)".
    - `[a-z].` at start of line (followed by content) is a huruf item start.
    - Empty lines are skipped.
    - Everything else is appended to the current unit (word-wrap continuation).
    """
    # Cross-reference preceding tokens that indicate (N) is inline, NOT an ayat header
    _INLINE_PRECEDING = {"ayat", "pasal", "huruf", "dan"}

    lines = body.split("\n")
    # Each element is a list of tokens making up one logical line
    groups: list[list[str]] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        is_ayat_candidate = bool(re.match(r"^\(\d+\)\s*$", stripped) or re.match(r"^\(\d+\)\s+\S", stripped))
        is_huruf_start = bool(re.match(r"^[a-z]\.\s+\S", stripped))

        if is_huruf_start:
            groups.append([stripped])
            continue

        if is_ayat_candidate:
            # Check if preceding accumulated text ends with a cross-reference word
            if groups:
                # Get the last word of the last group's accumulated text
                last_token = groups[-1][-1].split()[-1].lower().rstrip(".,;:")
                if last_token in _INLINE_PRECEDING or last_token.isdigit():
                    # This (N) is an inline cross-reference — append to current group
                    groups[-1].append(stripped)
                    continue
            # Otherwise it's a real new ayat
            groups.append([stripped])
            continue

        # Regular continuation line
        if groups:
            groups[-1].append(stripped)
        else:
            groups.append([stripped])

    return "\n".join(" ".join(g) for g in groups)


def _parse_pasal_body(pasal_num: int, body: str) -> dict:
    """
    Parse a single Pasal body text into structured dict.
    Handles two formats:
    - Regular Pasal: (1) ... (2) ... ayat format
    - Pasal 1 (definitions): 1. text 2. text numbered list format → treated as huruf-like
    """
    body = body.strip()
    # Normalize word-wrap artifacts BEFORE regex extraction
    body = _normalize_pasal_body(body)
    ayat_items = _parse_ayat(body)

    # If no ayat detected, the Pasal is a single-block (common for Pasal tanpa ayat)
    if not ayat_items:
        # Check for Pasal 1-style numbered definitions (1.\n text 2.\n text)
        # These use integer. format at start of line
        numbered_pat = re.compile(
            r"(?:^|\n)(\d+)\.\s+(.*?)(?=\n\d+\.|\Z)",
            re.DOTALL,
        )
        numbered_items = numbered_pat.findall(body)
        if numbered_items and len(numbered_items) > 2:
            # This looks like a definitions list — store as synthetic ayat
            # with the item number as ayat.nomor, no sub-huruf
            for num_str, item_text in numbered_items:
                ayat_items.append({
                    "nomor": int(num_str),
                    "text": " ".join(item_text.split()),
                    "huruf": [],
                })

    huruf_top = _parse_huruf(body) if not ayat_items else []

    return {
        "nomor": pasal_num,
        "text_raw": body,
        "ayat": ayat_items,
        # top-level huruf for Pasal with no ayat but with lettered list
        "huruf_top": huruf_top,
    }


def parse_structure(text: str) -> dict:
    """
    Parse full document text into AST:
    {
        "bab": [
            {
                "nomor": "I",
                "judul": "KETENTUAN UMUM",
                "bagian": [{"nomor": "Kesatu", "judul": "Umum", "pasal_range": [4, 7]}],
                "pasal": [{"nomor": 1, "text_raw": "...", "ayat": [...]}]
            }
        ]
    }

    Raises no exceptions — logs warnings for malformed sections.
    """
    cleaned = _clean_text(text)
    # Drop the Penjelasan (elucidation) section — it repeats every Pasal as
    # non-normative explanation and inflates the parsed Pasal count ~2x.
    cleaned = _strip_penjelasan(cleaned)

    # ---------------------------------------------------------------------------
    # Step 1: Locate all BAB boundaries
    # ---------------------------------------------------------------------------
    bab_matches = list(BAB_PAT.finditer(cleaned))
    if not bab_matches:
        logger.warning("No BAB found in document — returning flat pasal list")
        # Fallback: parse all Pasal without BAB structure
        pasal_items = _extract_all_pasal(cleaned)
        return {"bab": [{"nomor": "?", "judul": "DOKUMEN", "bagian": [], "pasal": pasal_items}]}

    # ---------------------------------------------------------------------------
    # Step 1b: Recover Pasal that appear BEFORE the first BAB.
    # Short regulations often place the normative Pasal (Batang Tubuh) first and
    # only use BAB headings inside a trailing Lampiran. Those leading Pasal would
    # otherwise be dropped (BAB-scoped extraction only). The PASAL_PAT requires a
    # newline after the number, so preamble cross-refs like "Pasal 30 ayat (3)"
    # in Menimbang/Mengingat are naturally NOT matched here.
    # ---------------------------------------------------------------------------
    bab_list: list[dict] = []
    pre_bab_region = cleaned[: bab_matches[0].start()]
    pre_bab_pasal = _extract_all_pasal(pre_bab_region)
    if pre_bab_pasal:
        logger.info(
            "Recovered %d Pasal before first BAB: %s",
            len(pre_bab_pasal), [p["nomor"] for p in pre_bab_pasal],
        )
        bab_list.append({
            "nomor": "-",
            "judul": "KETENTUAN",
            "bagian": [],
            "pasal": pre_bab_pasal,
        })

    for idx, bm in enumerate(bab_matches):
        bab_start = bm.start()
        bab_end = bab_matches[idx + 1].start() if idx + 1 < len(bab_matches) else len(cleaned)
        bab_body = cleaned[bab_start:bab_end]

        bab_nomor = bm.group(1).strip()
        bab_judul = bm.group(2).strip()

        # ---------------------------------------------------------------------------
        # Step 3: Locate Bagian within this BAB
        # ---------------------------------------------------------------------------
        bagian_matches = list(BAGIAN_PAT.finditer(bab_body))
        bagian_list: list[dict] = []
        for b_idx, bagm in enumerate(bagian_matches):
            b_start = bagm.start()
            b_end = bagian_matches[b_idx + 1].start() if b_idx + 1 < len(bagian_matches) else len(bab_body)
            bagian_body = bab_body[b_start:b_end]
            bagian_nomor_raw = bagm.group(1)
            # Normalise: "kesatu" → "Kesatu"
            bagian_nomor = bagian_nomor_raw.capitalize()
            bagian_judul = bagm.group(2).strip()

            # Extract Pasal numbers that appear in this bagian slice
            pasal_nums_in_bagian = [
                int(m.group(1)) for m in re.finditer(r"Pasal\s+(\d+)", bagian_body)
            ]
            bagian_list.append({
                "nomor": bagian_nomor,
                "judul": bagian_judul,
                "pasal_range": [
                    min(pasal_nums_in_bagian) if pasal_nums_in_bagian else None,
                    max(pasal_nums_in_bagian) if pasal_nums_in_bagian else None,
                ],
            })

        # ---------------------------------------------------------------------------
        # Step 4: Extract Pasal within this BAB
        # ---------------------------------------------------------------------------
        pasal_items = _extract_all_pasal(bab_body)

        bab_list.append({
            "nomor": bab_nomor,
            "judul": bab_judul,
            "bagian": bagian_list,
            "pasal": pasal_items,
        })

    # ---------------------------------------------------------------------------
    # Step 4b: Cross-BAB dedup — within one document each Pasal number is unique.
    # Duplicates across BAB slices are parse artifacts (Lampiran repeat, BAB-
    # boundary overlap). Keep the FIRST occurrence (Batang Tubuh / earliest),
    # drop the rest.
    # ---------------------------------------------------------------------------
    seen_pasal: set[int] = set()
    for bab in bab_list:
        kept = []
        for p in bab["pasal"]:
            if p["nomor"] in seen_pasal:
                continue
            seen_pasal.add(p["nomor"])
            kept.append(p)
        bab["pasal"] = kept

    ast = {"bab": bab_list}

    # ---------------------------------------------------------------------------
    # Step 5: Quality checks
    # ---------------------------------------------------------------------------
    all_pasal_nums = [p["nomor"] for b in bab_list for p in b["pasal"]]
    logger.info(
        "Parsed %d BAB, %d total Pasal: %s",
        len(bab_list),
        len(all_pasal_nums),
        all_pasal_nums,
    )

    # Check for duplicate Pasal numbers (indicates parser split issue)
    if len(all_pasal_nums) != len(set(all_pasal_nums)):
        dupes = [n for n in all_pasal_nums if all_pasal_nums.count(n) > 1]
        logger.warning("Duplicate Pasal numbers detected: %s", sorted(set(dupes)))

    return ast


def _extract_all_pasal(text: str) -> list[dict]:
    """
    Find all Pasal N ... blocks in text and return list of parsed Pasal dicts.
    """
    pasal_items: list[dict] = []
    seen: set[int] = set()

    for m in PASAL_PAT.finditer(text):
        pasal_num = int(m.group(1))
        if pasal_num in seen:
            logger.warning("Duplicate Pasal %d in segment — skipping second occurrence", pasal_num)
            continue
        seen.add(pasal_num)
        body = m.group(2).strip()
        pasal_items.append(_parse_pasal_body(pasal_num, body))

    return pasal_items

"""System prompts (Bahasa Indonesia) with bilingual variant and user prompt builder.

Prompt files live in configs/prompts/:
  system_id.md       — Indonesian-only system prompt
  system_bilingual.md — Bilingual variant (adds EN response instructions)
  fewshot.md         — Few-shot examples (optional, injected when needed)

Context XML format (per spec Section 5.3 line 868-882):
  <konteks>
  [Pasal N — BAB X — Permensos 8/2023]
  {pasal_text}

  [Pasal M — ...]
  ...
  </konteks>

  <pertanyaan>
  {user_query}
  </pertanyaan>

  Bahasa jawaban yang diharapkan: {lang_q}
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default prompt dir — resolved relative to project root (CWD when running).
# Fallback: relative to this file's parent tree.
_PROMPT_DIR_CANDIDATES = [
    Path("configs/prompts"),
    Path(__file__).parent.parent.parent / "configs" / "prompts",
]


def _find_prompt_dir() -> Path:
    for candidate in _PROMPT_DIR_CANDIDATES:
        if candidate.exists():
            return candidate
    # Last resort: return first candidate and let the read fail with a clear message.
    return _PROMPT_DIR_CANDIDATES[0]


def _read_prompt_file(filename: str) -> str:
    prompt_dir = _find_prompt_dir()
    path = prompt_dir / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {path.resolve()}. "
            "Run from project root or ensure configs/prompts/ exists."
        )
    return path.read_text(encoding="utf-8").strip()


def load_system_prompt(lang: str = "id", include_fewshot: bool = False) -> str:
    """Load the system prompt for the given response language.

    Args:
        lang:            'id' → system_id.md, anything else → system_bilingual.md.
        include_fewshot: If True, append few-shot examples from fewshot.md.

    Returns:
        System prompt string ready to be used as the 'system' message content.
    """
    if lang == "id":
        base = _read_prompt_file("system_id.md")
    else:
        base = _read_prompt_file("system_bilingual.md")

    if include_fewshot:
        try:
            fewshot = _read_prompt_file("fewshot.md")
            base = base + "\n\n---\n\nContoh jawaban yang benar:\n\n" + fewshot
        except FileNotFoundError:
            logger.warning("fewshot.md not found — skipping few-shot injection.")

    return base


# Registry cache for doc labels (loaded lazily)
_REGISTRY_PATH_CANDIDATES = [
    Path("data/registry/documents.json"),
    Path(__file__).parent.parent.parent / "data" / "registry" / "documents.json",
]
_doc_registry_cache: dict[str, dict] | None = None


def _load_doc_registry() -> dict[str, dict]:
    global _doc_registry_cache
    if _doc_registry_cache is not None:
        return _doc_registry_cache
    import json
    for candidate in _REGISTRY_PATH_CANDIDATES:
        if candidate.exists():
            try:
                _doc_registry_cache = json.loads(candidate.read_text(encoding="utf-8"))
                return _doc_registry_cache
            except Exception as e:
                logger.warning("Failed to load registry %s: %s", candidate, e)
    _doc_registry_cache = {}
    return _doc_registry_cache


def doc_label(doc_id: str) -> str:
    """Human-friendly doc label from registry (e.g. 'Permensos 8/2023').

    Falls back to doc_id if not in registry.
    """
    if not doc_id:
        return "dokumen"
    reg = _load_doc_registry()
    entry = reg.get(doc_id, {})
    jenis = (entry.get("jenis_regulasi") or "").replace("_", " ").title()
    nomor = entry.get("nomor")
    tahun = entry.get("tahun")
    if jenis and nomor and tahun:
        return f"{jenis} {nomor}/{tahun}"
    return entry.get("title") or doc_id


def _format_chunk_header(chunk: dict[str, Any]) -> str:
    """Build the bracketed header for a parent chunk in the context block.

    Two shapes:
      - Regulasi (pasal is set): "[Pasal N — BAB X — <doc label>]"
      - Reference doc (pasal is None): "[<section title> — hal. P — <doc label>]"
    The source doc label comes from each chunk's own doc_id (multi-doc).
    """
    label = doc_label(chunk.get("doc_id", ""))
    pasal = chunk.get("pasal")

    # Non-Pasal reference doc (SOP / statistik / RPJMN)
    if pasal is None:
        section = chunk.get("section_title") or "Bagian"
        page = chunk.get("source_page")
        parts = [str(section)]
        if page:
            parts.append(f"hal. {page}")
        parts.append(label)
        return "[" + " — ".join(parts) + "]"

    # Regulasi (Pasal-based)
    bab = chunk.get("bab", "")
    bagian = chunk.get("bagian", "")
    is_always_on = chunk.get("is_always_on", False)
    parts = [f"Pasal {pasal}"]
    if bab:
        parts.append(f"BAB {bab}")
    if bagian:
        parts.append(f"Bagian {bagian}")
    parts.append(label)
    if is_always_on:
        parts.append("definisi")

    return "[" + " — ".join(parts) + "]"


def build_user_prompt(
    query: str,
    context: list[dict[str, Any]],
    lang: str = "id",
) -> str:
    """Build the user message with <konteks> XML wrapper and <pertanyaan>.

    Args:
        query:   Original user query (in original language — NOT translated).
        context: List of parent chunk dicts from RetrievalResult.parent_chunks.
                 Each dict must have at minimum a 'text' field.
        lang:    Response language hint injected at the end of the prompt.

    Returns:
        Formatted user prompt string.
    """
    if not context:
        logger.warning("build_user_prompt: empty context list — LLM will likely refuse.")

    # Build context block
    ctx_parts: list[str] = []
    for chunk in context:
        header = _format_chunk_header(chunk)
        text = chunk.get("text", "").strip()
        if text:
            ctx_parts.append(f"{header}\n{text}")

    context_block = "\n\n".join(ctx_parts)

    # Language hint string
    lang_hint = "Bahasa Indonesia" if lang == "id" else "Bahasa Inggris (English)"

    user_prompt = (
        f"<konteks>\n{context_block}\n</konteks>\n\n"
        f"<pertanyaan>\n{query}\n</pertanyaan>\n\n"
        f"Bahasa jawaban yang diharapkan: {lang_hint}"
    )

    return user_prompt


def build_messages(
    query: str,
    context: list[dict[str, Any]],
    lang: str = "id",
    include_fewshot: bool = False,
) -> list[dict[str, str]]:
    """Build the full messages list for LiteLLM / OpenAI chat format.

    Returns:
        [{"role": "system", "content": ...}, {"role": "user", "content": ...}]
    """
    system_content = load_system_prompt(lang=lang, include_fewshot=include_fewshot)
    user_content = build_user_prompt(query=query, context=context, lang=lang)

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]

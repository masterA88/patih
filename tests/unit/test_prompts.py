"""Unit tests for app.llm.prompts.

Checks:
- System prompts loadable (id and bilingual variants)
- User prompt format correct (XML wrapper, header, query present)
- Context properly escaped / formatted
- build_messages returns correct role structure
"""

import os
import sys
from pathlib import Path

import pytest

# Ensure we run from project root so configs/prompts/ resolves correctly
_project_root = Path(__file__).parent.parent.parent
os.chdir(_project_root)
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from app.llm.prompts import (
    build_messages,
    build_user_prompt,
    load_system_prompt,
)


# ---------------------------------------------------------------------------
# System prompt loading
# ---------------------------------------------------------------------------

def test_load_system_prompt_id():
    prompt = load_system_prompt(lang="id")
    assert isinstance(prompt, str)
    assert len(prompt) > 100, "ID system prompt unexpectedly short"
    # Must mention the regulation
    assert "Permensos" in prompt or "Peraturan Menteri" in prompt
    # Must mention citation format
    assert "Pasal" in prompt


def test_load_system_prompt_bilingual():
    prompt = load_system_prompt(lang="en")
    assert isinstance(prompt, str)
    assert len(prompt) > 100
    # Bilingual prompt should mention English response
    assert "English" in prompt or "bilingual" in prompt.lower() or "bahasa" in prompt.lower()


def test_load_system_prompt_unknown_lang_uses_bilingual():
    # Any non-'id' lang should fall back to bilingual
    prompt = load_system_prompt(lang="fr")
    bilingual_prompt = load_system_prompt(lang="en")
    assert prompt == bilingual_prompt


def test_load_system_prompt_with_fewshot():
    prompt = load_system_prompt(lang="id", include_fewshot=True)
    assert isinstance(prompt, str)
    # Should contain the base + fewshot separator
    assert len(prompt) > 200


# ---------------------------------------------------------------------------
# User prompt building
# ---------------------------------------------------------------------------

_sample_chunks = [
    {
        "chunk_id": "permensos-8-2023__parent__1",
        "pasal": 1,
        "bab": "I",
        "bagian": "",
        "is_always_on": True,
        "text": (
            "Dalam Peraturan Menteri ini yang dimaksud dengan:\n"
            "5. Korban Tindak Pidana Perdagangan Orang yang selanjutnya disebut "
            "Korban TPPO adalah seseorang yang mengalami penderitaan psikis, mental, "
            "fisik, seksual, ekonomi, dan/atau sosial yang diakibatkan tindak pidana "
            "perdagangan orang."
        ),
    },
    {
        "chunk_id": "permensos-8-2023__parent__5",
        "pasal": 5,
        "bab": "II",
        "bagian": "",
        "is_always_on": False,
        "text": (
            "(1) Eksploitasi sebagaimana dimaksud dalam Pasal 4 huruf d meliputi:\n"
            "a. eksploitasi seksual;\nb. eksploitasi tenaga atau jasa secara paksa;"
        ),
    },
]


def test_user_prompt_contains_konteks_tags():
    prompt = build_user_prompt("apa itu korban TPPO?", _sample_chunks, lang="id")
    assert "<konteks>" in prompt
    assert "</konteks>" in prompt


def test_user_prompt_contains_pertanyaan_tags():
    query = "apa itu korban TPPO?"
    prompt = build_user_prompt(query, _sample_chunks, lang="id")
    assert "<pertanyaan>" in prompt
    assert query in prompt


def test_user_prompt_contains_pasal_headers():
    prompt = build_user_prompt("test", _sample_chunks, lang="id")
    assert "Pasal 1" in prompt
    assert "Pasal 5" in prompt


def test_user_prompt_contains_chunk_text():
    prompt = build_user_prompt("test", _sample_chunks, lang="id")
    assert "Korban TPPO" in prompt
    assert "eksploitasi" in prompt


def test_user_prompt_lang_hint_id():
    prompt = build_user_prompt("test", _sample_chunks, lang="id")
    assert "Indonesia" in prompt


def test_user_prompt_lang_hint_en():
    prompt = build_user_prompt("test", _sample_chunks, lang="en")
    assert "English" in prompt or "Inggris" in prompt


def test_user_prompt_empty_context():
    # Should not crash — just empty konteks block
    prompt = build_user_prompt("test query", [], lang="id")
    assert "<konteks>" in prompt
    assert "<pertanyaan>" in prompt


def test_user_prompt_always_on_header():
    prompt = build_user_prompt("test", _sample_chunks, lang="id")
    # Pasal 1 is_always_on — header should mention 'definisi'
    assert "definisi" in prompt


# ---------------------------------------------------------------------------
# build_messages
# ---------------------------------------------------------------------------

def test_build_messages_structure():
    messages = build_messages("test", _sample_chunks, lang="id")
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


def test_build_messages_system_content():
    messages = build_messages("test", _sample_chunks, lang="id")
    system_content = messages[0]["content"]
    assert "Pasal" in system_content
    assert len(system_content) > 100


def test_build_messages_user_content():
    query = "apa saja bentuk eksploitasi?"
    messages = build_messages(query, _sample_chunks, lang="id")
    user_content = messages[1]["content"]
    assert query in user_content
    assert "<konteks>" in user_content

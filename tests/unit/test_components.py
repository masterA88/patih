"""Unit tests for app/ui/components.py.

Tests for build_confidence_badge() — pure function, no Chainlit runtime needed.
Tests for build_citation_cards() — requires mocking cl.Text construction since
Chainlit's Element.__post_init__ tries to access the active session context.
The fixture 'mock_cl_element' patches chainlit.element.Element.__post_init__
to bypass this, which is the correct pattern for unit-testing Chainlit element
factories outside of a running server.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

import chainlit as cl
import chainlit.element as _cl_elem


# ---------------------------------------------------------------------------
# Shared fixture: bypass Chainlit context requirement for Element construction
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_cl_element():
    """Patch Element.__post_init__ so cl.Text can be constructed without a
    running Chainlit server/session context.  Applied per-test."""

    def _patched_post_init(self):
        self.persisted = False
        self.updatable = False
        self.thread_id = "test-thread-id"
        if not getattr(self, "url", None) and not getattr(self, "path", None) and not getattr(self, "content", None):
            raise ValueError("Must provide url, path or content to instantiate element")

    with patch.object(_cl_elem.Element, "__post_init__", _patched_post_init):
        yield


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _badge():
    from app.ui.components import build_confidence_badge
    return build_confidence_badge


def _cards():
    from app.ui.components import build_citation_cards
    return build_citation_cards


# ---------------------------------------------------------------------------
# build_confidence_badge — pure function (no Chainlit context required)
# ---------------------------------------------------------------------------

class TestBuildConfidenceBadge:
    def test_none_returns_empty(self):
        assert _badge()(None) == ""

    def test_error_dict_returns_grey_badge(self):
        result = _badge()({"error": "some exception"})
        assert "⚪" in result
        assert "Validasi tidak tersedia" in result

    def test_hitl_false_returns_green(self):
        val = {
            "hitl_flag": False,
            "citation_accuracy": 1.0,
            "eg_score": 1.0,
            "rp_score": 1.0,
            "hitl_reasons": [],
        }
        result = _badge()(val)
        assert "🟢" in result
        assert "Tinggi" in result

    def test_hitl_true_high_scores_returns_yellow(self):
        val = {
            "hitl_flag": True,
            "citation_accuracy": 0.90,
            "eg_score": 0.92,
            "rp_score": 0.85,
            "hitl_reasons": ["low_rp"],
        }
        result = _badge()(val)
        assert "🟡" in result
        assert "Sedang" in result

    def test_hitl_true_low_citation_acc_returns_red(self):
        val = {
            "hitl_flag": True,
            "citation_accuracy": 0.50,
            "eg_score": 0.80,
            "rp_score": 0.70,
            "hitl_reasons": ["invalid_citation", "low_eg", "low_rp"],
        }
        result = _badge()(val)
        assert "🔴" in result
        assert "Rendah" in result
        assert "invalid_citation" in result

    def test_hitl_true_exactly_at_yellow_threshold(self):
        """citation_accuracy == 0.80 and eg_score == 0.90 → yellow (boundary inclusive)."""
        val = {
            "hitl_flag": True,
            "citation_accuracy": 0.80,
            "eg_score": 0.90,
            "rp_score": 0.85,
            "hitl_reasons": [],
        }
        result = _badge()(val)
        assert "🟡" in result

    def test_hitl_true_citation_acc_just_below_threshold_returns_red(self):
        """citation_accuracy == 0.79 → red."""
        val = {
            "hitl_flag": True,
            "citation_accuracy": 0.79,
            "eg_score": 0.95,
            "rp_score": 0.95,
            "hitl_reasons": ["invalid_citation"],
        }
        result = _badge()(val)
        assert "🔴" in result

    def test_missing_optional_fields_default_to_safe(self):
        """Partial validation dict (only hitl_flag=False) → green badge."""
        result = _badge()({"hitl_flag": False})
        assert "🟢" in result

    def test_empty_reasons_red_badge_no_crash(self):
        """hitl_flag=True, low scores, empty reasons list → red badge, no KeyError."""
        val = {
            "hitl_flag": True,
            "citation_accuracy": 0.0,
            "eg_score": 0.0,
            "rp_score": 0.0,
            "hitl_reasons": [],
        }
        result = _badge()(val)
        assert "🔴" in result


# ---------------------------------------------------------------------------
# build_citation_cards — requires mock_cl_element fixture + tmp parent_lookup
# ---------------------------------------------------------------------------

class TestBuildCitationCards:
    """Tests that patch Element.__post_init__ to bypass Chainlit context."""

    @pytest.fixture(autouse=True)
    def reset_cache(self):
        """Reset module-level _parent_lookup cache before each test."""
        import app.ui.components as comp
        original = comp._parent_lookup
        comp._parent_lookup = None
        yield
        comp._parent_lookup = original

    def test_returns_empty_when_lookup_missing(self, tmp_path, monkeypatch, mock_cl_element):
        import app.ui.components as comp
        monkeypatch.setattr(comp, "_PARENT_LOOKUP_PATH", tmp_path / "nonexistent.json")
        comp._parent_lookup = None
        result = _cards()([1, 2], None)
        assert result == []

    def test_returns_text_elements_for_valid_pasals(self, tmp_path, monkeypatch, mock_cl_element):
        import app.ui.components as comp

        lookup = {
            "permensos-8-2023::pasal1": {
                "chunk_id": "permensos-8-2023::pasal1",
                "bab": "I",
                "bagian": None,
                "pasal": 1,
                "text": "Definisi TPPO.",
                "source_page": 3,
            },
            "permensos-8-2023::pasal5": {
                "chunk_id": "permensos-8-2023::pasal5",
                "bab": "II",
                "bagian": "Pertama",
                "pasal": 5,
                "text": "Bentuk eksploitasi.",
                "source_page": 7,
            },
        }
        lookup_path = tmp_path / "parent_lookup.json"
        lookup_path.write_text(json.dumps(lookup), encoding="utf-8")
        monkeypatch.setattr(comp, "_PARENT_LOOKUP_PATH", lookup_path)
        comp._parent_lookup = None

        elements = _cards()([1, 5], None)

        assert len(elements) == 2
        assert all(isinstance(e, cl.Text) for e in elements)
        names = [e.name for e in elements]
        assert "Pasal 1" in names
        assert "Pasal 5" in names

    def test_caps_at_max_citation_cards(self, tmp_path, monkeypatch, mock_cl_element):
        import app.ui.components as comp

        lookup = {
            f"permensos-8-2023::pasal{i}": {
                "bab": "I", "bagian": None, "pasal": i,
                "text": f"Teks pasal {i}.", "source_page": i,
            }
            for i in range(1, 15)
        }
        lookup_path = tmp_path / "parent_lookup.json"
        lookup_path.write_text(json.dumps(lookup), encoding="utf-8")
        monkeypatch.setattr(comp, "_PARENT_LOOKUP_PATH", lookup_path)
        comp._parent_lookup = None

        elements = _cards()(list(range(1, 15)), None)
        assert len(elements) <= comp._MAX_CITATION_CARDS

    def test_skips_unknown_pasal_ids(self, tmp_path, monkeypatch, mock_cl_element):
        import app.ui.components as comp

        lookup = {
            "permensos-8-2023::pasal1": {
                "bab": "I", "bagian": None, "pasal": 1,
                "text": "Teks.", "source_page": 3,
            }
        }
        lookup_path = tmp_path / "parent_lookup.json"
        lookup_path.write_text(json.dumps(lookup), encoding="utf-8")
        monkeypatch.setattr(comp, "_PARENT_LOOKUP_PATH", lookup_path)
        comp._parent_lookup = None

        # pasal 99 does not exist in lookup
        elements = _cards()([1, 99], None)
        assert len(elements) == 1
        assert elements[0].name == "Pasal 1"

    def test_deduplicates_repeated_pasals(self, tmp_path, monkeypatch, mock_cl_element):
        import app.ui.components as comp

        lookup = {
            "permensos-8-2023::pasal1": {
                "bab": "I", "bagian": None, "pasal": 1,
                "text": "Teks.", "source_page": 3,
            }
        }
        lookup_path = tmp_path / "parent_lookup.json"
        lookup_path.write_text(json.dumps(lookup), encoding="utf-8")
        monkeypatch.setattr(comp, "_PARENT_LOOKUP_PATH", lookup_path)
        comp._parent_lookup = None

        elements = _cards()([1, 1, 1], None)
        assert len(elements) == 1

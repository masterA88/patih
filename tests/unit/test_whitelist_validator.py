"""Unit tests for whitelist_validator.py.

Uses mock Chroma collection for speed (no I/O, no embedder needed).
Spec: build-spec Section 5.4 line 990.

Mock strategy:
  - chroma_collection.get(where=..., include=..., limit=...) returns dict with
    "ids" list. Non-empty ids -> match found. Empty ids -> not found.
  - doc_registry matches real documents.json structure with n_pasal=34.
"""

import pytest
from unittest.mock import MagicMock

from app.validators.whitelist_validator import validate_citations


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DOC_REGISTRY = {
    "permensos-8-2023": {
        "doc_id": "permensos-8-2023",
        "n_pasal": 34,
    }
}

DOC_ID = "permensos-8-2023"


def _make_chroma_mock(found: bool):
    """Return a mock chroma collection where get() returns found/not-found."""
    mock = MagicMock()
    if found:
        mock.get.return_value = {"ids": ["some-chunk-id"], "metadatas": [{}]}
    else:
        mock.get.return_value = {"ids": [], "metadatas": []}
    return mock


def _citation(pasal, ayat=None, huruf=None):
    return {
        "pasal": pasal,
        "ayat": ayat,
        "huruf": huruf,
        "raw": f"Pasal {pasal}",
        "char_start": 0,
        "char_end": 0,
    }


# ---------------------------------------------------------------------------
# Tests — range check
# ---------------------------------------------------------------------------

class TestRangeCheck:
    def test_pasal_within_range_valid(self):
        """Pasal 5 (n_pasal=34) -> range check passes, Chroma found -> True."""
        chroma = _make_chroma_mock(found=True)
        result = validate_citations([_citation(5)], DOC_ID, chroma, DOC_REGISTRY)
        assert result == [True]

    def test_pasal_exceeds_n_pasal_invalid(self):
        """Pasal 99 (n_pasal=34) -> range check fails -> False without Chroma query."""
        chroma = _make_chroma_mock(found=True)  # would be True if queried
        result = validate_citations([_citation(99)], DOC_ID, chroma, DOC_REGISTRY)
        assert result == [False]
        chroma.get.assert_not_called()

    def test_pasal_zero_invalid(self):
        """Pasal 0 -> invalid (below 1)."""
        chroma = _make_chroma_mock(found=True)
        result = validate_citations([_citation(0)], DOC_ID, chroma, DOC_REGISTRY)
        assert result == [False]
        chroma.get.assert_not_called()

    def test_pasal_boundary_34_valid(self):
        """Pasal 34 (= n_pasal) -> range check passes -> True if Chroma found."""
        chroma = _make_chroma_mock(found=True)
        result = validate_citations([_citation(34)], DOC_ID, chroma, DOC_REGISTRY)
        assert result == [True]

    def test_pasal_35_invalid(self):
        """Pasal 35 (> n_pasal=34) -> False."""
        chroma = _make_chroma_mock(found=True)
        result = validate_citations([_citation(35)], DOC_ID, chroma, DOC_REGISTRY)
        assert result == [False]


# ---------------------------------------------------------------------------
# Tests — Chroma existence check
# ---------------------------------------------------------------------------

class TestChromaCheck:
    def test_pasal_ayat_found(self):
        """Pasal 5 ayat (2) -> Chroma returns match -> True."""
        chroma = _make_chroma_mock(found=True)
        result = validate_citations([_citation(5, ayat=2)], DOC_ID, chroma, DOC_REGISTRY)
        assert result == [True]

    def test_pasal_ayat_not_found(self):
        """Pasal 5 ayat (99) -> Chroma returns no match -> False."""
        chroma = _make_chroma_mock(found=False)
        result = validate_citations([_citation(5, ayat=99)], DOC_ID, chroma, DOC_REGISTRY)
        assert result == [False]

    def test_pasal_ayat_huruf_found(self):
        """Pasal 5 ayat (2) huruf a -> Chroma match -> True."""
        chroma = _make_chroma_mock(found=True)
        result = validate_citations([_citation(5, ayat=2, huruf="a")], DOC_ID, chroma, DOC_REGISTRY)
        assert result == [True]

    def test_pasal_ayat_huruf_not_found(self):
        """Pasal 5 ayat (2) huruf z -> no match (z doesn't exist) -> False."""
        chroma = _make_chroma_mock(found=False)
        result = validate_citations([_citation(5, ayat=2, huruf="z")], DOC_ID, chroma, DOC_REGISTRY)
        assert result == [False]

    def test_pasal_only_no_chroma_match(self):
        """Pasal 5 only (no ayat) -> Chroma returns no match -> False.
        This can happen if parse silently failed for a valid pasal number.
        """
        chroma = _make_chroma_mock(found=False)
        result = validate_citations([_citation(5)], DOC_ID, chroma, DOC_REGISTRY)
        assert result == [False]


# ---------------------------------------------------------------------------
# Tests — multiple citations
# ---------------------------------------------------------------------------

class TestMultipleCitations:
    def test_mixed_valid_invalid(self):
        """Pasal 5 (found) + Pasal 99 (range fail) -> [True, False]."""
        chroma = _make_chroma_mock(found=True)
        citations = [_citation(5), _citation(99)]
        result = validate_citations(citations, DOC_ID, chroma, DOC_REGISTRY)
        assert result == [True, False]

    def test_all_valid(self):
        """Two valid citations -> both True."""
        chroma = _make_chroma_mock(found=True)
        citations = [_citation(5), _citation(10)]
        result = validate_citations(citations, DOC_ID, chroma, DOC_REGISTRY)
        assert result == [True, True]

    def test_empty_citations(self):
        """Empty input -> empty output."""
        chroma = _make_chroma_mock(found=True)
        result = validate_citations([], DOC_ID, chroma, DOC_REGISTRY)
        assert result == []

    def test_chroma_exception_conservative_false(self):
        """If Chroma raises an exception, result should be conservative False."""
        chroma = MagicMock()
        chroma.get.side_effect = RuntimeError("connection error")
        result = validate_citations([_citation(5)], DOC_ID, chroma, DOC_REGISTRY)
        assert result == [False]


# ---------------------------------------------------------------------------
# Tests — where clause construction (verify query parameters)
# ---------------------------------------------------------------------------

class TestWhereClauseConstruction:
    def test_pasal_only_uses_no_ayat_filter(self):
        """Pasal-only citation -> where clause should NOT include 'ayat' key."""
        chroma = _make_chroma_mock(found=True)
        validate_citations([_citation(5)], DOC_ID, chroma, DOC_REGISTRY)
        call_kwargs = chroma.get.call_args.kwargs
        where = call_kwargs.get("where", {})
        # Verify $and conditions don't include ayat
        conditions = where.get("$and", [])
        condition_keys = [list(c.keys())[0] for c in conditions]
        assert "ayat" not in condition_keys

    def test_pasal_ayat_includes_ayat_filter(self):
        """Pasal+ayat citation -> where clause includes ayat with int value."""
        chroma = _make_chroma_mock(found=True)
        validate_citations([_citation(5, ayat=2)], DOC_ID, chroma, DOC_REGISTRY)
        call_kwargs = chroma.get.call_args.kwargs
        where = call_kwargs.get("where", {})
        conditions = where.get("$and", [])
        ayat_conditions = [c for c in conditions if "ayat" in c]
        assert len(ayat_conditions) == 1
        # ayat should be queried as int 2, not string "2"
        assert ayat_conditions[0]["ayat"]["$eq"] == 2

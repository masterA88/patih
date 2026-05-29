"""
Unit tests for cross_ref_resolver.py — regex match cases.

Tests cover:
  - "Pasal 5 ayat (2)" → extracts [5]
  - "Pasal 1" standalone → extracts [1]
  - "Pasal 10" standalone → extracts [10]
  - "Pasal-Pasal" plural form (no number directly after) → NOT matched
  - Multiple Pasals in one text
  - resolve_cross_refs() integration with a fake parent_lookup
"""

from __future__ import annotations

import pytest

from app.retrieval.cross_ref_resolver import extract_pasal_refs, resolve_cross_refs


class TestExtractPasalRefs:
    def test_pasal_with_ayat(self):
        """'Pasal 5 ayat (2)' → [5]."""
        text = "sebagaimana dimaksud dalam Pasal 5 ayat (2) Peraturan ini"
        result = extract_pasal_refs(text)
        assert result == [5]

    def test_pasal_standalone(self):
        """'Pasal 1' without ayat → [1]."""
        text = "merujuk kepada Pasal 1 tentang definisi"
        result = extract_pasal_refs(text)
        assert result == [1]

    def test_pasal_10_standalone(self):
        """'Pasal 10' → [10]."""
        text = "ketentuan Pasal 10 berlaku secara mutatis mutandis"
        result = extract_pasal_refs(text)
        assert result == [10]

    def test_pasal_plural_form_no_number(self):
        """
        'Pasal-Pasal' (plural noun) should NOT be extracted.
        The regex requires \\d+ after 'Pasal\\s+', and 'Pasal-Pasal' has a hyphen.
        """
        text = "Pasal-Pasal berikut berlaku untuk PMI bermasalah"
        result = extract_pasal_refs(text)
        assert result == [], f"Expected no match for 'Pasal-Pasal', got {result}"

    def test_multiple_pasals_in_text(self):
        """Multiple distinct Pasal references → deduplicated sorted list."""
        text = "Pasal 3, Pasal 1 ayat (2), dan Pasal 3 berlaku di sini"
        result = extract_pasal_refs(text)
        assert result == [1, 3]  # sorted, deduped

    def test_pasal_high_number(self):
        """'Pasal 34' → [34]."""
        text = "pengaturan lebih lanjut ada di Pasal 34"
        result = extract_pasal_refs(text)
        assert result == [34]

    def test_no_pasal_reference(self):
        """Text with no Pasal references → []."""
        text = "Dalam hal ini, kementerian sosial bertanggung jawab"
        result = extract_pasal_refs(text)
        assert result == []

    def test_pasal_with_large_ayat(self):
        """'Pasal 15 ayat (3)' → [15]."""
        text = "diatur dalam Pasal 15 ayat (3) peraturan ini"
        result = extract_pasal_refs(text)
        assert result == [15]

    def test_multiple_ayat_same_pasal(self):
        """'Pasal 5 ayat (1)' and 'Pasal 5 ayat (2)' → [5] (deduped)."""
        text = "Pasal 5 ayat (1) dan Pasal 5 ayat (2) mengatur eksploitasi"
        result = extract_pasal_refs(text)
        assert result == [5]

    def test_pasal_in_heading_context(self):
        """'BAB II Pasal 4' context → [4]."""
        text = "BAB II KETENTUAN UMUM Pasal 4 tentang asesmen"
        result = extract_pasal_refs(text)
        assert result == [4]


class TestResolveCrossRefs:
    """Integration tests for resolve_cross_refs() with fake parent_lookup."""

    @pytest.fixture
    def parent_lookup(self) -> dict:
        """Minimal fake parent lookup table for 5 Pasals."""
        return {
            f"permensos-8-2023::pasal{n}": {
                "chunk_id": f"permensos-8-2023::pasal{n}",
                "pasal": n,
                "chunk_type": "parent",
                "text": f"Teks Pasal {n} tentang sesuatu yang penting.",
                "text_for_embed": f"passage: Pasal {n}",
            }
            for n in range(1, 6)
        }

    def test_resolves_referenced_pasal(self, parent_lookup):
        """A parent chunk that references Pasal 3 causes Pasal 3 to be added."""
        parent_with_ref = {
            "chunk_id": "permensos-8-2023::pasal5",
            "pasal": 5,
            "text": "berlaku sebagaimana dimaksud dalam Pasal 3 ayat (1)",
            "text_for_embed": "passage: Pasal 5",
        }
        parents = [(parent_with_ref, 0.9)]
        already_included = {"permensos-8-2023::pasal5"}

        result = resolve_cross_refs(
            parents, parent_lookup, already_included, doc_id="permensos-8-2023"
        )
        assert len(result) == 1
        assert result[0]["pasal"] == 3

    def test_dedup_already_included(self, parent_lookup):
        """A referenced Pasal already in already_included is not added again."""
        parent_with_ref = {
            "chunk_id": "permensos-8-2023::pasal5",
            "pasal": 5,
            "text": "berlaku sebagaimana dimaksud dalam Pasal 1",
            "text_for_embed": "passage: Pasal 5",
        }
        parents = [(parent_with_ref, 0.9)]
        # Pasal 1 already in context
        already_included = {
            "permensos-8-2023::pasal1",
            "permensos-8-2023::pasal5",
        }
        result = resolve_cross_refs(
            parents, parent_lookup, already_included, doc_id="permensos-8-2023"
        )
        assert result == []

    def test_cap_at_max_refs(self, parent_lookup):
        """No more than max_refs new Pasals are added."""
        # Parent text references 4 Pasals, but cap is 3
        parent_with_refs = {
            "chunk_id": "permensos-8-2023::pasal5",
            "pasal": 5,
            "text": "lihat Pasal 1, Pasal 2, Pasal 3, dan Pasal 4 untuk detailnya",
            "text_for_embed": "passage: Pasal 5",
        }
        parents = [(parent_with_refs, 0.9)]
        already_included = {"permensos-8-2023::pasal5"}

        result = resolve_cross_refs(
            parents, parent_lookup, already_included,
            doc_id="permensos-8-2023", max_refs=3
        )
        assert len(result) == 3

    def test_no_cross_refs(self, parent_lookup):
        """Parent with no Pasal references returns empty list."""
        parent_no_ref = {
            "chunk_id": "permensos-8-2023::pasal5",
            "pasal": 5,
            "text": "penanganan PMI bermasalah dilakukan secara holistik",
            "text_for_embed": "passage: Pasal 5",
        }
        parents = [(parent_no_ref, 0.9)]
        already_included = {"permensos-8-2023::pasal5"}

        result = resolve_cross_refs(
            parents, parent_lookup, already_included, doc_id="permensos-8-2023"
        )
        assert result == []

    def test_referenced_pasal_not_in_lookup(self, parent_lookup):
        """If referenced Pasal not in parent_lookup, skip silently."""
        parent_with_ref = {
            "chunk_id": "permensos-8-2023::pasal5",
            "pasal": 5,
            # References Pasal 99 which doesn't exist in our 5-item lookup
            "text": "lihat Pasal 99 untuk pengaturan lebih lanjut",
            "text_for_embed": "passage: Pasal 5",
        }
        parents = [(parent_with_ref, 0.9)]
        already_included = {"permensos-8-2023::pasal5"}

        result = resolve_cross_refs(
            parents, parent_lookup, already_included, doc_id="permensos-8-2023"
        )
        assert result == []

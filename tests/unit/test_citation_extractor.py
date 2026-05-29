"""Unit tests for citation_extractor.py.

10+ response samples asserting extraction correctness.
Spec: build-spec Section 5.4 line 989.
"""

import pytest
from app.validators.citation_extractor import extract_citations


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _pasal_nums(citations):
    return [c["pasal"] for c in citations]


def _ayat_nums(citations):
    return [c["ayat"] for c in citations]


def _huruf_list(citations):
    return [c["huruf"] for c in citations]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBasicExtraction:
    def test_pasal_only(self):
        """'Pasal 5' without ayat/huruf -> 1 citation, ayat=None, huruf=None."""
        cits = extract_citations("Hal ini diatur dalam Pasal 5.")
        assert len(cits) == 1
        assert cits[0]["pasal"] == 5
        assert cits[0]["ayat"] is None
        assert cits[0]["huruf"] is None

    def test_pasal_with_ayat(self):
        """'Pasal 5 ayat (2)' -> ayat populated, huruf=None."""
        cits = extract_citations("Lihat Pasal 5 ayat (2) untuk detailnya.")
        assert len(cits) == 1
        assert cits[0]["pasal"] == 5
        assert cits[0]["ayat"] == 2
        assert cits[0]["huruf"] is None

    def test_pasal_with_ayat_and_huruf(self):
        """'Pasal 5 ayat (2) huruf b' -> all fields populated."""
        cits = extract_citations("Berdasarkan Pasal 5 ayat (2) huruf b tentang eksploitasi.")
        assert len(cits) == 1
        assert cits[0]["pasal"] == 5
        assert cits[0]["ayat"] == 2
        assert cits[0]["huruf"] == "b"

    def test_case_insensitive_upper(self):
        """'PASAL 1' (fully uppercase) -> match."""
        cits = extract_citations("Sesuai PASAL 1 peraturan ini.")
        assert len(cits) == 1
        assert cits[0]["pasal"] == 1

    def test_case_insensitive_mixed(self):
        """'Pasal' with varying case -> match."""
        cits = extract_citations("pAsAl 3 mengatur ruang lingkup.")
        assert len(cits) == 1
        assert cits[0]["pasal"] == 3


class TestMultipleCitations:
    def test_explicit_two_pasals(self):
        """'Pasal 5 dan Pasal 6' -> 2 explicit citations (both have 'Pasal' prefix)."""
        cits = extract_citations("Diatur dalam Pasal 5 dan Pasal 6.")
        assert len(cits) == 2
        assert _pasal_nums(cits) == [5, 6]

    def test_implicit_second_pasal_not_captured(self):
        """'Pasal 5 dan 6' -> Phase 1: only Pasal 5 captured (6 has no 'Pasal' prefix)."""
        cits = extract_citations("Pasal 5 dan 6 mengatur hal ini.")
        # Phase 1 documented trade-off: bare '6' not captured
        assert len(cits) == 1
        assert cits[0]["pasal"] == 5

    def test_list_of_citations(self):
        """Multiple citations in a list -> all captured."""
        text = (
            "Berikut bentuk eksploitasi:\n"
            "a. Pelacuran (Pasal 5 ayat (2) huruf a)\n"
            "b. Kerja paksa (Pasal 5 ayat (2) huruf b)\n"
            "c. Perbudakan (Pasal 5 ayat (2) huruf c)\n"
        )
        cits = extract_citations(text)
        assert len(cits) == 3
        assert all(c["pasal"] == 5 for c in cits)
        assert all(c["ayat"] == 2 for c in cits)
        assert _huruf_list(cits) == ["a", "b", "c"]

    def test_multi_huruf_in_one_phrase_captures_first_only(self):
        """'Pasal 5 ayat (2) huruf a, huruf b, huruf c' -> Phase 1: only 'huruf a' captured.
        This is a documented Phase 1 trade-off; Phase 2 should handle multi-huruf listing.
        """
        cits = extract_citations("Sumber: Pasal 5 ayat (2) huruf a, huruf b, huruf c.")
        # Phase 1: regex captures Pasal 5 ayat 2 huruf a only
        assert len(cits) == 1
        assert cits[0]["pasal"] == 5
        assert cits[0]["ayat"] == 2
        assert cits[0]["huruf"] == "a"


class TestEdgeCases:
    def test_pasal_without_number_no_match(self):
        """'Pasal' text without a following digit -> no match."""
        cits = extract_citations("Peraturan ini mengacu pada beberapa Pasal terkait.")
        assert len(cits) == 0

    def test_pasal_range_captures_start_only(self):
        """'Pasal-Pasal 5 sampai 7' -> captures 'Pasal 5', range expansion skipped."""
        cits = extract_citations("Sebagaimana Pasal-Pasal 5 sampai 7 menyatakan.")
        assert len(cits) >= 1
        assert cits[0]["pasal"] == 5

    def test_refusal_text_no_citations(self):
        """Standard refusal text should produce 0 citations."""
        refusal = (
            "Informasi yang Anda tanyakan tidak diatur secara spesifik "
            "dalam Peraturan Menteri Sosial Nomor 8 Tahun 2023. "
            "Untuk pertanyaan ini, mohon merujuk ke peraturan lain."
        )
        cits = extract_citations(refusal)
        assert len(cits) == 0

    def test_duplicate_citations_preserved(self):
        """Same Pasal cited twice -> both preserved (dedup is caller's job)."""
        text = "Pasal 5 mengatur X. Berdasarkan Pasal 5, Y juga berlaku."
        cits = extract_citations(text)
        assert len(cits) == 2
        assert all(c["pasal"] == 5 for c in cits)

    def test_char_positions_correct(self):
        """char_start and char_end should point to correct substring."""
        text = "XXX Pasal 10 ayat (3) YYY"
        cits = extract_citations(text)
        assert len(cits) == 1
        c = cits[0]
        assert text[c["char_start"]:c["char_end"]] == c["raw"]

    def test_large_pasal_number(self):
        """Multi-digit Pasal number extracted correctly."""
        cits = extract_citations("Peraturan ini memiliki Pasal 34 sebagai penutup.")
        assert len(cits) == 1
        assert cits[0]["pasal"] == 34

    def test_raw_field_matches_text(self):
        """raw field matches the original matched text."""
        cits = extract_citations("Pasal 7 ayat (1) huruf c")
        assert len(cits) == 1
        assert cits[0]["raw"] == "Pasal 7 ayat (1) huruf c"

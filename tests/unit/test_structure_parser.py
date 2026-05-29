"""Unit tests for structure_parser.py."""

import re
import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from app.ingest.structure_parser import (
    parse_structure,
    _normalize_pasal_body,
    _strip_penjelasan,
)


# ---------------------------------------------------------------------------
# Penjelasan stripping
# ---------------------------------------------------------------------------

def test_strip_penjelasan_truncates_late_marker():
    """PENJELASAN past 40% of the doc should truncate the body there."""
    body = "Pasal 1\nisi.\n" * 50  # ~600 chars of body
    text = body + "\nPENJELASAN\nATAS UNDANG-UNDANG\nPasal 1\npenjelasan isi.\n" * 1
    stripped = _strip_penjelasan(text)
    assert "PENJELASAN" not in stripped
    assert stripped.endswith("isi.\n") or stripped.rstrip().endswith("isi.")


def test_strip_penjelasan_keeps_early_marker():
    """PENJELASAN in the first 40% (e.g. omnibus / TOC) must NOT truncate."""
    text = "PENJELASAN\n" + ("Pasal 1\nisi panjang sekali.\n" * 100)
    stripped = _strip_penjelasan(text)
    assert stripped == text  # unchanged


def test_strip_penjelasan_no_marker_noop():
    text = "BAB I\nPasal 1\nisi.\n"
    assert _strip_penjelasan(text) == text


def test_parse_structure_drops_penjelasan_pasal_duplicates():
    """A doc whose Pasal repeat in a late PENJELASAN must not double-count."""
    body = (
        "BAB I\nKETENTUAN UMUM\n\n"
        "Pasal 1\n(1) Definisi pertama yang cukup panjang untuk mengisi ruang.\n\n"
        "Pasal 2\n(1) Ketentuan kedua yang juga cukup panjang sebagai isi.\n\n"
    )
    # Pad body so PENJELASAN lands past 40%
    body = body + ("Pasal 3\n(1) isi tambahan yang panjang.\n\n" * 10)
    penjelasan = "PENJELASAN\nATAS\n\nPasal 1\nCukup jelas.\n\nPasal 2\nCukup jelas.\n"
    ast = parse_structure(body + penjelasan)
    all_pasal = [p["nomor"] for b in ast["bab"] for p in b["pasal"]]
    # Pasal 1 and 2 appear once each (3 appears once too); no Penjelasan repeats
    assert all_pasal.count(1) == 1
    assert all_pasal.count(2) == 1


# ---------------------------------------------------------------------------
# Golden tests
# ---------------------------------------------------------------------------

GOLDEN_MINIMAL = (
    "BAB I\n"
    "KETENTUAN UMUM\n\n"
    "Pasal 1\n"
    "(1) Lorem ipsum sit amet.\n"
    "(2) Lorem ipsum dolor sit amet.\n"
)


def test_golden_minimal_bab_count():
    ast = parse_structure(GOLDEN_MINIMAL)
    assert len(ast["bab"]) == 1
    assert ast["bab"][0]["nomor"] == "I"
    assert ast["bab"][0]["judul"] == "KETENTUAN UMUM"


def test_golden_minimal_pasal_count():
    ast = parse_structure(GOLDEN_MINIMAL)
    bab = ast["bab"][0]
    assert len(bab["pasal"]) == 1
    assert bab["pasal"][0]["nomor"] == 1


def test_golden_minimal_ayat_count():
    ast = parse_structure(GOLDEN_MINIMAL)
    pasal = ast["bab"][0]["pasal"][0]
    assert len(pasal["ayat"]) == 2
    assert pasal["ayat"][0]["nomor"] == 1
    assert pasal["ayat"][1]["nomor"] == 2


def test_golden_multi_bab():
    text = (
        "BAB I\nKETENTUAN UMUM\n\n"
        "Pasal 1\n(1) Definisi satu.\n\n"
        "BAB II\nPENANGANAN\n\n"
        "Pasal 2\n(1) Tata cara penanganan.\n"
        "(2) Dilaksanakan oleh pejabat.\n\n"
        "Pasal 3\nPelaksana penanganan adalah Menteri.\n"
    )
    ast = parse_structure(text)
    assert len(ast["bab"]) == 2
    assert ast["bab"][1]["nomor"] == "II"
    pasal_nums = [p["nomor"] for p in ast["bab"][1]["pasal"]]
    assert 2 in pasal_nums
    assert 3 in pasal_nums


def test_golden_huruf_in_ayat():
    text = (
        "BAB I\nUMUM\n\n"
        "Pasal 1\n"
        "(1) Penanganan dilakukan melalui:\n"
        "a. rehabilitasi sosial;\n"
        "b. jaminan sosial;\n"
        "c. perlindungan sosial.\n"
        "(2) Penanganan diatur lebih lanjut oleh Menteri.\n"
    )
    ast = parse_structure(text)
    pasal = ast["bab"][0]["pasal"][0]
    assert len(pasal["ayat"]) == 2
    ayat1 = pasal["ayat"][0]
    assert len(ayat1["huruf"]) == 3
    assert ayat1["huruf"][0]["nomor"] == "a"
    assert ayat1["huruf"][2]["nomor"] == "c"


def test_inline_cross_ref_not_false_ayat():
    """
    Cross-references like 'sebagaimana dimaksud pada ayat\\n(1)\\nMenteri...'
    must NOT create a duplicate ayat (1).
    """
    text = (
        "BAB I\nUMUM\n\n"
        "Pasal 7\n"
        "(1) Menteri melakukan pemulangan bagi Korban.\n"
        "(2) Dalam melakukan pemulangan sebagaimana dimaksud pada ayat\n"
        "(1) Menteri berkoordinasi dengan kementerian terkait.\n"
    )
    ast = parse_structure(text)
    pasal = ast["bab"][0]["pasal"][0]
    ayat_nums = [a["nomor"] for a in pasal["ayat"]]
    # Should only have ayat 1 and 2 — not a duplicate (1) from the cross-ref
    assert ayat_nums == [1, 2], f"Expected [1, 2], got {ayat_nums}"


def test_no_bab_returns_fallback():
    """Documents without BAB markers return a fallback structure."""
    text = "Pasal 1\n(1) Isi satu.\nPasal 2\n(1) Isi dua.\n"
    ast = parse_structure(text)
    assert len(ast["bab"]) == 1
    assert len(ast["bab"][0]["pasal"]) == 2


def test_bagian_detected():
    text = (
        "BAB II\nPENANGANAN\n\n"
        "Bagian Kesatu\nUmum\n\n"
        "Pasal 4\nPenanganan dilakukan berdasarkan asesmen.\n\n"
        "Bagian Kedua\nTahapan\n\n"
        "Pasal 5\n(1) Tahapan dimulai dari asesmen.\n"
    )
    ast = parse_structure(text)
    bab = ast["bab"][0]
    assert len(bab["bagian"]) == 2
    assert bab["bagian"][0]["nomor"] in ("Kesatu", "Ke satu")
    assert bab["bagian"][1]["nomor"] in ("Kedua",)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_huruf_i_not_false_positive():
    """The letter 'i' appearing inside a word should not trigger huruf parsing."""
    text = (
        "BAB I\nUMUM\n\n"
        "Pasal 1\n"
        "(1) Ini adalah definisi tentang tindak pidana perdagangan orang di Indonesia.\n"
    )
    ast = parse_structure(text)
    ayat = ast["bab"][0]["pasal"][0]["ayat"][0]
    # 'i' inside words like 'ini', 'di' should NOT generate huruf items
    assert ayat["huruf"] == []


def test_pasal_without_ayat_has_empty_ayat_list():
    text = (
        "BAB I\nUMUM\n\n"
        "Pasal 16\n"
        "Ketentuan lebih lanjut ditetapkan oleh Pejabat Tinggi Madya.\n"
    )
    ast = parse_structure(text)
    pasal = ast["bab"][0]["pasal"][0]
    # No (1) markers → ayat list should be empty
    assert pasal["ayat"] == []


def test_all_34_pasal_from_permensos8():
    """Integration-lite: verify parse_structure returns 34 Pasal from real PDF text."""
    import json
    from pathlib import Path
    parsed_path = (
        Path(__file__).parent.parent.parent
        / "data"
        / "parsed"
        / "permensos-8-2023.json"
    )
    # Skip if file doesn't exist (CI without PDF)
    if not parsed_path.exists():
        pytest.skip("Parsed output not available")

    with open(parsed_path, encoding="utf-8") as f:
        data = json.load(f)

    all_pasal = [p for b in data["ast"]["bab"] for p in b["pasal"]]
    assert len(all_pasal) == 34, f"Expected 34 Pasal, got {len(all_pasal)}"


# ---------------------------------------------------------------------------
# Normalizer unit tests
# ---------------------------------------------------------------------------

def test_normalize_collapses_word_wrap():
    body = "(1) \nMenteri dapat\n melakukan\n pemulangan.\n"
    normalized = _normalize_pasal_body(body)
    lines = [l for l in normalized.split("\n") if l.strip()]
    assert len(lines) == 1
    assert "Menteri dapat melakukan pemulangan." in lines[0]


def test_normalize_cross_ref_not_split():
    body = (
        "(2) \nDalam melakukan pemulangan sebagaimana dimaksud \npada \nayat \n(1) \n"
        "Menteri \nberkoordinasi."
    )
    normalized = _normalize_pasal_body(body)
    # Only ONE line starting with (2) — the (1) must be inline
    ayat_starts = [l for l in normalized.split("\n") if re.match(r"^\(\d+\)", l.strip())]
    assert len(ayat_starts) == 1, f"Expected 1 ayat start, got {ayat_starts}"


def test_normalize_two_real_ayat_kept_separate():
    body = "(1) Pertama.\n(2) Kedua.\n"
    normalized = _normalize_pasal_body(body)
    ayat_starts = [l for l in normalized.split("\n") if re.match(r"^\(\d+\)", l.strip())]
    assert len(ayat_starts) == 2


# ---------------------------------------------------------------------------
# Property-based tests (Hypothesis)
# ---------------------------------------------------------------------------

@given(
    n_bab=st.integers(min_value=1, max_value=5),
    pasal_per_bab=st.integers(min_value=1, max_value=4),
    ayat_per_pasal=st.integers(min_value=0, max_value=3),
)
@settings(
    max_examples=40,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_property_parse_roundtrip(n_bab: int, pasal_per_bab: int, ayat_per_pasal: int):
    """
    Randomly generated valid BAB/Pasal/Ayat structures parse correctly.
    Total Pasal count must match expected, and all Pasal numbers must be unique.
    """
    _roman = ["I", "II", "III", "IV", "V"]

    lines: list[str] = []
    pasal_counter = 1
    expected_pasal_count = n_bab * pasal_per_bab

    for b in range(n_bab):
        roman = _roman[b % len(_roman)]
        lines.append(f"BAB {roman}")
        lines.append(f"JUDUL BAB {roman}")
        lines.append("")

        for p in range(pasal_per_bab):
            lines.append(f"Pasal {pasal_counter}")
            if ayat_per_pasal > 0:
                for a in range(1, ayat_per_pasal + 1):
                    lines.append(f"({a}) Teks ayat {a} dari Pasal {pasal_counter}.")
            else:
                lines.append(f"Teks Pasal {pasal_counter} tanpa ayat.")
            lines.append("")
            pasal_counter += 1

    text = "\n".join(lines)
    ast = parse_structure(text)

    all_pasal = [p for b in ast["bab"] for p in b["pasal"]]
    assert len(all_pasal) == expected_pasal_count, (
        f"Expected {expected_pasal_count}, got {len(all_pasal)}"
    )
    pasal_nums = [p["nomor"] for p in all_pasal]
    assert len(pasal_nums) == len(set(pasal_nums)), "Duplicate Pasal numbers"

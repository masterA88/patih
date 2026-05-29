"""Unit tests for app.llm.lang_detect.

Golden cases covering Indonesian queries, English queries, and mixed/ambiguous.
"""
import pytest
from app.llm.lang_detect import LangDetector, detect


@pytest.fixture(scope="module")
def detector():
    return LangDetector()


# ---------------------------------------------------------------------------
# Indonesian golden cases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "apa itu korban TPPO?",
    "siapa yang berwenang menangani korban perdagangan orang?",
    "bagaimana alur reintegrasi sosial bagi pekerja migran bermasalah?",
    "Apa saja bentuk eksploitasi menurut Permensos 8/2023?",
    "Sebutkan hak-hak korban TPPO yang diatur dalam peraturan ini.",
])
def test_detects_indonesian(detector, text):
    result = detector.detect(text)
    assert result == "id", f"Expected 'id' for: {text!r}, got {result!r}"


# ---------------------------------------------------------------------------
# English golden cases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "What are the forms of exploitation regulated in this ministerial regulation?",
    "Who has the authority to conduct needs assessment for trafficking victims?",
    "What social rehabilitation services are available for TPPO victims?",
    "How does the reintegration process work for problematic migrant workers?",
])
def test_detects_english(detector, text):
    result = detector.detect(text)
    assert result == "en", f"Expected 'en' for: {text!r}, got {result!r}"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_string_defaults_to_id(detector):
    assert detector.detect("") == "id"


def test_whitespace_only_defaults_to_id(detector):
    assert detector.detect("   \n\t  ") == "id"


def test_none_like_empty(detector):
    # detect() on empty string should gracefully return 'id'
    assert detector.detect("") == "id"


def test_short_query_indonesian(detector):
    # Very short queries — lingua may be uncertain, should default to 'id'
    result = detector.detect("apa itu?")
    # Either 'id' or 'mixed' is acceptable; must not be 'en'
    assert result in ("id", "mixed"), f"Got {result!r} for 'apa itu?'"


def test_short_query_english(detector):
    result = detector.detect("what is this?")
    assert result in ("en", "mixed"), f"Got {result!r} for 'what is this?'"


# ---------------------------------------------------------------------------
# Module-level detect() function
# ---------------------------------------------------------------------------

def test_module_detect_id():
    assert detect("apa itu korban TPPO?") == "id"


def test_module_detect_en():
    result = detect("What are the rights of trafficking victims?")
    assert result == "en"


def test_module_detect_empty():
    assert detect("") == "unknown"


def test_module_detect_none_string():
    # Module-level detect treats empty as unknown
    assert detect("") == "unknown"

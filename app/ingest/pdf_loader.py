"""PyMuPDF text-native extraction with Tesseract OCR fallback."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Quality thresholds
_MIN_CHARS_PER_PAGE = 50
_MAX_NON_ASCII_RATIO = 0.60

PARSE_METHOD_TEXT = "pymupdf_text"
PARSE_METHOD_OCR = "ocr_tesseract"
PARSE_METHOD_LOW_QUALITY = "pymupdf_text_low_quality"


def _assess_quality(pages: list[str]) -> tuple[bool, str]:
    """
    Returns (is_good, reason).
    is_good=False means the text quality is suspect and OCR should be tried.
    """
    if not pages:
        return False, "no_pages"

    total_chars = sum(len(p) for p in pages)
    avg_chars_per_page = total_chars / len(pages)

    if avg_chars_per_page < _MIN_CHARS_PER_PAGE:
        return False, f"avg_chars_per_page={avg_chars_per_page:.1f}<{_MIN_CHARS_PER_PAGE}"

    # Non-ASCII ratio check on concatenated text (sample up to 5000 chars)
    sample = "".join(pages)[:5000]
    if sample:
        non_ascii = sum(1 for c in sample if ord(c) > 127)
        ratio = non_ascii / len(sample)
        if ratio > _MAX_NON_ASCII_RATIO:
            return False, f"non_ascii_ratio={ratio:.2f}>{_MAX_NON_ASCII_RATIO}"

    return True, "ok"


def _configure_tesseract(pytesseract_mod) -> None:
    """Best-effort auto-config so OCR works without manual env setup.

    - Point pytesseract at the Windows default install if not already on PATH.
    - Set TESSDATA_PREFIX to the project-local models/tessdata (which ships the
      Indonesian `ind` language pack) if the env var isn't already set.
    """
    import os
    import shutil

    # Locate the tesseract binary
    if not shutil.which("tesseract"):
        for candidate in (
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ):
            if Path(candidate).exists():
                pytesseract_mod.pytesseract.tesseract_cmd = candidate
                break

    # Point at the local tessdata (with ind.traineddata) unless already set
    if not os.environ.get("TESSDATA_PREFIX"):
        local_tessdata = Path(__file__).parent.parent.parent / "models" / "tessdata"
        if local_tessdata.exists():
            os.environ["TESSDATA_PREFIX"] = str(local_tessdata)


def _try_ocr(pdf_path: Path) -> tuple[list[str], bool]:
    """
    Attempt Tesseract OCR on each page rendered as image.
    Returns (page_texts, success).
    success=False if Tesseract is not available.
    """
    try:
        import pytesseract
        import pymupdf
    except ImportError as e:
        logger.warning("OCR dependency missing: %s", e)
        return [], False

    _configure_tesseract(pytesseract)

    # Check Tesseract binary is accessible
    try:
        pytesseract.get_tesseract_version()
    except Exception as e:
        logger.warning(
            "Tesseract binary not found or not on PATH: %s. "
            "Install tesseract-ocr and add to PATH for OCR fallback.",
            e,
        )
        return [], False

    doc = pymupdf.open(str(pdf_path))
    page_texts: list[str] = []
    for page in doc:
        # Render at 2x resolution for better OCR accuracy
        mat = pymupdf.Matrix(2, 2)
        pix = page.get_pixmap(matrix=mat)
        # Convert to PIL Image via bytes
        import io
        from PIL import Image
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        text = pytesseract.image_to_string(img, lang="ind", config="--psm 6")
        page_texts.append(text)

    return page_texts, True


def load_pdf(pdf_path: Path, force_ocr: bool = False) -> tuple[str, dict]:
    """
    Load PDF and return (full_text, metadata).

    full_text:  pages concatenated with '\\n\\n--- page N ---\\n\\n' separators.
    metadata keys:
        n_pages: int
        parse_method: str  ("pymupdf_text" | "ocr_tesseract" | "pymupdf_text_low_quality")
        page_boundaries: list[int]  — char offset of each '--- page N ---' marker in full_text
        quality_ok: bool
        quality_reason: str
    """
    import pymupdf

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = pymupdf.open(str(pdf_path))
    n_pages = len(doc)

    # --- Primary: PyMuPDF text extraction ---
    raw_pages: list[str] = []
    for page in doc:
        raw_pages.append(page.get_text("text"))

    quality_ok, quality_reason = _assess_quality(raw_pages)

    if force_ocr or not quality_ok:
        logger.info(
            "Quality check failed (%s) or force_ocr=%s — attempting OCR.",
            quality_reason,
            force_ocr,
        )
        ocr_pages, ocr_ok = _try_ocr(pdf_path)
        if ocr_ok:
            page_texts = ocr_pages
            parse_method = PARSE_METHOD_OCR
        else:
            # Tesseract not available; use pymupdf text with low-quality flag
            logger.warning(
                "OCR unavailable — using PyMuPDF text despite quality warning."
            )
            page_texts = raw_pages
            parse_method = PARSE_METHOD_LOW_QUALITY
    else:
        page_texts = raw_pages
        parse_method = PARSE_METHOD_TEXT

    # Build full_text with page boundary markers
    segments: list[str] = []
    page_boundaries: list[int] = []
    cursor = 0

    for i, text in enumerate(page_texts, start=1):
        if i > 1:
            sep = f"\n\n--- page {i} ---\n\n"
            page_boundaries.append(cursor + len(segments[-1]) if segments else cursor)
            segments.append(sep)
            cursor += len(sep)
        segments.append(text)
        cursor += len(text)

    full_text = "".join(segments)

    metadata = {
        "n_pages": n_pages,
        "parse_method": parse_method,
        "page_boundaries": page_boundaries,
        "quality_ok": quality_ok,
        "quality_reason": quality_reason,
    }

    logger.info(
        "Loaded %s: %d pages, method=%s, quality_ok=%s",
        pdf_path.name,
        n_pages,
        parse_method,
        quality_ok,
    )
    return full_text, metadata

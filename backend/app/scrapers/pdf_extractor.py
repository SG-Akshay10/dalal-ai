"""PDF text extraction with pdfplumber (primary) and Tesseract OCR (fallback).

TDD NOTE: Write tests in tests/unit/test_pdf_extractor.py FIRST.
See tdd_guide_phase1.md for the full test code to copy.
"""
import logging
from io import BytesIO
from typing import Tuple

import pdfplumber
import pytesseract
import requests
from PIL import Image

logger = logging.getLogger(__name__)

# Minimum character count from pdfplumber to consider it a digital (non-scanned) PDF.
_MIN_DIGITAL_TEXT_LENGTH = 50


def extract_text_from_pdf(url: str) -> Tuple[str, float, bool]:
    """Download and extract text from a PDF URL.

    Attempts pdfplumber first (fast, accurate for digital PDFs).
    Falls back to Tesseract OCR if extracted text is too short or pdfplumber fails.

    Args:
        url: HTTP(S) URL pointing to a PDF document.

    Returns:
        Tuple of (text, confidence, ocr_used) where:
        - text: Extracted plain text content.
        - confidence: Float 0.0–1.0. 1.0 = clean digital PDF. <0.8 = OCR-only.
        - ocr_used: True if Tesseract OCR was invoked.

    Raises:
        ValueError: If the PDF cannot be downloaded (HTTP error).
    """
    # ── 1. Download PDF ────────────────────────────────────────────────────────
    try:
        response = requests.get(url, timeout=30, headers={"User-Agent": "StockSenseAI/1.0"})
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise ValueError(f"Failed to download PDF from {url}: {exc}") from exc

    pdf_bytes = BytesIO(response.content)

    # ── 2. Try pdfplumber (digital PDF) ────────────────────────────────────────
    try:
        with pdfplumber.open(pdf_bytes) as pdf:
            pages_text = [page.extract_text() or "" for page in pdf.pages]
            full_text = "\n".join(pages_text).strip()

        if len(full_text) >= _MIN_DIGITAL_TEXT_LENGTH:
            logger.debug("pdfplumber extracted %d chars from %s", len(full_text), url)
            return full_text, 1.0, False

        logger.info(
            "pdfplumber returned %d chars (below threshold) for %s — trying OCR",
            len(full_text),
            url,
        )
    except Exception as exc:
        logger.warning("pdfplumber failed for %s: %s — falling back to OCR", url, exc)

    # ── 3. Fallback: Tesseract OCR ─────────────────────────────────────────────
    pdf_bytes.seek(0)
    ocr_text = _ocr_pdf(pdf_bytes)
    confidence = 0.6 if ocr_text else 0.0
    logger.info("OCR extracted %d chars from %s (confidence=%.1f)", len(ocr_text), url, confidence)
    return ocr_text, confidence, True


def _ocr_pdf(pdf_bytes: BytesIO) -> str:
    """Run Tesseract OCR on each page of a PDF (rendered as images).

    Requires Tesseract to be installed on the system.
    On Ubuntu/Debian: sudo apt-get install tesseract-ocr
    """
    # pdfplumber can also render pages as PIL images for OCR
    pdf_bytes.seek(0)
    try:
        with pdfplumber.open(pdf_bytes) as pdf:
            page_texts = []
            for page in pdf.pages:
                img = page.to_image(resolution=200).original
                page_text = pytesseract.image_to_string(img, lang="eng")
                page_texts.append(page_text)
            return "\n".join(page_texts).strip()
    except Exception as exc:
        logger.error("OCR failed: %s", exc)
        return ""

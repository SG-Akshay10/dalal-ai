"""Unit tests for pdf_extractor.py — TDD Red-Green-Refactor.

Run: pytest tests/unit/test_pdf_extractor.py -v
Coverage: pytest tests/unit/test_pdf_extractor.py --cov=app/scrapers/pdf_extractor --cov-report=term-missing
"""
from unittest.mock import MagicMock, patch

import pytest
import requests

from app.scrapers.pdf_extractor import extract_text_from_pdf


class TestExtractTextFromPdf:
    """Tests for extract_text_from_pdf(url) -> (text, confidence, ocr_used)."""

    def _make_mock_response(self, content: bytes = b"%PDF-fake"):
        """Helper: build a mock requests.Response."""
        resp = MagicMock()
        resp.content = content
        resp.raise_for_status = MagicMock()
        return resp

    def test_digital_pdf_returns_text_without_ocr(self):
        """HAPPY PATH: pdfplumber extracts text → (text, 1.0, False)."""
        digital_text = ("RELIANCE INDUSTRIES Q3FY24 RESULTS " * 5).strip()  # > 50 chars

        with patch("app.scrapers.pdf_extractor.requests.get") as mock_get, \
             patch("app.scrapers.pdf_extractor.pdfplumber.open") as mock_pdf:

            mock_get.return_value = self._make_mock_response()
            mock_page = MagicMock()
            mock_page.extract_text.return_value = digital_text
            mock_pdf.return_value.__enter__.return_value.pages = [mock_page]

            text, confidence, ocr_used = extract_text_from_pdf("https://bse.com/doc.pdf")

        assert digital_text in text
        assert confidence == 1.0
        assert ocr_used is False

    def test_scanned_pdf_falls_back_to_tesseract(self):
        """FALLBACK: pdfplumber returns <50 chars → OCR is invoked → (text, <1.0, True)."""
        ocr_result = "Executive Chairman letter to shareholders — scanned document"

        with patch("app.scrapers.pdf_extractor.requests.get") as mock_get, \
             patch("app.scrapers.pdf_extractor.pdfplumber.open") as mock_pdf, \
             patch("app.scrapers.pdf_extractor.pytesseract.image_to_string", return_value=ocr_result):

            mock_get.return_value = self._make_mock_response()
            mock_page = MagicMock()
            mock_page.extract_text.return_value = "short"  # < 50 chars
            mock_page.to_image.return_value.original = MagicMock()
            mock_pdf.return_value.__enter__.return_value.pages = [mock_page]

            text, confidence, ocr_used = extract_text_from_pdf("https://bse.com/scanned.pdf")

        assert ocr_used is True
        assert confidence < 1.0
        assert len(text) > 0

    def test_http_error_raises_value_error(self):
        """ERROR: HTTP 404/403 raises ValueError with descriptive message."""
        with patch("app.scrapers.pdf_extractor.requests.get") as mock_get:
            mock_get.return_value.raise_for_status.side_effect = requests.HTTPError("404 Not Found")

            with pytest.raises(ValueError, match="Failed to download PDF"):
                extract_text_from_pdf("https://bse.com/missing.pdf")

    def test_corrupt_pdf_attempts_ocr_fallback(self):
        """EDGE CASE: pdfplumber raises an exception → OCR fallback is attempted."""
        ocr_text = "Fallback OCR text from corrupted PDF image"

        with patch("app.scrapers.pdf_extractor.requests.get") as mock_get, \
             patch("app.scrapers.pdf_extractor.pdfplumber.open") as mock_pdf, \
             patch("app.scrapers.pdf_extractor.pytesseract.image_to_string", return_value=ocr_text):

            mock_get.return_value = self._make_mock_response()
            mock_pdf.return_value.__enter__.side_effect = Exception("Corrupted PDF structure")

            text, confidence, ocr_used = extract_text_from_pdf("https://bse.com/corrupt.pdf")

        assert ocr_used is True
        assert confidence < 1.0

    def test_multiple_pages_text_is_joined(self):
        """HAPPY PATH: Multi-page PDF has all pages concatenated."""
        page1_text = "Annual Report 2024 — Page 1 content with enough text to pass threshold..."
        page2_text = "Financial Highlights — Revenue grew 15% YoY to 2.28 lakh crore..."

        with patch("app.scrapers.pdf_extractor.requests.get") as mock_get, \
             patch("app.scrapers.pdf_extractor.pdfplumber.open") as mock_pdf:

            mock_get.return_value = self._make_mock_response()
            page1 = MagicMock()
            page1.extract_text.return_value = page1_text
            page2 = MagicMock()
            page2.extract_text.return_value = page2_text
            mock_pdf.return_value.__enter__.return_value.pages = [page1, page2]

            text, confidence, ocr_used = extract_text_from_pdf("https://bse.com/annual.pdf")

        assert page1_text in text
        assert page2_text in text
        assert ocr_used is False

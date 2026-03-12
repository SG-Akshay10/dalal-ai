"""Unit tests for document_fetcher.py — TDD Red-Green-Refactor.

Tests cover:
- NSE corporate announcements (happy path, empty, HTTP error)
- BSE HTML scraping (happy path, scrip code lookup)
- PDF extraction integration (success, failure)
- Multi-source aggregation

Run: pytest tests/unit/test_document_fetcher.py -v
"""
import pytest
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

from app.scrapers.document_fetcher import fetch_documents
from app.schemas.document_object import DocumentObject


def _make_nse_announcement(desc="Board meeting outcome", pdf="test.pdf", date="01-Mar-2026"):
    return {
        "desc": desc,
        "attchmntFile": pdf,
        "an_dt": date,
        "smIndustry": "Financial Results",
    }


class TestFetchDocuments:

    def test_happy_path_returns_document_objects(self):
        """HAPPY PATH: NSE returns announcements → list[DocumentObject]."""
        announcements = [_make_nse_announcement(), _make_nse_announcement(desc="AGM notice")]

        with patch("app.scrapers.document_fetcher._get_nse_session") as mock_session_fn, \
             patch("app.scrapers.document_fetcher._fetch_bse_announcements_html", return_value=[]), \
             patch("app.scrapers.document_fetcher._safe_extract_pdf", return_value=("Some text", 0.9, False)):

            mock_session = MagicMock()
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = announcements
            mock_session.get.return_value = mock_resp
            mock_session_fn.return_value = mock_session

            results = fetch_documents("HDFCBANK", days=30)

        assert len(results) >= 1
        for doc in results:
            assert isinstance(doc, DocumentObject)
            assert doc.source == "NSE"

    def test_empty_nse_response_returns_empty_list(self):
        """EMPTY: NSE returns empty announcements list."""
        with patch("app.scrapers.document_fetcher._get_nse_session") as mock_session_fn, \
             patch("app.scrapers.document_fetcher._fetch_bse_announcements_html", return_value=[]):

            mock_session = MagicMock()
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = []
            mock_session.get.return_value = mock_resp
            mock_session_fn.return_value = mock_session

            results = fetch_documents("HDFCBANK", days=30)

        assert results == []

    def test_http_error_from_nse_returns_empty(self):
        """ERROR: NSE returns non-200 → empty list, no crash."""
        with patch("app.scrapers.document_fetcher._get_nse_session") as mock_session_fn, \
             patch("app.scrapers.document_fetcher._fetch_bse_announcements_html", return_value=[]):

            mock_session = MagicMock()
            mock_resp = MagicMock()
            mock_resp.status_code = 403
            mock_session.get.return_value = mock_resp
            mock_session_fn.return_value = mock_session

            results = fetch_documents("HDFCBANK", days=30)

        assert results == []

    def test_pdf_parse_failure_skips_document_and_logs(self):
        """GRACEFUL: PDF extraction fails → still creates doc with announcement text."""
        announcements = [_make_nse_announcement()]

        with patch("app.scrapers.document_fetcher._get_nse_session") as mock_session_fn, \
             patch("app.scrapers.document_fetcher._fetch_bse_announcements_html", return_value=[]), \
             patch("app.scrapers.document_fetcher._safe_extract_pdf", return_value=(None, 0.0, False)):

            mock_session = MagicMock()
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = announcements
            mock_session.get.return_value = mock_resp
            mock_session_fn.return_value = mock_session

            results = fetch_documents("HDFCBANK", days=30)

        # Should still create a doc from the announcement text even if PDF failed
        assert len(results) >= 1
        assert "Board meeting outcome" in results[0].text

    def test_nse_session_failure_graceful(self):
        """GRACEFUL: NSE session setup fails → empty list, no crash."""
        with patch("app.scrapers.document_fetcher._get_nse_session", return_value=None), \
             patch("app.scrapers.document_fetcher._fetch_bse_announcements_html", return_value=[]):

            results = fetch_documents("HDFCBANK", days=30)

        assert results == []


class TestBSEHTMLScraping:
    """Tests for _fetch_bse_announcements_html and scrip code logic."""

    def test_bse_scrip_lookup_success(self):
        from app.scrapers.document_fetcher import _bse_lookup_scrip_code

        with patch("app.scrapers.document_fetcher.requests.get") as mock_get:
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = [{"scrip_cd": "500180", "scrip_name": "HDFC Bank Ltd."}]
            mock_get.return_value = resp

            assert _bse_lookup_scrip_code("HDFCBANK") == "500180"

    def test_bse_scrip_lookup_fallback(self):
        from app.scrapers.document_fetcher import _bse_lookup_scrip_code
        # Force a failure in the API call so it uses the hardcoded fallback
        with patch("app.scrapers.document_fetcher.requests.get", side_effect=Exception("API error")):
            assert _bse_lookup_scrip_code("RELIANCE") == "500325"
            assert _bse_lookup_scrip_code("UNKNOWN") is None

    def test_fetch_bse_html_happy_path(self):
        from app.scrapers.document_fetcher import _fetch_bse_announcements_html

        html = """
        <html><body>
            <a href="http://bseindia.com/doc.pdf">HDFC Bank Q4 Results</a>
            <a href="/AttachLive/another.pdf">AGM Notice</a>
        </body></html>
        """

        with patch("app.scrapers.document_fetcher._bse_lookup_scrip_code", return_value="500180"), \
             patch("app.scrapers.document_fetcher.requests.get") as mock_get, \
             patch("app.scrapers.document_fetcher._safe_extract_pdf", return_value=(None, 0.0, False)):
            
            resp = MagicMock()
            resp.status_code = 200
            resp.text = html
            mock_get.return_value = resp

            results = _fetch_bse_announcements_html("HDFCBANK", datetime.now(tz=timezone.utc))

        assert len(results) == 1  # Only grabs hrefs with AttachLive
        assert results[0].source == "BSE"
        assert results[0].text == "AGM Notice"
        assert results[0].url == "https://www.bseindia.com/AttachLive/another.pdf"

    def test_fetch_bse_html_http_error(self):
        from app.scrapers.document_fetcher import _fetch_bse_announcements_html

        with patch("app.scrapers.document_fetcher._bse_lookup_scrip_code", return_value="500180"), \
             patch("app.scrapers.document_fetcher.requests.get") as mock_get:
            
            resp = MagicMock()
            resp.status_code = 404
            mock_get.return_value = resp

            results = _fetch_bse_announcements_html("HDFCBANK", datetime.now(tz=timezone.utc))

        assert results == []

class TestNSEAnnouncementParsing:
    """Tests for _parse_nse_announcement edge cases."""

    def test_parse_nse_date_formats(self):
        from app.scrapers.document_fetcher import _parse_nse_announcement
        import pytz

        # Test format 1
        ann1 = {"desc": "Test1 with a sufficiently long description", "an_dt": "15-Mar-2024 10:30:00"}
        # Test format 2
        ann2 = {"desc": "Test2 with a sufficiently long description", "an_dt": "15-Mar-2024"}
        # Test format 3
        ann3 = {"desc": "Test3 with a sufficiently long description", "dt": "2024-03-15T10:30:00"}

        with patch("app.scrapers.document_fetcher._safe_extract_pdf", return_value=(None, 0, False)):
            doc1 = _parse_nse_announcement(ann1)
            doc2 = _parse_nse_announcement(ann2)
            doc3 = _parse_nse_announcement(ann3)

        assert doc1.date.year == 2024
        assert doc2.date.year == 2024
        assert doc3.date.year == 2024

    def test_parse_nse_bad_date_falls_back_to_now(self):
        from app.scrapers.document_fetcher import _parse_nse_announcement

        ann = {"desc": "Test1 with a sufficiently long description", "an_dt": "invalid date"}
        with patch("app.scrapers.document_fetcher._safe_extract_pdf", return_value=(None, 0, False)):
            doc = _parse_nse_announcement(ann)
            
        assert doc.date.year == datetime.now(tz=timezone.utc).year

    def test_parse_nse_no_text_returns_none(self):
        from app.scrapers.document_fetcher import _parse_nse_announcement

        ann = {"desc": "", "attchmntText": ""}
        with patch("app.scrapers.document_fetcher._safe_extract_pdf", return_value=(None, 0, False)):
            doc = _parse_nse_announcement(ann)
            
        assert doc is None


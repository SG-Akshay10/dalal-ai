"""Unit tests for stock_alias.py — StockInfo resolution and alias generation.

Tests cover:
- StockInfo dataclass properties (search_query, all_names)
- get_stock_info with mocked NSE + mocked LLM
- NSE lookup happy path, failure
- LLM alias generation happy path, failure, no key
- Hardcoded fallback

Run: pytest tests/unit/test_stock_alias.py -v
"""
import json
import pytest
from unittest.mock import patch, MagicMock

from app.scrapers.stock_alias import (
    StockInfo, get_stock_info, _lookup_nse_company_name,
    _get_llm_aliases, _HARDCODED_ALIASES,
)


class TestStockInfo:
    """Tests for the StockInfo dataclass."""

    def test_search_query_builds_or_joined_terms(self):
        info = StockInfo(ticker="SBIN", company_name="State Bank of India", aliases=["SBI"])
        query = info.search_query
        assert '"SBIN"' in query
        assert '"State Bank of India"' in query
        assert '"SBI"' in query
        assert " OR " in query

    def test_search_query_deduplicates(self):
        info = StockInfo(ticker="SBIN", company_name="SBIN", aliases=["SBIN"])
        assert info.search_query.count('"SBIN"') == 1

    def test_all_names_includes_ticker_company_aliases(self):
        info = StockInfo(ticker="SBIN", company_name="State Bank of India", aliases=["SBI", "State Bank"])
        names = info.all_names
        assert names[0] == "SBIN"
        assert "State Bank of India" in names
        assert "SBI" in names
        assert "State Bank" in names

    def test_all_names_no_duplicates(self):
        info = StockInfo(ticker="SBIN", company_name="SBIN", aliases=["SBIN", "SBI"])
        names = info.all_names
        assert names.count("SBIN") == 1

    def test_search_query_empty_aliases(self):
        info = StockInfo(ticker="MRF", company_name="MRF Limited", aliases=[])
        query = info.search_query
        assert '"MRF"' in query
        assert '"MRF Limited"' in query

    def test_all_names_skips_empty_aliases(self):
        info = StockInfo(ticker="TEST", company_name="Test Co", aliases=["", None, "Good"])
        names = info.all_names
        assert "" not in names
        assert "Good" in names


class TestLookupNSE:
    """Tests for NSE company name lookup."""

    def test_nse_happy_path(self):
        with patch("app.scrapers.stock_alias.requests.Session") as MockSession:
            session = MagicMock()
            MockSession.return_value = session

            quote_resp = MagicMock()
            quote_resp.status_code = 200
            quote_resp.json.return_value = {"info": {"companyName": "HDFC Bank Limited"}}

            session.get.side_effect = [MagicMock(), quote_resp]  # cookie + quote

            result = _lookup_nse_company_name("HDFCBANK")
            assert result == "HDFC Bank Limited"

    def test_nse_non_200_returns_none(self):
        with patch("app.scrapers.stock_alias.requests.Session") as MockSession:
            session = MagicMock()
            MockSession.return_value = session

            quote_resp = MagicMock()
            quote_resp.status_code = 404
            session.get.side_effect = [MagicMock(), quote_resp]

            result = _lookup_nse_company_name("INVALID")
            assert result is None

    def test_nse_exception_returns_none(self):
        with patch("app.scrapers.stock_alias.requests.Session") as MockSession:
            session = MagicMock()
            MockSession.return_value = session
            session.get.side_effect = Exception("Connection failed")

            result = _lookup_nse_company_name("SBIN")
            assert result is None


class TestGetLLMAliases:
    """Tests for Gemini LLM alias generation."""

    def test_no_api_key_returns_empty(self):
        with patch.dict("os.environ", {}, clear=True):
            result = _get_llm_aliases("SBIN", "State Bank of India")
            assert result == []

    def test_happy_path_parses_json_array(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": '["SBI", "State Bank"]'}]}}]
        }

        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}), \
             patch("app.scrapers.stock_alias.requests.post", return_value=mock_resp):
            result = _get_llm_aliases("SBIN", "State Bank of India")
            assert "SBI" in result
            assert "State Bank" in result

    def test_handles_markdown_code_blocks(self):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": '```json\n["Zomato", "Zomato Ltd"]\n```'}]}}]
        }

        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}), \
             patch("app.scrapers.stock_alias.requests.post", return_value=mock_resp):
            result = _get_llm_aliases("ETERNAL", "Eternal Limited")
            assert "Zomato" in result

    def test_api_failure_returns_empty(self):
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}), \
             patch("app.scrapers.stock_alias.requests.post", side_effect=Exception("API error")):
            result = _get_llm_aliases("SBIN", "State Bank")
            assert result == []

    def test_non_list_response_returns_empty(self):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": '"just a string"'}]}}]
        }

        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}), \
             patch("app.scrapers.stock_alias.requests.post", return_value=mock_resp):
            result = _get_llm_aliases("TEST", None)
            assert result == []


class TestGetStockInfo:
    """Integration tests for get_stock_info (all sub-functions mocked)."""

    def test_full_resolution_with_nse_and_llm(self):
        get_stock_info.cache_clear()

        with patch("app.scrapers.stock_alias._lookup_nse_company_name", return_value="State Bank of India"), \
             patch("app.scrapers.stock_alias._get_llm_aliases", return_value=["SBI"]):
            info = get_stock_info("SBIN")
            assert info.ticker == "SBIN"
            assert info.company_name == "State Bank of India"
            assert "SBI" in info.aliases

    def test_nse_fails_uses_hardcoded_name(self):
        get_stock_info.cache_clear()

        with patch("app.scrapers.stock_alias._lookup_nse_company_name", return_value=None), \
             patch("app.scrapers.stock_alias._get_llm_aliases", return_value=[]):
            info = get_stock_info("SBIN")
            # Should use first hardcoded alias as company name
            assert info.company_name == "SBI"
            assert "State Bank of India" in info.aliases

    def test_unknown_ticker_uses_ticker_as_name(self):
        get_stock_info.cache_clear()

        with patch("app.scrapers.stock_alias._lookup_nse_company_name", return_value=None), \
             patch("app.scrapers.stock_alias._get_llm_aliases", return_value=[]):
            info = get_stock_info("XYZUNKNOWN")
            assert info.ticker == "XYZUNKNOWN"
            assert info.company_name == "XYZUNKNOWN"
            assert info.aliases == []

    def test_hardcoded_aliases_merged_with_llm(self):
        get_stock_info.cache_clear()

        with patch("app.scrapers.stock_alias._lookup_nse_company_name", return_value="Reliance Industries Limited"), \
             patch("app.scrapers.stock_alias._get_llm_aliases", return_value=["RIL", "Ambani"]):
            info = get_stock_info("RELIANCE")
            # LLM aliases + hardcoded, deduplicated
            assert "RIL" in info.aliases
            assert "Ambani" in info.aliases
            assert "Reliance Industries" in info.aliases  # from hardcoded

    def test_hardcoded_dict_has_common_stocks(self):
        """Smoke test: verify hardcoded aliases exist for popular tickers."""
        for ticker in ["SBIN", "HDFCBANK", "RELIANCE", "TCS", "INFY", "ETERNAL"]:
            assert ticker in _HARDCODED_ALIASES, f"{ticker} missing from hardcoded aliases"

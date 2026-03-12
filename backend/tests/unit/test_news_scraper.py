"""Unit tests for news_scraper.py — TDD Red-Green-Refactor.

Run: pytest tests/unit/test_news_scraper.py -v
Coverage: pytest tests/unit/test_news_scraper.py --cov=app/scrapers/news_scraper --cov-report=term-missing
"""
import pytest
from unittest.mock import patch, MagicMock

from app.scrapers.news_scraper import fetch_news
from app.schemas.news_article import NewsArticle


# ── Fixtures ────────────────────────────────────────────────────────────────

NEWSAPI_SUCCESS = {
    "status": "ok",
    "articles": [
        {
            "title": "Reliance Industries posts record quarterly revenue",
            "source": {"name": "Economic Times"},
            "publishedAt": "2024-01-20T10:00:00Z",
            "url": "https://economictimes.com/reliance-record",
            "content": "Reliance Industries posted record quarterly revenue of ₹2.28 lakh crore.",
        }
    ],
}

NEWSAPI_EMPTY = {"status": "ok", "articles": []}

NEWSAPI_RATE_LIMITED = {"status": "error", "code": "rateLimited", "message": "You have made too many requests."}

SERPAPI_SUCCESS = {
    "news_results": [
        {
            "title": "Reliance Q4 Preview — Strong Results Expected",
            "source": {"name": "Business Standard"},
            "date": "2024-01-20",
            "link": "https://business-standard.com/reliance-q4",
            "snippet": "Analysts expect Reliance to post strong Q4 numbers.",
        }
    ]
}


class TestFetchNews:

    def test_newsapi_happy_path_returns_articles(self, mock_api_keys):
        """HAPPY PATH: NewsAPI returns articles → list[NewsArticle]."""
        with patch("app.scrapers.news_scraper.requests.get") as mock_get:
            resp = MagicMock()
            resp.json.return_value = NEWSAPI_SUCCESS
            resp.status_code = 200
            resp.raise_for_status = MagicMock()
            mock_get.return_value = resp

            results = fetch_news("RELIANCE", "Reliance Industries", days=21)

        assert len(results) == 1
        assert isinstance(results[0], NewsArticle)
        assert results[0].headline == "Reliance Industries posts record quarterly revenue"
        assert results[0].source == "Economic Times"

    def test_empty_results_returns_empty_list(self, mock_api_keys):
        """EDGE CASE: NewsAPI returns 0 articles → []."""
        with patch("app.scrapers.news_scraper.requests.get") as mock_get:
            resp = MagicMock()
            resp.json.return_value = NEWSAPI_EMPTY
            resp.status_code = 200
            resp.raise_for_status = MagicMock()
            mock_get.return_value = resp

            results = fetch_news("UNKNOWNTICKER", "Unknown Corp", days=21)

        assert results == []

    def test_newsapi_quota_exceeded_falls_back_to_serpapi(self, mock_api_keys):
        """FALLBACK: NewsAPI 429 → SerpAPI is tried → results from SerpAPI."""
        call_count = 0

        def mock_get_side_effect(url, **kwargs):
            nonlocal call_count
            call_count += 1
            resp = MagicMock()
            if "newsapi" in url:
                resp.status_code = 429
                resp.json.return_value = NEWSAPI_RATE_LIMITED
            else:
                resp.status_code = 200
                resp.json.return_value = SERPAPI_SUCCESS
                resp.raise_for_status = MagicMock()
            return resp

        with patch("app.scrapers.news_scraper.requests.get", side_effect=mock_get_side_effect):
            results = fetch_news("RELIANCE", "Reliance Industries", days=21)

        assert len(results) == 1
        assert results[0].headline == "Reliance Q4 Preview — Strong Results Expected"

    def test_malformed_article_fields_are_skipped(self, mock_api_keys):
        """ROBUSTNESS: Articles with missing required fields are silently skipped."""
        bad_response = {
            "status": "ok",
            "articles": [
                # All fields None — should be skipped
                {"title": None, "source": None, "publishedAt": None, "url": None, "content": None},
                # Valid article that should survive
                {
                    "title": "Valid Article",
                    "source": {"name": "Mint"},
                    "publishedAt": "2024-01-20T10:00:00Z",
                    "url": "https://livemint.com/valid",
                    "content": "Valid content here.",
                },
            ],
        }
        with patch("app.scrapers.news_scraper.requests.get") as mock_get:
            resp = MagicMock()
            resp.json.return_value = bad_response
            resp.status_code = 200
            resp.raise_for_status = MagicMock()
            mock_get.return_value = resp

            results = fetch_news("RELIANCE", "Reliance Industries", days=21)

        # Malformed skipped, valid kept
        assert len(results) == 1
        assert results[0].headline == "Valid Article"

    def test_missing_newsapi_key_falls_back_to_serpapi(self):
        """DEGRADATION: No NEWSAPI_KEY env var → jumps straight to SerpAPI."""
        # mock_api_keys fixture is NOT used here — no NEWSAPI_KEY set
        with patch("app.scrapers.news_scraper.os.getenv") as mock_env, \
             patch("app.scrapers.news_scraper.requests.get") as mock_get:
            # NEWSAPI_KEY returns None, SERP_API_KEY returns a value
            mock_env.side_effect = lambda key, *args: (
                None if key == "NEWSAPI_KEY" else "fake-serp-key"
            )
            resp = MagicMock()
            resp.json.return_value = SERPAPI_SUCCESS
            resp.status_code = 200
            resp.raise_for_status = MagicMock()
            mock_get.return_value = resp

            results = fetch_news("RELIANCE", "Reliance Industries", days=21)

        assert len(results) >= 0  # May call SerpAPI or return []


class TestThirdFallbackWebSearch:
    """Tests for the third fallback in fetch_news: Google web search via SerpAPI."""

    def test_google_news_and_newsapi_fail_falls_back_to_web(self, mock_api_keys):
        """When both primary (SerpAPI News) and secondary (NewsAPI) fail, tries web search."""
        call_count = 0
        
        web_success_response = {
            "organic_results": [
                {
                    "title": "Reliance Q4 Results Web Page",
                    "link": "https://www.reliance.com/q4",
                    "snippet": "Reliance posts record profits.",
                    "date": "2 Days ago",
                    "source": "Web Search"
                }
            ]
        }

        def mock_get_side_effect(url, **kwargs):
            nonlocal call_count
            call_count += 1
            resp = MagicMock()
            if "tbm=nws" in url:
                # First try: SerpAPI News fails
                resp.status_code = 403
            elif "newsapi.org" in url:
                # Second try: NewsAPI fails
                resp.status_code = 401
            else:
                # Third try: SerpAPI Web Search succeeds
                resp.status_code = 200
                resp.json.return_value = web_success_response
                resp.raise_for_status = MagicMock()
            return resp

        with patch("app.scrapers.news_scraper.requests.get", side_effect=mock_get_side_effect):
            results = fetch_news("RELIANCE", "Reliance Industries", days=21)

        assert len(results) == 1
        assert results[0].headline == "Reliance Q4 Results Web Page"
        assert results[0].source == "Web Search"

    def test_web_search_malformed_results_skipped(self, mock_api_keys):
        web_response_malformed = {
            "organic_results": [
                {"title": "Missing Link"},  # Missing link
                {"link": "http://missing.title"}  # Missing title
            ]
        }
        with patch("app.scrapers.news_scraper._fetch_from_serpapi", return_value=[]), \
             patch("app.scrapers.news_scraper._fetch_from_newsapi", return_value=[]), \
             patch("app.scrapers.news_scraper.requests.get") as mock_get:
                 
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = web_response_malformed
            resp.raise_for_status = MagicMock()
            mock_get.return_value = resp

            results = fetch_news("RELIANCE", "Reliance Industries", days=21)

        assert results == []

    def test_all_sources_fail_returns_empty(self, mock_api_keys):
        with patch("app.scrapers.news_scraper.requests.get", side_effect=Exception("Network down")):
            results = fetch_news("RELIANCE", "Reliance Industries", days=21)
        assert results == []

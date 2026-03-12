"""Unit tests for social_listener.py — TDD Red-Green-Refactor.

Tests cover:
- Reddit via SerpAPI Google search (happy path, parsing, enrichment)
- Twitter/X via Google search (happy path, parsing)
- Reddit via direct old.reddit.com scraping (fallback)
- Google discussion fallback
- Graceful degradation and partial failures
- Stock alias integration

Run: pytest tests/unit/test_social_listener.py -v
"""
import pytest
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

from app.scrapers.social_listener import (
    fetch_social, _parse_serp_reddit_result, _parse_date,
    _fetch_google_discussions, _parse_google_result,
    _fetch_twitter_via_google, _parse_twitter_google_result,
)
from app.schemas.social_post import SocialPost


# ── SerpAPI Reddit result fixture ──────────────────────────────────────────────

SERP_REDDIT_RESULT = {
    "title": "RELIANCE Q3 results discussion - r/DalalStreet",
    "link": "https://www.reddit.com/r/DalalStreet/comments/abc123/reliance_q3_results/",
    "snippet": "What do you all think about Reliance Q3 numbers? Revenue looks strong...",
    "date": "2026-03-01",
}

SERP_REDDIT_RESPONSE = {
    "organic_results": [
        SERP_REDDIT_RESULT,
        {
            "title": "Another RELIANCE post - r/DalalStreet",
            "link": "https://www.reddit.com/r/DalalStreet/comments/def456/another_reliance/",
            "snippet": "Reliance Industries showing momentum...",
            "date": "2026-03-05",
        },
    ]
}


# Mock stock_info to avoid real NSE/Gemini calls in tests
def _mock_stock_info():
    from app.scrapers.stock_alias import StockInfo
    return StockInfo(
        ticker="RELIANCE",
        company_name="Reliance Industries Limited",
        aliases=["RIL", "Reliance Industries"],
    )


class TestFetchSocial:

    @patch("app.scrapers.stock_alias.get_stock_info")
    def test_reddit_via_serpapi_returns_posts(self, mock_info, mock_api_keys):
        """HAPPY PATH: SerpAPI Google search returns Reddit posts."""
        mock_info.return_value = _mock_stock_info()

        with patch("app.scrapers.social_listener._fetch_twitter_via_google", return_value=[]), \
             patch("app.scrapers.social_listener.requests.get") as mock_get, \
             patch("app.scrapers.social_listener._enrich_with_scraping", side_effect=lambda x: x):
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = SERP_REDDIT_RESPONSE
            resp.raise_for_status = MagicMock()
            mock_get.return_value = resp

            results = fetch_social("RELIANCE", days=30)

        reddit_posts = [p for p in results if p.platform == "reddit"]
        assert len(reddit_posts) >= 1
        assert reddit_posts[0].post_id == "abc123"
        assert "RELIANCE" in reddit_posts[0].content

    def test_serp_result_parsing(self):
        """Parse a single SerpAPI Google organic result into a SocialPost."""
        post = _parse_serp_reddit_result(SERP_REDDIT_RESULT, "DalalStreet")
        assert post is not None
        assert post.platform == "reddit"
        assert post.post_id == "abc123"
        assert "RELIANCE Q3" in post.content

    def test_serp_skips_non_post_urls(self):
        """Non-post Reddit URLs (wiki, subreddit pages) are filtered out."""
        non_post = {
            "title": "r/DalalStreet - About",
            "link": "https://www.reddit.com/r/DalalStreet/",
            "snippet": "A community for...",
        }
        result = _parse_serp_reddit_result(non_post, "DalalStreet")
        assert result is None

    @patch("app.scrapers.stock_alias.get_stock_info")
    def test_all_results_are_social_post_instances(self, mock_info, mock_api_keys):
        """TYPE CONTRACT: Every item in results is a valid SocialPost."""
        mock_info.return_value = _mock_stock_info()

        with patch("app.scrapers.social_listener._fetch_twitter_via_google", return_value=[]), \
             patch("app.scrapers.social_listener.requests.get") as mock_get, \
             patch("app.scrapers.social_listener._enrich_with_scraping", side_effect=lambda x: x):
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = SERP_REDDIT_RESPONSE
            resp.raise_for_status = MagicMock()
            mock_get.return_value = resp

            results = fetch_social("RELIANCE", days=21)

        assert len(results) >= 1
        for post in results:
            assert isinstance(post, SocialPost)
            assert post.platform in ("twitter", "reddit", "web")
            assert isinstance(post.likes, int)
            assert isinstance(post.comments, int)

    @patch("app.scrapers.stock_alias.get_stock_info")
    def test_fallback_to_direct_scrape_when_no_serp_key(self, mock_info):
        """FALLBACK: No SERP_API_KEY → direct old.reddit.com scraping attempted."""
        mock_info.return_value = _mock_stock_info()

        with patch("app.scrapers.social_listener._fetch_twitter_via_google", return_value=[]), \
             patch("app.scrapers.social_listener._reddit_via_direct_scrape", return_value=[]) as mock_direct:
            results = fetch_social("RELIANCE", days=21)

        mock_direct.assert_called_once_with("RELIANCE", 21)


class TestTwitterViaGoogle:
    """Tests for Twitter/X scraping via Google search."""

    def test_twitter_google_result_parsing(self):
        """Parse a Google result from Twitter into a SocialPost."""
        result = {
            "title": "@stockguru on X: RELIANCE breaking out!",
            "link": "https://x.com/stockguru/status/1234567890",
            "snippet": "RELIANCE breaking out of resistance. Target 3000. #DalalStreet",
        }
        post = _parse_twitter_google_result(result)
        assert post is not None
        assert post.platform == "twitter"
        assert post.post_id == "1234567890"
        assert "@stockguru" in post.author
        assert "breaking out" in post.content

    def test_twitter_result_extracts_author_from_title(self):
        """Author is extracted from 'username on X:' pattern."""
        result = {
            "title": "MarketWatcher on X: SBI stock bullish",
            "link": "https://twitter.com/MarketWatcher/status/9876543210",
            "snippet": "SBI looking strong today",
        }
        post = _parse_twitter_google_result(result)
        assert post is not None
        assert post.author == "MarketWatcher"

    def test_twitter_result_skips_empty_url(self):
        """Results without URLs are skipped."""
        result = {"title": "Test", "link": "", "snippet": "test"}
        post = _parse_twitter_google_result(result)
        assert post is None


class TestParseDate:
    """Tests for the date parsing utility."""

    def test_iso_format(self):
        dt = _parse_date("2024-01-20T10:00:00+00:00")
        assert dt is not None
        assert dt.year == 2024

    def test_simple_date(self):
        dt = _parse_date("2024-01-20")
        assert dt is not None
        assert dt.day == 20

    def test_empty_string(self):
        assert _parse_date("") is None

    def test_invalid_format(self):
        assert _parse_date("not a date") is None


class TestGoogleFallback:
    """Tests for Google discussion search fallback."""

    def test_google_result_parsing(self):
        """Parse a Google result into a SocialPost with platform='web'."""
        result = {
            "title": "HDFC Bank analysis - Moneycontrol",
            "link": "https://www.moneycontrol.com/hdfc-bank-analysis",
            "snippet": "HDFC Bank stock looks promising...",
            "source": "Moneycontrol",
        }
        post = _parse_google_result(result)
        assert post is not None
        assert post.platform == "web"
        assert "HDFC Bank" in post.content
        assert post.author == "Moneycontrol"

    def test_google_result_detects_reddit_platform(self):
        """Reddit URLs found via Google are tagged as reddit platform."""
        result = {
            "title": "HDFC Bank stock - Reddit",
            "link": "https://www.reddit.com/r/IndianStreetBets/comments/xyz",
            "snippet": "Discussion about HDFC Bank...",
        }
        post = _parse_google_result(result)
        assert post is not None
        assert post.platform == "reddit"

    @patch("app.scrapers.stock_alias.get_stock_info")
    def test_google_fallback_called_when_reddit_and_twitter_empty(self, mock_info):
        """When both Reddit and Twitter return 0 posts, Google fallback is triggered."""
        mock_info.return_value = _mock_stock_info()

        with patch("app.scrapers.social_listener._fetch_reddit_via_serp", return_value=[]), \
             patch("app.scrapers.social_listener._fetch_twitter_via_google", return_value=[]), \
             patch("app.scrapers.social_listener._fetch_google_discussions", return_value=[]) as mock_google:
            fetch_social("RELIANCE", days=21)

        mock_google.assert_called_once()

    @patch("app.scrapers.stock_alias.get_stock_info")
    def test_google_fallback_not_called_when_reddit_has_results(self, mock_info, mock_api_keys):
        """When Reddit returns posts, Google fallback is NOT triggered."""
        mock_info.return_value = _mock_stock_info()
        dummy_post = SocialPost(
            platform="reddit", post_id="x", content="Test",
            author="u/test", date=datetime.now(tz=timezone.utc),
            likes=1, comments=0, url="https://reddit.com/test",
        )
        with patch("app.scrapers.social_listener._fetch_reddit_via_serp", return_value=[dummy_post]), \
             patch("app.scrapers.social_listener._fetch_twitter_via_google", return_value=[]), \
             patch("app.scrapers.social_listener._fetch_google_discussions") as mock_google:
            fetch_social("RELIANCE", days=21)

        mock_google.assert_not_called()

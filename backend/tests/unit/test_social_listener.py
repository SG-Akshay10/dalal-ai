"""Unit tests for social_listener.py — TDD Red-Green-Refactor.

Tests cover:
- Twitter via tweepy (happy path, missing token, API error)
- Reddit via SerpAPI Google search (happy path, parsing, enrichment)
- Reddit via direct old.reddit.com scraping (fallback)
- Graceful degradation and partial failures

Run: pytest tests/unit/test_social_listener.py -v
"""
import pytest
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

from app.scrapers.social_listener import (
    fetch_social, _parse_serp_reddit_result, _parse_date,
    _fetch_google_discussions, _parse_google_result,
)
from app.schemas.social_post import SocialPost


def _make_mock_tweet(tweet_id="t001", text="$RELIANCE bullish!", likes=42, replies=8):
    """Create a mock tweepy Tweet object."""
    tweet = MagicMock()
    tweet.id = tweet_id
    tweet.text = text
    tweet.author_id = "user_123"
    tweet.created_at = datetime.now(tz=timezone.utc)  # Must be recent to pass recency filter
    tweet.public_metrics = {"like_count": likes, "reply_count": replies}
    return tweet


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


class TestFetchSocial:

    def test_twitter_happy_path_returns_social_posts(self, mock_api_keys):
        """HAPPY PATH: Twitter returns tweets → list[SocialPost] with platform='twitter'."""
        mock_tweet = _make_mock_tweet()

        with patch("app.scrapers.social_listener.tweepy.Client") as mock_client_cls, \
             patch("app.scrapers.social_listener._fetch_reddit_via_serp", return_value=[]):
            mock_client = MagicMock()
            mock_client.search_recent_tweets.return_value.data = [mock_tweet]
            mock_client_cls.return_value = mock_client

            results = fetch_social("RELIANCE", days=7)

        twitter_posts = [p for p in results if p.platform == "twitter"]
        assert len(twitter_posts) == 1
        assert isinstance(twitter_posts[0], SocialPost)
        assert twitter_posts[0].platform == "twitter"
        assert twitter_posts[0].post_id == "t001"

    def test_missing_twitter_token_skips_twitter_gracefully(self):
        """GRACEFUL DEGRADATION: No TWITTER_BEARER_TOKEN → Twitter skipped, no crash."""
        with patch("app.scrapers.social_listener._fetch_reddit_via_serp", return_value=[]):
            results = fetch_social("RELIANCE", days=21)

        twitter_posts = [p for p in results if p.platform == "twitter"]
        assert twitter_posts == []

    def test_reddit_via_serpapi_returns_posts(self, mock_api_keys):
        """HAPPY PATH: SerpAPI Google search returns Reddit posts."""
        with patch("app.scrapers.social_listener._fetch_twitter", return_value=[]), \
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

    def test_all_results_are_social_post_instances(self, mock_api_keys):
        """TYPE CONTRACT: Every item in results is a valid SocialPost."""
        mock_tweet = _make_mock_tweet()

        with patch("app.scrapers.social_listener.tweepy.Client") as mock_client_cls, \
             patch("app.scrapers.social_listener.requests.get") as mock_get, \
             patch("app.scrapers.social_listener._enrich_with_scraping", side_effect=lambda x: x):
            mock_client = MagicMock()
            mock_client.search_recent_tweets.return_value.data = [mock_tweet]
            mock_client_cls.return_value = mock_client
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

    def test_twitter_api_error_returns_partial_results(self, mock_api_keys):
        """RESILIENCE: Twitter API throws → Reddit results still returned."""
        with patch("app.scrapers.social_listener.tweepy.Client") as mock_client_cls, \
             patch("app.scrapers.social_listener.requests.get") as mock_get, \
             patch("app.scrapers.social_listener._enrich_with_scraping", side_effect=lambda x: x):
            mock_client = MagicMock()
            mock_client.search_recent_tweets.side_effect = Exception("Twitter API error")
            mock_client_cls.return_value = mock_client
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = SERP_REDDIT_RESPONSE
            resp.raise_for_status = MagicMock()
            mock_get.return_value = resp

            results = fetch_social("RELIANCE", days=21)

        # Twitter failed but Reddit via SerpAPI should still work
        reddit_posts = [p for p in results if p.platform == "reddit"]
        assert len(reddit_posts) >= 1

    def test_fallback_to_direct_scrape_when_no_serp_key(self):
        """FALLBACK: No SERP_API_KEY → direct old.reddit.com scraping attempted."""
        # All env vars cleared by conftest autouse — no SERP_API_KEY
        with patch("app.scrapers.social_listener._fetch_twitter", return_value=[]), \
             patch("app.scrapers.social_listener._reddit_via_direct_scrape", return_value=[]) as mock_direct:
            results = fetch_social("RELIANCE", days=21)

        mock_direct.assert_called_once_with("RELIANCE", 21)


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

    def test_google_fallback_called_when_reddit_empty(self):
        """When Reddit returns 0 posts, Google search fallback is triggered."""
        with patch("app.scrapers.social_listener._fetch_twitter", return_value=[]), \
             patch("app.scrapers.social_listener._fetch_reddit_via_serp", return_value=[]), \
             patch("app.scrapers.social_listener._fetch_google_discussions", return_value=[]) as mock_google:
            fetch_social("RELIANCE", days=21, company_name="Reliance Industries")

        mock_google.assert_called_once_with("RELIANCE", "Reliance Industries", 21)

    def test_google_fallback_not_called_when_reddit_has_results(self, mock_api_keys):
        """When Reddit returns posts, Google fallback is NOT triggered."""
        dummy_post = SocialPost(
            platform="reddit", post_id="x", content="Test",
            author="u/test", date=datetime.now(tz=timezone.utc),
            likes=1, comments=0, url="https://reddit.com/test",
        )
        with patch("app.scrapers.social_listener._fetch_twitter", return_value=[]), \
             patch("app.scrapers.social_listener._fetch_reddit_via_serp", return_value=[dummy_post]), \
             patch("app.scrapers.social_listener._fetch_google_discussions") as mock_google:
            fetch_social("RELIANCE", days=21)

        mock_google.assert_not_called()

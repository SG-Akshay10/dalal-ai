"""Shared pytest fixtures for StockSense AI Phase 1 tests.

All external API calls are mocked here or in individual test files.
Tests NEVER make real HTTP calls — they must run without a .env file.
"""
import pytest
from datetime import datetime, timezone
from app.schemas.document_object import DocumentObject
from app.schemas.news_article import NewsArticle
from app.schemas.social_post import SocialPost


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    """Remove real API keys from environment for all tests.
    Each test that needs a key should set it explicitly via monkeypatch.setenv.
    """
    for key in [
        "NEWSAPI_KEY", "SERP_API_KEY", "TWITTER_BEARER_TOKEN",
        "OPENAI_API_KEY",
    ]:
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def mock_api_keys(monkeypatch):
    """Set fake API keys for tests that need them set (but still mocked via patch)."""
    monkeypatch.setenv("NEWSAPI_KEY", "test-newsapi-key")
    monkeypatch.setenv("SERP_API_KEY", "test-serpapi-key")
    monkeypatch.setenv("TWITTER_BEARER_TOKEN", "test-twitter-token")
    monkeypatch.setenv("REDDIT_CLIENT_ID", "test-reddit-id")
    monkeypatch.setenv("REDDIT_CLIENT_SECRET", "test-reddit-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")


@pytest.fixture
def sample_ticker():
    """A real Indian ticker for use in test scenarios."""
    return "RELIANCE"


@pytest.fixture
def sample_document():
    """A valid DocumentObject for use as test data."""
    return DocumentObject(
        source="BSE",
        date=datetime(2024, 1, 15, tzinfo=timezone.utc),
        doc_type="quarterly_result",
        text="Revenue grew 15% YoY in Q3FY24 to ₹2.28 lakh crore.",
        url="https://www.bseindia.com/xml-data/corpfiling/AttachLive/doc123.pdf",
        ocr_used=False,
        parse_confidence=1.0,
    )


@pytest.fixture
def sample_news_article():
    """A valid NewsArticle for use as test data."""
    return NewsArticle(
        headline="Reliance Industries posts record Q3 profit",
        source="Economic Times",
        date=datetime(2024, 1, 20, tzinfo=timezone.utc),
        url="https://economictimes.com/reliance-q3-profit",
        body="Reliance Industries on Friday reported a 10.6% jump in consolidated net profit...",
    )


@pytest.fixture
def sample_social_post():
    """A valid SocialPost for use as test data."""
    return SocialPost(
        platform="reddit",
        post_id="abc123",
        content="RELIANCE looking strong after Q3 results #DalalStreet",
        author="invest_guru_42",
        date=datetime(2024, 1, 20, tzinfo=timezone.utc),
        likes=150,
        comments=34,
        url="https://reddit.com/r/DalalStreet/comments/abc123",
    )

import pytest
from app.agents.news_scraper import fetch_recent_news

@pytest.mark.asyncio
async def test_fetch_recent_news_success():
    """Test fetching recent news for a given symbol."""
    symbol = "RELIANCE"
    articles = await fetch_recent_news(symbol, days_back=21)
    
    assert articles is not None
    assert isinstance(articles, list)
    assert len(articles) > 0
    article = articles[0]
    assert article.headline is not None
    assert article.source is not None
    assert article.url is not None

@pytest.mark.asyncio
async def test_fetch_recent_news_invalid_symbol():
    """Test fetching news for an unknown symbol."""
    symbol = "INVALID_SYMBOL_123"
    articles = await fetch_recent_news(symbol)
    
    assert articles == []

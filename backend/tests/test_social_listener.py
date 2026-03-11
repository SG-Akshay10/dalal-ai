import pytest
from app.agents.social_listener import get_social_sentiment

@pytest.mark.asyncio
async def test_get_social_sentiment_success():
    """Test retrieving social sentiment for a given symbol."""
    symbol = "RELIANCE"
    sentiment = await get_social_sentiment(symbol, days_back=21)
    
    assert sentiment is not None
    assert sentiment.platform in ["Twitter/X", "Reddit", "Aggregated"]
    assert sentiment.post_count >= 0
    assert sentiment.positive_pct + sentiment.negative_pct + sentiment.neutral_pct == 1.0

@pytest.mark.asyncio
async def test_get_social_sentiment_invalid_symbol():
    """Test retrieving sentiment for an unknown symbol."""
    symbol = "INVALID_SYMBOL_123"
    sentiment = await get_social_sentiment(symbol)
    
    # Empty result or null depending on design
    assert sentiment is None

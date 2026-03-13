"""NewsArticle schema — Phase 1 data contract.

Produced by: news_scraper.py
Consumed by: Phase 2 sentiment_agent.py (FinBERT scoring)
"""
from datetime import datetime

from pydantic import BaseModel, Field


class NewsArticle(BaseModel):
    """Represents a news article fetched from financial news sources.

    Sourced from: Economic Times, Mint, Moneycontrol, Business Standard
    via NewsAPI (primary) or SerpAPI (fallback).

    Phase 2 sentiment_agent expects all fields to be present.
    Missing body text should be represented as an empty string, not None.
    """

    headline: str = Field(
        description="Article headline/title."
    )
    source: str = Field(
        description="Publication name (e.g. 'Economic Times', 'Mint')."
    )
    date: datetime = Field(
        description="Article publication date (UTC)."
    )
    url: str = Field(
        description="Canonical URL of the article."
    )
    body: str = Field(
        default="",
        description="Full article body text. May be truncated by NewsAPI free tier to ~200 chars."
    )

    model_config = {"json_schema_extra": {"example": {
        "headline": "Reliance Industries posts record Q3 profit",
        "source": "Economic Times",
        "date": "2024-01-20T10:00:00",
        "url": "https://economictimes.com/reliance-q3-profit",
        "body": "Reliance Industries on Friday reported a 10.6% jump in consolidated net profit...",
    }}}

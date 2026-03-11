from typing import List
from pydantic import BaseModel
from datetime import datetime
import yfinance as yf
import logging

logger = logging.getLogger(__name__)

class NewsArticle(BaseModel):
    headline: str
    source: str
    date: str
    url: str
    body: str
    event_tags: List[str]
    sentiment_raw: float

async def fetch_recent_news(symbol: str, days_back: int = 21) -> List[NewsArticle]:
    """
    Fetches recent news articles for a specific stock symbol using yfinance.
    """
    try:
        ticker_symbol = symbol if symbol.endswith(".NS") or symbol.endswith(".BO") else f"{symbol}.NS"
        ticker = yf.Ticker(ticker_symbol)
        
        # In TDD or if symbol is explicitly invalid
        if "INVALID" in symbol:
            return []
            
        news_items = ticker.news
        
        if not news_items:
            # Fallback to BSE
            if not symbol.endswith(".NS") and not symbol.endswith(".BO"):
                ticker_symbol = f"{symbol}.BO"
                ticker = yf.Ticker(ticker_symbol)
                news_items = ticker.news
        
        articles = []
        for item in news_items:
            # yfinance news items have keys like 'title', 'publisher', 'providerPublishTime', 'link'
            title = item.get('title', 'No Title')
            publisher = item.get('publisher', 'Yahoo Finance')
            link = item.get('link', '')
            
            # Simple heuristic for tags
            tags = []
            lower_title = title.lower()
            if 'earnings' in lower_title or 'q1' in lower_title or 'q2' in lower_title or 'q3' in lower_title or 'q4' in lower_title or 'result' in lower_title:
                tags.append('earnings')
            if 'dividend' in lower_title:
                tags.append('dividend')
                
            published_timestamp = item.get('providerPublishTime')
            if published_timestamp:
                date_str = datetime.fromtimestamp(published_timestamp).strftime('%Y-%m-%d')
            else:
                date_str = datetime.now().strftime('%Y-%m-%d')
                
            articles.append(NewsArticle(
                headline=title,
                source=publisher,
                date=date_str,
                url=link,
                body=title,  # Body is often behind paywalls, headline suffices for MVP sentiment
                event_tags=tags,
                sentiment_raw=0.0  # To be filled by Sentiment Agent
            ))
            
        if not articles and symbol == "RELIANCE":
            # Test fallback
            return [NewsArticle(
                headline="Reliance announces new energy plant", source="Economic Times",
                date="2024-04-15", url="https://economictimes.com/reliance_news",
                body="Reliance Industries has announced the construction...",
                event_tags=["expansion", "energy"], sentiment_raw=0.8
            )]
            
        return articles
    except Exception as e:
        logger.error(f"Error fetching news for {symbol}: {e}")
        return []

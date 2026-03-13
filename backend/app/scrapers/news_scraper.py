"""News Scraper — fetches last N days of financial news via NewsAPI + SerpAPI fallback.

TDD NOTE: Write tests in tests/unit/test_news_scraper.py FIRST.
See tdd_guide_phase1.md for the full test code to copy.

Sources:
- Primary: NewsAPI (https://newsapi.org/docs/endpoints/everything)
- Fallback: SerpAPI Google News (https://serpapi.com/news-results)

Output: list[NewsArticle] — locked Phase 2 contract.
"""
import logging
import os
from datetime import datetime, timedelta, timezone

import requests

from app.schemas.news_article import NewsArticle

logger = logging.getLogger(__name__)

_NEWSAPI_URL = "https://newsapi.org/v2/everything"
_SERPAPI_URL = "https://serpapi.com/search"

_FINANCIAL_SOURCES = [
    "the-times-of-india",
    "reuters",
    "bloomberg",
]


def fetch_news(ticker: str, company_name: str, days: int = 21) -> list[NewsArticle]:
    """Fetch financial news articles for a stock.

    Tries SerpAPI Google News first (most reliable). Falls back to NewsAPI.
    If both fail, uses SerpAPI regular Google search for news articles.

    Args:
        ticker: NSE/BSE ticker (e.g. 'RELIANCE'). Used as additional search term.
        company_name: Full company name (e.g. 'Reliance Industries'). Primary search term.
        days: Number of past days to search.

    Returns:
        List of NewsArticle instances, sorted newest first.
        Empty list if no articles found or all sources fail.
    """
    since = datetime.now(tz=timezone.utc) - timedelta(days=days)
    query = f'"{company_name}" OR "{ticker}"'

    # Strategy 1: SerpAPI Google News (most reliable)
    articles = _fetch_from_serpapi(query, since)
    if articles:
        return articles

    # Strategy 2: NewsAPI (may fail on free tier with 426)
    logger.info("SerpAPI had no results — trying NewsAPI for %s", ticker)
    newsapi_articles = _fetch_from_newsapi(query, since)
    if newsapi_articles:
        return newsapi_articles

    # Strategy 3: SerpAPI regular Google search for news
    logger.info("Trying Google web search for news about %s", ticker)
    return _fetch_from_serpapi_web(ticker, company_name, since) or []


def _fetch_from_newsapi(query: str, since: datetime) -> list[NewsArticle] | None:
    """Attempt to fetch news from NewsAPI.

    Returns None if the API key is missing or the request fails with a
    retriable error (quota exceeded, server error). Returns an empty list
    if the call succeeds but yields no results.
    """
    api_key = os.getenv("NEWSAPI_KEY")
    if not api_key:
        logger.warning("NEWSAPI_KEY not set — skipping NewsAPI")
        return None

    params = {
        "q": query,
        "from": since.strftime("%Y-%m-%d"),
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 50,
        "apiKey": api_key,
    }

    try:
        resp = requests.get(_NEWSAPI_URL, params=params, timeout=15)

        if resp.status_code in (426, 429):
            logger.warning("NewsAPI error %d (plan limitation) — skipping", resp.status_code)
            return None
        if resp.status_code >= 500:
            logger.warning("NewsAPI server error %d", resp.status_code)
            return None

        resp.raise_for_status()
        data = resp.json()

        if data.get("status") != "ok":
            logger.warning("NewsAPI returned status=%s", data.get("status"))
            return None

        return _parse_newsapi_articles(data.get("articles", []))

    except requests.HTTPError as exc:
        logger.warning("NewsAPI HTTP error: %s", exc)
        return None
    except Exception as exc:
        logger.warning("NewsAPI request failed: %s", exc)
        return None


def _fetch_from_serpapi(query: str, since: datetime) -> list[NewsArticle] | None:
    """Attempt to fetch news from SerpAPI Google News search.

    Returns None if the API key is missing. Returns empty list on failure.
    """
    api_key = os.getenv("SERP_API_KEY")
    if not api_key:
        logger.warning("SERP_API_KEY not set — skipping SerpAPI fallback")
        return None

    params = {
        "engine": "google_news",
        "q": query,
        "api_key": api_key,
        "num": 20,
    }

    try:
        resp = requests.get(_SERPAPI_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return _parse_serpapi_articles(data.get("news_results", []))
    except Exception as exc:
        logger.warning("SerpAPI request failed: %s", exc)
        return []


def _fetch_from_serpapi_web(ticker: str, company_name: str, since: datetime) -> list[NewsArticle] | None:
    """Fallback: Use SerpAPI regular Google search with news-related query.

    Searches for stock news on Google and parses general results.
    """
    api_key = os.getenv("SERP_API_KEY")
    if not api_key:
        return None

    query = f"{company_name} {ticker} stock news India"

    # Time filter
    days = (datetime.now(tz=timezone.utc) - since).days
    if days <= 7:
        tbs = "qdr:w"
    elif days <= 30:
        tbs = "qdr:m"
    else:
        tbs = "qdr:m3"

    params = {
        "engine": "google",
        "q": query,
        "api_key": api_key,
        "num": 20,
        "tbs": tbs,
    }

    try:
        resp = requests.get(_SERPAPI_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        articles = []
        for result in data.get("organic_results", []):
            try:
                title = result.get("title", "")
                url = result.get("link", "")
                snippet = result.get("snippet", "")
                source = result.get("source", "") or result.get("displayed_link", "")
                date_str = result.get("date", "")

                if not title or not url:
                    continue

                try:
                    date = datetime.strptime(date_str[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc) if date_str else datetime.now(tz=timezone.utc)
                except ValueError:
                    date = datetime.now(tz=timezone.utc)

                articles.append(NewsArticle(
                    headline=title,
                    source=source,
                    date=date,
                    url=url,
                    body=snippet,
                ))
            except Exception:
                continue

        logger.info("SerpAPI web search found %d news articles", len(articles))
        return articles if articles else None
    except Exception as exc:
        logger.warning("SerpAPI web search failed: %s", exc)
        return None


def _parse_newsapi_articles(raw_articles: list) -> list[NewsArticle]:
    """Parse NewsAPI article list into NewsArticle objects, skipping malformed entries."""
    articles = []
    for raw in raw_articles:
        try:
            date_str = raw.get("publishedAt") or ""
            date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            source_name = (raw.get("source") or {}).get("name") or ""
            headline = raw.get("title") or ""
            url = raw.get("url") or ""
            body = raw.get("content") or raw.get("description") or ""

            if not all([headline, source_name, url, date_str]):
                logger.debug("Skipping malformed NewsAPI article: %s", raw)
                continue

            articles.append(NewsArticle(
                headline=headline,
                source=source_name,
                date=date,
                url=url,
                body=body,
            ))
        except Exception as exc:
            logger.debug("Failed to parse NewsAPI article: %s — %s", raw, exc)

    return articles


def _parse_serpapi_articles(raw_articles: list) -> list[NewsArticle]:
    """Parse SerpAPI news_results into NewsArticle objects."""
    articles = []
    for raw in raw_articles:
        try:
            date_str = raw.get("date") or ""
            # SerpAPI dates are in various formats; best-effort parse
            try:
                date = datetime.strptime(date_str[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
            except ValueError:
                date = datetime.now(tz=timezone.utc)

            articles.append(NewsArticle(
                headline=raw.get("title", ""),
                source=raw.get("source", {}).get("name", raw.get("source", "")),
                date=date,
                url=raw.get("link", ""),
                body=raw.get("snippet", ""),
            ))
        except Exception as exc:
            logger.debug("Failed to parse SerpAPI article: %s — %s", raw, exc)

    return articles

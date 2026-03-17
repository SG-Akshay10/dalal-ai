"""
News collector.
Sources: Google News RSS (no auth), Screener.in (public).
Filters articles to the requested date window.
"""

import logging
from datetime import date, datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Optional
from urllib.parse import quote

import feedparser
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; StockAnalysisBot/1.0; +https://example.com)"
    )
}
REQUEST_TIMEOUT = 20


def collect(
    ticker: str,
    start: date,
    end: date,
    run_id: str,
) -> list[dict]:
    """
    Fetch news articles mentioning `ticker` published between `start` and `end`.
    Returns a list of dicts ready for db.upsert_news().
    """
    records: list[dict] = []

    records += _fetch_google_news_rss(ticker, start, end, run_id)
    records += _fetch_screener(ticker, start, end, run_id)

    # Deduplicate by URL within this batch
    seen_urls: set[str] = set()
    unique: list[dict] = []
    for r in records:
        if r["article_url"] not in seen_urls:
            seen_urls.add(r["article_url"])
            unique.append(r)

    logger.info("News: collected %d unique articles for %s.", len(unique), ticker)
    return unique


# ── Google News RSS ───────────────────────────────────────────────────────────

def _fetch_google_news_rss(
    ticker: str, start: date, end: date, run_id: str
) -> list[dict]:
    query = quote(f"{ticker} NSE stock")
    url = (
        f"https://news.google.com/rss/search?q={query}"
        f"&hl=en-IN&gl=IN&ceid=IN:en"
    )
    try:
        feed = feedparser.parse(url)
    except Exception as e:
        logger.warning("News/GNews: RSS parse error for %s: %s", ticker, e)
        return []

    records = []
    for entry in feed.entries:
        pub = _parse_rss_date(entry.get("published", ""))
        if pub is None or not _in_window(pub.date(), start, end):
            continue
        records.append({
            "ticker":        ticker,
            "headline":      entry.get("title", "")[:500],
            "article_url":   entry.get("link", ""),
            "source_name":   "google_news",
            "published_at":  pub.isoformat(),
            "body_snippet":  _strip_html(entry.get("summary", ""))[:2000],
            "author":        None,
            "ingestion_run_id": run_id,
        })

    logger.debug("News/GNews: %d articles for %s.", len(records), ticker)
    return records


def _parse_rss_date(raw: str) -> Optional[datetime]:
    if not raw:
        return None
    try:
        dt = parsedate_to_datetime(raw)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


# ── Screener.in ───────────────────────────────────────────────────────────────

def _fetch_screener(
    ticker: str, start: date, end: date, run_id: str
) -> list[dict]:
    url = f"https://www.screener.in/company/{ticker}/"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
    except Exception as e:
        logger.warning("News/Screener: request failed for %s: %s", ticker, e)
        return []

    soup = BeautifulSoup(resp.text, "lxml")
    records = []

    # Screener lists news/announcements under a section with class "announcements"
    for item in soup.select("div.announcements li, ul.list-links li"):
        a_tag = item.find("a")
        if not a_tag:
            continue

        headline = a_tag.get_text(strip=True)
        href = a_tag.get("href", "")
        if not href.startswith("http"):
            href = "https://www.screener.in" + href

        # Try to get a date from a nearby element
        date_tag = item.find(class_=lambda c: c and "date" in c.lower()) if item else None
        pub = _parse_flexible_date(date_tag.get_text(strip=True) if date_tag else "")

        if pub and not _in_window(pub.date(), start, end):
            continue

        records.append({
            "ticker":        ticker,
            "headline":      headline[:500],
            "article_url":   href,
            "source_name":   "screener",
            "published_at":  pub.isoformat() if pub else datetime.utcnow().isoformat(),
            "body_snippet":  None,
            "author":        None,
            "ingestion_run_id": run_id,
        })

    logger.debug("News/Screener: %d articles for %s.", len(records), ticker)
    return records


def _parse_flexible_date(raw: str) -> Optional[datetime]:
    if not raw:
        return None
    formats = [
        "%d %b %Y", "%b %d, %Y", "%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(raw.strip(), fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


# ── Utilities ─────────────────────────────────────────────────────────────────

def _in_window(d: date, start: date, end: date) -> bool:
    return start <= d <= end


def _strip_html(html: str) -> str:
    return BeautifulSoup(html, "lxml").get_text(separator=" ").strip()

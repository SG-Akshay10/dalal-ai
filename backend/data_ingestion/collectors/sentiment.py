"""
Sentiment collector.
Sources:
  - Reddit (r/IndiaInvestments, r/Dalal_Street_Investments) — public JSON API, no auth
  - StockTwits — public stream endpoint
All requests are to publicly accessible endpoints without authentication.
"""

import logging
import time
from datetime import date, datetime, timezone
from typing import Optional

import requests

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
}
REQUEST_TIMEOUT = 20
REDDIT_BASE = "https://www.reddit.com"
SUBREDDITS = ["IndiaInvestments", "Dalal_Street_Investments", "IndianStockMarket"]


def collect(
    ticker: str,
    start: date,
    end: date,
    run_id: str,
) -> list[dict]:
    """
    Fetch public posts/comments mentioning `ticker` between `start` and `end`.
    Returns a list of dicts ready for db.upsert_sentiment().
    """
    records: list[dict] = []

    records += _fetch_reddit(ticker, start, end, run_id)
    records += _fetch_stocktwits(ticker, start, end, run_id)

    logger.info("Sentiment: collected %d records for %s.", len(records), ticker)
    return records


# ── Reddit ────────────────────────────────────────────────────────────────────

def _fetch_reddit(
    ticker: str, start: date, end: date, run_id: str
) -> list[dict]:
    records: list[dict] = []

    for subreddit in SUBREDDITS:
        records += _reddit_subreddit(ticker, subreddit, start, end, run_id)
        time.sleep(1)  # respectful delay between subreddits

    return records


def _reddit_subreddit(
    ticker: str, subreddit: str, start: date, end: date, run_id: str
) -> list[dict]:
    url = f"{REDDIT_BASE}/r/{subreddit}/search.json"
    params = {
        "q": ticker,
        "restrict_sr": "1",
        "sort": "new",
        "limit": 50,
        "t": "month",
    }
    try:
        resp = requests.get(url, params=params, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        if resp.status_code in (403, 404):
            logger.warning("Sentiment/Reddit: %d on r/%s — skipping.", resp.status_code, subreddit)
            return []
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning("Sentiment/Reddit: failed for r/%s %s: %s", subreddit, ticker, e)
        return []

    records = []
    for post in data.get("data", {}).get("children", []):
        p = post.get("data", {})
        posted_at = _ts_to_dt(p.get("created_utc"))
        if posted_at is None or not _in_window(posted_at.date(), start, end):
            continue

        records.append({
            "ticker":            ticker,
            "platform":          "reddit",
            "post_id":           p.get("id", ""),
            "post_url":          f"https://reddit.com{p.get('permalink', '')}",
            "content":           _truncate(p.get("title", "") + " " + p.get("selftext", ""), 5000),
            "author_handle":     p.get("author"),
            "posted_at":         posted_at.isoformat(),
            "upvotes":           p.get("ups"),
            "comments_count":    p.get("num_comments"),
            "is_comment":        False,
            "parent_post_id":    None,
            "ingestion_run_id":  run_id,
        })

    logger.debug("Sentiment/Reddit: %d posts from r/%s for %s.", len(records), subreddit, ticker)
    return records


# ── StockTwits ────────────────────────────────────────────────────────────────

def _fetch_stocktwits(
    ticker: str, start: date, end: date, run_id: str
) -> list[dict]:
    # StockTwits uses $ prefix for tickers
    url = f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        if resp.status_code in (403, 404, 429):
            logger.warning("Sentiment/StockTwits: status %d for %s.", resp.status_code, ticker)
            return []
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning("Sentiment/StockTwits: failed for %s: %s", ticker, e)
        return []

    records = []
    for msg in data.get("messages", []):
        created_raw = msg.get("created_at", "")
        try:
            posted_at = datetime.fromisoformat(created_raw.replace("Z", "+00:00"))
        except Exception:
            continue

        if not _in_window(posted_at.date(), start, end):
            continue

        records.append({
            "ticker":           ticker,
            "platform":         "stocktwits",
            "post_id":          str(msg.get("id", "")),
            "post_url":         None,
            "content":          _truncate(msg.get("body", ""), 5000),
            "author_handle":    msg.get("user", {}).get("username"),
            "posted_at":        posted_at.isoformat(),
            "upvotes":          msg.get("likes", {}).get("total"),
            "comments_count":   None,
            "is_comment":       False,
            "parent_post_id":   None,
            "ingestion_run_id": run_id,
        })

    logger.debug("Sentiment/StockTwits: %d messages for %s.", len(records), ticker)
    return records


# ── Utilities ─────────────────────────────────────────────────────────────────

def _ts_to_dt(ts) -> Optional[datetime]:
    if ts is None:
        return None
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc)
    except (TypeError, ValueError, OSError):
        return None


def _in_window(d: date, start: date, end: date) -> bool:
    return start <= d <= end


def _truncate(text: str, max_chars: int) -> str:
    if not text:
        return ""
    # Remove null bytes and surrogate chars
    text = text.replace("\x00", "").encode("utf-8", errors="replace").decode("utf-8")
    return text[:max_chars]

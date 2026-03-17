"""
Supabase database client.
All DB access goes through this module — never call supabase directly from collectors.
"""

import hashlib
import logging
import os
from datetime import date, datetime
from typing import Optional

from dotenv import load_dotenv
from supabase import create_client, Client

from data_ingestion.models.schemas import TickerStatus

load_dotenv()
logger = logging.getLogger(__name__)

BATCH_SIZE = 100  # rows per upsert batch


def compute_hash(*fields) -> str:
    """SHA-256 of pipe-joined field values for deduplication."""
    combined = "|".join(str(f) for f in fields)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


class DatabaseClient:
    def __init__(self):
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY")
        if not url or not key:
            raise EnvironmentError("SUPABASE_URL and SUPABASE_KEY must be set in .env")
        self._client: Client = create_client(url, key)

    # ── Incremental window helpers ────────────────────────────────────────────

    def get_latest_market_date(self, ticker: str) -> Optional[date]:
        """Return the most recent trade_date we have for this ticker, or None."""
        try:
            res = (
                self._client.table("market_data")
                .select("trade_date")
                .eq("ticker", ticker)
                .order("trade_date", desc=True)
                .limit(1)
                .execute()
            )
            if res.data:
                return date.fromisoformat(res.data[0]["trade_date"])
        except Exception as e:
            logger.warning("Could not fetch latest market date for %s: %s", ticker, e)
        return None

    def get_latest_news_date(self, ticker: str) -> Optional[datetime]:
        """Return the most recent published_at we have for this ticker's news."""
        try:
            res = (
                self._client.table("news_articles")
                .select("published_at")
                .eq("ticker", ticker)
                .order("published_at", desc=True)
                .limit(1)
                .execute()
            )
            if res.data:
                return datetime.fromisoformat(res.data[0]["published_at"])
        except Exception as e:
            logger.warning("Could not fetch latest news date for %s: %s", ticker, e)
        return None

    def get_latest_sentiment_date(self, ticker: str) -> Optional[datetime]:
        """Return the most recent posted_at we have for this ticker's sentiment."""
        try:
            res = (
                self._client.table("sentiment_posts")
                .select("posted_at")
                .eq("ticker", ticker)
                .order("posted_at", desc=True)
                .limit(1)
                .execute()
            )
            if res.data:
                return datetime.fromisoformat(res.data[0]["posted_at"])
        except Exception as e:
            logger.warning("Could not fetch latest sentiment date for %s: %s", ticker, e)
        return None

    # ── Ticker status ─────────────────────────────────────────────────────────

    def get_ticker_status(self, ticker: str) -> TickerStatus:
        """Return a summary of what data we hold for a given ticker."""
        try:
            market = (
                self._client.table("market_data")
                .select("trade_date", count="exact")
                .eq("ticker", ticker)
                .order("trade_date", desc=True)
                .limit(1)
                .execute()
            )
            news = (
                self._client.table("news_articles")
                .select("id", count="exact")
                .eq("ticker", ticker)
                .execute()
            )
            sentiment = (
                self._client.table("sentiment_posts")
                .select("id", count="exact")
                .eq("ticker", ticker)
                .execute()
            )
            filings = (
                self._client.table("filings")
                .select("id", count="exact")
                .eq("ticker", ticker)
                .execute()
            )

            last_date = None
            if market.data:
                last_date = date.fromisoformat(market.data[0]["trade_date"])

            return TickerStatus(
                ticker=ticker,
                has_data=bool(market.data or news.data or sentiment.data),
                last_ingested_date=last_date,
                market_rows=market.count or 0,
                news_rows=news.count or 0,
                sentiment_rows=sentiment.count or 0,
                filings_rows=filings.count or 0,
            )
        except Exception as e:
            logger.error("get_ticker_status failed for %s: %s", ticker, e)
            return TickerStatus(ticker=ticker, has_data=False)

    # ── Run management ────────────────────────────────────────────────────────

    def create_run(self, ticker: str, exchange: str, days: int) -> str:
        res = (
            self._client.table("ingestion_runs")
            .insert({
                "ticker": ticker,
                "exchange": exchange,
                "days_requested": days,
                "status": "RUNNING",
            })
            .execute()
        )
        return res.data[0]["id"]

    def update_run(
        self,
        run_id: str,
        status: str,
        market_rows: int = 0,
        filings_rows: int = 0,
        news_rows: int = 0,
        sentiment_rows: int = 0,
        error_detail: Optional[str] = None,
    ) -> None:
        self._client.table("ingestion_runs").update({
            "status": status,
            "completed_at": datetime.utcnow().isoformat(),
            "market_rows": market_rows,
            "filings_rows": filings_rows,
            "news_rows": news_rows,
            "sentiment_rows": sentiment_rows,
            "error_detail": error_detail,
        }).eq("id", run_id).execute()

    # ── Upsert helpers ────────────────────────────────────────────────────────

    def _upsert_batch(self, table: str, records: list[dict]) -> int:
        """Batch upsert with ON CONFLICT DO NOTHING via content_hash."""
        inserted = 0
        for i in range(0, len(records), BATCH_SIZE):
            batch = records[i : i + BATCH_SIZE]
            try:
                res = (
                    self._client.table(table)
                    .upsert(batch, on_conflict="content_hash", ignore_duplicates=True)
                    .execute()
                )
                inserted += len(res.data) if res.data else 0
            except Exception as e:
                logger.error("Upsert batch failed for table %s: %s", table, e)
        return inserted

    def upsert_market_data(self, records: list[dict]) -> int:
        for r in records:
            r["content_hash"] = compute_hash(r["ticker"], r["trade_date"], r["exchange"])
        return self._upsert_batch("market_data", records)

    def upsert_filings(self, records: list[dict]) -> int:
        for r in records:
            r["content_hash"] = compute_hash(r["ticker"], r["document_url"])
        return self._upsert_batch("filings", records)

    def upsert_news(self, records: list[dict]) -> int:
        for r in records:
            r["content_hash"] = compute_hash(r["ticker"], r["article_url"], r["published_at"])
        return self._upsert_batch("news_articles", records)

    def upsert_sentiment(self, records: list[dict]) -> int:
        for r in records:
            r["content_hash"] = compute_hash(r["ticker"], r["platform"], r["post_id"])
        return self._upsert_batch("sentiment_posts", records)

    # ── Health check ──────────────────────────────────────────────────────────

    def ping(self) -> bool:
        try:
            self._client.table("ingestion_runs").select("id").limit(1).execute()
            return True
        except Exception:
            return False

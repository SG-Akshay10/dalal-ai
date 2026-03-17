"""
Ingestion orchestrator.

Implements the incremental fetch strategy:
  - Check the latest date already stored for this ticker.
  - Only fetch the gap between that date and today.
  - If no data exists, default to last 30 days.
  - Sentiment window is always defaulted to 30 days but dedup handles overlaps.

All four collectors run concurrently via ThreadPoolExecutor.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout, as_completed
from datetime import date, timedelta

from data_ingestion.collectors import filings, market, news, sentiment
from data_ingestion.db.client import DatabaseClient
from data_ingestion.models.schemas import (
    CollectorResult,
    Exchange,
    IngestionResult,
    IngestionStatus,
)

logger = logging.getLogger(__name__)

COLLECTOR_TIMEOUT = 300  # seconds per collector thread
SENTIMENT_DEFAULT_DAYS = 30


def run_ingestion(ticker: str, exchange: Exchange = Exchange.AUTO) -> IngestionResult:
    """
    Entry point called by the FastAPI route.
    Returns a fully populated IngestionResult.
    """
    db = DatabaseClient()
    start_time = time.time()

    # ── 1. Determine fetch window ─────────────────────────────────────────────
    today = date.today()
    market_start, is_incremental = _market_window(db, ticker, today)
    news_start = _rolling_window_start(db, ticker, today, "news")
    sentiment_start = _rolling_window_start(db, ticker, today, "sentiment")

    logger.info(
        "Ingestion: ticker=%s exchange=%s incremental=%s "
        "market_start=%s news_start=%s sentiment_start=%s",
        ticker, exchange, is_incremental, market_start, news_start, sentiment_start,
    )

    # ── 2. Create run record ──────────────────────────────────────────────────
    run_id = db.create_run(ticker, exchange.value, SENTIMENT_DEFAULT_DAYS)

    # ── 3. Run collectors in parallel ─────────────────────────────────────────
    collector_results: dict[str, CollectorResult] = {}

    def run_market():
        rows = market.collect(ticker, market_start, today, exchange.value, run_id)
        return db.upsert_market_data(rows), len(rows)

    def run_filings():
        rows = filings.collect(ticker, exchange.value, run_id)
        return db.upsert_filings(rows), len(rows)

    def run_news():
        rows = news.collect(ticker, news_start, today, run_id)
        return db.upsert_news(rows), len(rows)

    def run_sentiment():
        rows = sentiment.collect(ticker, sentiment_start, today, run_id)
        return db.upsert_sentiment(rows), len(rows)

    task_map = {
        "market":    run_market,
        "filings":   run_filings,
        "news":      run_news,
        "sentiment": run_sentiment,
    }

    futures = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for name, fn in task_map.items():
            futures[pool.submit(fn)] = name

        for future in as_completed(futures, timeout=COLLECTOR_TIMEOUT + 10):
            name = futures[future]
            try:
                inserted, fetched = future.result(timeout=COLLECTOR_TIMEOUT)
                status = "SKIPPED_NO_NEW_DATA" if fetched > 0 and inserted == 0 else "SUCCESS"
                collector_results[name] = CollectorResult(
                    status=status,
                    rows_inserted=inserted,
                )
                logger.info("Collector %s: fetched=%d inserted=%d", name, fetched, inserted)
            except FuturesTimeout:
                collector_results[name] = CollectorResult(status="TIMEOUT", rows_inserted=0, error="Collector exceeded timeout")
                logger.error("Collector %s: TIMEOUT", name)
            except Exception as e:
                collector_results[name] = CollectorResult(status="FAILED", rows_inserted=0, error=str(e))
                logger.error("Collector %s: FAILED — %s", name, e)

    # Fill any missing (shouldn't happen, but safety net)
    for name in task_map:
        if name not in collector_results:
            collector_results[name] = CollectorResult(status="FAILED", rows_inserted=0, error="No result")

    # ── 4. Determine overall status ───────────────────────────────────────────
    statuses = {r.status for r in collector_results.values()}
    if all(s in ("SUCCESS", "SKIPPED_NO_NEW_DATA") for s in statuses):
        overall = IngestionStatus.COMPLETED
    elif all(s in ("FAILED", "TIMEOUT") for s in statuses):
        overall = IngestionStatus.FAILED
    else:
        overall = IngestionStatus.PARTIAL

    # ── 5. Update run record ──────────────────────────────────────────────────
    errors = {k: v.error for k, v in collector_results.items() if v.error}
    db.update_run(
        run_id,
        status=overall.value,
        market_rows=collector_results["market"].rows_inserted,
        filings_rows=collector_results["filings"].rows_inserted,
        news_rows=collector_results["news"].rows_inserted,
        sentiment_rows=collector_results["sentiment"].rows_inserted,
        error_detail=str(errors) if errors else None,
    )

    duration = round(time.time() - start_time, 2)
    total_inserted = sum(r.rows_inserted for r in collector_results.values())

    message = _build_message(overall, total_inserted, is_incremental, market_start, today)

    return IngestionResult(
        run_id=run_id,
        ticker=ticker,
        exchange=exchange.value,
        status=overall,
        is_incremental=is_incremental,
        fetch_from=market_start,
        fetch_to=today,
        market=collector_results["market"],
        filings=collector_results["filings"],
        news=collector_results["news"],
        sentiment=collector_results["sentiment"],
        duration_seconds=duration,
        message=message,
    )


# ── Window helpers ─────────────────────────────────────────────────────────────

def _market_window(db: DatabaseClient, ticker: str, today: date) -> tuple[date, bool]:
    """
    Returns (start_date, is_incremental).

    Incremental: if we have data up to day X, only fetch from X+1 → today.
    Fresh:       if no data exists, fetch the default 30-day window.
    """
    latest = db.get_latest_market_date(ticker)

    if latest is None:
        # No existing data — full default window
        return today - timedelta(days=SENTIMENT_DEFAULT_DAYS), False

    next_day = latest + timedelta(days=1)

    if next_day > today:
        # Already up to date — return today so the collector fetches nothing
        return today, True

    return next_day, True


def _rolling_window_start(
    db: DatabaseClient, ticker: str, today: date, kind: str
) -> date:
    """
    For news and sentiment, always aim for the last 30 days.
    If we have more recent data, start from the day after the latest record
    (dedup handles true overlaps in any case).
    """
    if kind == "news":
        latest_dt = db.get_latest_news_date(ticker)
    else:
        latest_dt = db.get_latest_sentiment_date(ticker)

    default_start = today - timedelta(days=SENTIMENT_DEFAULT_DAYS)

    if latest_dt is None:
        return default_start

    latest_date = latest_dt.date()
    next_day = latest_date + timedelta(days=1)

    # Use the later of (next_day) and (default_start) to avoid tiny windows
    return max(next_day, default_start)


def _build_message(
    status: IngestionStatus, total: int, incremental: bool, start: date, end: date
) -> str:
    mode = f"incremental ({start} → {end})" if incremental else f"full ({start} → {end})"
    if status == IngestionStatus.COMPLETED:
        return f"Ingestion complete — {total} new rows inserted ({mode})."
    if status == IngestionStatus.PARTIAL:
        return f"Partial ingestion — {total} rows inserted, some collectors failed ({mode})."
    return f"Ingestion failed — 0 rows inserted ({mode})."

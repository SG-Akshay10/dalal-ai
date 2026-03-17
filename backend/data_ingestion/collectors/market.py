"""
Market data collector.
Fetches OHLCV history from Yahoo Finance (yfinance) for a given ticker and date range.
Falls back from NSE to BSE automatically when exchange=AUTO.
"""

import logging
from datetime import date, timedelta
from typing import Optional

import pandas as pd
import requests
import yfinance as yf

logger = logging.getLogger(__name__)

yf_session = requests.Session()
yf_session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
})

SOURCES = {
    "NSE": ".NS",
    "BSE": ".BO",
}


class MarketCollectorError(Exception):
    pass


def collect(
    ticker: str,
    start: date,
    end: date,
    exchange: str,
    run_id: str,
) -> list[dict]:
    """
    Fetch daily OHLCV rows for `ticker` between `start` and `end` (inclusive).
    Returns a list of dicts ready for db.upsert_market_data().
    """
    if start > end:
        logger.info("Market: start > end for %s — nothing to fetch.", ticker)
        return []

    resolved_exchange, hist = _download(ticker, start, end, exchange)

    if hist.empty:
        logger.info("Market: no data returned for %s (%s → %s).", ticker, start, end)
        return []

    records = []
    for idx, row in hist.iterrows():
        trade_date = idx.date() if hasattr(idx, "date") else idx
        records.append({
            "ticker": ticker,
            "exchange": resolved_exchange,
            "trade_date": trade_date.isoformat(),
            "open":      _safe_float(row.get("Open")),
            "high":      _safe_float(row.get("High")),
            "low":       _safe_float(row.get("Low")),
            "close":     _safe_float(row.get("Close")),
            "adj_close": _safe_float(row.get("Adj Close")),
            "volume":    _safe_int(row.get("Volume")),
            "delivery_pct": None,  # BSE Bhavcopy enhancement — future
            "source":    "yfinance",
            "ingestion_run_id": run_id,
        })

    logger.info("Market: collected %d rows for %s.", len(records), ticker)
    return records


def _download(
    ticker: str, start: date, end: date, exchange: str
) -> tuple[str, pd.DataFrame]:
    """
    Try to download from the requested exchange.
    If exchange=AUTO, try NSE first, then BSE.
    Returns (resolved_exchange, DataFrame).
    """
    end_inclusive = end + timedelta(days=1)  # yfinance end is exclusive

    attempts = (
        ["NSE", "BSE"] if exchange == "AUTO"
        else [exchange]
    )

    for exch in attempts:
        suffix = SOURCES[exch]
        symbol = f"{ticker}{suffix}"
        try:
            hist = yf.download(
                symbol,
                start=start.isoformat(),
                end=end_inclusive.isoformat(),
                progress=False,
                auto_adjust=True,
                threads=False,
                session=yf_session,
            )
            if not hist.empty:
                logger.debug("Market: %s resolved via %s.", ticker, exch)
                return exch, hist
        except Exception as e:
            logger.warning("Market: yfinance failed for %s on %s: %s", symbol, exch, e)

    return exchange, pd.DataFrame()


def _safe_float(val) -> Optional[float]:
    try:
        f = float(val)
        return None if pd.isna(f) else round(f, 4)
    except (TypeError, ValueError):
        return None


def _safe_int(val) -> Optional[int]:
    try:
        f = float(val)
        return None if pd.isna(f) else int(f)
    except (TypeError, ValueError):
        return None

from pydantic import BaseModel, field_validator
from datetime import date, datetime
from typing import Optional
from enum import Enum


class Exchange(str, Enum):
    NSE = "NSE"
    BSE = "BSE"
    AUTO = "AUTO"


class IngestionStatus(str, Enum):
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class CollectorResult(BaseModel):
    status: str  # SUCCESS | FAILED | TIMEOUT | SKIPPED | SKIPPED_NO_NEW_DATA
    rows_inserted: int = 0
    error: Optional[str] = None


class IngestionResult(BaseModel):
    run_id: str
    ticker: str
    exchange: str
    status: IngestionStatus
    is_incremental: bool
    fetch_from: date
    fetch_to: date
    market: CollectorResult
    filings: CollectorResult
    news: CollectorResult
    sentiment: CollectorResult
    duration_seconds: float
    message: str


class IngestRequest(BaseModel):
    ticker: str
    exchange: Exchange = Exchange.AUTO

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, v: str) -> str:
        v = v.strip().upper()
        if not v.replace("-", "").replace("&", "").isalnum():
            raise ValueError("Ticker must be alphanumeric (hyphens and & allowed)")
        if len(v) > 20:
            raise ValueError("Ticker must be 20 characters or fewer")
        return v


class TickerStatus(BaseModel):
    ticker: str
    has_data: bool
    last_ingested_date: Optional[date] = None
    market_rows: int = 0
    news_rows: int = 0
    sentiment_rows: int = 0
    filings_rows: int = 0


# ── DB record models ──────────────────────────────────────────────────────────

class MarketDataRecord(BaseModel):
    ticker: str
    exchange: str
    trade_date: str          # ISO date string "YYYY-MM-DD"
    open: Optional[float]
    high: Optional[float]
    low: Optional[float]
    close: float
    adj_close: Optional[float]
    volume: Optional[int]
    source: str
    content_hash: str
    ingestion_run_id: str


class FilingRecord(BaseModel):
    ticker: str
    exchange: str
    filing_type: str
    title: str
    document_url: str
    published_at: Optional[str]   # ISO datetime string
    period_end: Optional[str]     # ISO date string
    extracted_text: Optional[str]
    page_count: Optional[int]
    content_hash: str
    ingestion_run_id: str


class NewsArticleRecord(BaseModel):
    ticker: str
    headline: str
    article_url: str
    source_name: str
    published_at: str          # ISO datetime string
    body_snippet: Optional[str]
    author: Optional[str]
    content_hash: str
    ingestion_run_id: str


class SentimentPostRecord(BaseModel):
    ticker: str
    platform: str
    post_id: str
    post_url: Optional[str]
    content: str
    author_handle: Optional[str]
    posted_at: str             # ISO datetime string
    upvotes: Optional[int]
    comments_count: Optional[int]
    is_comment: bool
    parent_post_id: Optional[str]
    content_hash: str
    ingestion_run_id: str

"""
FastAPI application entry point.
All routes are prefixed with /api.
"""

import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Path
from fastapi.middleware.cors import CORSMiddleware

from data_ingestion.db.client import DatabaseClient
from data_ingestion.models.schemas import IngestRequest, IngestionResult, TickerStatus
from data_ingestion.orchestrator import run_ingestion

load_dotenv()

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Stock Analysis API",
    description="Phase 1: Data Ingestion",
    version="1.0.0",
)

ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["health"])
def health():
    """Quick liveness check."""
    db = DatabaseClient()
    return {"status": "ok", "database": "connected" if db.ping() else "unreachable"}


@app.post("/api/ingest", response_model=IngestionResult, tags=["ingestion"])
def ingest(request: IngestRequest):
    """
    Trigger data ingestion for a stock ticker.

    - Sentiment window is always the last 30 days.
    - If data already exists for this ticker, only the gap since the last
      ingestion is fetched (incremental mode).
    """
    logger.info("Ingest request: ticker=%s exchange=%s", request.ticker, request.exchange)
    try:
        result = run_ingestion(request.ticker, request.exchange)
        return result
    except EnvironmentError as e:
        raise HTTPException(status_code=500, detail=f"Configuration error: {e}")
    except Exception as e:
        logger.exception("Ingestion failed for %s", request.ticker)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/status/{ticker}", response_model=TickerStatus, tags=["ingestion"])
def get_status(
    ticker: str = Path(..., min_length=1, max_length=20, description="NSE/BSE ticker symbol"),
):
    """
    Return ingestion status and row counts for a given ticker.
    Does not trigger any new data collection.
    """
    ticker = ticker.upper()
    try:
        db = DatabaseClient()
        return db.get_ticker_status(ticker)
    except Exception as e:
        logger.exception("Status check failed for %s", ticker)
        raise HTTPException(status_code=500, detail=str(e))

export type IngestionStatus = "COMPLETED" | "PARTIAL" | "FAILED" | "RUNNING" | "SKIPPED";

export interface CollectorResult {
  status: string;
  rows_inserted: number;
  error: string | null;
}

export interface IngestionResult {
  run_id: string;
  ticker: string;
  exchange: string;
  status: IngestionStatus;
  is_incremental: boolean;
  fetch_from: string;
  fetch_to: string;
  market: CollectorResult;
  filings: CollectorResult;
  news: CollectorResult;
  sentiment: CollectorResult;
  duration_seconds: number;
  message: string;
}

export interface TickerStatus {
  ticker: string;
  has_data: boolean;
  last_ingested_date: string | null;
  market_rows: number;
  news_rows: number;
  sentiment_rows: number;
  filings_rows: number;
}

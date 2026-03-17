import type { IngestionResult, TickerStatus } from "@/types/api";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export async function ingestTicker(ticker: string): Promise<IngestionResult> {
  const res = await fetch(`${API_URL}/api/ingest`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ticker: ticker.trim().toUpperCase() }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Unknown error" }));
    throw new Error(err.detail ?? `HTTP ${res.status}`);
  }

  return res.json();
}

export async function fetchTickerStatus(ticker: string): Promise<TickerStatus> {
  const res = await fetch(
    `${API_URL}/api/status/${ticker.trim().toUpperCase()}`
  );

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Unknown error" }));
    throw new Error(err.detail ?? `HTTP ${res.status}`);
  }

  return res.json();
}

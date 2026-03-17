"use client";

import type { IngestionResult } from "@/types/api";
import CollectorRow from "./CollectorRow";

interface ResultCardProps {
  result: IngestionResult;
}

const OVERALL_STYLES = {
  COMPLETED: {
    border: "border-green-200",
    header: "bg-green-50",
    badge:  "bg-green-100 text-green-700",
    icon:   "✓",
  },
  PARTIAL: {
    border: "border-yellow-200",
    header: "bg-yellow-50",
    badge:  "bg-yellow-100 text-yellow-700",
    icon:   "⚠",
  },
  FAILED: {
    border: "border-red-200",
    header: "bg-red-50",
    badge:  "bg-red-100 text-red-700",
    icon:   "✕",
  },
  RUNNING: {
    border: "border-blue-200",
    header: "bg-blue-50",
    badge:  "bg-blue-100 text-blue-700",
    icon:   "…",
  },
  SKIPPED: {
    border: "border-gray-200",
    header: "bg-gray-50",
    badge:  "bg-gray-100 text-gray-600",
    icon:   "–",
  },
};

export default function ResultCard({ result }: ResultCardProps) {
  const styles = OVERALL_STYLES[result.status] ?? OVERALL_STYLES.SKIPPED;
  const totalRows =
    result.market.rows_inserted +
    result.filings.rows_inserted +
    result.news.rows_inserted +
    result.sentiment.rows_inserted;

  return (
    <div className={`rounded-xl border ${styles.border} overflow-hidden shadow-sm`}>
      {/* Header */}
      <div className={`${styles.header} px-5 py-4 flex items-start justify-between`}>
        <div>
          <div className="flex items-center gap-2">
            <span className="text-lg font-bold text-gray-900">{result.ticker}</span>
            <span className="text-xs text-gray-400 font-medium">{result.exchange}</span>
            <span
              className={`text-xs font-semibold px-2.5 py-0.5 rounded-full ${styles.badge}`}
            >
              {styles.icon} {result.status}
            </span>
            {result.is_incremental && (
              <span className="text-xs font-medium px-2 py-0.5 rounded-full bg-indigo-100 text-indigo-600">
                Incremental
              </span>
            )}
          </div>
          <p className="text-sm text-gray-500 mt-1">{result.message}</p>
        </div>
        <div className="text-right shrink-0 ml-4">
          <p className="text-2xl font-bold text-gray-800">
            {totalRows > 0 ? `+${totalRows.toLocaleString()}` : "0"}
          </p>
          <p className="text-xs text-gray-400">total rows</p>
        </div>
      </div>

      {/* Date range + duration */}
      <div className="px-5 py-2.5 bg-white border-b border-gray-100 flex items-center gap-6 text-xs text-gray-400">
        <span>
          Window:{" "}
          <span className="text-gray-600 font-medium">
            {result.fetch_from} → {result.fetch_to}
          </span>
        </span>
        <span>
          Duration:{" "}
          <span className="text-gray-600 font-medium">
            {result.duration_seconds}s
          </span>
        </span>
        <span>
          Run ID:{" "}
          <span className="text-gray-600 font-mono">{result.run_id.slice(0, 8)}…</span>
        </span>
      </div>

      {/* Collector breakdown */}
      <div className="px-5 py-1 bg-white">
        <CollectorRow label="Market data" result={result.market} />
        <CollectorRow label="Filings"     result={result.filings} />
        <CollectorRow label="News"        result={result.news} />
        <CollectorRow label="Sentiment"   result={result.sentiment} />
      </div>
    </div>
  );
}

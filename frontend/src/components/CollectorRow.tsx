"use client";

import type { CollectorResult } from "@/types/api";

interface CollectorRowProps {
  label: string;
  result: CollectorResult;
}

const STATUS_STYLES: Record<string, string> = {
  SUCCESS:              "bg-green-100 text-green-700",
  SKIPPED_NO_NEW_DATA:  "bg-blue-100 text-blue-700",
  FAILED:               "bg-red-100 text-red-700",
  TIMEOUT:              "bg-yellow-100 text-yellow-700",
};

const STATUS_LABELS: Record<string, string> = {
  SUCCESS:              "Done",
  SKIPPED_NO_NEW_DATA:  "Up to date",
  FAILED:               "Failed",
  TIMEOUT:              "Timed out",
};

export default function CollectorRow({ label, result }: CollectorRowProps) {
  const pill = STATUS_STYLES[result.status] ?? "bg-gray-100 text-gray-600";
  const pillLabel = STATUS_LABELS[result.status] ?? result.status;

  return (
    <div className="flex items-center justify-between py-2.5 border-b border-gray-100 last:border-0">
      <span className="text-sm text-gray-600 w-28">{label}</span>

      <span className={`text-xs font-medium px-2.5 py-0.5 rounded-full ${pill}`}>
        {pillLabel}
      </span>

      <span className="text-sm tabular-nums text-gray-500 w-20 text-right">
        {result.rows_inserted > 0
          ? `+${result.rows_inserted.toLocaleString()} rows`
          : "—"}
      </span>

      {result.error && (
        <span className="text-xs text-red-400 truncate max-w-[160px]" title={result.error}>
          {result.error}
        </span>
      )}
    </div>
  );
}

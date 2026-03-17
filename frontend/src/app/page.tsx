"use client";

import { useState, useRef } from "react";
import type { IngestionResult } from "@/types/api";
import { ingestTicker } from "@/lib/api";
import ResultCard from "@/components/ResultCard";

type Phase = "idle" | "loading" | "done" | "error";

export default function Home() {
  const [ticker, setTicker]   = useState("");
  const [phase, setPhase]     = useState<Phase>("idle");
  const [result, setResult]   = useState<IngestionResult | null>(null);
  const [error, setError]     = useState<string | null>(null);
  const inputRef              = useRef<HTMLInputElement>(null);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const t = ticker.trim().toUpperCase();
    if (!t) return;

    setPhase("loading");
    setResult(null);
    setError(null);

    try {
      const data = await ingestTicker(t);
      setResult(data);
      setPhase("done");
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
      setPhase("error");
    }
  }

  function handleReset() {
    setTicker("");
    setResult(null);
    setError(null);
    setPhase("idle");
    setTimeout(() => inputRef.current?.focus(), 50);
  }

  return (
    <main className="min-h-screen flex flex-col items-center justify-start pt-24 px-4 pb-16">

      {/* Header */}
      <div className="text-center mb-10">
        <h1 className="text-3xl font-bold text-gray-900 tracking-tight">
          Stock Data Ingestion
        </h1>
        <p className="text-gray-500 mt-2 text-sm max-w-sm mx-auto">
          Enter an NSE or BSE ticker to ingest market data, filings, news, and
          sentiment for the last 30 days.
        </p>
      </div>

      {/* Input form */}
      <form
        onSubmit={handleSubmit}
        className="w-full max-w-md flex gap-2"
      >
        <input
          ref={inputRef}
          type="text"
          value={ticker}
          onChange={(e) => setTicker(e.target.value.toUpperCase())}
          placeholder="e.g. RELIANCE, HDFCBANK, TCS"
          maxLength={20}
          disabled={phase === "loading"}
          className="
            flex-1 px-4 py-2.5 rounded-lg border border-gray-300 text-sm
            placeholder:text-gray-400 bg-white shadow-sm
            focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent
            disabled:opacity-50 disabled:cursor-not-allowed
            uppercase tracking-wider font-medium
          "
        />
        <button
          type="submit"
          disabled={phase === "loading" || !ticker.trim()}
          className="
            px-5 py-2.5 rounded-lg bg-indigo-600 text-white text-sm font-semibold
            hover:bg-indigo-700 active:bg-indigo-800
            disabled:opacity-50 disabled:cursor-not-allowed
            transition-colors shadow-sm
          "
        >
          {phase === "loading" ? "Ingesting…" : "Ingest"}
        </button>
      </form>

      {/* Hint */}
      <p className="text-xs text-gray-400 mt-2">
        Sentiment is always fetched for the last 30 days. Already-ingested data is skipped automatically.
      </p>

      {/* Loading state */}
      {phase === "loading" && (
        <div className="mt-10 flex flex-col items-center gap-3 text-gray-500">
          <div className="w-8 h-8 border-2 border-indigo-400 border-t-transparent rounded-full animate-spin" />
          <p className="text-sm">
            Running collectors for <span className="font-semibold text-gray-700">{ticker}</span>…
          </p>
          <p className="text-xs text-gray-400">This may take up to a minute.</p>
        </div>
      )}

      {/* Error state */}
      {phase === "error" && error && (
        <div className="mt-8 w-full max-w-md rounded-xl border border-red-200 bg-red-50 px-5 py-4">
          <p className="text-sm font-semibold text-red-700">Ingestion failed</p>
          <p className="text-sm text-red-500 mt-1">{error}</p>
          <button
            onClick={handleReset}
            className="mt-3 text-xs text-red-600 underline hover:no-underline"
          >
            Try again
          </button>
        </div>
      )}

      {/* Result */}
      {phase === "done" && result && (
        <div className="mt-8 w-full max-w-2xl flex flex-col gap-4">
          <ResultCard result={result} />
          <div className="text-center">
            <button
              onClick={handleReset}
              className="text-xs text-gray-400 hover:text-gray-600 underline hover:no-underline"
            >
              Ingest another ticker
            </button>
          </div>
        </div>
      )}
    </main>
  );
}

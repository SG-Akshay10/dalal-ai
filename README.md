# StockSense AI

> AI-powered Indian equity research assistant — generates comprehensive stock reports from regulatory filings, financial news, and social media sentiment.

**Current Status: Phase 1 — Scrape & See**

---

## Architecture Overview

| Layer | Technology | Introduced |
|-------|-----------|-----------|
| Data Collection | Python scrapers (pdfplumber, NewsAPI, tweepy, PRAW) | Phase 1 |
| Agent Pipeline | LangGraph + GPT-4o + FinBERT | Phase 2 |
| Vector Store | ChromaDB (OpenAI embeddings) | Phase 2 |
| API + Database | FastAPI + Supabase (Postgres) | Phase 2 |
| Gradio UI | Gradio (data inspection) | Phase 1 |
| Production UI | Next.js 14 + shadcn/ui | Phase 4 |
| Async Queue | Celery + Redis | Phase 5 |

---

## Phase 1 Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (`pip install uv`)
- Tesseract OCR: `sudo apt-get install tesseract-ocr tesseract-ocr-eng` (Ubuntu/Debian)
- macOS: `brew install tesseract`

### 1. Clone & install

```bash
git clone https://github.com/<your-org>/stocksense-ai.git
cd stocksense-ai/backend

# Install all dependencies (including dev tools)
uv sync --extra dev
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and fill in your API keys:
#   NEWSAPI_KEY      → https://newsapi.org (free tier: 100 req/day)
#   SERP_API_KEY     → https://serpapi.com (fallback for news)
#   TWITTER_BEARER_TOKEN → Twitter Developer Portal (v2 API)
#   REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET → https://www.reddit.com/prefs/apps
```

### 3. Run unit tests

```bash
# All unit tests (no API keys needed — all external calls mocked)
uv run pytest tests/unit/ -v

# With coverage report
uv run pytest tests/unit/ --cov=app/scrapers --cov=app/schemas --cov-report=term-missing
```

### 4. Launch Gradio UI

```bash
# Requires real API keys in .env
uv run python -m gradio_app.app
# Opens at: http://localhost:7860
```

Enter a ticker (e.g. `RELIANCE`) and company name (e.g. `Reliance Industries`), then click "Fetch" in each tab.

---

## Project Structure (Phase 1)

```
backend/
├── app/
│   ├── schemas/            # Pydantic data contracts (Phase 2 API)
│   │   ├── document_object.py
│   │   ├── news_article.py
│   │   └── social_post.py
│   └── scrapers/           # Phase 1 data collectors
│       ├── pdf_extractor.py     # pdfplumber + Tesseract OCR
│       ├── document_fetcher.py  # BSE / NSE / SEBI EDGAR
│       ├── news_scraper.py      # NewsAPI + SerpAPI fallback
│       └── social_listener.py  # Twitter/X + Reddit
├── gradio_app/
│   └── app.py              # Tabbed data inspection UI
├── tests/
│   ├── conftest.py         # Shared fixtures (env cleared, no real API calls)
│   └── unit/               # 20 unit tests across all scrapers
├── .env.example            # Required environment variables
├── pyproject.toml          # Dependencies + tool config
└── README.md
```

---

## API Keys Guide

| Key | Where to Get | Free Tier |
|-----|-------------|-----------|
| `NEWSAPI_KEY` | [newsapi.org](https://newsapi.org) | 100 req/day, last 30 days |
| `SERP_API_KEY` | [serpapi.com](https://serpapi.com) | 100 searches/month |
| `TWITTER_BEARER_TOKEN` | [developer.twitter.com](https://developer.twitter.com) | 500K tweet reads/month |
| `REDDIT_CLIENT_ID` + `SECRET` | [reddit.com/prefs/apps](https://www.reddit.com/prefs/apps) | Free (rate limited) |
| `OPENAI_API_KEY` | [platform.openai.com](https://platform.openai.com) | Pay-per-use (Phase 2+) |
| `SUPABASE_URL` + `SUPABASE_SERVICE_ROLE_KEY` (or `SUPABASE_SECRET_KEY` / `SUPABASE_KEY`) | [supabase.com](https://supabase.com) | Free tier available |

---

## Testing

```bash
# Unit tests only (Phase 1 — no real API calls)
uv run pytest tests/unit/ -v

# With coverage (must be ≥80%)
uv run pytest tests/unit/ --cov=app/scrapers --cov=app/schemas \
  --cov-report=term-missing --cov-fail-under=80

# Lint
uv run ruff check app/ tests/
```

---

## Roadmap

| Phase | Description | Timeline |
|-------|------------|---------|
| ✅ **1 — Scrape & See** | Data collectors + Gradio UI | Weeks 1–4 |
| 🔲 **2 — Agents & Analysis** | LangGraph pipeline + report generation | Weeks 5–7 |
| 🔲 **3 — Dockerise** | Docker Compose one-command setup | Weeks 8–9 |
| 🔲 **4 — Next.js UI** | Production frontend | Weeks 10–13 |
| 🔲 **5 — Redis & Scale** | Celery + async queue + caching | Weeks 14–16 |

---

## Legal Disclaimer

StockSense AI is a research and information tool only. Reports are for informational purposes only and do **not** constitute investment advice. This platform is **not** SEBI-registered. Always consult a SEBI-registered financial advisor before making investment decisions.

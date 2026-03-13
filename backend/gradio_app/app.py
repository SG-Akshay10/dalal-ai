"""Gradio tabbed UI for Phase 1 — StockSense AI Data Inspector.

Provides a simple inspection interface for the three Phase 1 scrapers:
- Tab 1: Documents (BSE/NSE regulatory filings)
- Tab 2: News (financial news articles)
- Tab 3: Social (Twitter/Reddit/Web posts)

The user only needs to enter a ticker symbol — company name and aliases
are auto-resolved via the stock_alias module (NSE lookup + Gemini LLM).

Run: python -m gradio_app.app   (from backend/ directory)
"""
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

# Load .env file BEFORE importing scrapers (they read env vars at import time)
load_dotenv()

import pandas as pd
import gradio as gr

from app.scrapers.document_fetcher import fetch_documents
from app.scrapers.news_scraper import fetch_news
from app.scrapers.social_listener import fetch_social
from app.scrapers.stock_alias import get_stock_info

import json
from app.agents.competitor_agent import identify_competitors
from app.agents.orchestrator import run_pipeline
from app.schemas.report import CompetitorAnalysis

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Column definitions for each tab ──────────────────────────────────────────

DOC_COLUMNS = ["source", "date", "doc_type", "text_preview", "url", "ocr_used", "parse_confidence"]
NEWS_COLUMNS = ["headline", "source", "date", "url", "body_preview"]
SOCIAL_COLUMNS = ["platform", "author", "date", "content_preview", "likes", "comments", "url"]


# ── Scraper wrappers for Gradio ───────────────────────────────────────────────

def _resolve_stock(ticker: str) -> str:
    """Resolve ticker → show the resolved info as a status message."""
    ticker = ticker.strip().upper()
    if not ticker:
        return ""
    info = get_stock_info(ticker)
    names = ", ".join(info.all_names[:6])
    return f"🔍 **{info.company_name}** — searching with: {names}"


def _fetch_and_display_documents(ticker: str, days: int) -> pd.DataFrame:
    """Fetch documents and format as a DataFrame for Gradio display."""
    ticker = ticker.strip().upper()
    if not ticker:
        return pd.DataFrame(columns=DOC_COLUMNS)

    logger.info("Fetching documents for %s (%d days)...", ticker, days)
    docs = fetch_documents(ticker, days=int(days))

    if not docs:
        return pd.DataFrame(columns=DOC_COLUMNS)

    rows = []
    for doc in docs:
        rows.append({
            "source": doc.source,
            "date": doc.date.strftime("%Y-%m-%d"),
            "doc_type": doc.doc_type,
            "text_preview": doc.text[:200] + "..." if len(doc.text) > 200 else doc.text,
            "url": doc.url,
            "ocr_used": "⚠️ OCR" if doc.ocr_used else "✅ Digital",
            "parse_confidence": f"{doc.parse_confidence:.0%}",
        })

    return pd.DataFrame(rows, columns=DOC_COLUMNS)


def _fetch_and_display_news(ticker: str, days: int) -> pd.DataFrame:
    """Fetch news — auto-resolves company name from ticker."""
    ticker = ticker.strip().upper()
    if not ticker:
        return pd.DataFrame(columns=NEWS_COLUMNS)

    info = get_stock_info(ticker)
    logger.info("Fetching news for %s (%s, %d days)...", ticker, info.company_name, days)
    articles = fetch_news(ticker, info.company_name, days=int(days))

    if not articles:
        return pd.DataFrame(columns=NEWS_COLUMNS)

    rows = []
    for article in articles:
        rows.append({
            "headline": article.headline,
            "source": article.source,
            "date": article.date.strftime("%Y-%m-%d"),
            "url": article.url,
            "body_preview": article.body[:200] + "..." if len(article.body) > 200 else article.body,
        })

    return pd.DataFrame(rows, columns=NEWS_COLUMNS)


def _fetch_and_display_social(ticker: str, days: int) -> pd.DataFrame:
    """Fetch social posts — auto-resolves all aliases from ticker."""
    ticker = ticker.strip().upper()
    if not ticker:
        return pd.DataFrame(columns=SOCIAL_COLUMNS)

    logger.info("Fetching social posts for %s (%d days)...", ticker, days)
    posts = fetch_social(ticker, days=int(days))

    if not posts:
        return pd.DataFrame(columns=SOCIAL_COLUMNS)

    platform_icons = {"twitter": "🐦 Twitter/X", "reddit": "🟠 Reddit", "web": "🌐 Web"}
    rows = []
    for post in posts:
        rows.append({
            "platform": platform_icons.get(post.platform, post.platform),
            "author": post.author,
            "date": post.date.strftime("%Y-%m-%d"),
            "content_preview": post.content[:200] + "..." if len(post.content) > 200 else post.content,
            "likes": post.likes,
            "comments": post.comments,
            "url": post.url,
        })

    return pd.DataFrame(rows, columns=SOCIAL_COLUMNS)


def _fetch_all(ticker: str, days: int):
    """Fetch all three data sources in parallel. Returns (docs_df, news_df, social_df, status)."""
    ticker = ticker.strip().upper()
    if not ticker:
        empty = (
            pd.DataFrame(columns=DOC_COLUMNS),
            pd.DataFrame(columns=NEWS_COLUMNS),
            pd.DataFrame(columns=SOCIAL_COLUMNS),
            "⚠️ Enter a ticker first",
        )
        return empty

    # Resolve stock info once (cached) so all fetchers use it
    info = get_stock_info(ticker)
    logger.info("⚡ Fetch All: %s (%s) — %d days, running in parallel...", ticker, info.company_name, days)

    results = {}

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            pool.submit(_fetch_and_display_documents, ticker, days): "docs",
            pool.submit(_fetch_and_display_news, ticker, days): "news",
            pool.submit(_fetch_and_display_social, ticker, days): "social",
        }

        for future in as_completed(futures):
            key = futures[future]
            try:
                results[key] = future.result()
            except Exception as exc:
                logger.warning("Fetch %s failed: %s", key, exc)
                if key == "docs":
                    results[key] = pd.DataFrame(columns=DOC_COLUMNS)
                elif key == "news":
                    results[key] = pd.DataFrame(columns=NEWS_COLUMNS)
                else:
                    results[key] = pd.DataFrame(columns=SOCIAL_COLUMNS)

    # Default fallbacks if empty
    docs_df = results.get("docs", pd.DataFrame(columns=DOC_COLUMNS))
    news_df = results.get("news", pd.DataFrame(columns=NEWS_COLUMNS))
    social_df = results.get("social", pd.DataFrame(columns=SOCIAL_COLUMNS))

    doc_count = len(docs_df)
    news_count = len(news_df)
    social_count = len(social_df)

    status = (
        f"✅ **{info.company_name}** — "
        f"📄 {doc_count} filings · 📰 {news_count} articles · 💬 {social_count} posts"
    )

    return docs_df, news_df, social_df, status

def _find_competitors_gradio(ticker: str, provider: str):
    ticker = ticker.strip().upper()
    if not ticker:
        return "{}"
    try:
        comps = identify_competitors(ticker, provider=provider)
        return comps.model_dump_json(indent=2)
    except Exception as e:
        return f'{{\n  "error": "{str(e)}"\n}}'

def _generate_report_gradio(ticker: str, comp_json: str, provider: str):
    ticker = ticker.strip().upper()
    if not ticker: return "Error: Please enter a ticker first."
    
    overridden = None
    if comp_json and comp_json.strip() and not comp_json.startswith("{ \n  \"error"):
        try:
            data = json.loads(comp_json)
            overridden = CompetitorAnalysis(**data)
        except Exception as e:
            return f"Error parsing competitor JSON: {e}"
            
    return run_pipeline(ticker, overridden_competitors=overridden, preferred_provider=provider)

# ── Gradio UI ─────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="StockSense AI — Phase 1 Data Inspector",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            """
            # 📊 StockSense AI — Phase 1 Data Inspector
            Enter any NSE/BSE ticker — company name, aliases, and search terms are auto-resolved.
            > **Examples:** `SBIN`, `ETERNAL`, `HDFCBANK`, `RELIANCE`, `TATAMOTORS`
            """
        )

        # ── Shared controls: just ticker + days ──────────────────────────────
        with gr.Row():
            ticker_input = gr.Textbox(
                label="NSE/BSE Ticker",
                placeholder="e.g. SBIN, ETERNAL, RELIANCE",
                scale=3,
            )
            days_input = gr.Slider(
                label="Days to look back", minimum=15, maximum=150, value=50, step=5, scale=2
            )
            fetch_btn = gr.Button("Fetch", variant="primary", scale=1)

        # Show resolved stock info / fetch status
        stock_info_display = gr.Markdown("")
        ticker_input.change(
            fn=_resolve_stock,
            inputs=[ticker_input],
            outputs=[stock_info_display],
        )

        # ── Tabs ─────────────────────────────────────────────────────────────
        with gr.Tabs():
            with gr.Tab("📄 Regulatory Filings"):
                docs_table = gr.DataFrame(
                    headers=DOC_COLUMNS,
                    label="BSE / NSE Documents",
                    wrap=True,
                )

            with gr.Tab("📰 Financial News"):
                news_table = gr.DataFrame(
                    headers=NEWS_COLUMNS,
                    label="Financial News Articles",
                    wrap=True,
                )

            with gr.Tab("💬 Social Media"):
                social_table = gr.DataFrame(
                    headers=SOCIAL_COLUMNS,
                    label="Twitter/X / Reddit / Web Posts",
                    wrap=True,
                )
            
            with gr.Tab("🧠 Phase 2 Agents"):
                gr.Markdown("Identify competitors, modify the JSON output if you'd like, and then run the full automated analysis pipeline.")
                
                with gr.Row():
                    provider_dropdown = gr.Dropdown(
                        choices=["gemini", "sarvam"],
                        value="gemini",
                        label="LLM Provider",
                        info="Choose which LLM to use for generation (Embeddings will use Gemini)",
                        scale=2
                    )
                
                with gr.Row():
                    comp_btn = gr.Button("1. Identify Competitors")
                    comp_text = gr.Textbox(label="Competitors JSON (Editable)", lines=10)
                
                gen_btn = gr.Button("2. Generate Full Report", variant="primary")
                report_out = gr.Markdown("The synthesized markdown report will appear here.")
                
                comp_btn.click(fn=_find_competitors_gradio, inputs=[ticker_input, provider_dropdown], outputs=[comp_text])
                gen_btn.click(fn=_generate_report_gradio, inputs=[ticker_input, comp_text, provider_dropdown], outputs=[report_out])

        # ── Single Fetch button wires to all three tables ────────────────────
        fetch_btn.click(
            fn=_fetch_all,
            inputs=[ticker_input, days_input],
            outputs=[docs_table, news_table, social_table, stock_info_display],
        )

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)

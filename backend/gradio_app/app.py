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
from dotenv import load_dotenv

# Load .env file BEFORE importing scrapers (they read env vars at import time)
load_dotenv()

import pandas as pd
import gradio as gr

from app.scrapers.document_fetcher import fetch_documents
from app.scrapers.news_scraper import fetch_news
from app.scrapers.social_listener import fetch_social
from app.scrapers.stock_alias import get_stock_info

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
                label="Days to look back", minimum=7, maximum=90, value=49, step=7, scale=2
            )

        # Show resolved stock info
        stock_info_display = gr.Markdown("")
        ticker_input.change(
            fn=_resolve_stock,
            inputs=[ticker_input],
            outputs=[stock_info_display],
        )

        # ── Tab 1: Documents ──────────────────────────────────────────────────
        with gr.Tabs():
            with gr.Tab("📄 Regulatory Filings"):
                docs_btn = gr.Button("Fetch Filings", variant="primary")
                docs_table = gr.DataFrame(
                    headers=DOC_COLUMNS,
                    label="BSE / NSE Documents",
                    wrap=True,
                )
                docs_btn.click(
                    fn=_fetch_and_display_documents,
                    inputs=[ticker_input, days_input],
                    outputs=[docs_table],
                )

            # ── Tab 2: News ───────────────────────────────────────────────────
            with gr.Tab("📰 Financial News"):
                news_btn = gr.Button("Fetch News", variant="primary")
                news_table = gr.DataFrame(
                    headers=NEWS_COLUMNS,
                    label="Financial News Articles",
                    wrap=True,
                )
                news_btn.click(
                    fn=_fetch_and_display_news,
                    inputs=[ticker_input, days_input],
                    outputs=[news_table],
                )

            # ── Tab 3: Social ─────────────────────────────────────────────────
            with gr.Tab("💬 Social Media"):
                social_btn = gr.Button("Fetch Social Posts", variant="primary")
                social_table = gr.DataFrame(
                    headers=SOCIAL_COLUMNS,
                    label="Twitter/X / Reddit / Web Posts",
                    wrap=True,
                )
                social_btn.click(
                    fn=_fetch_and_display_social,
                    inputs=[ticker_input, days_input],
                    outputs=[social_table],
                )

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)

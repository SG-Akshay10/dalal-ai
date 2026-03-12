"""Gradio tabbed UI for Phase 1 — StockSense AI Data Inspector.

Provides a simple inspection interface for the three Phase 1 scrapers:
- Tab 1: Documents (BSE/NSE/SEBI regulatory filings)
- Tab 2: News (financial news articles)
- Tab 3: Social (Twitter/Reddit posts)

Run: python -m gradio_app.app   (from backend/ directory)
     or: gradio gradio_app/app.py
"""
import logging
import pandas as pd
import gradio as gr

from app.scrapers.document_fetcher import fetch_documents
from app.scrapers.news_scraper import fetch_news
from app.scrapers.social_listener import fetch_social

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Column definitions for each tab ──────────────────────────────────────────

DOC_COLUMNS = ["source", "date", "doc_type", "text_preview", "url", "ocr_used", "parse_confidence"]
NEWS_COLUMNS = ["headline", "source", "date", "url", "body_preview"]
SOCIAL_COLUMNS = ["platform", "author", "date", "content_preview", "likes", "comments", "url"]


# ── Scraper wrappers for Gradio ───────────────────────────────────────────────

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


def _fetch_and_display_news(ticker: str, company_name: str, days: int) -> pd.DataFrame:
    """Fetch news and format as a DataFrame for Gradio display."""
    ticker = ticker.strip().upper()
    company_name = company_name.strip()
    if not ticker or not company_name:
        return pd.DataFrame(columns=NEWS_COLUMNS)

    logger.info("Fetching news for %s (%s, %d days)...", ticker, company_name, days)
    articles = fetch_news(ticker, company_name, days=int(days))

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


def _fetch_and_display_social(ticker: str, company_name: str, days: int) -> pd.DataFrame:
    """Fetch social posts and format as a DataFrame for Gradio display."""
    ticker = ticker.strip().upper()
    company_name = company_name.strip()
    if not ticker:
        return pd.DataFrame(columns=SOCIAL_COLUMNS)

    logger.info("Fetching social posts for %s (%d days)...", ticker, days)
    posts = fetch_social(ticker, days=int(days), company_name=company_name)

    if not posts:
        return pd.DataFrame(columns=SOCIAL_COLUMNS)

    platform_icons = {"twitter": "🐦 Twitter", "reddit": "🟠 Reddit", "web": "🌐 Web"}
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
            Inspect raw scraped data for any NSE/BSE listed company.
            Enter a ticker and company name to fetch results from BSE/NSE filings, financial news, and social media.
            > **Note:** Requires valid API keys in `.env` file. See `.env.example`.
            """
        )

        # ── Shared controls ───────────────────────────────────────────────────
        with gr.Row():
            ticker_input = gr.Textbox(
                label="NSE/BSE Ticker", placeholder="e.g. RELIANCE", scale=2
            )
            company_input = gr.Textbox(
                label="Company Name (for news search)",
                placeholder="e.g. Reliance Industries",
                scale=3,
            )
            days_input = gr.Slider(
                label="Days to look back", minimum=7, maximum=90, value=21, step=7, scale=2
            )

        # ── Tab 1: Documents ──────────────────────────────────────────────────
        with gr.Tabs():
            with gr.Tab("📄 Regulatory Filings"):
                docs_btn = gr.Button("Fetch Filings", variant="primary")
                docs_status = gr.Markdown("")
                docs_table = gr.DataFrame(
                    headers=DOC_COLUMNS,
                    label="BSE / NSE / SEBI Documents",
                    wrap=True,
                )
                docs_btn.click(
                    fn=lambda t, d: (_fetch_and_display_documents(t, d), f"✅ Fetched"),
                    inputs=[ticker_input, days_input],
                    outputs=[docs_table, docs_status],
                )

            # ── Tab 2: News ───────────────────────────────────────────────────
            with gr.Tab("📰 Financial News"):
                news_btn = gr.Button("Fetch News", variant="primary")
                news_status = gr.Markdown("")
                news_table = gr.DataFrame(
                    headers=NEWS_COLUMNS,
                    label="Financial News Articles",
                    wrap=True,
                )
                news_btn.click(
                    fn=lambda t, c, d: (_fetch_and_display_news(t, c, d), f"✅ Fetched"),
                    inputs=[ticker_input, company_input, days_input],
                    outputs=[news_table, news_status],
                )

            # ── Tab 3: Social ─────────────────────────────────────────────────
            with gr.Tab("💬 Social Media"):
                social_btn = gr.Button("Fetch Social Posts", variant="primary")
                social_status = gr.Markdown("")
                social_table = gr.DataFrame(
                    headers=SOCIAL_COLUMNS,
                    label="Twitter / Reddit Posts",
                    wrap=True,
                )
                social_btn.click(
                    fn=lambda t, c, d: (_fetch_and_display_social(t, c, d), f"✅ Fetched"),
                    inputs=[ticker_input, company_input, days_input],
                    outputs=[social_table, social_status],
                )

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)

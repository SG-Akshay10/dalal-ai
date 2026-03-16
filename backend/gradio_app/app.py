"""Gradio UI for StockSense AI — Automated Equity Research.

Run: python -m gradio_app.app   (from backend/ directory)
"""
import logging
import os
import tempfile

import gradio as gr
from dotenv import load_dotenv

from app.agents.orchestrator import run_pipeline
from app.scrapers.stock_alias import get_stock_info

load_dotenv()  # Load .env file

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


def _resolve_stock(ticker: str) -> str:
    """Resolve ticker → show the resolved info as a status message."""
    ticker = ticker.strip().upper()
    if not ticker:
        return ""
    info = get_stock_info(ticker)
    names = ", ".join(info.all_names[:6])
    return f"🔍 **{info.company_name}** — searching with: {names}"


def generate_full_report(ticker: str, days: int, provider: str):
    ticker = ticker.strip().upper()
    if not ticker:
        yield "⚠️ Error: Please enter a ticker first.", gr.update(visible=False)
        return

    info = get_stock_info(ticker)

    yield f"🔄 **Running pipeline for {info.company_name}**\n\nSteps:\n1. Scraping Data (Docs, News, Social)\n2. Finding Competitors\n3. Analyzing Data\n4. Generating Report\n\n*This might take a few minutes...*", gr.update(visible=False)

    # Run the entire pipeline synchronously
    report_md = run_pipeline(ticker, preferred_provider=provider, days=int(days))

    # Save to a temporary file with the ticker name for downloading
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".md")
    tmp.close()

    filename = f"{ticker}_Equity_Report.md"
    new_path = os.path.join(os.path.dirname(tmp.name), filename)
    os.rename(tmp.name, new_path)

    with open(new_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    yield report_md, gr.update(value=new_path, visible=True)


# ── Gradio UI ─────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="StockSense AI — Report Generator",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            """
            # 📊 StockSense AI — Automated Equity Research
            Enter any NSE/BSE ticker, configure parameters, and click **Generate Report**.
            > **Examples:** `SBIN`, `ETERNAL`, `HDFCBANK`, `RELIANCE`, `TATAMOTORS`
            """
        )

        with gr.Row():
            ticker_input = gr.Textbox(
                label="NSE/BSE Ticker",
                placeholder="e.g. SBIN, ETERNAL, RELIANCE",
                scale=3,
            )
            days_input = gr.Slider(
                label="Days to look back", minimum=15, maximum=150, value=30, step=5, scale=2
            )
            provider_dropdown = gr.Dropdown(
                choices=["sarvam"],
                value="sarvam",
                label="Model Name (Generation)",
                scale=2
            )
            gen_btn = gr.Button("Generate Report", variant="primary", scale=2)

        stock_info_display = gr.Markdown("")
        ticker_input.change(
            fn=_resolve_stock,
            inputs=[ticker_input],
            outputs=[stock_info_display],
        )

        with gr.Row():
            report_out = gr.Markdown("The synthesized markdown report will appear here.")

        with gr.Row():
            download_btn = gr.DownloadButton("Download Markdown Report", visible=False)

        gen_btn.click(
            fn=generate_full_report,
            inputs=[ticker_input, days_input, provider_dropdown],
            outputs=[report_out, download_btn],
        )

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)

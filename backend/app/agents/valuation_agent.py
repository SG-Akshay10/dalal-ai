import json
import logging
from langchain_core.prompts import PromptTemplate

from app.llm_provider import get_llm_client
from app.schemas.analysis import FundamentalAnalysis, ValuationAnalysis, CompetitorAnalysis
from app.services.live_data import live_data_service

logger = logging.getLogger(__name__)

VALUATION_PROMPT = """You are an expert equity research analyst.
Write a concise valuation verdict (2-3 sentences) for {ticker}.
Also designate the Margin of Safety ("High", "Moderate", "Low", "Negative").
Finally, determine how it trades against its own history ("Premium", "At Par", "Discount") and the sector ("Premium", "At Par", "Discount").

Current Mutliples:
P/E: {pe}
P/B: {pb}
EV/EBITDA: {ev_eb}
Market Cap: {mc} Cr

Fundamental Context:
Growth: {growth}
Margins: {margins}

Peer Comparison Context (Target vs Sector/Peers):
(Infer from general knowledge or assume at par if no specific detailed multiples for peers are provided)

Always return your response as a valid JSON object matching the following schema:
{{
  "valuation_verdict": "...",
  "margin_of_safety": "Moderate",
  "vs_own_history": "At Par",
  "vs_sector_median": "At Par",
  "premium_discount_pct": 0.0
}}
"""

def analyze_valuation(ticker: str, fundamentals: FundamentalAnalysis, competitors: CompetitorAnalysis, provider: str = None) -> ValuationAnalysis:
    """Run the Valuation Agent."""
    # 1. Fetch live market multiples
    market_data = live_data_service.fetch_market_data(ticker)
    
    if not market_data or market_data.get("current_price") is None:
        return ValuationAnalysis(
            current_price=None,
            market_cap_cr=None,
            pe_ttm=None, pe_forward=None, pb_ratio=None, ev_ebitda=None, price_to_sales=None, dividend_yield=None, peg_ratio=None,
            vs_own_history=None, vs_sector_median=None, premium_discount_pct=None,
            valuation_verdict="Cannot compute valuation without live market data.",
            margin_of_safety="Negative"
        )
        
    revenue_growth = fundamentals.revenue_trend[-1].yoy_growth_pct if fundamentals and fundamentals.revenue_trend else 0.0
    
    prompt = PromptTemplate.from_template(VALUATION_PROMPT)
    formatted_prompt = prompt.format(
        ticker=ticker,
        pe=market_data.get("pe_ttm", "N/A"),
        pb=market_data.get("pb_ratio", "N/A"),
        ev_eb=market_data.get("ev_ebitda", "N/A"),
        mc=market_data.get("market_cap_cr", "N/A"),
        growth=f"Revenue Growth: {revenue_growth}% YoY",
        margins=f"Net Margin: {fundamentals.net_margin}%" if fundamentals else "Unknown"
    )
    
    llm = get_llm_client(provider)
    try:
        response = llm.invoke([
            {"role": "system", "content": "You output JSON only."},
            {"role": "user", "content": formatted_prompt}
        ])
        content = response.content.replace("```json", "").replace("```", "").strip()
        data = json.loads(content)
        
        return ValuationAnalysis(
            current_price=market_data.get("current_price"),
            market_cap_cr=market_data.get("market_cap_cr"),
            pe_ttm=market_data.get("pe_ttm"),
            pe_forward=market_data.get("pe_forward"),
            pb_ratio=market_data.get("pb_ratio"),
            ev_ebitda=market_data.get("ev_ebitda"),
            price_to_sales=market_data.get("price_to_sales"),
            dividend_yield=market_data.get("dividend_yield"),
            peg_ratio=market_data.get("peg_ratio"),
            vs_own_history=data.get("vs_own_history", "At Par"),
            vs_sector_median=data.get("vs_sector_median", "At Par"),
            premium_discount_pct=data.get("premium_discount_pct", 0.0),
            valuation_verdict=data.get("valuation_verdict", "Valuation appears standard for current metrics."),
            margin_of_safety=data.get("margin_of_safety", "Moderate")
        )
    except Exception as e:
        logger.error(f"Error in valuation agent for {ticker}: {e}")
        return ValuationAnalysis(
            current_price=market_data.get("current_price"),
            market_cap_cr=market_data.get("market_cap_cr"),
            pe_ttm=market_data.get("pe_ttm"),
            pe_forward=None, pb_ratio=None, ev_ebitda=None, price_to_sales=None, dividend_yield=None, peg_ratio=None,
            vs_own_history="At Par", vs_sector_median="At Par", premium_discount_pct=0.0,
            valuation_verdict=f"Error computing valuation verdict: {str(e)}",
            margin_of_safety="Low"
        )

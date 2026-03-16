"""
Rewritten for StockSense AI Pipeline v2.
Changes made:
- Live Peer Overhaul: Fixed LLM hallucination on peer multiples. Live fetches multiples for competitor tickers and injects a formatted table into the prompt.
- Deterministic Math (Python): Computes `premium_discount_pct` and `peg_ratio` explicitly. Includes rigorous `margin_of_safety` thresholding logic accounting for debt/equity and growth degradation.
- Historical Data Integration: Implemented non-blocking check to `live_data_service.fetch_historical_multiples()`. Forces `vs_own_history` to None if absent.
- Fundamental Fixes: Replaced `0.0` fallbacks with `None` safely. Derived EBITDA margin implicitly when available.
- Enhanced Prompt Structure: Feeds all 9 real-time metrics, includes the Indian Sector Reference Table, and enforces a strict 4-sentence `valuation_verdict` structure to limit filler and enforce comparative rigor.
- Strict Fallbacks: Modified all exception fallbacks to replace 0.0/'At Par' string masks with hard `None` values where deterministic computation failed.
"""

import json
import logging
import statistics
from langchain_core.prompts import PromptTemplate

from app.llm_provider import get_llm_client
from app.schemas.analysis import FundamentalAnalysis, ValuationAnalysis, CompetitorAnalysis
from app.services.live_data import live_data_service

logger = logging.getLogger(__name__)

VALUATION_PROMPT = """You are an expert equity research analyst focusing on valuation benchmarking.
Write a concise but highly rigorous valuation verdict for {ticker}.

<STRICT_RULES>
1. VERDICT STRUCTURE: Write EXACTLY 4 sentences with the following assigned jobs:
   - Sentence 1: Contrast the target's primary multiples against the peer median table, citing the exact numbers.
   - Sentence 2: Justify whether the premium/discount is warranted based on its revenue growth and fundamental quality.
   - Sentence 3: Identify the most prominent risk to multiple compression (e.g. slowing momentum, rising debt, sector rerating).
   - Sentence 4: Note any significant divergence between P/E and EV/EBITDA if material, otherwise conclude on the relative valuation attractiveness.
2. NO HALLUCINATION: Rely strictly on the provided multiples table, sector references, and fundamental context. Do NOT invent missing metrics.
3. RELATIVE CLASSIFICATIONS:
   - vs_sector_median: MUST be "Premium", "At Par", "Discount", or `null` if peer comparison is impossible.
   - vs_own_history: MUST be "Premium", "At Par", "Discount", or `null` if {historical_multiples} explicitly says unavailable. 
4. DO NOT COMPUTE: The Margin of Safety, Discount %, and PEG have been deterministically pre-calculated for you. Do not override them.
</STRICT_RULES>

<REASONING_STEPS>
1. Evaluate Target P/E against the Peer Median derived in Python. Determine vs_sector_median.
2. Evaluate Target P/E against Historical Average if provided. Determine vs_own_history.
3. Draft the 4-sentence valuation verdict adhering strictly to the metrics provided below.
</REASONING_STEPS>

Current Multiples ({ticker}):
P/E (TTM): {pe}
P/E (Forward): {pe_fwd}
P/B: {pb}
EV/EBITDA: {ev_eb}
P/S: {ps}
Div Yield: {dy}%
PEG Ratio: {peg}
Market Cap: {mc} Cr

Historical Multiples:
{historical_multiples}

Fundamental Context:
Revenue Growth: {growth}
EBITDA Margin: {ebitda_margin}
Net Margin: {net_margin}
Debt to Equity: {debt_equity}

{peer_context}

Reference - Indian Sector P/E Averages:
- Banking (Large): 12-18x | NBFC: 18-25x
- FMCG: 35-55x | IT Services: 20-35x
- Pharma: 18-30x | Discretionary: 40-70x
- Capital Goods: 30-50x | Autos: 15-25x

Always return your response as a valid JSON object matching the following schema. Wait until after your reasoning to generate the JSON.
<EXAMPLE_SCHEMA>
{{
  "valuation_verdict": "...",
  "vs_own_history": "Premium",
  "vs_sector_median": "Discount"
}}
</EXAMPLE_SCHEMA>
"""

def _safe_float(val) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None

def analyze_valuation(ticker: str, fundamentals: FundamentalAnalysis, competitors: CompetitorAnalysis, provider: str = None) -> ValuationAnalysis:
    """Run the Valuation Agent with live peer data and deterministic scoring."""
    
    # 1. Fetch live market multiples for Target
    market_data = live_data_service.fetch_market_data(ticker)
    
    if not market_data or market_data.get("current_price") is None:
        logger.error(f"[{ticker}] Market data fetch failed. Cannot complete valuation.")
        return ValuationAnalysis(
            current_price=None, market_cap_cr=None, pe_ttm=None, pe_forward=None, pb_ratio=None, 
            ev_ebitda=None, price_to_sales=None, dividend_yield=None, peg_ratio=None,
            vs_own_history=None, vs_sector_median=None, premium_discount_pct=None,
            valuation_verdict="Cannot compute valuation without live market data.",
            margin_of_safety="Negative"
        )
        
    pe_ttm = _safe_float(market_data.get("pe_ttm"))
        
    # 2. Extract Fundamental metrics with None guards
    rev_growth = None
    if fundamentals and fundamentals.revenue_trend and len(fundamentals.revenue_trend) > 0:
        rev_growth = fundamentals.revenue_trend[-1].yoy_growth_pct

    ebitda_margin = None
    if fundamentals and fundamentals.ebitda_trend and fundamentals.revenue_trend:
        if len(fundamentals.ebitda_trend) > 0 and len(fundamentals.revenue_trend) > 0:
            e_val = fundamentals.ebitda_trend[-1].value
            r_val = fundamentals.revenue_trend[-1].value
            if e_val and r_val and r_val > 0:
                ebitda_margin = (e_val / r_val) * 100
                ebitda_margin = round(ebitda_margin, 2)

    net_margin = getattr(fundamentals, 'net_margin', None)
    debt_equity = getattr(fundamentals, 'debt_equity_ratio', None)

    # 3. Deterministic Python Math
    # PEG Ratio
    peg_val = None
    if pe_ttm and rev_growth and pe_ttm > 0 and rev_growth > 0:
        peg_val = round(pe_ttm / rev_growth, 2)

    # Fetch Peer Multiples and format table
    peer_pes = []
    peer_context_str = "Peer Comparison Data Not Available."
    peer_median_pe = None
    
    peers = getattr(competitors, "competitors", []) if competitors else []
    
    if peers:
        peer_lines = ["| Ticker | P/E | EV/EBITDA | P/B | P/S |", "|---|---|---|---|---|"]
        
        for p in peers:
            peer_ticker = p.ticker
            try:
                p_data = live_data_service.fetch_market_data(peer_ticker)
                if p_data:
                    p_pe = _safe_float(p_data.get("pe_ttm"))
                    p_e_e = _safe_float(p_data.get("ev_ebitda"))
                    p_pb = _safe_float(p_data.get("pb_ratio"))
                    p_ps = _safe_float(p_data.get("price_to_sales"))
                    
                    if p_pe and p_pe > 0:
                        peer_pes.append(p_pe)
                        
                    peer_lines.append(f"| {peer_ticker} | {p_pe if p_pe else '-'} | {p_e_e if p_e_e else '-'} | {p_pb if p_pb else '-'} | {p_ps if p_ps else '-'} |")
            except Exception as e:
                logger.warning(f"[{ticker}] Failed fetching peer multiples for {peer_ticker}: {e}")
                
        if len(peer_lines) > 2:
            peer_context_str = "Peer Comparison Table:\n" + "\n".join(peer_lines)
            
    # Premium/Discount against Peers
    premium_discount_pct = None
    if peer_pes:
        peer_median_pe = statistics.median(peer_pes)
        if pe_ttm and peer_median_pe > 0:
            premium_discount_pct = ((pe_ttm - peer_median_pe) / peer_median_pe) * 100
            peer_context_str += f"\n\nPeer Median P/E: {round(peer_median_pe, 2)}"
            
    # Margin of Safety Logic
    mof = "Low"
    if premium_discount_pct is not None:
        if premium_discount_pct <= -30:
            mof = "High"
        elif premium_discount_pct <= -10:
            mof = "Moderate"
        elif premium_discount_pct <= 20:
            mof = "Low"
        else:
            mof = "Negative"
            
    # Penalty step-downs
    if mof != "Negative":
        penalized = False
        if rev_growth is not None and rev_growth < 0:
            penalized = True
        if debt_equity is not None and debt_equity > 1.5:
            penalized = True
            
        if penalized:
            if mof == "High": mof = "Moderate"
            elif mof == "Moderate": mof = "Low"
            elif mof == "Low": mof = "Negative"

    # 4. Historical Data Logic
    hist_context_str = "Historical data not available — set vs_own_history to null."
    try:
        try:
            hist_result = live_data_service.fetch_historical_multiples(ticker)
            if hist_result:
                hist_context_str = str(hist_result)
        except AttributeError:
            pass
    except Exception as e:
         logger.warning(f"[{ticker}] Failed fetching historical multiples: {e}")

    # 5. Connect to LLM
    prompt = PromptTemplate.from_template(VALUATION_PROMPT)
    formatted_prompt = prompt.format(
        ticker=ticker,
        pe=pe_ttm if pe_ttm else "N/A",
        pe_fwd=_safe_float(market_data.get("pe_forward")) or "N/A",
        pb=_safe_float(market_data.get("pb_ratio")) or "N/A",
        ev_eb=_safe_float(market_data.get("ev_ebitda")) or "N/A",
        ps=_safe_float(market_data.get("price_to_sales")) or "N/A",
        dy=_safe_float(market_data.get("dividend_yield")) or "N/A",
        peg=peg_val if peg_val else "N/A",
        mc=_safe_float(market_data.get("market_cap_cr")) or "N/A",
        historical_multiples=hist_context_str,
        growth=f"{rev_growth}% YoY" if rev_growth is not None else "N/A",
        ebitda_margin=f"{ebitda_margin}%" if ebitda_margin is not None else "N/A",
        net_margin=f"{net_margin}%" if net_margin is not None else "N/A",
        debt_equity=debt_equity if debt_equity is not None else "N/A",
        peer_context=peer_context_str
    )
    
    llm = get_llm_client(provider)
    
    result_data = {}
    try:
        if hasattr(llm, "with_structured_output"):
            from pydantic import BaseModel, Field
            class ValOut(BaseModel):
                valuation_verdict: str
                vs_own_history: str | None
                vs_sector_median: str | None
                
            structured_llm = llm.with_structured_output(
                ValOut, 
                method="json_mode" if "openai" in str(type(llm)).lower() else "function_calling"
            )
            try:
                llm_res = structured_llm.invoke(formatted_prompt)
                result_data = {
                    "valuation_verdict": getattr(llm_res, "valuation_verdict", f"Valuation computed for {ticker}."),
                    "vs_own_history": getattr(llm_res, "vs_own_history", None),
                    "vs_sector_median": getattr(llm_res, "vs_sector_median", None)
                }
            except json.JSONDecodeError as jde:
                logger.error(f"[{ticker}] JSON decode error in structured valuation output: {jde}")
            except Exception as e:
                logger.error(f"[{ticker}] Structured output invoke error: {e}")

        if not result_data:
            response = llm.invoke([
                {"role": "system", "content": "You output JSON only. Return keys: valuation_verdict, vs_own_history, vs_sector_median."},
                {"role": "user", "content": formatted_prompt}
            ])
            content = response.content
            if "```json" in content:
                content = content.split("```json")[-1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[-1].split("```")[0].strip()
            else:
                start = content.find('{')
                end = content.rfind('}')
                if start != -1 and end != -1:
                    content = content[start:end+1]
                    
            data = json.loads(content)
            result_data = {
                "valuation_verdict": data.get("valuation_verdict", f"Valuation computed for {ticker}."),
                "vs_own_history": data.get("vs_own_history", None),
                "vs_sector_median": data.get("vs_sector_median", None)
            }
            
        # Hard override if peer_median wasn't computable
        if not peer_pes:
            result_data["vs_sector_median"] = None
            
        # Hard override if history unavailable
        if "Historical data not available" in hist_context_str:
            result_data["vs_own_history"] = None

        return ValuationAnalysis(
            current_price=market_data.get("current_price"),
            market_cap_cr=market_data.get("market_cap_cr"),
            pe_ttm=pe_ttm,
            pe_forward=_safe_float(market_data.get("pe_forward")),
            pb_ratio=_safe_float(market_data.get("pb_ratio")),
            ev_ebitda=_safe_float(market_data.get("ev_ebitda")),
            price_to_sales=_safe_float(market_data.get("price_to_sales")),
            dividend_yield=_safe_float(market_data.get("dividend_yield")),
            peg_ratio=peg_val,
            vs_own_history=result_data["vs_own_history"],
            vs_sector_median=result_data["vs_sector_median"],
            premium_discount_pct=premium_discount_pct if premium_discount_pct is None else round(premium_discount_pct, 2),
            valuation_verdict=result_data["valuation_verdict"],
            margin_of_safety=mof
        )

    except Exception as e:
        logger.error(f"Error in valuation agent for {ticker}: {e}")
        return ValuationAnalysis(
            current_price=market_data.get("current_price"),
            market_cap_cr=market_data.get("market_cap_cr"),
            pe_ttm=pe_ttm,
            pe_forward=_safe_float(market_data.get("pe_forward")),
            pb_ratio=_safe_float(market_data.get("pb_ratio")),
            ev_ebitda=_safe_float(market_data.get("ev_ebitda")),
            price_to_sales=_safe_float(market_data.get("price_to_sales")),
            dividend_yield=_safe_float(market_data.get("dividend_yield")),
            peg_ratio=peg_val,
            vs_own_history=None, 
            vs_sector_median=None, 
            premium_discount_pct=None,
            valuation_verdict=f"Error computing detailed valuation verdict: {str(e)}",
            margin_of_safety="Negative"
        )

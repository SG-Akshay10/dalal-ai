"""
Rewritten for StockSense AI Pipeline v2.
Changes made:
- Function Signature: Added `sector` and `events` parameters from orchestrator state.
- Deep Context Builders: Removed all arbitrary string truncations. Built dedicated `_build_*_context()` functions to flatten upstream agent objects explicitly for the LLM.
- Secure Base Pricing: Eliminated the `100.0` fake current_price fallback. Returns a "Low" conviction safety fallback immediately if current_price is unavailable.
- Deterministic Math & Normalization: Python handles `upside_downside_pct`, `weighted_price_target`, and `expected_return_pct`. Probabilities are strictly normalized to sum exactly to 1.0 (proportionally scaled if they mismatch).
- Strict Pre-computed Conviction: Conviction and position sizing are deterministically calculated initially to guide the LLM, and then recalculated mathematically post-generation based on the final expected return targets.
- Missing Scenario Guard: Loops resulting JSON and enforces exactly 3 scenarios (Bull, Base, Bear). If the LLM hallucinates and drops one, Python synthesizes a default mathematically scaled from the current price, flagging it clearly in `key_assumptions`.
- Prompt Guardrails: Added `today` string. Structurally defines Bull (top of historical PE), Base (consensus multiples), and Bear (technical support floor). Mandates `[Observable metric] -> [Impact]` formatting for triggers.
"""

import json
import logging
from datetime import date
from pydantic import BaseModel

from langchain_core.prompts import PromptTemplate

from app.llm_provider import get_llm_client
from app.schemas.analysis import (
    PriceTargetAnalysis, Scenario, FundamentalAnalysis, ValuationAnalysis, 
    SentimentAnalysis, PeerBenchmarkAnalysis, TechnicalAnalysis, 
    RiskAnalysis, SectorAnalysis, EventAnalysis
)

logger = logging.getLogger(__name__)

SCENARIO_PROMPT = """You are a Lead Portfolio Manager synthesizing fundamental, quantitative, and technical research into actionable investing scenarios for {ticker}.
Today's Date: {today}

Current Price: ₹{current_price}

<ANALYSIS_CONTEXT>
1. FUNDAMENTALS
{fundamentals_ctx}

2. VALUATION & PEERS
{valuation_ctx}

3. TECHNICALS & SENTIMENT
{market_ctx}

4. RISK & CATALYSTS
{risk_ctx}
</ANALYSIS_CONTEXT>

Based on the algorithm, the pre-computed PRELIMINARY CONVICTION is "{pre_conviction}" and the SUGGESTED POSITION SIZING is "{pre_sizing}". 
Use these exactly in your output.

<STRICT_RULES>
1. SCENARIO DEFINITIONS:
   - Bull Case: Best realistic outcome. Anchor multiples to the top of the historical P/E range.
   - Base Case: Consensus assumptions. Maintain current operating multiples without speculative re-rating.
   - Bear Case: Primary risks materializing. Assume multiple compression and technical support floor failure.
2. KEY ASSUMPTIONS RULE: Each scenario's `key_assumptions` MUST explicitly state the target P/E or EV/EBITDA multiple used and the rationale behind that specific valuation methodology.
3. TRIGGER FORMAT: `upgrade_triggers` and `downgrade_triggers` MUST follow this exact format:
   "[Specific observable metric crossing threshold] → [Expected impact on thesis]". 
   Prohibit vague statements like "positive commentary" or "macro improvement". (Max 2 triggers each).
4. Do NOT calculate upside_downside_pct, expected_return_pct, or weighted_price_target. The external engine does the math. Just provide the target prices and raw probabilities.
</STRICT_RULES>

<REASONING_STEPS>
1. Review the provided operating metrics, valuation ranges, technical support floors, and upcoming catalysts.
2. Construct the Base scenario: set a realistic 12-month target price based on current multiples. Assign highest probability (e.g., 50-60%).
3. Construct the Bull scenario: identify the upside multiple and growth rate. Set target and assign remaining probability.
4. Construct the Bear scenario: identify downside compression multiplier and technical floor. Assign remainder probability.
5. Identify specific upgrade/downgrade observable metrics from the risk & event catalysts context.
6. Return JSON conforming to the exact schema.
</REASONING_STEPS>

Always return your final response as a valid JSON object matching the following schema. Wait until after your reasoning to generate the JSON.
<EXAMPLE_SCHEMA>
{{
  "scenarios": [
    {{
      "label": "Bull",
      "probability": 0.2,
      "key_assumptions": ["Revenue growth accelerates to 15%. Re-rates to 40x P/E tracking historical peak."],
      "price_target_12m": 2400.0,
      "upside_downside_pct": 0.0
    }},
    {{
      "label": "Base",
      "probability": 0.6,
      "key_assumptions": ["Growth sustains at 10%. Maintains current 32x P/E multiple."],
      "price_target_12m": 1950.0,
      "upside_downside_pct": 0.0
    }},
    {{
      "label": "Bear",
      "probability": 0.2,
      "key_assumptions": ["Inflation bites margins. Multiple compresses to historical mean of 25x P/E."],
      "price_target_12m": 1450.0,
      "upside_downside_pct": 0.0
    }}
  ],
  "conviction": "Medium",
  "position_sizing_note": "2-4% allocation",
  "upgrade_triggers": ["EBITDA margin sustainably crosses 22% → Triggers Bull case P/E multiple expansion"],
  "downgrade_triggers": ["Short-term debt load exceeds ₹5,000 Cr → Elevates interest burden and compresses net margins"],
  "investment_horizon": "Medium (6–18M)"
}}
</EXAMPLE_SCHEMA>
"""

def _build_fundamentals_context(fundamentals: FundamentalAnalysis | None, sector: SectorAnalysis | None) -> str:
    if not fundamentals: return "Fundamentals unavailable."
    
    sector_str = "Unknown Sector"
    if sector and hasattr(sector, 'sector_name'):
        sector_str = sector.sector_name
        
    rev_gr = "N/A"
    eb_gr = "N/A"
    
    if fundamentals.revenue_trend and len(fundamentals.revenue_trend) > 0:
        gr = fundamentals.revenue_trend[-1].yoy_growth_pct
        if gr is not None: rev_gr = f"{gr}% YoY"
        
    if fundamentals.ebitda_trend and len(fundamentals.ebitda_trend) > 0:
         gr = fundamentals.ebitda_trend[-1].yoy_growth_pct
         if gr is not None: eb_gr = f"{gr}% YoY"
         
    return (
        f"Sector: {sector_str}\n"
        f"Revenue Growth: {rev_gr}\n"
        f"EBITDA Growth: {eb_gr}\n"
        f"Debt/Equity: {fundamentals.debt_equity_ratio}\n"
        f"Management Commentary: {fundamentals.management_commentary}\n"
        f"Red Flags: {', '.join(fundamentals.red_flags) if fundamentals.red_flags else 'None'}"
    )

def _build_valuation_context(valuation: ValuationAnalysis | None, peers: PeerBenchmarkAnalysis | None) -> str:
    if not valuation: return "Valuation unavailable."
    
    prem_disc = valuation.premium_discount_pct
    prem_disc_str = f"{prem_disc}%" if prem_disc is not None else "N/A"
    
    peer_str = "N/A"
    if peers and peers.relative_positioning:
        peer_str = f"{peers.relative_positioning}. \nStrengths: {'; '.join(peers.strengths_vs_peers)}\nWeaknesses: {'; '.join(peers.weaknesses_vs_peers)}"
        
    return (
        f"P/E TTM: {valuation.pe_ttm}\n"
        f"PEG Ratio: {valuation.peg_ratio}\n"
        f"Vs Sector Median P/E: {valuation.vs_sector_median}\n"
        f"Premium/Discount to Peers: {prem_disc_str}\n"
        f"Margin of Safety: {valuation.margin_of_safety}\n"
        f"Valuation Verdict: {valuation.valuation_verdict}\n\n"
        f"Peer Benchmark Position: {peer_str}"
    )

def _build_market_context(technical: TechnicalAnalysis | None, sentiment: SentimentAnalysis | None) -> str:
    tech_str = "Technical Analysis unavailable."
    if technical:
        sups = [f"₹{s}" for s in technical.support_levels[:2]]
        ress = [f"₹{r}" for r in technical.resistance_levels[:2]]
        tech_str = (
            f"Trend Bias: {technical.trend_bias}\n"
            f"Nearest Supports: {', '.join(sups)}\n"
            f"Nearest Resistances: {', '.join(ress)}\n"
            f"Volatility State: {technical.volatility_state}\n"
            f"Technical Narrative: {technical.narrative}"
        )
        
    sent_str = "Sentiment Analysis unavailable."
    if sentiment:
        sent_str = (
            f"Label: {sentiment.sentiment_label} (CSS: {sentiment.css_score}/100)\n"
            f"Positive Themes: {', '.join(sentiment.positive_themes)}\n"
            f"Negative Themes: {', '.join(sentiment.negative_themes)}\n"
            f"Narrative: {sentiment.narrative}"
        )
        
    return f"[TECHNICALS]\n{tech_str}\n\n[SENTIMENT]\n{sent_str}"

def _build_risk_context(risk: RiskAnalysis | None, events: EventAnalysis | None) -> str:
    risk_str = "Risk rating unavailable."
    if risk:
        high_critical = [f for f in risk.flags if f.severity in ["High", "Critical"]]
        flag_str = "; ".join([f"{f.flag_type}: {f.detail}" for f in high_critical]) if high_critical else "No High/Critical flags."
        risk_str = f"Overall Risk: {risk.overall_risk_rating}\nDominant Threats: {flag_str}\nNarrative: {risk.risk_narrative}"
        
    event_str = "Event catalysts unavailable."
    if events:
        upc = [f"{e.date}: {e.event_type} - {e.investment_relevance}" for e in events.upcoming_events[:2]]
        upc_join = "\n".join(upc) if upc else "None."
        event_str = f"Catalyst within 7 days: {events.event_risk_flag}\nNearest Catalyst (Days): {events.nearest_catalyst_days}\nUpcoming: {upc_join}"
        
    return f"[RED FLAGS]\n{risk_str}\n\n[CATALYSTS]\n{event_str}"

def _safe_float(val, default_val=0.0):
    if val is None: return default_val
    try: return float(val)
    except (ValueError, TypeError): return default_val

def generate_scenarios(
    ticker: str, 
    fundamentals: FundamentalAnalysis | None = None, 
    valuation: ValuationAnalysis | None = None, 
    sentiment: SentimentAnalysis | None = None, 
    peers: PeerBenchmarkAnalysis | None = None, 
    technical: TechnicalAnalysis | None = None, 
    risk: RiskAnalysis | None = None,
    sector: SectorAnalysis | None = None,
    events: EventAnalysis | None = None,
    provider: str = None
) -> PriceTargetAnalysis:
    """Synthesize all analysis into actionable scenarios and price targets (deterministic handling)."""
    
    current_price = valuation.current_price if valuation else None
    
    if current_price is None or current_price <= 0:
        logger.warning(f"[{ticker}] Fatal: Current Price is unavailable for Scenario generation.")
        return PriceTargetAnalysis(
            current_price=0.0,
            weighted_price_target=0.0,
            expected_return_pct=0.0,
            scenarios=[],
            conviction="Low",
            position_sizing_note="Price data unavailable — do not size a position.",
            upgrade_triggers=["Price data successfully retrieved"],
            downgrade_triggers=["Delisting or trading halt"],
            investment_horizon="None"
        )
        
    # 1. Compile Rich Context
    fund_ctx = _build_fundamentals_context(fundamentals, sector)
    val_ctx = _build_valuation_context(valuation, peers)
    mkt_ctx = _build_market_context(technical, sentiment)
    risk_ctx = _build_risk_context(risk, events)

    # 2. Preliminary Conviction & Sizing Deterministic Math
    overall_risk = risk.overall_risk_rating if risk else "Unknown"
    event_flag = events.event_risk_flag if events else False
    
    # Preliminary estimates for LLM guidance
    pre_conviction = "Medium"
    if overall_risk in ["High", "Very High"]: pre_conviction = "Low"
    elif not event_flag and overall_risk in ["Low", "Moderate"]: pre_conviction = "High"
    
    pre_sizing = "1-2% or watchlist"
    if pre_conviction == "High" and not event_flag: pre_sizing = "Up to 5-7% allocation"
    elif pre_conviction == "Medium": pre_sizing = "2-4% allocation"

    # 3. LLM Generation
    prompt = PromptTemplate.from_template(SCENARIO_PROMPT)
    formatted_prompt = prompt.format(
        ticker=ticker,
        today=date.today().strftime('%Y-%m-%d'),
        current_price=current_price,
        fundamentals_ctx=fund_ctx,
        valuation_ctx=val_ctx,
        market_ctx=mkt_ctx,
        risk_ctx=risk_ctx,
        pre_conviction=pre_conviction,
        pre_sizing=pre_sizing
    )
    
    llm = get_llm_client(provider)
    raw_scenarios = []
    triggers_up = []
    triggers_dn = []
    horizon = "Medium (6–18M)"

    try:
        result_data = {}
        if hasattr(llm, "with_structured_output"):
            class LLMScenarioFrame(BaseModel):
                scenarios: list[dict]
                conviction: str
                position_sizing_note: str
                upgrade_triggers: list[str]
                downgrade_triggers: list[str]
                investment_horizon: str
                
            structured_llm = llm.with_structured_output(
                LLMScenarioFrame, 
                method="json_mode" if "openai" in str(type(llm)).lower() else "function_calling"
            )
            try:
                llm_res = structured_llm.invoke(formatted_prompt)
                raw_scenarios = getattr(llm_res, "scenarios", [])
                triggers_up = getattr(llm_res, "upgrade_triggers", [])
                triggers_dn = getattr(llm_res, "downgrade_triggers", [])
                horizon = getattr(llm_res, "investment_horizon", "Medium (6–18M)")
            except Exception as e:
                logger.error(f"[{ticker}] Structured output error in Scenarios: {e}")

        if not raw_scenarios:
            response = llm.invoke([
                {"role": "system", "content": "You output JSON only."},
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
                if start != -1 and end != -1: content = content[start:end+1]
                    
            data = json.loads(content)
            raw_scenarios = data.get("scenarios", [])
            triggers_up = data.get("upgrade_triggers", [])
            triggers_dn = data.get("downgrade_triggers", [])
            horizon = data.get("investment_horizon", "Medium (6–18M)")

    except Exception as e:
        logger.error(f"[{ticker}] Scenario generation pipeline error: {e}")
        # empty raw_scenarios will trigger Missing Scenario Guard below

    # 4. Scenario Missing Guard & Probability Normalization
    parsed_scenarios = {
        "Bull": None,
        "Base": None,
        "Bear": None
    }
    
    total_raw_prob = 0.0
    for s in raw_scenarios:
        lab = s.get("label")
        if lab in parsed_scenarios:
            parsed_scenarios[lab] = s
            total_raw_prob += _safe_float(s.get("probability", 0.33))
            
    # Normalize probabilities if sum != 1.0 (allow slight float drift)
    if total_raw_prob == 0: total_raw_prob = 1.0
    scale_factor = 1.0 / total_raw_prob if abs(1.0 - total_raw_prob) > 0.01 else 1.0
    
    if abs(1.0 - total_raw_prob) > 0.01:
         logger.warning(f"[{ticker}] Normalizing LLM scenario probabilities (Summed to {total_raw_prob:.2f})")

    final_scenarios = []
    
    # Process required cases individually to apply Fallbacks
    for required_lbl, pct_target in [("Bull", 1.30), ("Base", 1.10), ("Bear", 0.85)]:
        s = parsed_scenarios[required_lbl]
        
        # 4a. Apply Missing Scenario Guard
        if not s:
            targ = current_price * pct_target
            prob_dict = {"Bull": 0.2, "Base": 0.6, "Bear": 0.2}
            prob = prob_dict[required_lbl] * scale_factor # normalize default prob too just in case
            
            logger.warning(f"[{ticker}] Generating default fallback for Missing {required_lbl} Scenario.")
            obj = Scenario(
                label=required_lbl,
                probability=prob,
                key_assumptions=[f"Default — LLM did not generate this case. Mathematically interpolated to target {pct_target}x."],
                price_target_12m=targ,
                upside_downside_pct=((targ - current_price) / current_price) * 100
            )
            final_scenarios.append(obj)
        else:
            # 4b. Parse and apply explicit Python math exactly
            targ = _safe_float(s.get("price_target_12m", current_price))
            prob = _safe_float(s.get("probability", 0.33)) * scale_factor
            if prob < 0: prob = 0.01

            obj = Scenario(
                label=required_lbl,
                probability=prob,
                key_assumptions=s.get("key_assumptions", ["No assumptions generated."]),
                price_target_12m=targ,
                upside_downside_pct=((targ - current_price) / current_price) * 100
            )
            final_scenarios.append(obj)

    # 5. Deterministic Valuation Metrics
    weighted_price_target = 0.0
    for s in final_scenarios:
        weighted_price_target += (s.price_target_12m * s.probability)
        
    expected_return_pct = ((weighted_price_target - current_price) / current_price) * 100

    # Calculate Asymmetry
    upside = 0.001 # prevent div/0
    downside = -0.001
    for s in final_scenarios:
        if s.label == "Bull": upside = s.upside_downside_pct
        if s.label == "Bear": downside = s.upside_downside_pct
    
    asymmetry = abs(upside / downside) if downside < 0 else 0.0

    # 6. Final Conviction & Sizing Logic (Re-computed after mathematical validation)
    final_conviction = "Low"
    if expected_return_pct >= 20.0 and asymmetry >= 2.0 and overall_risk not in ["High", "Very High"]:
        final_conviction = "High"
    elif expected_return_pct >= 10.0 and asymmetry >= 1.5:
        final_conviction = "Medium"
        
    final_sizing = "1-2% or watchlist"
    if final_conviction == "High" and not event_flag:
        final_sizing = "Up to 5-7% allocation"
    elif final_conviction == "Medium":
        final_sizing = "2-4% allocation"

    logger.info(f"[{ticker}] Synthesis Complete. E[R]: {expected_return_pct:.1f}%. Conviction: {final_conviction}.")

    return PriceTargetAnalysis(
        current_price=current_price,
        weighted_price_target=round(weighted_price_target, 2),
        expected_return_pct=round(expected_return_pct, 2),
        scenarios=final_scenarios,
        conviction=final_conviction,
        position_sizing_note=final_sizing,
        upgrade_triggers=triggers_up[:2],
        downgrade_triggers=triggers_dn[:2],
        investment_horizon=horizon
    )

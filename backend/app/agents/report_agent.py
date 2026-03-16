"""
Rewritten for StockSense AI Pipeline v2.
Changes made:
- Persona Change: Switched from "institutional-grade" to a retail investor persona (short sentences, plain English, no jargon).
- Context Builders: Removed `model_dump_json(indent=2)`. Wrote explicit `_ctx_*` builder functions to extract precise context (added freshness warnings for Fundamentals, and 🔴/🟡 icons for Risk severities).
- Report Headers: Injects `date.today()`, `conviction`, `weighted_price_target`, and a mapped `Action Verdict` (✅ Accumulate | ⚖ Hold | ❌ Avoid).
- Section Titles: Modernized to retail-friendly headers ("The Big Picture", "Is the Trend Your Friend?", etc).
- Writing Rules: Added rigid Prompt Constraints forcing quantitative anchors, 3-6 sentence targets per section, mandatory data gap warnings, and strict section-level formatting (like the Revenue table in Fundamentals, and the specific Buy/Resistance output in Technicals).
- Execution & Safety: Switched to Message-List LLM Invoke. Added a sub-500 character sanity check to warn about LLM response truncation. Inserted the required legal disclaimer via the prompt template boundary.
"""

import logging
from datetime import date
from langchain_core.prompts import PromptTemplate

from app.llm_provider import get_llm_client
from app.schemas.analysis import (
    CompetitorAnalysis, FundamentalAnalysis, SectorAnalysis, SentimentAnalysis,
    TechnicalAnalysis, ValuationAnalysis, RiskAnalysis, EventAnalysis,
    PeerBenchmarkAnalysis, PriceTargetAnalysis
)

logger = logging.getLogger(__name__)

REPORT_PROMPT = """You are a Lead Equity Research Analyst writing directly for retail investors. 
Synthesize the insights from the provided analysis into a clear, professional stock research report for {ticker}.
Write in plain, persuasive language — use short sentences, avoid financial jargon, and always put the verdict first.

Today's Date: {today}

<WRITING_RULES>
1. NUMBER ANCHORING: Anchor every claim to a specific number from the context (never say "strong growth" without the actual percentage).
2. TONE: Plain language — short sentences, no jargon, verdict first.
3. DATA GAPS: If the context for a section indicates the data is "Unavailable", you MUST write exactly one sentence stating the data is unavailable. Do not invent data and do not silently omit the section.
4. LENGTH: Each section must contain 3-6 sentences of prose, plus tables or bullet points where structured data is available.

SECTION-SPECIFIC INSTRUCTIONS:
- Section 3 (Fundamental Analysis): You MUST include a standard quarterly tracking table formatted exactly as: `| Quarter | Revenue | YoY% | EBITDA | PAT |`. State management guidance verbatim if present in context.
- Section 4 (Is the Trend Your Friend?): You MUST output the specific technical zones exactly formatted as "Buy zone/Support: ₹X — ₹Y" and "Target/Resistance: ₹X — ₹Y" as explicit named items.
- Section 5 (Is It Cheap or Expensive?): You MUST include the peer comparison table formatting provided in the context. Explain the premium or discount in plain English. Explicitly explain what the Margin of Safety label means for a buyer today.
- Section 8 (Calendar Check): If the context flags an event risk (True), you MUST open the section with exactly: "⚠ HIGH PRIORITY: A significant catalyst is within 7 days. Review before acting."
- Section 9 (Risks & Red Flags): For every High or Critical risk flag provided, you must translate it into a plain-English investor implication (explain exactly what this could mean for the stock price).
</WRITING_RULES>

<ANALYSIS_CONTEXT>
{context}
</ANALYSIS_CONTEXT>

Format Requirements:
-----------------------
# {ticker} Equity Research Report
**Date:** {today} | **Conviction:** {conviction} | **Target:** ₹{target} ({expected_return}%) | **Verdict:** {action_verdict}

## 1. The Big Picture
[Write a strong executive summary blending fundamentals, technicals, and valuation. State the clear bottom-line verdict for retail investors.]

## 2. Price Targets & Scenarios
Extract Bull/Base/Bear scenarios from the context.
- **Bull:** ₹[Target] — [Key triggers summary]
- **Base:** ₹[Target] — [Key triggers summary]
- **Bear:** ₹[Target] — [Key triggers summary]

## 3. Fundamental Analysis
[Summarize revenue, margin, and profitability. Highlight FCF/debt. Include Quarterly Table.]

## 4. Is the Trend Your Friend?
[Summarize trend bias, RSI, and MACD. Output exact Buy zone/Support and Target/Resistance items.]

## 5. Is It Cheap or Expensive?
[Include peer table. Explain premium/discount. Explain Margin of Safety in plain English.]

## 6. Sector & Macro Tailwinds
[Summarize sector growth, index performance, and regulatory context.]

## 7. What Does the Market Think?
[Highlight sentiment distribution, news trajectory, and alternatives.]

## 8. Calendar Check
[List nearest catalyst days and upcoming catalysts. Include 7-day warning if applicable.]

## 9. Risks & Red Flags
[Detail red flags and translate High/Critical implications strictly.]

---
*This report was generated by StockSense AI. For informational purposes only. Not financial advice. Always do your own research before investing.*
"""

def _ctx_fundamentals(f: FundamentalAnalysis | None) -> str:
    if not f: return "Fundamentals: Unavailable"
    
    parts = ["[FUNDAMENTALS]"]
    if hasattr(f, "data_freshness_days"):
        if f.data_freshness_days == -1 or f.data_freshness_days > 90:
            parts.append(f"⚠ WARNING: Financial data may be stale ({f.data_freshness_days} days old).")
            
    # Attempt to format a raw table string for the LLM to use
    parts.append("Quarterly Trend Data:")
    for i in range(len(f.revenue_trend)):
        rev = f.revenue_trend[i]
        eb = f.ebitda_trend[i] if f.ebitda_trend and i < len(f.ebitda_trend) else None
        pat = f.pat_trend[i] if f.pat_trend and i < len(f.pat_trend) else None
        
        r_val = rev.value if rev else "N/A"
        r_gw = f"{rev.yoy_growth_pct}%" if rev and rev.yoy_growth_pct is not None else "N/A"
        e_val = eb.value if eb else "N/A"
        p_val = pat.value if pat else "N/A"
        
        period = rev.period if rev else f"Q{i}"
        parts.append(f"- {period}: Rev ₹{r_val}Cr (YoY: {r_gw}), EBITDA ₹{e_val}Cr, PAT ₹{p_val}Cr")
        
    parts.append(f"Debt/Equity: {f.debt_equity_ratio}")
    parts.append(f"Net Margin: {f.net_margin}%")
    parts.append(f"FCF Commentary: {f.fcf_commentary}")
    parts.append(f"Management Guidance: {f.management_commentary}")
    return "\n".join(parts)

def _ctx_valuation(v: ValuationAnalysis | None) -> str:
    if not v: return "Valuation: Unavailable"
    return (
        f"[VALUATION]\n"
        f"Current Price: ₹{v.current_price}\n"
        f"P/E TTM: {v.pe_ttm}\n"
        f"EV/EBITDA: {v.ev_ebitda}\n"
        f"P/B: {v.price_to_book}\n"
        f"PEG Ratio: {v.peg_ratio}\n"
        f"Sector Median P/E: {v.vs_sector_median}\n"
        f"Premium/Discount to Peers: {v.premium_discount_pct}%\n"
        f"Margin of Safety: {v.margin_of_safety}\n"
        f"Verdict: {v.valuation_verdict}"
    )

def _ctx_technical(t: TechnicalAnalysis | None) -> str:
    if not t: return "Technical: Unavailable"
    sups = [f"₹{x}" for x in t.support_levels[:2]]
    ress = [f"₹{x}" for x in t.resistance_levels[:2]]
    return (
        f"[TECHNICAL]\n"
        f"Trend Bias: {t.trend_bias}\n"
        f"RSI 14: {t.rsi_14}\n"
        f"MACD Signal: {t.macd_signal}\n"
        f"Volume Trend: {t.volume_trend}\n"
        f"Nearest Support Zones: {', '.join(sups)}\n"
        f"Nearest Resistance Targets: {', '.join(ress)}\n"
        f"Volatility: {t.volatility_state}\n"
        f"Narrative: {t.narrative}"
    )

def _ctx_peers(p: PeerBenchmarkAnalysis | None) -> str:
    if not p: return "Peer Benchmark: Unavailable"
    # Format the peer rows properly
    rows_str = []
    for r in p.rows:
        rows_str.append(f"{r.ticker}: P/E {r.pe_ratio}, EV/EBT {r.ev_ebitda}, RevGr {r.revenue_growth_yoy}%, EBITDA% {r.ebitda_margin}%, ROE {r.roe}%, D/E {r.debt_equity}")
        
    return (
        f"[PEER BENCHMARK]\n"
        f"Positioning: {p.relative_positioning}\n"
        f"Strengths: {'; '.join(p.strengths_vs_peers)}\n"
        f"Weaknesses: {'; '.join(p.weaknesses_vs_peers)}\n"
        f"Peer Table Raw Data:\n" + "\n".join(rows_str) + "\n"
        f"Narrative: {p.narrative}"
    )

def _ctx_sector(s: SectorAnalysis | None) -> str:
    if not s: return "Sector: Unavailable"
    return (
        f"[SECTOR]\n"
        f"Name: {s.sector_name}\n"
        f"Growth Stage: {s.growth_stage}\n"
        f"Index YTD: {s.index_performance_ytd}%\n"
        f"FII Flow Trend: {s.fii_flow_trend}\n"
        f"Policy Shifts: {s.policy_regulatory_shifts}\n"
        f"Narrative: {s.sector_narrative}"
    )

def _ctx_sentiment(s: SentimentAnalysis | None) -> str:
    if not s: return "Sentiment: Unavailable"
    return (
        f"[SENTIMENT]\n"
        f"Label: {s.sentiment_label} (Score: {s.css_score}/100)\n"
        f"Pos/Neu/Neg Distribution: {s.positive_pct}% / {s.neutral_pct}% / {s.negative_pct}%\n"
        f"Positive Themes: {', '.join(s.positive_themes)}\n"
        f"Negative Themes: {', '.join(s.negative_themes)}\n"
        f"Narrative: {s.narrative}"
    )

def _ctx_risk(r: RiskAnalysis | None) -> str:
    if not r: return "Risk: Unavailable"
    flag_lines = []
    for f in r.flags:
        if f.severity in ["Critical", "High"]:
            flag_lines.append(f"🔴 {f.severity}: {f.flag_type} - {f.detail}")
        elif f.severity == "Medium":
            flag_lines.append(f"🟡 {f.severity}: {f.flag_type} - {f.detail}")
        # omit low for brevity
        
    return (
        f"[RISK & RED FLAGS]\n"
        f"Overall Rating: {r.overall_risk_rating}\n"
        f"Promoter Pledged: {r.promoter_pledge_pct}%\n"
        f"Key Flags:\n" + "\n".join(flag_lines) + "\n"
        f"Narrative: {r.risk_narrative}"
    )

def _ctx_events(e: EventAnalysis | None) -> str:
    if not e: return "Events: Unavailable"
    up_events = [f"{x.date}: {x.event_type} - {x.investment_relevance}" for x in e.upcoming_events[:3]]
    rec_events = [f"{x.date}: {x.event_type} - {x.investment_relevance}" for x in e.recent_events[:3]]
    
    return (
        f"[EVENTS]\n"
        f"Event Risk Flag (7 Days): {e.event_risk_flag}\n"
        f"Nearest Catalyst Days: {e.nearest_catalyst_days}\n"
        f"Upcoming:\n" + "\n".join(up_events) + "\n"
        f"Recent:\n" + "\n".join(rec_events)
    )

def _ctx_scenario(s: PriceTargetAnalysis | None) -> str:
    if not s: return "Scenarios: Unavailable"
    scen_str = []
    for sc in s.scenarios:
        scen_str.append(f"- {sc.label} (Prob {sc.probability}): ₹{sc.price_target_12m} ({sc.upside_downside_pct}%) - {', '.join(sc.key_assumptions)}")
        
    return (
        f"[SCENARIOS]\n"
        f"Conviction: {s.conviction}\n"
        f"Target: ₹{s.weighted_price_target} (E[r]: {s.expected_return_pct}%)\n"
        f"Horizon: {s.investment_horizon}\n"
        f"Position Sizing: {s.position_sizing_note}\n"
        f"Triggers Up: {', '.join(s.upgrade_triggers)}\n"
        f"Triggers Down: {', '.join(s.downgrade_triggers)}\n"
        f"Scenarios:\n" + "\n".join(scen_str)
    )

def generate_report(
    ticker: str,
    fundamentals: FundamentalAnalysis | None,
    sentiment: SentimentAnalysis | None,
    sector: SectorAnalysis | None,
    competitors: CompetitorAnalysis | None,
    technical: TechnicalAnalysis | None = None,
    valuation: ValuationAnalysis | None = None,
    risk: RiskAnalysis | None = None,
    events: EventAnalysis | None = None,
    peer_benchmark: PeerBenchmarkAnalysis | None = None,
    scenario: PriceTargetAnalysis | None = None,
    provider: str | None = None,
) -> str:
    """Consolidation agent creating the final V2 format Markdown report."""
    
    # Bundle contexts using custom builders
    context_parts = [
        _ctx_scenario(scenario),
        _ctx_fundamentals(fundamentals),
        _ctx_valuation(valuation),
        _ctx_technical(technical),
        _ctx_peers(peer_benchmark),
        _ctx_sector(sector),
        _ctx_sentiment(sentiment),
        _ctx_risk(risk),
        _ctx_events(events)
    ]
    
    context_blob = "\n\n".join(context_parts)
    
    if not context_blob.strip():
        logger.error(f"[{ticker}] Report context is entirely empty.")
        return f"# Error Generating Report for {ticker}\n\nNone of the sub-agents returned data."
        
    # Headers logic
    today_str = date.today().strftime("%d %B %Y")
    
    conviction = scenario.conviction if scenario else "Unknown"
    target = scenario.weighted_price_target if scenario else "N/A"
    ex_ret = scenario.expected_return_pct if scenario else "N/A"
    
    # Map Action Verdict
    action_verdict = "⚖ Hold"
    if conviction == "High" and ex_ret != "N/A" and ex_ret >= 15:
        action_verdict = "✅ Accumulate"
    elif ex_ret != "N/A" and ex_ret < 5:
        action_verdict = "❌ Avoid"
    elif conviction == "Low":
        action_verdict = "❌ Avoid"
    
    prompt = PromptTemplate.from_template(REPORT_PROMPT)
    formatted_prompt = prompt.format(
        ticker=ticker, 
        today=today_str,
        conviction=conviction,
        target=target,
        expected_return=ex_ret,
        action_verdict=action_verdict,
        context=context_blob
    )
    
    llm = get_llm_client(provider)
    
    try:
        # Switch to message-list format enforcing Markdown completeness
        messages = [
            {"role": "system", "content": "Write the complete research equity report in standard Markdown formatting. Do not truncate any section. Follow structure strictly."},
            {"role": "user", "content": formatted_prompt}
        ]
        
        response = llm.invoke(messages)
        content = response.content.strip()
        
        # Sanity Check
        if len(content) < 500:
            logger.warning(f"[{ticker}] LLM Report output seems artificially truncated (Length: {len(content)}).")
            
        return content
        
    except Exception as e:
        logger.error(f"[{ticker}] Final report generation failed: {e}")
        return f"# Error Generating Final Report for {ticker}\n\n{str(e)}"

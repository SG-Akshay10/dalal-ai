from langchain_core.prompts import PromptTemplate

from app.llm_provider import get_llm_client
from app.schemas.analysis import (
    CompetitorAnalysis,
    FundamentalAnalysis,
    SectorAnalysis,
    SentimentAnalysis,
    TechnicalAnalysis,
    ValuationAnalysis,
    RiskAnalysis,
    EventAnalysis,
    PeerBenchmarkAnalysis,
    PriceTargetAnalysis
)

REPORT_PROMPT = """You are a top-tier Lead Equity Research Analyst. 
Synthesize the insights from 10 different agent analyses into a deep, professional institutional-grade stock research report for {ticker}.
Write in clear, precise markdown format following the specific structure below. Be persuasive, distinct, and avoid using flowery or generic AI language. 

Provided Analysis Context:
-----------------------
{context}

Format Requirements:
-----------------------
# [Ticker] Equity Research Report
**Date:** [Current Date] | **Conviction:** [High/Medium/Low from Scenario] | **Target:** [Price Target from Scenario]

## 1. Executive Summary & Verdict
Write a strong executive summary blending fundamentals, technicals, and valuation. State the clear bottom-line verdict for investors.

## 2. Price Targets & Scenarios
Extract Bull/Base/Bear scenarios from the context.
- **Bull:** [Upside] [Key triggers]
- **Base:** [Upside] [Key triggers]
- **Bear:** [Downside] [Key triggers]

## 3. Fundamental Analysis
Summarize the revenue, margin, and profitability trends. Highlight FCF and debt.

## 4. Technical Setup
Summarize the trend bias, RSI, Support/Resistance and MACD.

## 5. Peer Benchmarking & Valuation
Summarize how it screens against peers (Premium/Discount). Highlight key multiples. State the valuation verdict.

## 6. Sector & Macro Tailwinds
Summarize sector growth, index performance, and regulatory/policy context.

## 7. Sentiment & Alternative Data
Highlight the FinBERT distribution, news trajectory, and social media themes.

## 8. Key Corporate Events
List the nearest catalyst days and important upcoming catalysts.

## 9. Risks & Red Flags
Detail any red flags (Promoter Pledge, Audit Issues, SEBI actions). State the overall risk rating.

---
Ensure the formatting uses standard Markdown headers, lists, and bold text for readability. No extra pleasantries.
"""

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
    
    # Bundle contexts to feed into the prompt
    context_parts = []
    
    if scenario:
        context_parts.append(f"Scenarios & Targets: {scenario.model_dump_json(indent=2)}")
    if fundamentals:
        context_parts.append(f"Fundamentals: {fundamentals.model_dump_json(indent=2)}")
    if valuation:
        context_parts.append(f"Valuation: {valuation.model_dump_json(indent=2)}")
    if technical:
        context_parts.append(f"Technical: {technical.model_dump_json(indent=2)}")
    if peer_benchmark:
        context_parts.append(f"Peer Benchmark: {peer_benchmark.model_dump_json(indent=2)}")
    if sector:
        context_parts.append(f"Sector: {sector.model_dump_json(indent=2)}")
    if sentiment:
        context_parts.append(f"Sentiment: {sentiment.model_dump_json(indent=2)}")
    if risk:
        context_parts.append(f"Risk: {risk.model_dump_json(indent=2)}")
    if events:
        context_parts.append(f"Events: {events.model_dump_json(indent=2)}")
        
    context_blob = "\n\n".join(context_parts)
    
    if not context_blob.strip():
        return f"# Error Generating Report for {ticker}\n\nNone of the sub-agents returned data."
    
    prompt = PromptTemplate.from_template(REPORT_PROMPT)
    formatted_prompt = prompt.format(ticker=ticker, context=context_blob)
    
    llm = get_llm_client(provider)
    
    # We don't force JSON here since we want a formatted Markdown string.
    try:
        response = llm.invoke(formatted_prompt)
        return response.content
    except Exception as e:
        return f"# Error Generating Final Report for {ticker}\n{str(e)}"

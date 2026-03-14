import json
import logging
from langchain_core.prompts import PromptTemplate

from app.llm_provider import get_llm_client
from app.schemas.analysis import PeerBenchmarkAnalysis, CompetitorAnalysis, PeerRow
from app.services.live_data import live_data_service

logger = logging.getLogger(__name__)

PEER_BENCHMARK_PROMPT = """You are an expert equity research analyst.
Examine the following peer comparison table for the target stock ({ticker}) and its closest competitors.

{table_data}

Write a concise narrative comparing the target against its peers. Focus on where it leads and where it lags.
Assign the Relative Positioning constraint to one of: "Market Leader", "In Line With Peers", "Lags Peers".

Always return your response as a valid JSON object matching the following schema:
{{
  "relative_positioning": "In Line With Peers",
  "strengths_vs_peers": ["...", "..."],
  "weaknesses_vs_peers": ["...", "..."],
  "narrative": "..."
}}
"""

def build_peer_benchmark(ticker: str, competitors: CompetitorAnalysis, provider: str = None) -> PeerBenchmarkAnalysis:
    """Build the peer benchmarking table and analysis."""
    
    rows = []
    
    # 1. Fetch data for target
    target_data = live_data_service.fetch_market_data(ticker)
    rows.append(PeerRow(
        ticker=ticker,
        name=ticker,
        revenue_growth_yoy=None, # In a fuller implementation, this comes from Fundamentals
        ebitda_margin=None,
        net_margin=None,
        roe=None,
        debt_equity=None,
        pe_ratio=target_data.get("pe_ttm"),
        ev_ebitda=target_data.get("ev_ebitda")
    ))
    
    # 2. Fetch data for peers
    if competitors and competitors.competitors:
        for comp in competitors.competitors:
            comp_data = live_data_service.fetch_market_data(comp.ticker)
            rows.append(PeerRow(
                ticker=comp.ticker,
                name=comp.name,
                revenue_growth_yoy=None,
                ebitda_margin=None,
                net_margin=None,
                roe=None,
                debt_equity=None,
                pe_ratio=comp_data.get("pe_ttm"),
                ev_ebitda=comp_data.get("ev_ebitda")
            ))
            
    # Format table data for LLM
    table_lines = ["Ticker | P/E TTM | EV/EBITDA"]
    for r in rows:
        pe = round(r.pe_ratio, 2) if r.pe_ratio else "N/A"
        ev = round(r.ev_ebitda, 2) if r.ev_ebitda else "N/A"
        table_lines.append(f"{r.ticker} | {pe} | {ev}")
        
    table_data = "\n".join(table_lines)
    
    prompt = PromptTemplate.from_template(PEER_BENCHMARK_PROMPT)
    formatted_prompt = prompt.format(ticker=ticker, table_data=table_data)
    
    llm = get_llm_client(provider)
    try:
        response = llm.invoke([
            {"role": "system", "content": "You output JSON only."},
            {"role": "user", "content": formatted_prompt}
        ])
        content = response.content.replace("```json", "").replace("```", "").strip()
        data = json.loads(content)
        
        return PeerBenchmarkAnalysis(
            rows=rows,
            relative_positioning=data.get("relative_positioning", "In Line With Peers"),
            strengths_vs_peers=data.get("strengths_vs_peers", []),
            weaknesses_vs_peers=data.get("weaknesses_vs_peers", []),
            narrative=data.get("narrative", f"Peer comparison complete for {ticker}.")
        )
    except Exception as e:
        logger.error(f"Error in Peer Benchmarking Agent for {ticker}: {e}")
        return PeerBenchmarkAnalysis(
            rows=rows,
            relative_positioning="In Line With Peers",
            strengths_vs_peers=[],
            weaknesses_vs_peers=[],
            narrative=f"Error generating peer narrative: {str(e)}"
        )

import logging
import time
from datetime import datetime, timezone
from typing import TypedDict, Any

from langgraph.graph import END, StateGraph

# Agent functions
from app.agents.fundamental_agent import extract_fundamentals
from app.agents.sentiment_agent import analyze_sentiment
from app.agents.competitor_agent import identify_competitors
from app.agents.sector_agent import analyze_sector
from app.agents.technical_agent import analyze_technical
from app.agents.valuation_agent import analyze_valuation
from app.agents.risk_redflag_agent import analyze_risk_and_redflags
from app.agents.event_detection_agent import analyze_events
from app.agents.peer_benchmark_agent import build_peer_benchmark
from app.agents.scenario_agent import generate_scenarios
from app.agents.report_agent import generate_report

# Schemas
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

# Phase 1 scrapers
from app.scrapers.document_fetcher import fetch_documents
from app.scrapers.news_scraper import fetch_news
from app.scrapers.social_listener import fetch_social

logger = logging.getLogger(__name__)

class ReportState(TypedDict, total=False):
    """The unified state passed around the LangGraph V2 DAG."""
    ticker: str
    company_name: str
    days: int | None

    # Phase 1 Data
    raw_documents: list
    raw_news: list
    raw_social: list

    # Phase 2 & 3 Agent Outputs
    fundamentals: FundamentalAnalysis | None
    sentiment: SentimentAnalysis | None
    competitors: CompetitorAnalysis | None
    sector: SectorAnalysis | None
    technical: TechnicalAnalysis | None
    valuation: ValuationAnalysis | None
    risk: RiskAnalysis | None
    events: EventAnalysis | None
    
    # Tier 2 outputs
    peer_benchmark: PeerBenchmarkAnalysis | None
    scenario: PriceTargetAnalysis | None

    # Final Output
    final_report_markdown: str
    errors: list[str]
    preferred_provider: str | None
    skipped_agents: list[str]
    report_generated_at: str | None

def get_fallback_analysis(schema_cls: Any, node_name: str) -> Any:
    """Helper to instantiate fallback empty schemas to prevent DAG crash."""
    try:
        if schema_cls == FundamentalAnalysis: return FundamentalAnalysis(revenue_trend=[], ebitda_trend=[], pat_trend=[], gross_margin=None, net_margin=None, fcf_commentary="", debt_equity_ratio=None, management_commentary="", red_flags=[f"Agent node '{node_name}' crashed — data unavailable."], data_freshness_days=0)
        if schema_cls == SentimentAnalysis: return SentimentAnalysis(composite_sentiment_score=0, label="Neutral", positive_themes=[], negative_themes=[], narrative=f"Agent node '{node_name}' crashed — data unavailable.", news_article_count=0, social_post_count=0, finbert_distribution={})
        if schema_cls == CompetitorAnalysis: return CompetitorAnalysis(competitors=[], primary_sector="Unknown", business_description="")
        if schema_cls == SectorAnalysis: return SectorAnalysis(sector_name="Unknown", growth_stage="Mature", index_performance_ytd=None, fii_flow_trend="Neutral", policy_tailwinds=[], policy_headwinds=[], regulatory_summary=f"Agent node '{node_name}' crashed — data unavailable.", sector_narrative=f"Agent node '{node_name}' crashed — data unavailable.")
        if schema_cls == TechnicalAnalysis: return TechnicalAnalysis(trend_bias="Insufficient Data", rsi_14=None, rsi_signal="Neutral", macd_signal="Neutral", bollinger_position="Mid", support_levels=[], resistance_levels=[], volume_trend="Neutral", narrative=f"Agent node '{node_name}' crashed — data unavailable.")
        if schema_cls == ValuationAnalysis: return ValuationAnalysis(current_price=None, market_cap_cr=None, pe_ttm=None, pe_forward=None, pb_ratio=None, ev_ebitda=None, price_to_sales=None, dividend_yield=None, peg_ratio=None, vs_own_history=None, vs_sector_median=None, premium_discount_pct=None, valuation_verdict=f"Agent node '{node_name}' crashed — data unavailable.", margin_of_safety="Low")
        if schema_cls == RiskAnalysis: return RiskAnalysis(flags=[], overall_risk_rating="Unknown", risk_narrative=f"Agent node '{node_name}' crashed — data unavailable.", promoter_pledge_pct=None)
        if schema_cls == EventAnalysis: return EventAnalysis(upcoming_events=[], recent_events=[], nearest_catalyst_days=None, event_risk_flag=False)
        if schema_cls == PeerBenchmarkAnalysis: return PeerBenchmarkAnalysis(rows=[], relative_positioning="In Line With Peers", strengths_vs_peers=[], weaknesses_vs_peers=[], narrative=f"Agent node '{node_name}' crashed — data unavailable.")
    except Exception:
        pass
    return None

def robust_agent_wrapper(state: ReportState, agent_func: callable, state_key: str, schema_cls: Any, **kwargs) -> ReportState:
    """Wraps an agent invocation to safely catch and mock output on failures."""
    if state_key in state.get("skipped_agents", []):
         logger.info(f"Skipping {state_key} as per supervisor logic.")
         return {state_key: get_fallback_analysis(schema_cls, state_key)}
         
    logger.info(f"→ Starting node '{state_key}' for {state['ticker']}")
    start_time = time.monotonic()
    
    try:
        result = agent_func(**kwargs)
        elapsed = time.monotonic() - start_time
        logger.info(f"✓ Node '{state_key}' completed in {elapsed:.2f}s")
        return {state_key: result}
    except Exception as e:
        elapsed = time.monotonic() - start_time
        logger.error(f"✗ Node '{state_key}' failed after {elapsed:.2f}s: {e}")
        
        errors = state.get("errors", [])
        errors.append(f"Node '{state_key}' failed: {e}")
        
        return {
            state_key: get_fallback_analysis(schema_cls, state_key),
            "errors": errors
        }

# 1. Scraping Node
def node_scrape_data(state: ReportState) -> ReportState:
    ticker = state["ticker"]
    company_name = state.get("company_name", ticker)
    logger.info(f"→ Starting node 'scrape' for {ticker}")
    start_time = time.monotonic()
    try:
        docs_days = state.get("days") or 90
        news_social_days = state.get("days") or 21
        docs = fetch_documents(ticker=ticker, days=docs_days)
        news = fetch_news(ticker=ticker, company_name=company_name, days=news_social_days)
        social = fetch_social(ticker=ticker, days=news_social_days)
        elapsed = time.monotonic() - start_time
        logger.info(f"✓ Node 'scrape' completed in {elapsed:.2f}s")
        return {"raw_documents": docs, "raw_news": news, "raw_social": social}
    except Exception as e:
        elapsed = time.monotonic() - start_time
        logger.error(f"✗ Node 'scrape' failed after {elapsed:.2f}s: {e}")
        errors = state.get("errors", [])
        errors.append(f"Scraping failed: {e}")
        return {
            "raw_documents": [], 
            "raw_news": [], 
            "raw_social": [],
            "errors": errors
        }

# 2. Tier 1 Nodes
def node_fundamental(state: ReportState):
    return robust_agent_wrapper(state, extract_fundamentals, "fundamentals", FundamentalAnalysis, ticker=state["ticker"], provider=state.get("preferred_provider"))

def node_sentiment(state: ReportState):
    return robust_agent_wrapper(state, analyze_sentiment, "sentiment", SentimentAnalysis, ticker=state["ticker"], news=state.get("raw_news", []), social=state.get("raw_social", []), provider=state.get("preferred_provider"))

def node_competitor(state: ReportState):
    if state.get("competitors"): return {"competitors": state["competitors"]}
    return robust_agent_wrapper(state, identify_competitors, "competitors", CompetitorAnalysis, ticker=state["ticker"], provider=state.get("preferred_provider"))

def node_sector(state: ReportState):
    return robust_agent_wrapper(state, analyze_sector, "sector", SectorAnalysis, ticker=state["ticker"], provider=state.get("preferred_provider"))

def node_technical(state: ReportState):
    return robust_agent_wrapper(state, analyze_technical, "technical", TechnicalAnalysis, ticker=state["ticker"], days=state.get("days") or 90, provider=state.get("preferred_provider"))

def node_valuation(state: ReportState):
    return robust_agent_wrapper(state, analyze_valuation, "valuation", ValuationAnalysis, ticker=state["ticker"], fundamentals=state.get("fundamentals"), competitors=state.get("competitors"), provider=state.get("preferred_provider"))

def node_risk(state: ReportState):
    return robust_agent_wrapper(state, analyze_risk_and_redflags, "risk", RiskAnalysis, ticker=state["ticker"], fundamentals=state.get("fundamentals"), news=state.get("raw_news", []), docs=state.get("raw_documents", []), provider=state.get("preferred_provider"))

def node_events(state: ReportState):
    return robust_agent_wrapper(state, analyze_events, "events", EventAnalysis, ticker=state["ticker"], news=state.get("raw_news", []), provider=state.get("preferred_provider"))

# 3. Tier 2 Nodes
def node_peer_benchmark(state: ReportState):
    return robust_agent_wrapper(state, build_peer_benchmark, "peer_benchmark", PeerBenchmarkAnalysis, ticker=state["ticker"], fundamentals=state.get("fundamentals"), company_name=state.get("company_name", state["ticker"]), competitors=state.get("competitors"), provider=state.get("preferred_provider"))

def node_scenario(state: ReportState):
    return robust_agent_wrapper(state, generate_scenarios, "scenario", PriceTargetAnalysis, 
             ticker=state["ticker"], fundamentals=state.get("fundamentals"), valuation=state.get("valuation"), 
             sentiment=state.get("sentiment"), peers=state.get("peer_benchmark"), technical=state.get("technical"), 
             risk=state.get("risk"), sector=state.get("sector"), events=state.get("events"), provider=state.get("preferred_provider"))

# 4. Report Node
def node_generate_report(state: ReportState) -> ReportState:
    logger.info(f"→ Starting node 'report' for {state['ticker']}")
    start_time = time.monotonic()
    
    report_generated_at = datetime.now(timezone.utc).isoformat()
    
    try:
        report = generate_report(
            ticker=state["ticker"],
            fundamentals=state.get("fundamentals"),
            sentiment=state.get("sentiment"),
            sector=state.get("sector"),
            competitors=state.get("competitors"),
            technical=state.get("technical"),
            valuation=state.get("valuation"),
            risk=state.get("risk"),
            events=state.get("events"),
            peer_benchmark=state.get("peer_benchmark"),
            scenario=state.get("scenario"),
            provider=state.get("preferred_provider")
        )
        
        errors = state.get("errors", [])
        if len(errors) >= 3:
            warning_header = "> ⚠ **WARNING:** Multiple analysis components failed during generation. This report may be incomplete.\n\n"
            report = warning_header + report
            
        elapsed = time.monotonic() - start_time
        logger.info(f"✓ Node 'report' completed in {elapsed:.2f}s")
        return {"final_report_markdown": report, "report_generated_at": report_generated_at}
    except Exception as e:
        elapsed = time.monotonic() - start_time
        logger.error(f"✗ Node 'report' failed after {elapsed:.2f}s: {e}")
        return {"final_report_markdown": f"# Error\n\nFailed to assemble final report: {e}", "report_generated_at": report_generated_at}


def build_orchestrator() -> StateGraph:
    """Build the V2 architecture LangGraph DAG."""
    workflow = StateGraph(ReportState)

    workflow.add_node("scrape", node_scrape_data)
    
    # Tier 1
    workflow.add_node("fundamental", node_fundamental)
    workflow.add_node("sentiment", node_sentiment)
    workflow.add_node("competitor", node_competitor)
    workflow.add_node("sector", node_sector)
    workflow.add_node("technical", node_technical)
    workflow.add_node("events", node_events)
    
    # Tier 1 deps (wait for scrape)
    for t1 in ["fundamental", "sentiment", "competitor", "sector", "technical", "events"]:
        workflow.add_edge("scrape", t1)
        
    # Tier 2 outputs feed downstream
    workflow.add_node("risk", node_risk)
    workflow.add_node("valuation", node_valuation)
    workflow.add_node("peer_benchmark", node_peer_benchmark)
    
    # new DAG dependencies
    workflow.add_edge("fundamental", "risk")
    workflow.add_edge("scrape", "risk") # needs docs/news
    
    workflow.add_edge("fundamental", "valuation")
    workflow.add_edge("competitor", "valuation")
    
    workflow.add_edge("fundamental", "peer_benchmark")
    workflow.add_edge("competitor", "peer_benchmark")
    
    workflow.add_node("scenario", node_scenario)
    
    # Scenario waits for critical variables ensuring they completed
    workflow.add_edge("peer_benchmark", "scenario")
    workflow.add_edge("fundamental", "scenario")
    workflow.add_edge("sentiment", "scenario")
    workflow.add_edge("technical", "scenario")
    workflow.add_edge("valuation", "scenario")
    workflow.add_edge("risk", "scenario")
    workflow.add_edge("sector", "scenario")
    workflow.add_edge("events", "scenario")
    
    # Report
    workflow.add_node("report", node_generate_report)
    workflow.add_edge("scenario", "report")
    workflow.add_edge("report", END)

    workflow.set_entry_point("scrape")
    
    app = workflow.compile()
    return app

def _determine_skipped_agents(ticker: str) -> list[str]:
    skipped = []
    if "BEES" in ticker.upper() or "ETF" in ticker.upper():
        skipped.extend(["competitors", "peer_benchmark"])
    return skipped

def run_pipeline(ticker: str, company_name: str = None, overridden_competitors=None, preferred_provider: str = None, days: int = None) -> str:
    app = build_orchestrator()
    skipped = _determine_skipped_agents(ticker)

    initial_state = {
        "ticker": ticker,
        "company_name": company_name or ticker,
        "days": days,
        "raw_documents": [],
        "raw_news": [],
        "raw_social": [],
        "fundamentals": None,
        "sentiment": None,
        "competitors": overridden_competitors,
        "sector": None,
        "technical": None,
        "valuation": None,
        "risk": None,
        "events": None,
        "peer_benchmark": None,
        "scenario": None,
        "final_report_markdown": "",
        "errors": [],
        "preferred_provider": preferred_provider,
        "skipped_agents": skipped,
        "report_generated_at": None
    }

    start_time = time.monotonic()
    result_state = app.invoke(initial_state)
    elapsed = time.monotonic() - start_time
    
    errors = result_state.get("errors", [])
    logger.info(f"Pipeline completed in {elapsed:.2f}s with {len(errors)} errors.")
    
    return result_state.get("final_report_markdown", "Error: No report generated.")

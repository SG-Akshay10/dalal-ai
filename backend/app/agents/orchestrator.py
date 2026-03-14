import logging
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
    error: str | None
    preferred_provider: str | None
    skipped_agents: list[str]

def get_fallback_analysis(schema_cls: Any) -> Any:
    """Helper to instantiate fallback empty schemas to prevent DAG crash."""
    try:
        if schema_cls == FundamentalAnalysis: return FundamentalAnalysis(revenue_trend=[], ebitda_trend=[], pat_trend=[], gross_margin=0.0, net_margin=0.0, fcf_commentary="", debt_equity_ratio=0.0, management_commentary="", red_flags=["Node crashed"], data_freshness_days=0)
        if schema_cls == SentimentAnalysis: return SentimentAnalysis(composite_sentiment_score=0, label="Neutral", positive_themes=[], negative_themes=[], narrative="Node crashed", news_article_count=0, social_post_count=0, finbert_distribution={})
        if schema_cls == CompetitorAnalysis: return CompetitorAnalysis(competitors=[], primary_sector="Unknown", business_description="")
        if schema_cls == SectorAnalysis: return SectorAnalysis(sector_name="Unknown", growth_stage="Mature", index_performance_ytd=0.0, fii_flow_trend="Neutral", policy_tailwinds=[], policy_headwinds=[], regulatory_summary="Node crashed")
        if schema_cls == TechnicalAnalysis: return TechnicalAnalysis(trend_bias="Insufficient Data", rsi_14=0.0, rsi_signal="Neutral", macd_signal="Bullish", bollinger_position="Mid", support_levels=[], resistance_levels=[], volume_trend="Neutral", narrative="Node crashed")
        if schema_cls == ValuationAnalysis: return ValuationAnalysis(current_price=None, market_cap_cr=None, pe_ttm=None, pe_forward=None, pb_ratio=None, ev_ebitda=None, price_to_sales=None, dividend_yield=None, peg_ratio=None, vs_own_history="At Par", vs_sector_median="At Par", premium_discount_pct=0.0, valuation_verdict="Node crashed", margin_of_safety="Low")
        if schema_cls == RiskAnalysis: return RiskAnalysis(flags=[], overall_risk_rating="Unknown", risk_narrative="Node crashed", promoter_pledge_pct=None)
        if schema_cls == EventAnalysis: return EventAnalysis(upcoming_events=[], recent_events=[], nearest_catalyst_days=None, event_risk_flag=False)
        if schema_cls == PeerBenchmarkAnalysis: return PeerBenchmarkAnalysis(rows=[], relative_positioning="In Line With Peers", strengths_vs_peers=[], weaknesses_vs_peers=[], narrative="Node crashed")
    except Exception:
        pass
    return None

def robust_agent_wrapper(state: ReportState, agent_func: callable, state_key: str, schema_cls: Any, **kwargs) -> ReportState:
    """Wraps an agent invocation to safely catch and mock output on failures."""
    logger.info(f"Running node: {state_key} for {state['ticker']}")
    
    if state_key in state.get("skipped_agents", []):
         logger.info(f"Skipping {state_key} as per supervisor logic.")
         return {state_key: get_fallback_analysis(schema_cls)}
         
    try:
        result = agent_func(**kwargs)
        return {state_key: result}
    except Exception as e:
        logger.error(f"FATAL ERROR in node {state_key}: {e}")
        return {state_key: get_fallback_analysis(schema_cls)}

# 1. Scraping Node
def node_scrape_data(state: ReportState) -> ReportState:
    ticker = state["ticker"]
    company_name = state.get("company_name", ticker)
    try:
        docs_days = state.get("days") or 90
        news_social_days = state.get("days") or 21
        docs = fetch_documents(ticker=ticker, days=docs_days)
        news = fetch_news(ticker=ticker, company_name=company_name, days=news_social_days)
        social = fetch_social(ticker=ticker, days=news_social_days)
        return {"raw_documents": docs, "raw_news": news, "raw_social": social}
    except Exception as e:
        logger.error(f"Scraping failed: {e}")
        return {"error": f"Scraping failed: {str(e)}"}

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
    return robust_agent_wrapper(state, analyze_risk_and_redflags, "risk", RiskAnalysis, ticker=state["ticker"], news=state.get("raw_news", []), docs=state.get("raw_documents", []), provider=state.get("preferred_provider"))

def node_events(state: ReportState):
    return robust_agent_wrapper(state, analyze_events, "events", EventAnalysis, ticker=state["ticker"], news=state.get("raw_news", []), provider=state.get("preferred_provider"))

# 3. Tier 2 Nodes
def node_peer_benchmark(state: ReportState):
    return robust_agent_wrapper(state, build_peer_benchmark, "peer_benchmark", PeerBenchmarkAnalysis, ticker=state["ticker"], competitors=state.get("competitors"), provider=state.get("preferred_provider"))

def node_scenario(state: ReportState):
    return robust_agent_wrapper(state, generate_scenarios, "scenario", PriceTargetAnalysis, 
             ticker=state["ticker"], fundamentals=state.get("fundamentals"), valuation=state.get("valuation"), 
             sentiment=state.get("sentiment"), peers=state.get("peer_benchmark"), technical=state.get("technical"), 
             risk=state.get("risk"), provider=state.get("preferred_provider"))

# 4. Report Node
def node_generate_report(state: ReportState) -> ReportState:
    if state.get("error"):
        return {"final_report_markdown": f"# Error\n\nPipeline failed earlier: {state['error']}"}

    # For now, report agent primarily relies on basic inputs due to its prompt definition,
    # but the DAG supports everything. We pass the 4 core things, V2 elements can be added to report generator soon.
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
    return {"final_report_markdown": report}


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
    workflow.add_node("valuation", node_valuation)
    workflow.add_node("risk", node_risk)
    workflow.add_node("events", node_events)
    
    # Tier 2
    workflow.add_node("peer_benchmark", node_peer_benchmark)
    workflow.add_node("scenario", node_scenario)
    
    # Report
    workflow.add_node("report", node_generate_report)

    # Edges
    workflow.set_entry_point("scrape")
    
    # Scrape -> Tier 1
    for t1 in ["fundamental", "sentiment", "competitor", "sector", "technical", "valuation", "risk", "events"]:
        workflow.add_edge("scrape", t1)
        
    # Tier 1 outputs feed downstream
    workflow.add_edge("competitor", "peer_benchmark")
    
    # Scenario waits for critical variables ensuring they completed
    # (In standard LangGraph, it'll wait for explicitly mapped downstream nodes. 
    # Usually you'd use parallel map/reduce or explicit edge lists.)
    workflow.add_edge("peer_benchmark", "scenario")
    workflow.add_edge("fundamental", "scenario")
    workflow.add_edge("sentiment", "scenario")
    workflow.add_edge("technical", "scenario")
    workflow.add_edge("valuation", "scenario")
    workflow.add_edge("risk", "scenario")
    
    # All terminal analyses must link to report
    workflow.add_edge("scenario", "report")
    workflow.add_edge("sector", "report")
    workflow.add_edge("events", "report")

    workflow.add_edge("report", END)

    app = workflow.compile()
    return app


def run_pipeline(ticker: str, company_name: str = None, overridden_competitors=None, preferred_provider: str = None, days: int = None) -> str:
    app = build_orchestrator()
    
    # Basic supervisor logic: conditionally skip for some variants if needed
    skipped = []
    if "BEES" in ticker.upper() or "ETF" in ticker.upper():
        skipped.extend(["competitors", "peer_benchmark", "risk"])

    initial_state = {
        "ticker": ticker,
        "company_name": company_name or ticker,
        "days": days,
        "raw_documents": [],
        "raw_news": [],
        "raw_social": [],
        "competitors": overridden_competitors,
        "final_report_markdown": "",
        "error": None,
        "preferred_provider": preferred_provider,
        "skipped_agents": skipped
    }

    result_state = app.invoke(initial_state)
    return result_state.get("final_report_markdown", "Error: No report generated.")

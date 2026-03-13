import logging
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END

# Import schemas
from app.schemas.report import FundamentalAnalysis, SentimentAnalysis, CompetitorAnalysis, SectorAnalysis

# Import agent functions
from app.agents.fundamental_agent import extract_fundamentals
from app.agents.sentiment_agent import analyze_sentiment
from app.agents.competitor_agent import identify_competitors
from app.agents.sector_agent import analyze_sector
from app.agents.report_agent import generate_report

# Import Phase 1 scrapers
from app.scrapers.document_fetcher import fetch_documents
from app.scrapers.news_scraper import fetch_news
from app.scrapers.social_listener import fetch_social

logger = logging.getLogger(__name__)

class ReportState(TypedDict, total=False):
    """The unified state passed around the LangGraph DAG."""
    ticker: str
    company_name: str
    days: Optional[int]
    
    # Phase 1 Data (Optional initially)
    raw_documents: list
    raw_news: list
    raw_social: list
    
    # Phase 2 Agent Outputs
    fundamentals: Optional[FundamentalAnalysis]
    sentiment: Optional[SentimentAnalysis]
    competitors: Optional[CompetitorAnalysis]
    sector: Optional[SectorAnalysis]
    
    # Final Output
    final_report_markdown: str
    error: Optional[str]
    preferred_provider: Optional[str]

# 1. Scraping Nodes
def node_scrape_data(state: ReportState) -> ReportState:
    """Fetch raw documents, news, and social data."""
    ticker = state["ticker"]
    company_name = state.get("company_name", ticker)
    
    logger.info(f"Node: Scrape Data - fetching for {ticker}")
    
    try:
        docs_days = state.get("days") or 90
        news_social_days = state.get("days") or 21
        
        docs = fetch_documents(ticker=ticker, days=docs_days)
        news = fetch_news(ticker=ticker, company_name=company_name, days=news_social_days)
        social = fetch_social(ticker=ticker, days=news_social_days)
        
        return {
            "raw_documents": docs,
            "raw_news": news,
            "raw_social": social,
        }
    except Exception as e:
        logger.error(f"Error in scraping node: {str(e)}")
        return {"error": f"Scraping failed: {str(e)}"}

# 2. Analysis Nodes
def node_fundamental_analysis(state: ReportState) -> ReportState:
    logger.info(f"Node: Fundamental Analysis for {state['ticker']}")
    
    # Note: In a full pipeline, we would ingest docs to Chroma here 
    # if they aren't already ingrained. For simplicity we assume ingest 
    # happens before, or we can just rely on the existing vector store.
    
    fundamentals = extract_fundamentals(state["ticker"], provider=state.get("preferred_provider"))
    return {"fundamentals": fundamentals}

def node_sentiment_analysis(state: ReportState) -> ReportState:
    logger.info(f"Node: Sentiment Analysis for {state['ticker']}")
    sentiment = analyze_sentiment(
        ticker=state["ticker"], 
        news=state.get("raw_news", []), 
        social=state.get("raw_social", []),
        provider=state.get("preferred_provider")
    )
    return {"sentiment": sentiment}

def node_competitor_analysis(state: ReportState) -> ReportState:
    logger.info(f"Node: Competitor Analysis for {state['ticker']}")
    if state.get("competitors"):
        # Allow overriding competitors before running pipeline
        return {"competitors": state["competitors"]}
    competitors = identify_competitors(state["ticker"], provider=state.get("preferred_provider"))
    return {"competitors": competitors}

def node_sector_analysis(state: ReportState) -> ReportState:
    logger.info(f"Node: Sector Analysis for {state['ticker']}")
    sector = analyze_sector(state["ticker"], provider=state.get("preferred_provider"))
    return {"sector": sector}

# 3. Report Synthesis Node
def node_generate_report(state: ReportState) -> ReportState:
    logger.info(f"Node: Generate Report for {state['ticker']}")
    
    # Ensure all required inputs are present
    if state.get("error"):
        return {"final_report_markdown": f"# Error\n\nPipeline failed earlier: {state['error']}"}
        
    report = generate_report(
        ticker=state["ticker"],
        fundamentals=state["fundamentals"],
        sentiment=state["sentiment"],
        sector=state["sector"],
        competitors=state["competitors"],
        provider=state.get("preferred_provider")
    )
    return {"final_report_markdown": report}

def build_orchestrator() -> StateGraph:
    """Build and compile the LangGraph DAG for the report pipeline."""
    workflow = StateGraph(ReportState)
    
    # Define Nodes
    workflow.add_node("scrape", node_scrape_data)
    workflow.add_node("fundamental", node_fundamental_analysis)
    workflow.add_node("sentiment", node_sentiment_analysis)
    workflow.add_node("competitor", node_competitor_analysis)
    workflow.add_node("sector", node_sector_analysis)
    workflow.add_node("report", node_generate_report)
    
    # Define Edges (DAG structure)
    workflow.set_entry_point("scrape")
    
    # Once scrape is done, we can run all analysis nodes in parallel
    # (LangGraph handles parallelism implicitly if branches diverge)
    workflow.add_edge("scrape", "fundamental")
    workflow.add_edge("scrape", "sentiment")
    workflow.add_edge("scrape", "competitor")
    workflow.add_edge("scrape", "sector")
    
    # Once all analysis nodes are complete, join them at the report node
    workflow.add_edge("fundamental", "report")
    workflow.add_edge("sentiment", "report")
    workflow.add_edge("competitor", "report")
    workflow.add_edge("sector", "report")
    
    # Final path
    workflow.add_edge("report", END)
    
    # Compile the graph
    app = workflow.compile()
    
    return app

def run_pipeline(ticker: str, company_name: str = None, overridden_competitors=None, preferred_provider: str = None, days: int = None) -> str:
    """Execute the full DAG for a given ticker."""
    app = build_orchestrator()
    
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
        "final_report_markdown": "",
        "error": None,
        "preferred_provider": preferred_provider
    }
    
    result_state = app.invoke(initial_state)
    return result_state.get("final_report_markdown", "Error: No report generated.")

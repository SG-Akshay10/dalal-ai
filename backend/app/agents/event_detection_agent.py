import json
import logging
from datetime import datetime

from langchain_core.prompts import PromptTemplate

from app.llm_provider import get_llm_client
from app.schemas.analysis import EventAnalysis
from app.schemas.news_article import NewsArticle
from app.services.live_data import live_data_service

logger = logging.getLogger(__name__)

EVENT_PROMPT = """You are an expert equity research analyst.
Identify the most crucial recent and upcoming corporate events for {ticker} that a stock investor must know about.

Upcoming Events (from calendar):
{calendar_events}

Recent News Headlines (extract major events from here):
{news_events}

Filter these down to only the most impactful events (e.g. earnings, dividends, M&A, regulatory notices).
Determine the `nearest_catalyst_days` (how many days until the next important upcoming event). If none, use null.
Set `event_risk_flag` to true if there is a high-impact catalyst occurring in the next 7 days.

Always return your response as a valid JSON object matching the following schema:
{{
  "upcoming_events": [
    {{"event_type": "...", "date": "...", "description": "...", "investment_relevance": "..."}}
  ],
  "recent_events": [
    {{"event_type": "...", "date": "...", "description": "...", "investment_relevance": "..."}}
  ],
  "nearest_catalyst_days": 12,
  "event_risk_flag": false
}}
"""

def analyze_events(ticker: str, news: list[NewsArticle], provider: str = None) -> EventAnalysis:
    """Run Event Detection Agent."""
    
    calendar_data = live_data_service.fetch_corporate_events(ticker)
    upcoming_cal = calendar_data.get("upcoming_events", [])
    
    cal_text = "\n".join([f"- {e['date']}: {e['event_type']} ({e['description']})" for e in upcoming_cal]) if upcoming_cal else "No upcoming calendar events found."
    
    news_text = "\n".join([f"- {n.date.strftime('%Y-%m-%d') if hasattr(n, 'date') and n.date else 'Recent'}: {n.headline}" for n in news[:15]]) if news else "No recent news found."
    
    prompt = PromptTemplate.from_template(EVENT_PROMPT)
    formatted_prompt = prompt.format(ticker=ticker, calendar_events=cal_text, news_events=news_text)
    
    llm = get_llm_client(provider)
    try:
        response = llm.invoke([
            {"role": "system", "content": "You output JSON only."},
            {"role": "user", "content": formatted_prompt}
        ])
        content = response.content.replace("```json", "").replace("```", "").strip()
        data = json.loads(content)
        
        return EventAnalysis(**data)
    except Exception as e:
        logger.error(f"Error in Event Detection Agent for {ticker}: {e}")
        return EventAnalysis(
            upcoming_events=[],
            recent_events=[],
            nearest_catalyst_days=None,
            event_risk_flag=False
        )

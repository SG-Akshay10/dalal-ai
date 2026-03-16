"""
Rewritten for StockSense AI Pipeline v2.
Changes made:
- Calendar API Fetch: Wrapped in isolated try/except. A calendar failure proceeds cleanly to news-only extraction.
- Deterministic Event Risks (Python): Nearest catalyst days and severe 7-day risk flags are manually parsed and computed 
  in python matching strictly against High Impact upcoming event categorizations. Dates are rigorously parsed.
- Advanced Retrieval: Initiates 3 new targeted queries for news items (Orders/JV, Credit Ratings, Index formatting) expanding footprint. deduplicates 
  news and injected docs by 120-char fingerprints.
- Controlled Prompt Environment: Hardcoded exact enumerations for Upcoming and Recent `event_type` schemas.
- Rigorous Logic Framework: Implemented 5 ASC limit rules for Upcoming and 5 DESC limit rules for Recent. Enforced rigid formatting on 
  the `investment_relevance` clause ("What happened. Why it matters."). Added strict noise filtering rule ("Would an institutional investor change their position").
- Fallbacks: Created `_build_fallback_upcoming` to map raw dict entries manually into structured `CorporateEvent` objects keeping raw calendar
  data from catastrophic JSON collapses. Per-event try/except loop filters out single malformed LLM responses without dropping the whole node.
"""

import json
import logging
from datetime import datetime, date

from langchain_core.prompts import PromptTemplate

from app.llm_provider import get_llm_client
from app.schemas.analysis import EventAnalysis, CorporateEvent
from app.schemas.news_article import NewsArticle
from app.vector_store.retriever import retrieve_documents
from app.services.live_data import live_data_service

logger = logging.getLogger(__name__)

EVENT_PROMPT = """You are an expert equity research analyst specializing in event-driven catalysts.
Identify the most crucial recent and upcoming corporate events for {ticker} that a stock investor must know about.

Today's Date is: {today}

<STRICT_RULES>
1. NOISE REDUCTION FILTER: Only include an event if the answer to "Would an institutional investor change their position based on this?" is Yes. Discard routine board meetings, minor CSR events, and generic sector noise.
2. CONTROLLED VOCABULARY - `event_type`:
   - Upcoming Events MUST be one of: [Quarterly Earnings Release, Board Meeting, AGM/EGM, Dividend Record Date, Stock Split/Bonus Date, FPO/Offer for Sale, Rights Issue, QIP, Lock-in Expiry, NCLT/IBC Hearing, FDA/Regulatory Audit, Court Hearing, Product Launch, Major Investor Presentation].
   - Recent Events MUST be one of: [Order Win / Contract, M&A / Joint Venture, Leadership Change, Regulatory Penalty / SEBI Order, Credit Rating Change, Analyst Meet / Concall, Promoter Stake Change, Auditor Resignation, Plant Shutdown / Force Majeure, Product Approval (FDA/CE), Index Inclusion / Exclusion, Fundraise Completed, Default / Rating Downgrade].
3. DATE FORMAT: Always output `date` as `YYYY-MM-DD`. If only a general timeframe is known (e.g., "Late Q3"), use the first date of that month. If totally unknown, use `null`.
4. INVESTMENT RELEVANCE: Max ONE sentence. Must follow this format: "[What will happen / happened]. [Why it matters — magnitude, direction, or uncertainty]."
5. CONSTRAINTS:
   - limit `upcoming_events` to a maximum of 5 distinct events. Sort ASCENDING by date (closest first).
   - limit `recent_events` to a maximum of 5 distinct events that occurred BEFORE today. Sort DESCENDING by date (newest first).
</STRICT_RULES>

<REASONING_STEPS>
1. Separate all context into Future (Upcoming) and Past (Recent) relative to {today}.
2. Discard all noise events not meeting the institutional relevance filter.
3. Map remaining events to exactly one of the allowed controlled vocabulary strings.
4. Format dates rigidly to YYYY-MM-DD.
5. Draft exactly one-sentence `investment_relevance` matching the required format.
6. Apply the maximum limits (5 each) sorting Ascending (Upcoming) and Descending (Recent).
</REASONING_STEPS>

Upcoming Events (from calendar API):
{calendar_events}

Recent Target News & Corporate Filings:
{news_events}

Always return your final response as a valid JSON object matching the following schema structure. Wait until after your reasoning to generate the JSON.
<EXAMPLE_SCHEMA>
{{
  "upcoming_events": [
    {{"event_type": "Quarterly Earnings Release", "date": "2024-10-25", "description": "Q2FY25 Results Declaration.", "investment_relevance": "Expected to announce results. Crucial test for sustaining volume growth amid raw material inflation."}}
  ],
  "recent_events": [
    {{"event_type": "Order Win / Contract", "date": "2024-09-15", "description": "Company secured ₹500 Cr order from Ministry of Defence.", "investment_relevance": "Secured major defence contract. Significantly increases revenue visibility and strengthens the government order book backlog."}}
  ]
}}
</EXAMPLE_SCHEMA>
"""

def parse_date(date_str: str) -> date | None:
    if not date_str:
        return None
    try:
        # Common formats loosely
        for fmt in ["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%d-%m-%Y", "%d/%m/%Y"]:
            try:
                return datetime.strptime(date_str[:10], fmt).date()
            except ValueError:
                continue
    except Exception:
        pass
    return None

def _build_fallback_upcoming(raw_cal: list[dict]) -> list[CorporateEvent]:
    """Fallback generator mapping raw dicts to schema safely on LLM failure."""
    fallback_events = []
    for limit, e in enumerate(raw_cal):
        if limit >= 5: break
        try:
            fallback_events.append(CorporateEvent(
                event_type=str(e.get("event_type", "Unknown Event"))[:50],
                date=e.get("date"),
                description=str(e.get("description", "Calendar entry"))[:200],
                investment_relevance="LLM enrichment unavailable — raw calendar entry."
            ))
        except Exception:
            pass
    return fallback_events

def analyze_events(ticker: str, news: list[NewsArticle], provider: str = None) -> EventAnalysis:
    """Run Deterministic Event Detection Agent."""
    today = date.today()
    
    # 1. Safely Fetch Calendar Data
    upcoming_cal = []
    try:
        calendar_data = live_data_service.fetch_corporate_events(ticker)
        upcoming_cal = calendar_data.get("upcoming_events", [])
    except Exception as e:
        logger.warning(f"[{ticker}] Calendar fetch failed: {e}. Proceeding with news extraction only.")

    # 2. Targeted Risk Retrieval for Context Expansion
    queries = [
        f"{ticker} order win contract JV acquisition joint venture",
        f"{ticker} credit rating upgrade downgrade CRISIL ICRA CARE",
        f"{ticker} index inclusion exclusion Nifty Sensex FPO rights QIP"
    ]
    
    raw_docs = []
    for q in queries:
        try:
            raw_docs.extend(retrieve_documents(q, ticker=ticker, top_k=2))
        except Exception as e:
            logger.warning(f"[{ticker}] Targeted event retrieval failed for '{q}': {e}")
            
    # Deduplicate retrieved contextual docs by fingerprint
    unique_docs = []
    seen = set()
    for d in raw_docs:
        text = d.get('text', getattr(d, 'page_content', str(d)))
        if not text: continue
        fp = text[:120].strip()
        if fp not in seen:
            seen.add(fp)
            unique_docs.append(d)

    # 3. Format inputs for LLM
    cal_text = "No upcoming calendar events found."
    if upcoming_cal:
        cal_text = "\n".join([f"- {e.get('date', 'Unknown')}: {e.get('event_type', 'Event')} ({e.get('description', '')})" for e in upcoming_cal])
        
    news_lines = []
    if news:
        for n in news[:10]:
            d_str = n.date.strftime('%Y-%m-%d') if getattr(n, 'date', None) else 'Recent'
            body_snip = (n.body or '')[:250].replace('\n', ' ')
            news_lines.append(f"- [{d_str}] {n.headline}: {body_snip}")
            
    if unique_docs:
        for d in unique_docs[:5]:
            text_snip = d.get('text', getattr(d, 'page_content', str(d)))[:250].replace('\n', ' ')
            news_lines.append(f"- [Context]: {text_snip}")
            
    news_text = "\n".join(news_lines) if news_lines else "No recent news found."

    prompt = PromptTemplate.from_template(EVENT_PROMPT)
    formatted_prompt = prompt.format(
        ticker=ticker, 
        today=today.strftime('%Y-%m-%d'),
        calendar_events=cal_text, 
        news_events=news_text
    )
    
    llm = get_llm_client(provider)
    
    result_upcoming = []
    result_recent = []

    try:
        if hasattr(llm, "with_structured_output"):
            from pydantic import BaseModel, Field
            class EventArray(BaseModel):
                upcoming_events: list[dict]
                recent_events: list[dict]
                
            structured_llm = llm.with_structured_output(
                EventArray, 
                method="json_mode" if "openai" in str(type(llm)).lower() else "function_calling"
            )
            try:
                llm_res = structured_llm.invoke(formatted_prompt)
                
                # Single-event try/catch processing for resilience
                for r in getattr(llm_res, "upcoming_events", []):
                    try:
                        result_upcoming.append(CorporateEvent(**r))
                    except Exception as ve:
                        logger.debug(f"Pydantic validtion failed on upcoming event {r}: {ve}")
                        
                for r in getattr(llm_res, "recent_events", []):
                    try:
                        result_recent.append(CorporateEvent(**r))
                    except Exception as ve:
                        logger.debug(f"Pydantic validation failed on recent event {r}: {ve}")
            except Exception as e:
                logger.error(f"[{ticker}] Structured output error in Event Detection: {e}")

        # Fallback raw parse
        if not result_upcoming and not result_recent:
            response = llm.invoke([
                {"role": "system", "content": "You output JSON only. Return keys 'upcoming_events' and 'recent_events'."},
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
            
            for r in data.get("upcoming_events", []):
                try: result_upcoming.append(CorporateEvent(**r))
                except Exception: pass
            for r in data.get("recent_events", []):
                try: result_recent.append(CorporateEvent(**r))
                except Exception: pass
                
    except Exception as e:
        logger.error(f"Error in Event Agent for {ticker}: {e}")
        # Build safe fallback on LLM complete failure
        result_upcoming = _build_fallback_upcoming(upcoming_cal)
        result_recent = []

    # Final deterministic safety net: Date sorting & Truncation (in case LLM ignored rules)
    result_upcoming.sort(key=lambda x: x.date if x.date else "9999-12-31")
    result_upcoming = result_upcoming[:5]
    
    result_recent.sort(key=lambda x: x.date if x.date else "0000-00-00", reverse=True)
    result_recent = result_recent[:5]

    # --- DETERMINISTIC FLAG CALCULATOR in Python ---
    HIGH_IMPACT_UPCOMING = [
        "Quarterly Earnings Release", "Board Meeting", "FPO/Offer for Sale", 
        "Rights Issue", "QIP", "Lock-in Expiry", "NCLT/IBC Hearing"
    ]
    
    nearest_catalyst_days = None
    event_risk_flag = False
    
    for event in result_upcoming:
        d_val = parse_date(event.date)
        if d_val:
            days_delta = (d_val - today).days
            if days_delta >= 0:
                # Is it an important event?
                is_high_impact = event.event_type in HIGH_IMPACT_UPCOMING
                
                if is_high_impact:
                    # Capture nearest days
                    if nearest_catalyst_days is None or days_delta < nearest_catalyst_days:
                        nearest_catalyst_days = days_delta
                        
                    # Set 7-day risk flag
                    if days_delta <= 7:
                        event_risk_flag = True

    return EventAnalysis(
        upcoming_events=result_upcoming,
        recent_events=result_recent,
        nearest_catalyst_days=nearest_catalyst_days,
        event_risk_flag=event_risk_flag
    )

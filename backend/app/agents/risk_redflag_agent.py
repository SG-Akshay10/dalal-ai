import json
import logging
from langchain_core.prompts import PromptTemplate

from app.llm_provider import get_llm_client
from app.schemas.analysis import RiskAnalysis, RiskFlag
from app.services.live_data import live_data_service

logger = logging.getLogger(__name__)

RISK_CHECKLIST_PROMPT = """You are an expert auditor and equity researcher.
Review the following excerpts from filings and news for the ticker {ticker}.

Your job is strictly to evaluate these specific risk items:
1. Audit Qualification: Is there a non-clean audit opinion or "going concern" doubt?
2. CFO/CEO exit without succession plan recently?
3. SEBI notices or enforcement actions recently?

Context (News & Doc excerpts):
{context}

Always return your response as a valid JSON object matching the following schema. If you do not have evidence for a flag, do not invent one, but set its severity to "Unknown".
{{
  "flags": [
    {{"flag_type": "Audit Qualification", "severity": "Low", "detail": "Clean report", "source": "Filings"}},
    {{"flag_type": "Management Exit", "severity": "Unknown", "detail": "No data", "source": "News"}},
    {{"flag_type": "SEBI Action", "severity": "Unknown", "detail": "No data", "source": "News"}}
  ],
  "overall_risk_rating": "Moderate",
  "risk_narrative": "..."
}}
"""

def analyze_risk_and_redflags(ticker: str, news, docs, provider: str = None) -> RiskAnalysis:
    """Checklist driven risk agent."""
    
    # 1. Check live quantitative risk (Promoter Pledge)
    shareholding = live_data_service.fetch_shareholding_patterns(ticker)
    pledge_pct = shareholding.get("promoter_pledged_pct")
    
    quant_flags = []
    if pledge_pct is not None:
        severity = "High" if pledge_pct > 40 else "Medium" if pledge_pct > 20 else "Low"
        quant_flags.append(RiskFlag(
            flag_type="Promoter Pledge", 
            severity=severity, 
            detail=f"Promoter pledge stands at {pledge_pct}%", 
            source="NSE Shareholding API"
        ))

    # Prepare context for LLM qualitative checks
    news_text = "\n".join([f"- {n.headline}" for n in news[:10]]) if news else "No news."
    docs_text = "\n".join([f"- {getattr(d, 'text', str(d))[:200]}" for d in docs[:3]]) if docs else "No docs."
    context = f"NEWS:\n{news_text}\n\nDOC EXCERPTS:\n{docs_text}"
    
    prompt = PromptTemplate.from_template(RISK_CHECKLIST_PROMPT)
    formatted_prompt = prompt.format(ticker=ticker, context=context)
    
    llm = get_llm_client(provider)
    try:
        response = llm.invoke([
            {"role": "system", "content": "You output JSON only."},
            {"role": "user", "content": formatted_prompt}
        ])
        content = response.content.replace("```json", "").replace("```", "").strip()
        data = json.loads(content)
        
        flags = quant_flags + [RiskFlag(**f) for f in data.get("flags", [])]
        
        # Adjust overall rating based on pledge
        rating = data.get("overall_risk_rating", "Moderate")
        if pledge_pct and pledge_pct > 40:
            rating = "High"
            
        return RiskAnalysis(
            flags=flags,
            overall_risk_rating=rating,
            risk_narrative=data.get("risk_narrative", ""),
            promoter_pledge_pct=pledge_pct
        )
        
    except Exception as e:
        logger.error(f"Error in risk agent for {ticker}: {e}")
        return RiskAnalysis(
            flags=quant_flags,
            overall_risk_rating="Unknown",
            risk_narrative=f"Error processing risks: {str(e)}",
            promoter_pledge_pct=pledge_pct
        )

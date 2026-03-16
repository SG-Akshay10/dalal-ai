"""
Rewritten for StockSense AI Pipeline v2.
Changes made:
- Architecture: Implemented a 3-layer risk framework (Quantitative, Fundamental, Qualitative).
- Retrievals: Use 3 targeted queries + deduplication alongside provided docs. Expanded doc chunks to 500 chars, news to 300 chars.
- Quantitative (Python): Evaluates promoter pledge levels and QoQ trend safely.
- Fundamental (Python): Evaluates FCF keywords, multi-quarter revenue declines, and debt/equity leverage.
- Qualitative (LLM): Rigid 8-item checklist. Disabled "Unknown" severity; enforces "Low" if no evidence.
- Deterministic Rating: Computes Overall Risk Rating based cleanly on combined flag severities (>=1 Critical or >=2 High -> Very High, etc.)
- Fallback: Pre-calculated Layer 1 & 2 flags survive LLM failure seamlessly.
"""

import json
import logging
from langchain_core.prompts import PromptTemplate

from app.llm_provider import get_llm_client
from app.schemas.analysis import RiskAnalysis, RiskFlag, FundamentalAnalysis
from app.vector_store.retriever import retrieve_documents
from app.services.live_data import live_data_service

logger = logging.getLogger(__name__)

RISK_CHECKLIST_PROMPT = """You are an expert fundamental auditor and equity researcher.
Review the following excerpts from corporate filings, auditor reports, and recent news for the ticker {ticker}.

Your job is strictly to evaluate these 8 specific risk qualitative items. 
<STRICT_RULES>
1. CHECKLIST ORDER: You MUST return exactly 8 flags in the precise order requested.
2. NO "UNKNOWN" SEVERITY: Valid severities are ONLY "Low", "Medium", "High", "Critical". 
3. NO EVIDENCE FALLBACK: If there is no evidence for a flag in the context, you MUST set severity to "Low" and detail to exactly: "No evidence found in reviewed documents and news." 
4. DO NOT INVENT RISKS: Base decisions solely on the provided text.
</STRICT_RULES>

<RISK_ITEMS_TO_EVALUATE>
1. Audit Qualification (qualified opinion, emphasis of matter)
2. Going Concern Doubt (separate from audit qual — auditor/mgmt statement)
3. Related-Party Transactions > threshold
4. CFO/CEO Exit Without Succession Plan
5. SEBI Notice or Enforcement Action
6. Short-Term Debt Spike > 50% QoQ
7. Inventory/Receivables Growing Faster Than Revenue
8. NCLT/Insolvency Proceedings
</RISK_ITEMS_TO_EVALUATE>

<REASONING_STEPS>
1. Scan the text specifically for audit qualifications or emphasis of matter.
2. Scan for "going concern" language.
3. Check for mentions of excessive related-party transactions (RPT) or loans to directors.
4. Look for C-suite exits (CFO/CEO).
5. Check for SEBI action, show-cause notices, or penalties.
6. Look for short-term borrowing spikes (>50%).
7. Check working capital ratios (inventory/receivables vs revenue growth).
8. Look for NCLT or insolvency admissions.
9. Format exactly 8 flags matching these items.
</REASONING_STEPS>

Context (News & Doc excerpts):
{context}

Always return your final response as a valid JSON object matching the following schema. Wait until after your reasoning to generate the JSON.
<EXAMPLE_SCHEMA>
{{
  "llm_flags": [
    {{"flag_type": "Audit Qualification", "severity": "Low", "detail": "No evidence found in reviewed documents and news.", "source": "LLM text review"}},
    {{"flag_type": "Going Concern Doubt", "severity": "Low", "detail": "No evidence found in reviewed documents and news.", "source": "LLM text review"}}
  ],
  "risk_narrative": "A concise 3-sentence summary of the highest risks identified across the text."
}}
</EXAMPLE_SCHEMA>
"""

def analyze_risk_and_redflags(ticker: str, news, docs, fundamentals: FundamentalAnalysis | None = None, provider: str = None) -> RiskAnalysis:
    """Checklist driven risk agent with 3-Layer architecture (Quant, Fundamental, LLM Qualitative)."""
    
    layer1_flags = []
    layer2_flags = []
    
    # ---------------------------------------------------------
    # LAYER 1: Quantitative (Live Data - Promoter Pledge)
    # ---------------------------------------------------------
    try:
        shareholding = live_data_service.fetch_shareholding_patterns(ticker)
        pledge_pct = shareholding.get("promoter_pledged_pct")
        pledge_pct_prev = shareholding.get("promoter_pledged_pct_previous")
        
        if pledge_pct is not None:
            sev = "Low"
            if pledge_pct > 40:
                sev = "High"
            elif pledge_pct > 20:
                sev = "Medium"
                
            layer1_flags.append(RiskFlag(
                flag_type="Promoter Pledge Level", 
                severity=sev, 
                detail=f"Curr pledge: {pledge_pct}%. (>40% High, 20-40% Med)", 
                source="Quantitative Shareholding"
            ))
            
            if pledge_pct_prev is not None:
                inc = pledge_pct - pledge_pct_prev
                if inc > 5:
                    layer1_flags.append(RiskFlag(
                        flag_type="Promoter Pledge Trend", 
                        severity="High", 
                        detail=f"Pledge increased rapidly by {inc:.1f}% QoQ (>5% threshold).", 
                        source="Quantitative Shareholding"
                    ))
                elif inc > 2:
                    layer1_flags.append(RiskFlag(
                        flag_type="Promoter Pledge Trend", 
                        severity="Medium", 
                        detail=f"Pledge increased by {inc:.1f}% QoQ (>2% threshold).", 
                        source="Quantitative Shareholding"
                    ))
    except Exception as e:
        logger.warning(f"[{ticker}] Error fetching shareholding for Quant Risk Layer: {e}")
        pledge_pct = None

    # ---------------------------------------------------------
    # LAYER 2: Fundamental-derived
    # ---------------------------------------------------------
    if fundamentals:
        # FCF Negativity
        fcf_comm = getattr(fundamentals, 'fcf_commentary', '').lower()
        if fcf_comm and any(kw in fcf_comm for kw in ["negative free cash", "cash burn", "operating cash flow negative"]):
            layer2_flags.append(RiskFlag(
                flag_type="Negative Free Cash Flow",
                severity="Medium",
                detail="Keywords indicating free cash flow burn were identified in fundamental commentary.",
                source="Fundamental Extraction"
            ))
            
        # Multi-quarter revenue decline
        if fundamentals.revenue_trend and len(fundamentals.revenue_trend) >= 2:
            decline_count = 0
            for r in fundamentals.revenue_trend:
                if r.yoy_growth_pct is not None and r.yoy_growth_pct < 0:
                    decline_count += 1
            if decline_count >= 2:
                layer2_flags.append(RiskFlag(
                    flag_type="Revenue Contraction",
                    severity="High",
                    detail=f"Revenue declined YoY for {decline_count} quarters.",
                    source="Fundamental Extraction"
                ))
                
        # Leverage
        de_ratio = getattr(fundamentals, 'debt_equity_ratio', None)
        if de_ratio is not None:
            if de_ratio > 2.5:
                layer2_flags.append(RiskFlag(
                    flag_type="High Leverage",
                    severity="High",
                    detail=f"D/E ratio of {de_ratio} exceeds the critical 2.5x threshold.",
                    source="Fundamental Extraction"
                ))
            elif de_ratio > 1.5:
                layer2_flags.append(RiskFlag(
                    flag_type="Elevated Leverage",
                    severity="Medium",
                    detail=f"D/E ratio of {de_ratio} exceeds the elevated 1.5x threshold.",
                    source="Fundamental Extraction"
                ))

    # ---------------------------------------------------------
    # LAYER 3: LLM Qualitative (Retrieval & Processing)
    # ---------------------------------------------------------
    queries = [
        f"{ticker} audit opinion qualification going concern auditor remarks",
        f"{ticker} related party transactions RPT director loans key managerial",
        f"{ticker} SEBI notice enforcement penalty short-term debt borrowings NCLT insolvency"
    ]
    
    raw_docs = list(docs) if docs else []
    for q in queries:
        try:
            ret_docs = retrieve_documents(q, ticker=ticker, top_k=3)
            raw_docs.extend(ret_docs)
        except Exception as e:
            logger.warning(f"[{ticker}] Retrieval failed for risk query '{q}': {e}")
            
    # Deduplicate qualitative docs
    unique_docs = []
    seen = set()
    for d in raw_docs:
        text = d.get('text', getattr(d, 'page_content', str(d)))
        if not text: continue
        fp = text[:120].strip()
        if fp not in seen:
            seen.add(fp)
            unique_docs.append(d)

    news_text = "No news."
    if news:
        news_text = "\n".join([f"- {n.headline}: {(n.body or '')[:300]}" for n in news[:10]])
        
    docs_text = "No docs."
    if unique_docs:
        docs_text = "\n".join([f"- {d.get('text', getattr(d, 'page_content', str(d)))[:500]}" for d in unique_docs[:8]])
        
    context = f"NEWS:\n{news_text}\n\nDOC EXCERPTS:\n{docs_text}"

    prompt = PromptTemplate.from_template(RISK_CHECKLIST_PROMPT)
    formatted_prompt = prompt.format(ticker=ticker, context=context)
    
    llm = get_llm_client(provider)
    llm_flags = []
    narrative = ""

    try:
        result_data = {}
        if hasattr(llm, "with_structured_output"):
            from pydantic import BaseModel
            class LLMRiskOut(BaseModel):
                llm_flags: list[dict]
                risk_narrative: str
                
            structured_llm = llm.with_structured_output(
                LLMRiskOut, 
                method="json_mode" if "openai" in str(type(llm)).lower() else "function_calling"
            )
            try:
                llm_res = structured_llm.invoke(formatted_prompt)
                result_data = {
                    "llm_flags": getattr(llm_res, "llm_flags", []),
                    "risk_narrative": getattr(llm_res, "risk_narrative", "")
                }
            except Exception as e:
                logger.error(f"[{ticker}] Structured output invoke error in Risk LLM: {e}")

        if not result_data:
            response = llm.invoke([
                {"role": "system", "content": "You output JSON only. Must contain 'llm_flags' and 'risk_narrative'."},
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
            result_data = {
                "llm_flags": data.get("llm_flags", []),
                "risk_narrative": data.get("risk_narrative", "")
            }

        # Convert LLM flag dicts to RiskFlag objects, ensuring source mapping
        for f in result_data.get("llm_flags", []):
            try:
                # Patch any "Unknown" severity hallucinated by LLM
                sev = f.get("severity", "Low")
                if sev not in ["Low", "Medium", "High", "Critical"]:
                    sev = "Low"
                    
                llm_flags.append(RiskFlag(
                    flag_type=f.get("flag_type", "Qualitative Risk"),
                    severity=sev,
                    detail=f.get("detail", "No evidence found in reviewed documents and news."),
                    source="News & Filings Review"
                ))
            except Exception:
                pass
                
        narrative = result_data.get("risk_narrative", "")
        
    except Exception as e:
        logger.error(f"[{ticker}] Error in LLM Qualitative Risk step: {e}")
        # Build fallback narrative from high severity computed flags if any
        high_flags = [f for f in (layer1_flags + layer2_flags) if f.severity in ["High", "Critical"]]
        if high_flags:
            narrative = f"Qualitative text analysis failed. Quantitative and Fundamental metrics exhibit significant risks, specifically: {', '.join([f.flag_type for f in high_flags])}."
        else:
            narrative = f"Qualitative text analysis failed. No severe quantitative or fundamental risks detected automatically. (Error: {str(e)})"
            
        # Ensure 8 placeholder flags to satisfy the contract even on failure
        default_items = [
            "Audit Qualification", "Going Concern Doubt", "Related-Party Transactions", 
            "CFO/CEO Exit", "SEBI Action", "Short-Term Debt Spike", 
            "Inventory/Receivables Growth", "NCLT/Insolvency"
        ]
        for item in default_items:
            llm_flags.append(RiskFlag(
                flag_type=item,
                severity="Low",
                detail="No evidence found in reviewed documents and news.",
                source="Fallback Generation"
            ))

    # Combine all layers
    all_flags = layer1_flags + layer2_flags + llm_flags

    # ---------------------------------------------------------
    # Overall Risk Rating (Deterministic Python Math)
    # ---------------------------------------------------------
    crit_count = sum(1 for f in all_flags if f.severity == "Critical")
    high_count = sum(1 for f in all_flags if f.severity == "High")
    med_count  = sum(1 for f in all_flags if f.severity == "Medium")
    
    if crit_count > 0 or high_count >= 2:
        overall_rating = "Very High"
    elif high_count == 1:
        overall_rating = "High"
    elif med_count >= 2:
        overall_rating = "Moderate"
    else:
        overall_rating = "Low"
        
    logger.info(f"[{ticker}] Evaluated Risk Rating: {overall_rating}. (Cr: {crit_count}, Hi: {high_count}, Med: {med_count})")

    return RiskAnalysis(
        flags=all_flags,
        overall_risk_rating=overall_rating,
        risk_narrative=narrative,
        promoter_pledge_pct=pledge_pct
    )

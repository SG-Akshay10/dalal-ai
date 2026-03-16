"""
Rewritten for StockSense AI Pipeline v2.
Changes made:
- Prompt Quality: Removed instructions inviting hallucination, added STRICT RULES and REASONING STEPS blocks,
  provided India-market grounded few-shot examples, and enforced null constraints strictly.
- Code Quality: Used 3 targeted retrieval queries, deduplicated chunks by 120-char fingerprint, added a
  < MIN_THRESHOLD guard (>= 3 chunks), replaced 0.0 with None for numeric fallbacks, separated 
  json.JSONDecodeError handling from general Exceptions, and added comprehensive logging.
- `data_freshness_days` is now computed deterministically in Python and injected into the fallbacks.
"""

import json
import logging
from datetime import datetime, timezone

from langchain_core.prompts import PromptTemplate

from app.llm_provider import get_llm_client
from app.schemas.analysis import FundamentalAnalysis
from app.vector_store.retriever import retrieve_documents

logger = logging.getLogger(__name__)

FUNDAMENTAL_PROMPT = """You are an expert equity research analyst covering the Indian stock market.
Analyze the following retrieved documents for the stock {ticker} and extract the key fundamental information.

<STRICT_RULES>
1. DATA ABSENCE: If a field cannot be populated from the provided context, it MUST be set to null. Do not use default values (like 0.0) and do not invent figures.
2. UNIT NORMALISATION: All monetary values MUST be in INR Crores. If the document uses Lakhs or Millions, you must convert them to Crores and note the conversion in the `red_flags` list. Convert percentages to a plain float (e.g., 15.5% -> 15.5).
3. FORMATTING: Enforce "Q1FY25" style for quarters (Apr-Jun=Q1, Jul-Sep=Q2, Oct-Dec=Q3, Jan-Mar=Q4).
4. GROWTH CALCULATION: `yoy_growth_pct` and `qoq_growth_pct` MUST be null if the comparison period is absent. Never compute growth from a single data point.
5. COMMENTARY LIMITS:
   - `management_commentary`: Paraphrase only, maximum 3 sentences. MUST include specific numeric guidance if stated (e.g. "12-14% revenue growth guidance for FY26").
   - `red_flags`: Include ONLY material concerns (e.g., audit qualifications, margin compression for 3+ quarters, D/E spikes, going concern language). No generic market or macro risks.
   - `fcf_commentary`: Brief analysis of Free Cash Flow generation or cash burn.
6. PRECOMPUTED VALUES: You are provided with `data_freshness_days` = {precomputed_data_freshness_days}. Use this directly in your output schema, do not override it.
7. NO HALLUCINATION: Extract only what is present in the text. No verbatim quotes.
</STRICT_RULES>

<REASONING_STEPS>
Perform the following chain of thought before generating the schema object:
1. Identify the company and available financial periods present in the text.
2. Extract absolute figures (Revenue, EBITDA, PAT) and convert them to INR Crores if needed.
3. Identify relevant growth percentages (YoY, QoQ). State their absence if missing.
4. Extract margin percentages (Gross Margin, Net Margin).
5. Extract Debt-to-Equity ratio, if stated.
6. Summarise Management Commentary (max 3 sentences) focusing on specific guidance numbers.
7. Filter any red flags to only include material corporate or fundamental risks.
</REASONING_STEPS>

Retrieved Documents for {ticker}:
{documents}

Always return your final response as a valid JSON object matching the following schema structure. Wait until after your reasoning to generate the JSON.

<EXAMPLE_SCHEMA>
{{
  "revenue_trend": [{{"quarter": "Q1FY25", "value": 2450.5, "yoy_growth_pct": 12.4, "qoq_growth_pct": 2.1}}],
  "ebitda_trend": [{{"quarter": "Q1FY25", "value": 450.0, "yoy_growth_pct": null, "qoq_growth_pct": null}}],
  "pat_trend": [{{"quarter": "Q1FY25", "value": 210.5, "yoy_growth_pct": 8.5, "qoq_growth_pct": -1.2}}],
  "gross_margin": 45.2,
  "net_margin": 8.6,
  "fcf_commentary": "Strong operating cash flow of 300 Cr fully covered capex requirements.",
  "debt_equity_ratio": 0.45,
  "management_commentary": "Management expects 12-14% revenue growth guidance for FY26 driven by new product launches. Margins are expected to remain stable at current levels. Capex for the next year is planned at 500 Cr.",
  "red_flags": ["Auditor qualified trade receivables of 50 Cr.", "Report uses Millions; converted to Crores for analysis."],
  "data_freshness_days": {precomputed_data_freshness_days}
}}
</EXAMPLE_SCHEMA>
"""


def parse_date(date_str: str):
    """Attempt to parse common date formats for freshness calculation."""
    if not date_str:
        return None
    try:
        if "T" in date_str:
            return datetime.fromisoformat(date_str.replace("Z", "+00:00")).date()
        else:
            return datetime.strptime(date_str[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def extract_fundamentals(ticker: str, provider: str = None) -> FundamentalAnalysis:
    """Run the fundamental agent pipeline: Retrieve docs -> use LLM -> Return structured analysis."""
    
    # 1. Retrieve using 3 separate targeted queries
    queries = [
        f"{ticker} quarterly financial results earnings revenue profit margins",
        f"{ticker} concall transcript management commentary guidance outlook",
        f"{ticker} annual report free cash flow debt capex working capital"
    ]
    
    raw_docs = []
    for q in queries:
        try:
            docs = retrieve_documents(q, ticker=ticker, top_k=5)
            raw_docs.extend(docs)
        except Exception as e:
            logger.warning(f"[{ticker}] Retrieval failed for query '{q}': {e}")
            
    # 2. Deduplicate chunks by first-120-char fingerprint and determine latest document date
    unique_docs = []
    seen_fingerprints = set()
    latest_date = None
    
    for d in raw_docs:
        text = d.get('text', '')
        if not text:
            continue
            
        fingerprint = text[:120].strip()
        if fingerprint not in seen_fingerprints:
            seen_fingerprints.add(fingerprint)
            unique_docs.append(d)
            
            # Parse dates for freshness computation
            date_str = d.get('metadata', {}).get('date')
            parsed_date = parse_date(date_str)
            if parsed_date:
                if latest_date is None or parsed_date > latest_date:
                    latest_date = parsed_date

    # Deterministically calculate freshness
    if latest_date:
        data_freshness_days = (datetime.now(timezone.utc).date() - latest_date).days
    else:
        # -1 when no parseable date is found
        data_freshness_days = -1

    logger.info(f"[{ticker}] Retrieved {len(unique_docs)} unique chunks after deduplication. Calculated freshness: {data_freshness_days} days.")

    # 3. Guard pattern: Require >= 3 chunks
    if len(unique_docs) < 3:
        logger.warning(f"[{ticker}] Insufficient chunks retrieved ({len(unique_docs)} < 3). Triggering early fallback.")
        return FundamentalAnalysis(
            revenue_trend=[],
            ebitda_trend=[],
            pat_trend=[],
            gross_margin=None,
            net_margin=None,
            fcf_commentary="Insufficient documents retrieved to run analysis.",
            debt_equity_ratio=None,
            management_commentary="Insufficient continuous chunks to summarise.",
            red_flags=[f"Low retrieval volume ({len(unique_docs)} chunks)"],
            data_freshness_days=data_freshness_days
        )

    docs_text = "\n\n".join([
        f"Source: {d.get('metadata', {}).get('source')} | Date: {d.get('metadata', {}).get('date')}\n{d.get('text')}" 
        for d in unique_docs
    ])

    prompt = PromptTemplate.from_template(FUNDAMENTAL_PROMPT)
    formatted_prompt = prompt.format(
        ticker=ticker, 
        documents=docs_text, 
        precomputed_data_freshness_days=data_freshness_days
    )

    llm = get_llm_client(provider)

    try:
        # Default with_structured_output path
        if hasattr(llm, "with_structured_output"):
            structured_llm = llm.with_structured_output(
                FundamentalAnalysis, 
                method="json_mode" if "openai" in str(type(llm)).lower() else "function_calling"
            )
            try:
                result = structured_llm.invoke(formatted_prompt)
                
                # Override LLM freshness calculation just in case
                if hasattr(result, "data_freshness_days"):
                    setattr(result, "data_freshness_days", data_freshness_days)
                elif isinstance(result, dict):
                    result["data_freshness_days"] = data_freshness_days
                    
                logger.info(f"[{ticker}] Fundamental node completed successfully via structured output.")
                return result
            except json.JSONDecodeError as jde:
                logger.error(f"[{ticker}] JSON decode error in structured output: {jde}")
            except Exception as e:
                logger.error(f"[{ticker}] Structured output invoke error: {e}")

        # Fallback raw call and JSON parse if structured output fails or is not available
        response = llm.invoke([
            {"role": "system", "content": "Follow the prompt carefully. Make sure your final output is a valid JSON. You may output chain-of-thought before your JSON format, but isolate the JSON properly."},
            {"role": "user", "content": formatted_prompt}
        ])

        # Clean markdown codeblocks and find JSON chunk
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
        data['data_freshness_days'] = data_freshness_days
        
        result = FundamentalAnalysis(**data)
        logger.info(f"[{ticker}] Fundamental node completed successfully via raw invoke fallback.")
        return result

    except json.JSONDecodeError as jde:
        # Separated JSONDecodeError
        logger.error(f"[{ticker}] JSONDecodeError processing fundamental response: {jde}")
        return FundamentalAnalysis(
            revenue_trend=[],
            ebitda_trend=[],
            pat_trend=[],
            gross_margin=None,
            net_margin=None,
            fcf_commentary=None,
            debt_equity_ratio=None,
            management_commentary=None,
            red_flags=[f"Agent error processing fundamental data (JSONDecodeError): {str(jde)}"],
            data_freshness_days=data_freshness_days
        )
    except Exception as e:
        # General Exceptions
        logger.error(f"[{ticker}] Agent generic error: {e}")
        return FundamentalAnalysis(
            revenue_trend=[],
            ebitda_trend=[],
            pat_trend=[],
            gross_margin=None,
            net_margin=None,
            fcf_commentary=None,
            debt_equity_ratio=None,
            management_commentary=None,
            red_flags=[f"Agent error processing fundamental data: {str(e)}"],
            data_freshness_days=data_freshness_days
        )

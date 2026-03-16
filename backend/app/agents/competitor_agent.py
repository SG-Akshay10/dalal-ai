"""
Rewritten for StockSense AI Pipeline v2.
Changes made:
- Prompt Quality: Added STRICT RULES and REASONING STEPS blocks. Defined "direct competitor" strictly (3-tier screening), 
  mandated NSE/BSE ticker formats without suffixes, and added rules for unique business exceptions (e.g. IRCTC, MCX).
  Specified fields like `business_description` (min 3 sentences covering segments, models) and `rationale` (segment overlap).
- Code Quality: Uses 3 targeted queries (business segments, competitors, industry landscape) with deduplication by 
  first-120-char fingerprint. Unlike fundamental_agent, this does NOT short-circuit for empty docs, relying instead on the 
  LLM's training knowledge and strict verification rules.
- Fallbacks replace 0.0 with None where applicable and include robust JSON parsing error separation. 
- Extensive logging added, specifically logging the identified ticker list.
"""

import json
import logging

from langchain_core.prompts import PromptTemplate

from app.llm_provider import get_llm_client
from app.schemas.analysis import CompetitorAnalysis
from app.vector_store.retriever import retrieve_documents

logger = logging.getLogger(__name__)

COMPETITOR_PROMPT = """You are an expert equity research analyst specializing in the Indian stock market.
Identify 2-3 direct listed competitors for the company with ticker {ticker}.

<STRICT_RULES>
1. DEFINITION OF COMPETITOR (3-Tier Screen):
   - Tier 1: Do they share the SAME primary business segment (>50% revenue overlap)?
   - Tier 2: Are they listed on NSE or BSE?
   - Tier 3: Are they direct competitive peers (not just in the same broad sector)? 
   If a company fails this screen, DO NOT list them.
2. TICKER FORMAT: Ticker must be the exact NSE/BSE symbol WITHOUT suffixes (e.g., use "TCS", not "TCS.NS" or "TCS.BO").
   Reference anchors: RELIANCE, TCS, HDFCBANK, INFY, ICICIBANK, SBIN, ITC, BHARTIARTL, L&T, BAJFINANCE, HINDUNILVR, AXISBANK, MARUTI, SUNPHARMA, TATAMOTORS.
3. UNIQUE BUSINESS EXCEPTION: If the target company has no listed peer with a highly similar business model (e.g. IRCTC, MCX, BSE, CDSL, CAMS), return an EMPTY competitors list `[]` and explain this in the `business_description`. DO NOT fabricate distant peers.
4. BUSINESS DESCRIPTION: Minimum 3 sentences. Must explicitly cover: The primary segment and its revenue share (if known), key products/services, geographic footprint, and the business model type (B2B/B2C/B2G).
5. RATIONALE: Must answer specifically: What revenue segment overlaps, and what share of the competitor's business competes directly with the target?
6. NO HALLUCINATION: Only list factual, existing Indian listed entities.
</STRICT_RULES>

<REASONING_STEPS>
Perform the following chain of thought before generating the schema object:
1. Synthesise the target company's business model from the context and define its primary segment.
2. Identify the B2B/B2C/B2G nature and general product mix.
3. Scan for direct competitors. Check the 3-Tier screen for each potential competitor.
4. Check for the "Unique Business Exception". If true, clear the competitor list.
5. Format the tickers correctly (no suffixes).
6. Draft the business description (min 3 sentences) and the rationale for each valid competitor.
</REASONING_STEPS>

Use the following context from recent documents to inform your analysis (note: if context is sparse, use your training data but still adhere strictly to the rules above):
Retrieved Documents for {ticker}:
{documents}

Always return your final response as a valid JSON object matching the following schema structure. Wait until after your reasoning to generate the JSON.

<EXAMPLE_SCHEMA>
{{
  "competitors": [
    {{
      "ticker": "INFY",
      "name": "Infosys Limited",
      "exchange": "NSE",
      "rationale": "Over 60% of both companies' revenue comes from BFSI and Retail IT Services, competing directly for large US enterprise contracts."
    }},
    {{
      "ticker": "HCLTECH",
      "name": "HCL Technologies Limited",
      "exchange": "NSE",
      "rationale": "Direct competitor in Infrastructure Management Services and Digital Engineering, with a highly overlapping global B2B client base."
    }}
  ],
  "primary_sector": "IT Services & Consulting",
  "business_description": "TCS is India's largest IT services provider, deriving the majority of its revenue from application development and maintenance in the BFSI sector. The company primarily operates a B2B model serving large global enterprises across North America and Europe. Key service lines include cloud migration, cognitive business operations, and enterprise software implementation."
}}
</EXAMPLE_SCHEMA>
"""

def identify_competitors(ticker: str, provider: str = None) -> CompetitorAnalysis:
    """Run the competitor identification agent."""
    
    # 1. Retrieve using 3 separate targeted queries
    queries = [
        f"{ticker} business segments revenue mix products services",
        f"{ticker} competitors peer companies market share industry landscape",
        f"{ticker} industry overview listed peers sector comparison benchmark"
    ]
    
    raw_docs = []
    for q in queries:
        try:
            docs = retrieve_documents(q, ticker=ticker, top_k=5)
            raw_docs.extend(docs)
        except Exception as e:
            logger.warning(f"[{ticker}] Retrieval failed for query '{q}': {e}")
            
    # 2. Deduplicate chunks by first-120-char fingerprint
    unique_docs = []
    seen_fingerprints = set()
    
    for d in raw_docs:
        text = d.get('text', '')
        if not text:
            continue
            
        fingerprint = text[:120].strip()
        if fingerprint not in seen_fingerprints:
            seen_fingerprints.add(fingerprint)
            unique_docs.append(d)

    logger.info(f"[{ticker}] Retrieved {len(unique_docs)} unique chunks after deduplication for competitor extraction.")
    # No short-circuit for empty docs on Competitor Agent as per instructions.

    docs_text = "\n\n".join([
        f"Source: {d.get('metadata', {}).get('source')}\n{d.get('text')}" 
        for d in unique_docs
    ])

    prompt = PromptTemplate.from_template(COMPETITOR_PROMPT)
    formatted_prompt = prompt.format(ticker=ticker, documents=docs_text)

    llm = get_llm_client(provider)

    try:
        # Default with_structured_output path
        if hasattr(llm, "with_structured_output"):
            structured_llm = llm.with_structured_output(
                CompetitorAnalysis, 
                method="json_mode" if "openai" in str(type(llm)).lower() else "function_calling"
            )
            try:
                result = structured_llm.invoke(formatted_prompt)
                
                # Log tickers on structured path
                tickers = [c.ticker for c in getattr(result, "competitors", [])]
                logger.info(f"[{ticker}] Competitor node identified tickers: {tickers}")
                
                return result
            except json.JSONDecodeError as jde:
                logger.error(f"[{ticker}] JSON decode error in structured output: {jde}")
            except Exception as e:
                logger.error(f"[{ticker}] Structured output invoke error: {e}")

        # Fallback raw call and JSON parse
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
        result = CompetitorAnalysis(**data)
        
        # Log tickers on raw JSON parsing path
        tickers = [c.ticker for c in getattr(result, "competitors", [])]
        logger.info(f"[{ticker}] Competitor node completed raw invoke. Identified tickers: {tickers}")
        
        return result

    except json.JSONDecodeError as jde:
        logger.error(f"[{ticker}] JSONDecodeError processing competitor response: {jde}")
        return CompetitorAnalysis(
            competitors=[], 
            primary_sector="Unknown", 
            business_description=f"Agent error parsing competitor data: {str(jde)}"
        )
    except Exception as e:
        logger.error(f"[{ticker}] Agent generic error: {e}")
        return CompetitorAnalysis(
            competitors=[], 
            primary_sector="Unknown", 
            business_description=f"Agent error identifying competitors: {str(e)}"
        )

"""
Rewritten for StockSense AI Pipeline v2.
Changes made:
- Retrieval: Uses 3 queries covering industry TAM, policy/regulation, and index/FII flows. Deduplicates by 120-char fingerprint.
- Live Data: Implemented non-blocking attempt to call `live_data_service.fetch_sector_data(ticker)`.
- Prompt Quality & STRICT RULES: 
  - Controlled vocabulary for `sector_name` (~28 NSE-aligned options).
  - Explicit multi-criteria definitions for `growth_stage` (Emerging, High Growth, Mature, Declining) with % thresholds and Indian examples.
  - Strict null enforcement on `index_performance_ytd`.
  - Enforced `(inferred)` suffix string for `fii_flow_trend` if not explicitly given.
  - Strict formatting for `policy_tailwinds` and `policy_headwinds` (Policy Name: 1-sentence impact).
  - `regulatory_summary` rule to explicitly name SEBI/RBI/MoF/DPIIT, note intent, and a forward risk/opportunity.
- Code Reliability: Robust python parsing fallback with specific JSONDecodeError handling. Fallback assigns `index_performance_ytd=None` and `sector_name="Unknown — agent error"` strictly.
"""

import json
import logging

from langchain_core.prompts import PromptTemplate
from app.llm_provider import get_llm_client
from app.schemas.analysis import SectorAnalysis
from app.vector_store.retriever import retrieve_documents

logger = logging.getLogger(__name__)

SECTOR_PROMPT = """You are an expert equity research analyst focusing on macroeconomics and sector analysis in India.
Analyze the operating sector for the company with ticker {ticker}.

<STRICT_RULES>
1. SECTOR NAME: Choose ONLY from this controlled vocabulary:
   - "IT Services & Software", "Banking & Financial Services", "Healthcare Services & Hospitals", "Pharmaceuticals & Biotechnology", "Defence & Aerospace", "Automobile & Auto Components", "FMCG & Consumer Durables", "Telecommunications", "Oil, Gas & Consumable Fuels", "Metals & Mining", "Power & Renewable Energy", "Construction & Engineering", "Real Estate & Infrastructure", "Capital Goods & Manufacturing", "Chemicals & Petrochemicals", "Textiles, Apparel & Footwear", "Media & Entertainment", "Retail & E-Commerce", "Transportation & Logistics", "Agrochemicals & Fertilizers", "Building Materials & Cement", "Hotels, Resorts & Tourism", "Financial Institutions (NBFCs)", "Insurance", "Diversified", "Utilities", "Consumer Services", "Paper & Forest Products".
2. GROWTH STAGE Criteria:
   - "Emerging": Nascent industry, high revenue growth potential (>25%), low current profitability. (e.g. EV, Green Hydrogen, Drone Tech)
   - "High Growth": Expanding rapidly, strong growth (15-25%), expanding margins. (e.g. Specialty Chemicals, Data Centers, Renewable Energy)
   - "Mature": Established, stable single-digit growth (5-10%), high cash generation, consolidated market. (e.g. FMCG, Large-cap IT, Private Banking)
   - "Declining": Negative or stagnant growth (<5%), secular headwinds. (e.g. Print Media, Thermal Power)
3. INDEX PERFORMANCE YTD: This MUST be `null` unless a specific numeric figure explicitly appears in the documents or live data. NEVER invent or estimate this field. (Convert percentages to a plain float, e.g. 15.5)
4. FII FLOW TREND: If explicitly stated, state "Strong Inflow", "Moderate Inflow", "Neutral", "Moderate Outflow", or "Strong Outflow". If you have to infer it from macro context, you MUST append " (inferred)" to the value (e.g. "Moderate Inflow (inferred)").
5. TAILWINDS & HEADWINDS: Must be formatted exactly as "[Policy/Event Name]: [One sentence investment impact]". Absolutely no single-word entries.
6. REGULATORY SUMMARY: Write EXACTLY 3-5 sentences. You must explicitly name at least one Indian regulator or ministry (e.g. SEBI, RBI, MoF, DPIIT). Detail the direction of regulatory intent and state one forward-looking risk or opportunity.
7. NO HALLUCINATION: Rely strictly on provided documents and live data. Do not hallucinate metrics. 
</STRICT_RULES>

<REASONING_STEPS>
1. Review the data to map the company to one of the 28 exact sector names.
2. Evaluate TAM/Growth to classify into the 4 strict Growth Stages.
3. Check specifically for an exact numeric index performance YTD value. If none exists, note it will be null.
4. Assess FII/DII data to determine flow trend. If inferred, note the required suffix.
5. Identify specific regulatory policies, events, schemes, or headwinds. Format them as [Name]: [Impact].
6. Draft the 3-5 sentence regulatory summary naming the specific regulator and future intent.
</REASONING_STEPS>

Retrieved Document Context:
{documents}

Live Market/Macro Data (if connected):
{live_data}

Always return your final response as a valid JSON object matching the following schema. Wait until after your reasoning to generate the JSON.

<EXAMPLE_SCHEMA>
{{
  "sector_name": "Banking & Financial Services",
  "growth_stage": "Mature",
  "index_performance_ytd": 12.5,
  "fii_flow_trend": "Moderate Inflow",
  "policy_tailwinds": ["Union Budget: Increased infrastructure capex driving corporate credit demand.", "RBI Rate Pause: Supporting stable net interest margins for large private banks."],
  "policy_headwinds": ["RBI Unsecured Guidelines: Increased risk weights tightening capital availability for retail lending."],
  "regulatory_summary": "The RBI continues to prioritize systemic stability, tightening regulations around unsecured retail credit and NBFC exposures. This regulatory intent aims to curb over-leveraging while maintaining liquidity in productive sectors. A key forward-looking risk is further RBI intervention in deposit pricing if credit-deposit ratio imbalances persist, potentially squeezing margins for smaller lenders."
}}
</EXAMPLE_SCHEMA>
"""

def analyze_sector(ticker: str, provider: str = None) -> SectorAnalysis:
    """Run the sector analysis agent incorporating historical context and live macro data."""
    
    # 1. Fetch live sector data (non-blocking)
    live_data_payload = "Not available"
    try:
        from app.services import live_data_service
        try:
            live_result = live_data_service.fetch_sector_data(ticker)
            if live_result:
                live_data_payload = str(live_result)
        except AttributeError:
            # Service module exists but method is not implemented yet
            pass
        except Exception as e:
            logger.warning(f"[{ticker}] Error fetching live sector data: {e}")
    except ImportError:
        pass

    # 2. Retrieve using 3 targeted queries
    queries = [
        f"{ticker} sector industry overview market size growth TAM",
        f"{ticker} government policy regulatory PLI scheme budget SEBI circular",
        f"{ticker} sector index performance FII DII institutional flows macro headwinds tailwinds"
    ]
    
    raw_docs = []
    for q in queries:
        try:
            docs = retrieve_documents(q, ticker=ticker, top_k=5)
            raw_docs.extend(docs)
        except Exception as e:
            logger.warning(f"[{ticker}] Retrieval failed for query '{q}': {e}")
            
    # 3. Deduplicate chunks by first-120-char fingerprint
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

    logger.info(f"[{ticker}] Retrieved {len(unique_docs)} unique chunks for sector extraction.")

    if not unique_docs and live_data_payload == "Not available":
        return SectorAnalysis(
            sector_name="Unknown — agent error",
            growth_stage="Mature",
            index_performance_ytd=None,
            fii_flow_trend="Neutral (inferred)",
            policy_tailwinds=[],
            policy_headwinds=[],
            regulatory_summary="Insufficient documents and no live data to analyze sector."
        )

    docs_text = "\n\n".join([
        f"Source: {d.get('metadata', {}).get('source')}\n{d.get('text')}" 
        for d in unique_docs
    ])

    prompt = PromptTemplate.from_template(SECTOR_PROMPT)
    formatted_prompt = prompt.format(
        ticker=ticker, 
        documents=docs_text, 
        live_data=live_data_payload
    )

    llm = get_llm_client(provider)

    try:
        # Structured LLM Path
        if hasattr(llm, "with_structured_output"):
            structured_llm = llm.with_structured_output(
                SectorAnalysis, 
                method="json_mode" if "openai" in str(type(llm)).lower() else "function_calling"
            )
            try:
                result = structured_llm.invoke(formatted_prompt)
                logger.info(f"[{ticker}] Sector node completed successfully via structured output. Sector: {getattr(result, 'sector_name', 'Unknown')}")
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
        result = SectorAnalysis(**data)
        logger.info(f"[{ticker}] Sector node completed successfully via raw invoke fallback. Sector: {result.sector_name}")
        return result

    except json.JSONDecodeError as jde:
        logger.error(f"[{ticker}] JSONDecodeError processing sector response: {jde}")
        return SectorAnalysis(
            sector_name="Unknown — agent error",
            growth_stage="Mature",
            index_performance_ytd=None,
            fii_flow_trend="Neutral (inferred)",
            policy_tailwinds=[],
            policy_headwinds=[],
            regulatory_summary=f"Agent error processing sector data (JSONDecodeError): {str(jde)}"
        )
    except Exception as e:
        logger.error(f"[{ticker}] Agent generic error in sector analysis: {e}")
        return SectorAnalysis(
            sector_name="Unknown — agent error",
            growth_stage="Mature",
            index_performance_ytd=None,
            fii_flow_trend="Neutral (inferred)",
            policy_tailwinds=[],
            policy_headwinds=[],
            regulatory_summary=f"Agent error analyzing policy context: {str(e)}"
        )

import json
from app.llm_provider import get_llm_client
from app.vector_store.retriever import retrieve_documents
from app.schemas.report import SectorAnalysis
from langchain_core.prompts import PromptTemplate

SECTOR_PROMPT = """You are an expert equity research analyst focusing on macroeconomics and sector analysis in India.
Analyze the operating sector for the company with ticker {ticker}. 

Use the following context to assess the sector's performance, FII flows, and regulatory environment:
{documents}

Provide:
1. The sector name.
2. The current growth stage (e.g., Emerging, High Growth, Mature, Declining).
3. The sector's index performance and trend.
4. The policy/regulatory context (recent budgets, policies, or structural shifts).

Always return your response as a valid JSON object matching the following schema:
{{
  "sector_name": "...",
  "growth_stage": "...",
  "index_performance": "...",
  "policy_context": "..."
}}
"""

def analyze_sector(ticker: str, provider: str = None) -> SectorAnalysis:
    """Run the sector analysis agent."""
    query = f"{ticker} sector industry performance index policy FII flows"
    docs = retrieve_documents(query, ticker=ticker, top_k=5)
    
    docs_text = "\n\n".join([f"Source: {d['metadata'].get('source')}\n{d['text']}" for d in docs])
    
    if not docs_text:
        return SectorAnalysis(
            sector_name="Unknown",
            growth_stage="Unknown",
            index_performance="No data available",
            policy_context="No policy data available"
        )

    prompt = PromptTemplate.from_template(SECTOR_PROMPT)
    formatted_prompt = prompt.format(ticker=ticker, documents=docs_text)
    
    llm = get_llm_client(provider)
    
    try:
        if hasattr(llm, "with_structured_output"):
            structured_llm = llm.with_structured_output(SectorAnalysis, method="json_mode" if "openai" in str(type(llm)).lower() else "function_calling")
            try:
                result = structured_llm.invoke(formatted_prompt)
                return result
            except Exception:
                pass
                
        # Fallback raw call and JSON parse
        response = llm.invoke([
            {"role": "system", "content": "You output JSON only."},
            {"role": "user", "content": formatted_prompt}
        ])
        
        content = response.content.replace("```json", "").replace("```", "").strip()
        data = json.loads(content)
        return SectorAnalysis(**data)
        
    except Exception as e:
        return SectorAnalysis(
            sector_name="Error",
            growth_stage="Error",
            index_performance=f"Failed to analyze index performance: {str(e)}",
            policy_context="Error analyzing policy context."
        )

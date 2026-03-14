import json

from langchain_core.prompts import PromptTemplate

from app.llm_provider import get_llm_client
from app.schemas.analysis import CompetitorAnalysis
from app.vector_store.retriever import retrieve_documents

COMPETITOR_PROMPT = """You are an expert equity research analyst specializing in the Indian stock market.
Identify 2-3 direct listed competitors for the company with ticker {ticker}.

Use the following context from recent documents to understand the company's operating segments:
{documents}

For each competitor provide:
1. The stock ticker (e.g. TCS for Tata Consultancy Services)
2. The company name
3. A short rationale (1-2 sentences) explaining why they are a competitor.
4. The exchange it's listed on (NSE or BSE).

Also summarize the target company's primary sector and a brief description of its business segments.

Always return your response as a valid JSON object matching the following schema:
{{
  "competitors": [
    {{"ticker": "...", "name": "...", "exchange": "NSE", "rationale": "..."}}
  ],
  "primary_sector": "...",
  "business_description": "..."
}}
"""

def identify_competitors(ticker: str, provider: str = None) -> CompetitorAnalysis:
    """Run the competitor identification agent."""
    query = f"{ticker} core business segments operating products services"
    docs = retrieve_documents(query, ticker=ticker, top_k=5)

    docs_text = "\n\n".join([f"Source: {d['metadata'].get('source')}\n{d['text']}" for d in docs])

    prompt = PromptTemplate.from_template(COMPETITOR_PROMPT)
    formatted_prompt = prompt.format(ticker=ticker, documents=docs_text)

    llm = get_llm_client(provider)

    try:
        if hasattr(llm, "with_structured_output"):
            structured_llm = llm.with_structured_output(CompetitorAnalysis, method="json_mode" if "openai" in str(type(llm)).lower() else "function_calling")
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
        return CompetitorAnalysis(**data)

    except Exception:
        return CompetitorAnalysis(competitors=[], primary_sector="Unknown", business_description="Error identifying competitors")

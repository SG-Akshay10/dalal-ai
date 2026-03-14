import json

from langchain_core.prompts import PromptTemplate

from app.llm_provider import get_llm_client
from app.schemas.analysis import FundamentalAnalysis
from app.vector_store.retriever import retrieve_documents

FUNDAMENTAL_PROMPT = """You are an expert equity research analyst.
Analyze the following retrieved documents for the stock {ticker} and extract the key fundamental information.

Extract or estimate the following KPIs and trends based on the context:
1. Revenue, EBITDA, and PAT trends across quarters (YoY/QoQ).
2. Current gross margin and net margin percentage.
3. Free cash flow commentary.
4. Current Debt-to-Equity ratio.
5. Management commentary and guidance.
6. Key risks or red flags.

Retrieved Documents:
{documents}

Always return your response as a valid JSON object matching the following schema. Use 0.0 or empty strings if data is unavailable.
{{
  "revenue_trend": [{{"quarter": "...", "value": 0.0, "yoy_growth_pct": 0.0, "qoq_growth_pct": 0.0}}],
  "ebitda_trend": [{{"quarter": "...", "value": 0.0, "yoy_growth_pct": 0.0, "qoq_growth_pct": 0.0}}],
  "pat_trend": [{{"quarter": "...", "value": 0.0, "yoy_growth_pct": 0.0, "qoq_growth_pct": 0.0}}],
  "gross_margin": 0.0,
  "net_margin": 0.0,
  "fcf_commentary": "...",
  "debt_equity_ratio": 0.0,
  "management_commentary": "...",
  "red_flags": ["...", "..."],
  "data_freshness_days": 1
}}
"""

def extract_fundamentals(ticker: str, provider: str = None) -> FundamentalAnalysis:
    """Run the fundamental agent pipeline: Retrieve docs -> use LLM -> Return structured analysis."""
    # 1. Retrieve the top 10 relevant document chunks
    query = f"{ticker} quarterly financial results earnings revenue profit margins"
    docs = retrieve_documents(query, ticker=ticker, top_k=10)

    docs_text = "\n\n".join([f"Source: {d['metadata'].get('source')} | Date: {d['metadata'].get('date')}\n{d['text']}" for d in docs])

    if not docs_text:
        return FundamentalAnalysis(
            revenue_trend=[],
            ebitda_trend=[],
            pat_trend=[],
            gross_margin=0.0,
            net_margin=0.0,
            fcf_commentary="No documents available to summarize.",
            debt_equity_ratio=0.0,
            management_commentary="No documents available to summarize.",
            red_flags=["Insufficient data - report may be incomplete"],
            data_freshness_days=-1
        )

    prompt = PromptTemplate.from_template(FUNDAMENTAL_PROMPT)
    formatted_prompt = prompt.format(ticker=ticker, documents=docs_text)

    llm = get_llm_client(provider)

    # Use OpenAI/Gemini/Sarvam structured output functionality via Tool Calling or JSON Mode
    # For cross-provider compatibility, asking for JSON and parsing is safer in LangChain generic endpoints
    # Or we can try .with_structured_output() which works well on ChatOpenAI and ChatGoogleGenerativeAI
    try:
        if hasattr(llm, "with_structured_output"):
            structured_llm = llm.with_structured_output(FundamentalAnalysis, method="json_mode" if "openai" in str(type(llm)).lower() else "function_calling")
            try:
                result = structured_llm.invoke(formatted_prompt)
                return result
            except Exception:
                # Fallback to standard parsing
                pass

        # Fallback raw call and JSON parse if structured output fails or is not available
        response = llm.invoke([
            {"role": "system", "content": "You output JSON only."},
            {"role": "user", "content": formatted_prompt}
        ])

        # Clean markdown codeblocks if any
        content = response.content.replace("```json", "").replace("```", "").strip()
        data = json.loads(content)
        return FundamentalAnalysis(**data)

    except Exception as e:
        return FundamentalAnalysis(
            revenue_trend=[],
            ebitda_trend=[],
            pat_trend=[],
            gross_margin=0.0,
            net_margin=0.0,
            fcf_commentary="",
            debt_equity_ratio=0.0,
            management_commentary="",
            red_flags=[f"Agent error processing fundamental data: {str(e)}"],
            data_freshness_days=1
        )

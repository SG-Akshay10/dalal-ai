"""
Rewritten for StockSense AI Pipeline v2.
Changes made:
- Function Signature: Added `fundamentals` and `company_name` explicitly to process target row data.
- Target Row: Populates fundamental metrics (revenue_growth, ebitda_margin, net_margin, debt_equity) dynamically from the `FundamentalAnalysis` object instead of live calls, preventing duplication. ROE is extracted selectively.
- Peer Rows: Fetches market multiples (`fetch_market_data`) and attempts to fetch operating metrics via `fetch_financial_metrics()` securely.
- Deterministic Relative Positioning: Automatically calculates the win-rate across 6 competitive metrics (Rev Gr%, EBITDA%, Net Mg%, ROE%, D/E, P/E ratio). 
  Maps to "Market Leader" (>= 60%), "In Line With Peers" (>= 40%), or "Lags Peers" (< 40%) requiring a minimum 2 comparable metrics.
- Advanced Formatting: Outputs an 8-column markdown competitive benchmark table, including a dynamically calculated PEER MEDIAN row, explicitly highlighting ← TARGET.
- Prompt Additions: Enforced `<STRICT_RULES>` for strengths/weaknesses syntax (preventing N/A and requiring substantial deltas) and set a strict 5-sentence narrative structure constraint.
- Resilient Fallbacks: Forces `relative_positioning` metrics into results upon LLM JSON corruption and guarantees that the table `rows` data object is never lost.
"""

import json
import logging
import statistics
from langchain_core.prompts import PromptTemplate

from app.llm_provider import get_llm_client
from app.schemas.analysis import PeerBenchmarkAnalysis, CompetitorAnalysis, PeerRow, FundamentalAnalysis
from app.services.live_data import live_data_service

logger = logging.getLogger(__name__)

PEER_BENCHMARK_PROMPT = """You are an expert equity research analyst.
Examine the following peer comparison table for the target stock ({ticker}) against its closest competitors.

<STRICT_RULES>
1. STRENGTHS & WEAKNESSES FORMAT:
   - Each entry MUST follow this exact string format: "[metric] of [X]% vs peer median of [Y]% — [Z]bps [lead/lag], driven by [one-clause explanation]".
   - Do NOT include a metric if the Target vs Median relative difference is <= 5% (i.e., only mention material divergences).
   - Prohibit "N/A" entries. Only cite data that exists in the table. If there are no material strengths/weaknesses respectively, return an empty list `[]`.
2. NARRATIVE STRUCTURE: Write EXACTLY 5 sentences with these assigned jobs:
   - Sentence 1: State the overall relative positioning based on the pre-computed classification and cite the defining defining metric driving this position.
   - Sentence 2: Discuss the durability of the target's primary strength against the peer group.
   - Sentence 3: Identify the primary weakness or lag, and establish whether it is structural (permanent) or transitional (temporary).
   - Sentence 4: Compare the relative valuation multiples (P/E or EV/EBITDA) against the operating quality metrics (is it trading at a fair premium/discount?).
   - Sentence 5: Mention any outlier peer that may be distorting the peer median, or conclude on general sector alignment if none exist.
3. SPARSE DATA GUIDANCE: If greater than half the table cells are N/A, you must explicitly state this in the first sentence of your narrative and restrict your analysis strictly to the available data points.
4. NO HALLUCINATION: Rely strictly on the table below. Do not inject numbers not provided here.
</STRICT_RULES>

<REASONING_STEPS>
1. Review the comparative table and evaluate the degree of completeness. (Determine Sparse Data conditions).
2. Compute relative deltas for strengths (delta > 5%). Draft strings following the exact format.
3. Compute relative deltas for weaknesses (delta < -5% for margins/growth, or > 5% for D/E and P/E). Draft strings following the exact format.
4. Formulate the 5-sentence narrative based on the structural rules. 
</REASONING_STEPS>

Competitor Benchmark Data:
{table_data}

Target Relative Positioning Classification: {computed_position}

Always return your final response as a valid JSON object matching the following schema structure. Wait until after your reasoning to generate the JSON.
<EXAMPLE_SCHEMA>
{{
  "strengths_vs_peers": ["EBITDA Margin of 24.5% vs peer median of 18.2% — 630bps lead, driven by superior scale and lower raw material procurement costs."],
  "weaknesses_vs_peers": ["Revenue Growth of 8.2% vs peer median of 15.5% — 730bps lag, driven by delayed enterprise spending in key markets."],
  "narrative": "..."
}}
</EXAMPLE_SCHEMA>
"""

def _safe_float(val) -> float | None:
    if val is None:
        return None
    try:
        if isinstance(val, str) and (val.lower() == 'n/a' or val.lower() == 'nan'):
            return None
        return float(val)
    except (ValueError, TypeError):
        return None

def build_peer_benchmark(
    ticker: str, 
    competitors: CompetitorAnalysis, 
    fundamentals: FundamentalAnalysis | None = None,
    company_name: str | None = None,
    provider: str = None
) -> PeerBenchmarkAnalysis:
    """Build the peer benchmarking table and analysis."""
    
    rows = []
    
    # 1. Target Row Population
    target_data = live_data_service.fetch_market_data(ticker)
    
    rev_growth = None
    if fundamentals and fundamentals.revenue_trend and len(fundamentals.revenue_trend) > 0:
        rev_growth = _safe_float(fundamentals.revenue_trend[-1].yoy_growth_pct)

    ebitda_margin = None
    if fundamentals and fundamentals.ebitda_trend and fundamentals.revenue_trend:
        if len(fundamentals.ebitda_trend) > 0 and len(fundamentals.revenue_trend) > 0:
            e_val = fundamentals.ebitda_trend[-1].value
            r_val = fundamentals.revenue_trend[-1].value
            if e_val and r_val and r_val > 0:
                ebitda_margin = (e_val / r_val) * 100
                ebitda_margin = round(ebitda_margin, 2)

    net_margin = getattr(fundamentals, 'net_margin', None)
    debt_equity = getattr(fundamentals, 'debt_equity_ratio', None)
    
    # Try fetch_market_data for ROE first, then fallback to fetch_financial_metrics
    target_roe = _safe_float(target_data.get("roe"))
    if target_roe is None:
        try:
            fin_metrics = live_data_service.fetch_financial_metrics(ticker)
            if fin_metrics:
                target_roe = _safe_float(fin_metrics.get("roe"))
        except AttributeError:
            pass
        except Exception as e:
            logger.debug(f"[{ticker}] Target ROE fallback fetch failed: {e}")

    target_pe = _safe_float(target_data.get("pe_ttm"))
    target_ev_eb = _safe_float(target_data.get("ev_ebitda"))

    rows.append(PeerRow(
        ticker=ticker,
        name=company_name if company_name else ticker,
        revenue_growth_yoy=rev_growth,
        ebitda_margin=ebitda_margin,
        net_margin=net_margin,
        roe=target_roe,
        debt_equity=debt_equity,
        pe_ratio=target_pe,
        ev_ebitda=target_ev_eb
    ))

    # 2. Peer Row Population
    peers = getattr(competitors, "competitors", []) if competitors else []
    
    for comp in peers:
        comp_ticker = comp.ticker
        c_pe = None
        c_ev = None
        c_rev_growth = None
        c_ebitda_margin = None
        c_net_margin = None
        c_roe = None
        c_debt_equity = None
        
        # Multiples
        try:
            c_mkt_data = live_data_service.fetch_market_data(comp_ticker)
            if c_mkt_data:
                c_pe = _safe_float(c_mkt_data.get("pe_ttm"))
                c_ev = _safe_float(c_mkt_data.get("ev_ebitda"))
                
                # Check if ROE is incidentally in market data
                c_roe = _safe_float(c_mkt_data.get("roe")) 
        except Exception as e:
            logger.warning(f"[{comp_ticker}] Failed fetching peer market data: {e}")
            
        # Operating Metrics
        try:
            c_fin_metrics = live_data_service.fetch_financial_metrics(comp_ticker)
            if c_fin_metrics:
                c_rev_growth = _safe_float(c_fin_metrics.get("revenue_growth"))
                c_ebitda_margin = _safe_float(c_fin_metrics.get("ebitda_margin"))
                c_net_margin = _safe_float(c_fin_metrics.get("net_margin"))
                c_debt_equity = _safe_float(c_fin_metrics.get("debt_equity"))
                if c_roe is None:
                    c_roe = _safe_float(c_fin_metrics.get("roe"))
        except AttributeError:
            pass # Method might not exist yet
        except Exception as e:
            logger.warning(f"[{comp_ticker}] Failed fetching peer financial metrics: {e}")

        rows.append(PeerRow(
            ticker=comp_ticker,
            name=comp.name,
            revenue_growth_yoy=c_rev_growth,
            ebitda_margin=c_ebitda_margin,
            net_margin=c_net_margin,
            roe=c_roe,
            debt_equity=c_debt_equity,
            pe_ratio=c_pe,
            ev_ebitda=c_ev
        ))

    # 3. Deterministic Computations (Medians & Relative Positioning)
    medians = {}
    
    def calc_median(attr_name):
        vals = [_safe_float(getattr(r, attr_name)) for r in rows[1:]] # exclude target (index 0)
        vals = [v for v in vals if v is not None]
        return statistics.median(vals) if vals else None

    if len(rows) > 1:
        medians['revenue_growth_yoy'] = calc_median('revenue_growth_yoy')
        medians['ebitda_margin'] = calc_median('ebitda_margin')
        medians['net_margin'] = calc_median('net_margin')
        medians['roe'] = calc_median('roe')
        medians['debt_equity'] = calc_median('debt_equity')
        medians['pe_ratio'] = calc_median('pe_ratio')
        medians['ev_ebitda'] = calc_median('ev_ebitda')
    else:
        medians = {k: None for k in ['revenue_growth_yoy', 'ebitda_margin', 'net_margin', 'roe', 'debt_equity', 'pe_ratio', 'ev_ebitda']}

    # Scoring Relative Positioning against 6 metrics
    comparable_count = 0
    beat_count = 0

    higher_is_better = ['revenue_growth_yoy', 'ebitda_margin', 'net_margin', 'roe']
    for metric in higher_is_better:
        t_val = _safe_float(getattr(rows[0], metric))
        m_val = medians.get(metric)
        if t_val is not None and m_val is not None:
            comparable_count += 1
            if t_val > m_val:
                beat_count += 1

    lower_is_better = ['debt_equity', 'pe_ratio']
    for metric in lower_is_better:
        t_val = _safe_float(getattr(rows[0], metric))
        m_val = medians.get(metric)
        if t_val is not None and m_val is not None:
            comparable_count += 1
            # For debt and P/E, lower is better. Guard against 0.
            if t_val < m_val:
                beat_count += 1

    computed_position = "In Line With Peers"
    if comparable_count >= 2:
        beat_ratio = beat_count / comparable_count
        if beat_ratio >= 0.60:
            computed_position = "Market Leader"
        elif beat_ratio < 0.40:
            computed_position = "Lags Peers"

    logger.info(f"[{ticker}] Peer Positioning: {computed_position} (Beats {beat_count}/{comparable_count} metrics).")

    # 4. Construct Table For Prompt
    def fm(v, is_pct=False):
        if v is None: return "N/A"
        return f"{v:.1f}%" if is_pct else f"{v:.2f}"

    table_lines = [
        "| Ticker | Rev Gr% | EBITDA% | Net Mg% | ROE% | D/E | P/E | EV/EBITDA |",
        "|---|---|---|---|---|---|---|---|"
    ]
    
    for i, r in enumerate(rows):
        marker = " ← TARGET" if i == 0 else ""
        t_mark = f"{r.ticker}{marker}"
        table_lines.append(
            f"| {t_mark} "
            f"| {fm(r.revenue_growth_yoy, True)} "
            f"| {fm(r.ebitda_margin, True)} "
            f"| {fm(r.net_margin, True)} "
            f"| {fm(r.roe, True)} "
            f"| {fm(r.debt_equity)} "
            f"| {fm(r.pe_ratio)} "
            f"| {fm(r.ev_ebitda)} |"
        )
        
    if len(rows) > 1:
        table_lines.append(
            f"| PEER MEDIAN "
            f"| {fm(medians['revenue_growth_yoy'], True)} "
            f"| {fm(medians['ebitda_margin'], True)} "
            f"| {fm(medians['net_margin'], True)} "
            f"| {fm(medians['roe'], True)} "
            f"| {fm(medians['debt_equity'])} "
            f"| {fm(medians['pe_ratio'])} "
            f"| {fm(medians['ev_ebitda'])} |"
        )

    table_data = "\n".join(table_lines)

    # 5. Connect to LLM
    prompt = PromptTemplate.from_template(PEER_BENCHMARK_PROMPT)
    formatted_prompt = prompt.format(
        ticker=ticker, 
        table_data=table_data,
        computed_position=computed_position
    )
    
    llm = get_llm_client(provider)

    result_data = {}
    try:
        if hasattr(llm, "with_structured_output"):
            from pydantic import BaseModel, Field
            class PeerOut(BaseModel):
                strengths_vs_peers: list[str]
                weaknesses_vs_peers: list[str]
                narrative: str
                
            structured_llm = llm.with_structured_output(
                PeerOut, 
                method="json_mode" if "openai" in str(type(llm)).lower() else "function_calling"
            )
            try:
                llm_res = structured_llm.invoke(formatted_prompt)
                result_data = {
                    "strengths_vs_peers": getattr(llm_res, "strengths_vs_peers", []),
                    "weaknesses_vs_peers": getattr(llm_res, "weaknesses_vs_peers", []),
                    "narrative": getattr(llm_res, "narrative", f"Peer comparison narrative processed for {ticker}.")
                }
            except json.JSONDecodeError as jde:
                logger.error(f"[{ticker}] JSON decode error in structured peer output: {jde}")
            except Exception as e:
                logger.error(f"[{ticker}] Structured output invoke error: {e}")

        if not result_data:
            response = llm.invoke([
                {"role": "system", "content": "You output JSON only. Must contain keys: strengths_vs_peers, weaknesses_vs_peers, narrative."},
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
                "strengths_vs_peers": data.get("strengths_vs_peers", []),
                "weaknesses_vs_peers": data.get("weaknesses_vs_peers", []),
                "narrative": data.get("narrative", f"Peer comparison logic computed mathematically for {ticker}.")
            }

        return PeerBenchmarkAnalysis(
            rows=rows,
            relative_positioning=computed_position,
            strengths_vs_peers=result_data["strengths_vs_peers"],
            weaknesses_vs_peers=result_data["weaknesses_vs_peers"],
            narrative=result_data["narrative"]
        )

    except Exception as e:
        logger.error(f"Error in Peer Benchmarking Agent for {ticker}: {e}")
        return PeerBenchmarkAnalysis(
            rows=rows, # Preserve the table data regardless
            relative_positioning=computed_position, # Preserve fallback calculation
            strengths_vs_peers=[],
            weaknesses_vs_peers=[],
            narrative=f"Agent error generating narrative texts: {str(e)}"
        )

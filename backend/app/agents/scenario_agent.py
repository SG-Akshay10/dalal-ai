import json
import logging
from langchain_core.prompts import PromptTemplate

from app.llm_provider import get_llm_client
from app.schemas.analysis import PriceTargetAnalysis, Scenario

logger = logging.getLogger(__name__)

SCENARIO_PROMPT = """You are a Lead Portfolio Manager synthesizing research.
Take the following analysis summaries for {ticker} and construct a Bull, Base, and Bear scenario with explicit price targets.

Current Price: {current_price}

1. Fundamentals: {fundamentals}
2. Valuation Verdict: {valuation}
3. Sentiment: {sentiment}
4. Peer Benchmark: {peers}
5. Technical Trend: {technical}
6. Risk Rating: {risk}

Task:
Calculate a probability-weighted price target (probability must sum to 1.0).
Determine conviction (High, Medium, Low) and investment horizon.
Identify upgrade and downgrade triggers.

Always return your response as a valid JSON object matching exactly this schema:
{{
  "weighted_price_target": 0.0,
  "expected_return_pct": 0.0,
  "scenarios": [
    {{
      "label": "Bull",
      "probability": 0.2,
      "key_assumptions": ["..."],
      "price_target_12m": 0.0,
      "upside_downside_pct": 0.0
    }},
    {{
      "label": "Base",
      "probability": 0.6,
      "key_assumptions": ["..."],
      "price_target_12m": 0.0,
      "upside_downside_pct": 0.0
    }},
    {{
      "label": "Bear",
      "probability": 0.2,
      "key_assumptions": ["..."],
      "price_target_12m": 0.0,
      "upside_downside_pct": 0.0
    }}
  ],
  "conviction": "Medium",
  "position_sizing_note": "...",
  "upgrade_triggers": ["..."],
  "downgrade_triggers": ["..."],
  "investment_horizon": "Medium (6–18M)"
}}
"""

def generate_scenarios(
    ticker: str, 
    fundamentals, valuation, sentiment, peers, technical, risk, 
    provider: str = None
) -> PriceTargetAnalysis:
    """Synthesize all analysis into actionable scenarios and price targets."""
    
    current_price = valuation.current_price if valuation and valuation.current_price else 100.0  # fallback
    
    fund_text = fundamentals.management_commentary if fundamentals else "N/A"
    val_text = valuation.valuation_verdict if valuation else "N/A"
    sent_text = sentiment.narrative if sentiment else "N/A"
    peer_text = peers.relative_positioning if peers else "N/A"
    tech_text = technical.trend_bias if technical else "N/A"
    risk_text = risk.overall_risk_rating if risk else "N/A"
    
    prompt = PromptTemplate.from_template(SCENARIO_PROMPT)
    formatted_prompt = prompt.format(
        ticker=ticker,
        current_price=current_price,
        fundamentals=fund_text[:200],
        valuation=val_text[:200],
        sentiment=sent_text[:100],
        peers=peer_text,
        technical=tech_text,
        risk=risk_text
    )
    
    llm = get_llm_client(provider)
    try:
        response = llm.invoke([
            {"role": "system", "content": "You output JSON only."},
            {"role": "user", "content": formatted_prompt}
        ])
        content = response.content.replace("```json", "").replace("```", "").strip()
        data = json.loads(content)
        
        scenarios = [Scenario(**s) for s in data.get("scenarios", [])]
        
        return PriceTargetAnalysis(
            current_price=current_price,
            weighted_price_target=data.get("weighted_price_target", current_price),
            expected_return_pct=data.get("expected_return_pct", 0.0),
            scenarios=scenarios,
            conviction=data.get("conviction", "Low"),
            position_sizing_note=data.get("position_sizing_note", "No guidance provided."),
            upgrade_triggers=data.get("upgrade_triggers", []),
            downgrade_triggers=data.get("downgrade_triggers", []),
            investment_horizon=data.get("investment_horizon", "Medium (6–18M)")
        )
    except Exception as e:
        logger.error(f"Error in Scenario synthesis for {ticker}: {e}")
        # Return fallback
        scen = Scenario(label="Base", probability=1.0, key_assumptions=["Status quo"], price_target_12m=current_price, upside_downside_pct=0.0)
        return PriceTargetAnalysis(
            current_price=current_price,
            weighted_price_target=current_price,
            expected_return_pct=0.0,
            scenarios=[scen],
            conviction="Low",
            position_sizing_note="Error preventing scenario generation. Avoid sizing.",
            upgrade_triggers=[], downgrade_triggers=[], investment_horizon="Medium (6–18M)"
        )

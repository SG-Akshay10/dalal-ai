from pydantic import BaseModel, Field
from typing import Literal
from datetime import date

# --------------------------------------------------------------------------------
# 6.1 Fundamental Analysis
# --------------------------------------------------------------------------------
class QuarterlyKPI(BaseModel):
    quarter: str = Field(description="Quarter label, e.g., 'Q3 FY25'")
    value: float = Field(description="Value in standard units (e.g., Crores)")
    yoy_growth_pct: float | None = Field(description="Year-over-year growth percentage")
    qoq_growth_pct: float | None = Field(description="Quarter-over-quarter growth percentage")

class FundamentalAnalysis(BaseModel):
    revenue_trend: list[QuarterlyKPI] = Field(description="QoQ and YoY revenue figures")
    ebitda_trend: list[QuarterlyKPI] = Field(description="QoQ and YoY EBITDA figures")
    pat_trend: list[QuarterlyKPI] = Field(description="QoQ and YoY PAT figures")
    gross_margin: float = Field(description="Current gross margin percentage")
    net_margin: float = Field(description="Current net margin percentage")
    fcf_commentary: str = Field(description="Free cash flow narrative")
    debt_equity_ratio: float = Field(description="Current debt-to-equity ratio")
    management_commentary: str = Field(description="Summary of management guidance and strategy")
    red_flags: list[str] = Field(description="Any negative surprises or reversals indicated in typical filings")
    data_freshness_days: int = Field(description="How old is the most recent ingested doc in days")


# --------------------------------------------------------------------------------
# 6.2 Sentiment Analysis
# --------------------------------------------------------------------------------
class SentimentAnalysis(BaseModel):
    composite_sentiment_score: float = Field(description="-100 to +100")
    label: Literal["Strongly Bullish", "Bullish", "Neutral", "Bearish", "Strongly Bearish"]
    positive_themes: list[str] = Field(description="Top 3 positive themes")
    negative_themes: list[str] = Field(description="Top 3 negative themes")
    narrative: str = Field(description="Two-paragraph narrative synthesizing sentiment")
    news_article_count: int = Field(description="Number of news articles processed")
    social_post_count: int = Field(description="Number of social posts processed")
    finbert_distribution: dict = Field(description="Distribution of sentiment: {positive: %, negative: %, neutral: %}")


# --------------------------------------------------------------------------------
# 6.3 Competitor Analysis
# --------------------------------------------------------------------------------
class Competitor(BaseModel):
    ticker: str = Field(description="Ticker symbol")
    name: str = Field(description="Company name")
    exchange: Literal["NSE", "BSE"] = Field(default="NSE")
    rationale: str = Field(description="Why this company is a direct competitor")

class CompetitorAnalysis(BaseModel):
    competitors: list[Competitor] = Field(description="2-3 direct competitors")
    primary_sector: str = Field(description="Primary sector the company operates in")
    business_description: str = Field(description="Description of business segments")


# --------------------------------------------------------------------------------
# 6.4 Sector Analysis
# --------------------------------------------------------------------------------
class SectorAnalysis(BaseModel):
    sector_name: str
    growth_stage: Literal["Emerging", "High Growth", "Mature", "Declining"]
    index_performance_ytd: float = Field(description="Performance in %")
    fii_flow_trend: Literal["Strong Inflow", "Moderate Inflow", "Neutral", "Moderate Outflow", "Strong Outflow"]
    policy_tailwinds: list[str] = Field(description="Positive regulatory/policy factors")
    policy_headwinds: list[str] = Field(description="Negative regulatory/policy factors")
    regulatory_summary: str = Field(description="Summary of the policy/regulatory outlook")


# --------------------------------------------------------------------------------
# 6.5 Technical Analysis (New in V2)
# --------------------------------------------------------------------------------
class TechnicalAnalysis(BaseModel):
    trend_bias: Literal["Strong Uptrend", "Uptrend", "Sideways", "Downtrend", "Strong Downtrend", "Insufficient Data"]
    rsi_14: float
    rsi_signal: Literal["Overbought", "Neutral", "Oversold"]
    macd_signal: Literal["Bullish Crossover", "Bullish", "Bearish", "Bearish Crossover"]
    bollinger_position: Literal["Above Upper", "Near Upper", "Mid", "Near Lower", "Below Lower"]
    support_levels: list[float] = Field(description="2-3 key price support levels")
    resistance_levels: list[float] = Field(description="2-3 key price resistance levels")
    volume_trend: Literal["Expanding", "Contracting", "Neutral"]
    narrative: str = Field(description="LLM-written 2-3 sentence summary of the technicals")


# --------------------------------------------------------------------------------
# 6.6 Valuation Analysis (New in V2)
# --------------------------------------------------------------------------------
class ValuationAnalysis(BaseModel):
    current_price: float | None
    market_cap_cr: float | None = Field(description="In crores")
    pe_ttm: float | None
    pe_forward: float | None
    pb_ratio: float | None
    ev_ebitda: float | None
    price_to_sales: float | None
    dividend_yield: float | None
    peg_ratio: float | None
    vs_own_history: Literal["Premium", "At Par", "Discount"] | None
    vs_sector_median: Literal["Premium", "At Par", "Discount"] | None
    premium_discount_pct: float | None = Field(description="e.g. +32% premium to sector median")
    valuation_verdict: str = Field(description="LLM-written paragraph discussing valuation")
    margin_of_safety: Literal["High", "Moderate", "Low", "Negative"]


# --------------------------------------------------------------------------------
# 6.7 Peer Benchmark Analysis (New in V2)
# --------------------------------------------------------------------------------
class PeerRow(BaseModel):
    ticker: str
    name: str
    revenue_growth_yoy: float | None
    ebitda_margin: float | None
    net_margin: float | None
    roe: float | None
    debt_equity: float | None
    pe_ratio: float | None
    ev_ebitda: float | None

class PeerBenchmarkAnalysis(BaseModel):
    rows: list[PeerRow] = Field(description="Target stock + 2-3 peers")
    relative_positioning: Literal["Market Leader", "In Line With Peers", "Lags Peers"]
    strengths_vs_peers: list[str] = Field(description="Metrics where target leads")
    weaknesses_vs_peers: list[str] = Field(description="Metrics where target lags")
    narrative: str = Field(description="Synthesis of relative positioning")


# --------------------------------------------------------------------------------
# 6.8 Risk & Red Flag Analysis (New in V2)
# --------------------------------------------------------------------------------
class RiskFlag(BaseModel):
    flag_type: str = Field(description="Maps to checklist item")
    severity: Literal["Low", "Medium", "High", "Critical", "Unknown"]
    detail: str = Field(description="Specific finding from the source")
    source: str = Field(description="Where this was found")

class RiskAnalysis(BaseModel):
    flags: list[RiskFlag]
    overall_risk_rating: Literal["Low", "Moderate", "High", "Very High"]
    risk_narrative: str
    promoter_pledge_pct: float | None


# --------------------------------------------------------------------------------
# 6.9 Event Detection Analysis (New in V2)
# --------------------------------------------------------------------------------
class CorporateEvent(BaseModel):
    event_type: str
    date: date | str | None
    description: str
    investment_relevance: str = Field(description="Why this matters for the thesis")

class EventAnalysis(BaseModel):
    upcoming_events: list[CorporateEvent]
    recent_events: list[CorporateEvent]
    nearest_catalyst_days: int | None = Field(description="Days until next major event")
    event_risk_flag: bool = Field(description="True if high-impact event within 7 days")


# --------------------------------------------------------------------------------
# 6.10 Price Target & Scenario Analysis (New in V2)
# --------------------------------------------------------------------------------
class Scenario(BaseModel):
    label: Literal["Bull", "Base", "Bear"]
    probability: float = Field(description="Must sum to 1.0 across all three")
    key_assumptions: list[str]
    price_target_12m: float
    upside_downside_pct: float

class PriceTargetAnalysis(BaseModel):
    current_price: float
    weighted_price_target: float
    expected_return_pct: float
    scenarios: list[Scenario] = Field(description="Bull + Base + Bear")
    conviction: Literal["High", "Medium", "Low"]
    position_sizing_note: str = Field(description="e.g., 'suitable for up to 5% portfolio allocation'")
    upgrade_triggers: list[str] = Field(description="Conditions that would raise conviction")
    downgrade_triggers: list[str] = Field(description="Conditions that would lower conviction")
    investment_horizon: Literal["Short (< 6M)", "Medium (6–18M)", "Long (> 18M)"]

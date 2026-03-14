
from pydantic import BaseModel, Field


class KPI(BaseModel):
    name: str = Field(description="Name of the Key Performance Indicator (e.g., Revenue, EBITDA, PAT)")
    value: str = Field(description="Value of the KPI (e.g., '10.5 Cr')")
    trend: str = Field(description="YoY or QoQ trend (e.g., '+15% YoY')")

class FundamentalAnalysis(BaseModel):
    kpis: list[KPI] = Field(description="Top 5 KPIs extracted from the documents")
    red_flags: list[str] = Field(description="Any red flags, risks, or negative indicators mentioned")
    management_commentary: str = Field(description="A brief summary of management commentary and forward-looking statements")

class SentimentAnalysis(BaseModel):
    composite_score: int = Field(description="Composite Sentiment Score between -100 and +100")
    positive_themes: list[str] = Field(description="Top 3 positive themes")
    negative_themes: list[str] = Field(description="Top 3 negative themes")
    narrative: str = Field(description="Synthesis narrative about the market sentiment")

class Competitor(BaseModel):
    ticker: str = Field(description="Stock ticker of the competitor")
    name: str = Field(description="Company name of the competitor")
    rationale: str = Field(description="Why this company is a competitor")

class CompetitorAnalysis(BaseModel):
    competitors: list[Competitor] = Field(description="List of 2-3 identified competitors")

class SectorAnalysis(BaseModel):
    sector_name: str = Field(description="Name of the operating sector")
    growth_stage: str = Field(description="Current stage of the sector (e.g. Emerging, Mature, Declining)")
    index_performance: str = Field(description="Recent performance of the sector index")
    policy_context: str = Field(description="Regulatory or policy context affecting the sector")

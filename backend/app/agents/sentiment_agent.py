"""
Rewritten for StockSense AI Pipeline v2.
Changes made:
- Architecture: Split deterministic computations (CSS score, sentiment label, FinBERT distribution, weights) from LLM generation. 
  The LLM now only generates `positive_themes`, `negative_themes`, and a structured `narrative`.
- FinBERT Fix: Used `truncation=True, max_length=512` in the pipeline constructor instead of string slicing.
- Python Computation: Implemented credibility weights by source and recency decay (7-day half-life).
- CSS Logic: Computed directional score * credibility * recency. Clamped to [-100, 100]. Determined Label via Python thresholds.
- Data Guards: Early return if fewer than 5 valid items exist to prevent low-confidence thesis weighting.
- Prompt Rails: Added STRICT RULES for theme specificity (no generic labels) and defined a rigid 4-sentence structured narrative.
- Enforced Fallbacks: Re-injected all Python-computed values (scores, counts, distributions) into the final result object on both success and error paths.
"""

import json
import logging
import math
from datetime import datetime, timezone

from langchain_core.prompts import PromptTemplate
from transformers import pipeline

from app.llm_provider import get_llm_client
from app.schemas.news_article import NewsArticle
from app.schemas.analysis import SentimentAnalysis
from app.schemas.social_post import SocialPost

logger = logging.getLogger(__name__)

# Initialize FinBERT pipeline with truncation
try:
    sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert", truncation=True, max_length=512)
except Exception as e:
    logger.error(f"Failed to load FinBERT model: {e}")
    sentiment_pipeline = None

SENTIMENT_SYNTHESIS_PROMPT = """You are an expert financial sentiment analyst.
You are given a list of news articles and social media posts about the stock {ticker}.
These items have been pre-scored, weighted, and analyzed by a deterministic engine. 
The current Composite Sentiment Score (CSS) is {css_score:.1f} and the label is '{sentiment_label}'.
The distribution of the source data is: {dist}

Your task is ONLY to extract textual themes and write a narrative based on the provided data.

<STRICT_RULES>
1. THEME EXTRACTION: 
   - `positive_themes` and `negative_themes` MUST be specific noun phrases citing real events/data from the text.
   - NO generic labels (e.g., use "Q2FY25 earnings beat by 15%" instead of "Strong results").
   - NEVER invent or hallucinate themes. If there is no negative news, return an empty list `[]` for negative_themes. Do not invent filler.
2. NARRATIVE STRUCTURE: Write EXACTLY 4 sentences with the following assigned jobs:
   - Sentence 1: State the CSS score, label, and the volume of data analyzed.
   - Sentence 2: Summarize the top positive item or theme driving the score.
   - Sentence 3: Summarize the top negative item or risk factor.
   - Sentence 4: Note any significant divergence between news vs social media (or state if they align).
3. NO COMPUTATION: Do not adjust the CSS score, label, or distribution. You are ONLY generating strings.
</STRICT_RULES>

<REASONING_STEPS>
1. Review the provided data items to identify the most impactful positive and negative specific events.
2. Formulate 1-3 specific noun phrases for positive themes based on the highest weight items.
3. Formulate 1-3 specific noun phrases for negative themes based on the highest weight items.
4. Draft the 4-sentence narrative following the strict structure rules.
</REASONING_STEPS>

Scored Data:
{data_text}

Always return your response as a valid JSON object matching the following schema. Wait until after your reasoning to generate the JSON.

<EXAMPLE_SCHEMA>
{{
  "positive_themes": ["FY25 Revenue Guidance Upgrade", "Promoter Stake Increase via Open Market"],
  "negative_themes": ["Operating Margin Compression due to RM Costs"],
  "narrative": "The aggregate sentiment for RELIANCE is Bullish with a CSS of 45.2 across 24 recent items. The primary positive driver is management's upward revision of FY25 revenue guidance following strong festive demand. Conversely, sentiment is weighed down by concerns over operating margin compression tied to rising raw material costs. News coverage is overwhelmingly positive, while social sentiment shows slight divergence with retail caution around valuation."
}}
</EXAMPLE_SCHEMA>
"""

def parse_date(date_str: str):
    """Attempt to parse common date formats for recency calculation."""
    if not date_str:
        return None
    try:
        if "T" in date_str:
            return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        else:
            return datetime.strptime(date_str[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None

def get_credibility_weight(source: str) -> float:
    """Return credibility weight based on source/platform."""
    if not source:
        return 0.7
    
    src = source.lower()
    if any(x in src for x in ["reuters", "economic times", "economictimes", "mint", "livemint", "business standard", "businessstandard"]):
        return 1.0
    if any(x in src for x in ["moneycontrol", "cnbc"]):
        return 0.9
    if "zeebiz" in src:
        return 0.8
    if any(x in src for x in ["reddit", "twitter", "x", "stocktwits", "x.com"]):
        return 0.4
    if any(x in src for x in ["blog", "medium"]):
        return 0.6
        
    return 0.7

def get_recency_weight(date_obj) -> float:
    """Calculate exponential decay recency weight. w = 2^(-age_days / 7)"""
    if not date_obj:
        return 1.0 # No penalty if date is unknown
    
    now = datetime.now(timezone.utc)
    age_days = max(0, (now - date_obj).days)
    return math.pow(2, -age_days / 7.0)

def score_text_with_finbert(text: str) -> dict:
    """Score a single piece of text using FinBERT."""
    if not sentiment_pipeline or not text:
        return {"label": "neutral", "score": 0.0}

    try:
        # Note: truncation is handled by the pipeline constructor
        result = sentiment_pipeline(text)[0]
        return result
    except Exception as e:
        logger.debug(f"FinBERT scoring failed on text snippet, defaulting to neutral. Error: {e}")
        return {"label": "neutral", "score": 0.0}

def get_sentiment_label(css: float) -> str:
    """Map numerical CSS to a human-readable label."""
    if css >= 60:
        return "Strongly Bullish"
    elif css >= 20:
        return "Bullish"
    elif css >= -20:
        return "Neutral"
    elif css >= -60:
        return "Bearish"
    else:
        return "Strongly Bearish"

def analyze_sentiment(ticker: str, news: list[NewsArticle], social: list[SocialPost], provider: str = None) -> SentimentAnalysis:
    """Run the sentiment agent pipeline."""

    scored_items = []
    
    # 1. Process News
    for article in news:
        full_text = article.headline + ". " + (article.body or "")
        s = score_text_with_finbert(full_text)
        
        dt = parse_date(article.published_at)
        cred = get_credibility_weight(article.source)
        recency = get_recency_weight(dt)
        
        scored_items.append({
            "type": "news",
            "source": article.source,
            "text": article.headline,
            "sentiment_label": s["label"],
            "sentiment_score": s["score"],
            "credibility": cred,
            "recency": recency,
            "date_obj": dt
        })

    # 2. Process Social
    for post in social:
        s = score_text_with_finbert(post.content)
        
        dt = parse_date(post.posted_at)
        cred = get_credibility_weight(post.platform)
        recency = get_recency_weight(dt)
        
        scored_items.append({
            "type": "social",
            "source": post.platform,
            "text": post.content,
            "sentiment_label": s["label"],
            "sentiment_score": s["score"],
            "credibility": cred,
            "recency": recency,
            "date_obj": dt
        })

    total_items = len(scored_items)
    news_cnt = len(news)
    social_cnt = len(social)

    # Note: Short-circuit guard
    if total_items < 5:
        logger.warning(f"[{ticker}] Insufficient sentiment items ({total_items} < 5). Triggering early fallback.")
        return SentimentAnalysis(
            composite_sentiment_score=0.0,
            label="Neutral",
            positive_themes=[],
            negative_themes=[],
            narrative="Insufficient data — sentiment output should not be weighted in the final thesis.",
            news_article_count=news_cnt,
            social_post_count=social_cnt,
            finbert_distribution={"positive": 0, "negative": 0, "neutral": 100}
        )

    # 3. Compute CSS and Distributions Deterministically in Python
    pos_cnt = sum(1 for item in scored_items if item['sentiment_label'] == 'positive')
    neg_cnt = sum(1 for item in scored_items if item['sentiment_label'] == 'negative')
    neu_cnt = sum(1 for item in scored_items if item['sentiment_label'] == 'neutral')
    
    dist = {
        "positive": round((pos_cnt/total_items)*100, 1),
        "negative": round((neg_cnt/total_items)*100, 1),
        "neutral": round((neu_cnt/total_items)*100, 1),
    }

    sum_weighted_score = 0.0
    sum_normaliser = 0.0

    for item in scored_items:
        dir_score = 0.0
        if item['sentiment_label'] == 'positive':
            dir_score = item['sentiment_score']
        elif item['sentiment_label'] == 'negative':
            dir_score = -item['sentiment_score']
            
        weight = item['credibility'] * item['recency']
        sum_weighted_score += (dir_score * weight)
        sum_normaliser += weight

    css_raw = (sum_weighted_score / sum_normaliser) * 100 if sum_normaliser > 0 else 0.0
    css_final = max(-100.0, min(100.0, css_raw))
    computed_label = get_sentiment_label(css_final)
    
    logger.info(f"[{ticker}] Computed CSS: {css_final:.1f} ({computed_label}) from {total_items} items.")

    # 4. Prepare Context for LLM Narrative Generation
    # Sort by impact (credibility * recency) descending to give LLM best context
    scored_items.sort(key=lambda x: x['credibility'] * x['recency'], reverse=True)
    
    data_text_lines = []
    for idx, item in enumerate(scored_items):
        date_str = item['date_obj'].strftime('%Y-%m-%d') if item['date_obj'] else "Unknown Date"
        data_text_lines.append(
            f"[{idx+1}] [{item['type'].upper()} | {item['source']} | {date_str}] "
            f"- Sent: {item['sentiment_label']} (FinBERT: {item['sentiment_score']:.2f}) - {item['text']}"
        )

    data_text = "\n".join(data_text_lines)

    prompt = PromptTemplate.from_template(SENTIMENT_SYNTHESIS_PROMPT)
    formatted_prompt = prompt.format(
        ticker=ticker, 
        data_text=data_text, 
        css_score=css_final,
        sentiment_label=computed_label,
        dist=json.dumps(dist)
    )

    llm = get_llm_client(provider)
    result_data = {}

    try:
        # Structured LLM Path
        if hasattr(llm, "with_structured_output"):
            # Temporary schema override: Only query the LLM for themes and narrative since Python handles the rest
            from pydantic import BaseModel, Field
            class LLMOutputSchema(BaseModel):
                positive_themes: list[str] = Field(description="Specific positive themes grounded in text")
                negative_themes: list[str] = Field(description="Specific negative themes grounded in text")
                narrative: str = Field(description="4-sentence strict structured narrative")
                
            structured_llm = llm.with_structured_output(
                LLMOutputSchema, 
                method="json_mode" if "openai" in str(type(llm)).lower() else "function_calling"
            )
            
            try:
                llm_res = structured_llm.invoke(formatted_prompt)
                result_data = {
                    "positive_themes": getattr(llm_res, "positive_themes", []),
                    "negative_themes": getattr(llm_res, "negative_themes", []),
                    "narrative": getattr(llm_res, "narrative", "")
                }
            except json.JSONDecodeError as jde:
                logger.error(f"[{ticker}] JSON decode error in structured output: {jde}")
            except Exception as e:
                logger.error(f"[{ticker}] Structured output invoke error: {e}")

        # Fallback raw call if structured path failed or is unavailable
        if not result_data:
            response = llm.invoke([
                {"role": "system", "content": "Follow the prompt carefully. Output valid JSON containing 'positive_themes', 'negative_themes', and 'narrative' keys."},
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

            result_data = json.loads(content)

        # 5. Assemble final response forcing Python-computed metrics
        final_result = SentimentAnalysis(
            composite_sentiment_score=css_final,
            label=computed_label,
            positive_themes=result_data.get("positive_themes", []),
            negative_themes=result_data.get("negative_themes", []),
            narrative=result_data.get("narrative", f"Generated sentiment narrative for {ticker}."),
            news_article_count=news_cnt,
            social_post_count=social_cnt,
            finbert_distribution=dist
        )
        
        logger.info(f"[{ticker}] Sentiment node completed successfully.")
        return final_result

    except json.JSONDecodeError as jde:
        logger.error(f"[{ticker}] JSONDecodeError processing sentiment LLM response: {jde}")
        return SentimentAnalysis(
            composite_sentiment_score=css_final,
            label=computed_label,
            positive_themes=[],
            negative_themes=[],
            narrative=f"Agent error parsing sentiment text output (JSONDecodeError): {str(jde)}. Analysis based entirely on deterministic CSS.",
            news_article_count=news_cnt,
            social_post_count=social_cnt,
            finbert_distribution=dist
        )
    except Exception as e:
        logger.error(f"[{ticker}] Agent generic error in sentiment synthesis: {e}")
        return SentimentAnalysis(
            composite_sentiment_score=css_final,
            label=computed_label,
            positive_themes=[],
            negative_themes=[],
            narrative=f"Agent error generating sentiment text output: {str(e)}. Analysis based entirely on deterministic CSS.",
            news_article_count=news_cnt,
            social_post_count=social_cnt,
            finbert_distribution=dist
        )

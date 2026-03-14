import json

from langchain_core.prompts import PromptTemplate
from transformers import pipeline

from app.llm_provider import get_llm_client
from app.schemas.news_article import NewsArticle
from app.schemas.report import SentimentAnalysis
from app.schemas.social_post import SocialPost

# Initialize FinBERT pipeline
# Using ProsusAI/finbert for financial text sentiment
try:
    sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")
except Exception as e:
    # If there's an issue loading the model, we fall back to a mock or handle gracefully later
    print(f"Failed to load FinBERT model: {e}")
    sentiment_pipeline = None

SENTIMENT_SYNTHESIS_PROMPT = """You are an expert financial sentiment analyst.
You are given a list of news articles and social media posts about the stock {ticker}, along with their individual FinBERT sentiment scores.

Synthesize this data to provide an overall market sentiment analysis.
Provide:
1. A Composite Sentiment Score (CSS) between -100 and +100 based on the volume and intensity of positive vs negative news/social posts.
2. The top 3 positive themes.
3. The top 3 negative themes.
4. A brief narrative summarizing the current market sentiment.

Data:
{data_text}

Always return your response as a valid JSON object matching the following schema:
{{
  "composite_score": 0,
  "positive_themes": ["...", "...", "..."],
  "negative_themes": ["...", "...", "..."],
  "narrative": "..."
}}
"""

def score_text_with_finbert(text: str) -> dict:
    """Score a single piece of text using FinBERT."""
    if not sentiment_pipeline or not text:
        return {"label": "neutral", "score": 0.0}

    # Truncate text to max length if needed (FinBERT limit)
    truncated_text = text[:512]

    try:
        result = sentiment_pipeline(truncated_text)[0]
        return result
    except Exception:
        return {"label": "neutral", "score": 0.0}

def analyze_sentiment(ticker: str, news: list[NewsArticle], social: list[SocialPost], provider: str = None) -> SentimentAnalysis:
    """Run the sentiment agent pipeline."""

    # 1. Score all news and social posts using FinBERT
    scored_items = []

    for article in news:
        s = score_text_with_finbert(article.headline + ". " + (article.body or ""))
        scored_items.append({
            "type": "news",
            "source": article.source,
            "text": article.headline,
            "sentiment_label": s["label"],
            "sentiment_score": s["score"]
        })

    for post in social:
        s = score_text_with_finbert(post.content)
        scored_items.append({
            "type": "social",
            "platform": post.platform,
            "text": post.content,
            "sentiment_label": s["label"],
            "sentiment_score": s["score"]
        })

    # Prepare text for LLM
    data_text_lines = []
    for idx, item in enumerate(scored_items):
        data_text_lines.append(f"[{idx+1}] [{item['type'].upper()}] - Sent: {item['sentiment_label']} ({item['sentiment_score']:.2f}) - {item['text']}")

    data_text = "\n".join(data_text_lines)

    # Handle empty case
    if not data_text:
        return SentimentAnalysis(
            composite_score=0,
            positive_themes=["No positive themes"],
            negative_themes=["No negative themes"],
            narrative="No news or social data available to analyze sentiment."
        )

    # 2. Synthesize using LLM
    prompt = PromptTemplate.from_template(SENTIMENT_SYNTHESIS_PROMPT)
    formatted_prompt = prompt.format(ticker=ticker, data_text=data_text)

    llm = get_llm_client(provider)

    try:
        if hasattr(llm, "with_structured_output"):
            structured_llm = llm.with_structured_output(SentimentAnalysis, method="json_mode" if "openai" in str(type(llm)).lower() else "function_calling")
            try:
                result = structured_llm.invoke(formatted_prompt)
                return result
            except Exception:
                pass

        # Fallback raw call
        response = llm.invoke([
            {"role": "system", "content": "You output JSON only."},
            {"role": "user", "content": formatted_prompt}
        ])

        content = response.content.replace("```json", "").replace("```", "").strip()
        data = json.loads(content)
        return SentimentAnalysis(**data)

    except Exception as e:
        return SentimentAnalysis(
            composite_score=0,
            positive_themes=["Error"],
            negative_themes=["Error"],
            narrative=f"Agent error processing sentiment data: {str(e)}"
        )

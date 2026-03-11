from typing import List, Optional
from pydantic import BaseModel
import random
import logging

logger = logging.getLogger(__name__)

class TopPost(BaseModel):
    id: str
    text: str
    engagement: int

class SocialSentimentObject(BaseModel):
    platform: str
    query_period: int
    post_count: int
    positive_pct: float
    negative_pct: float
    neutral_pct: float
    top_posts: List[TopPost]
    trend_direction: str
    volume_delta_vs_prior_week: float

async def get_social_sentiment(symbol: str, days_back: int = 21) -> Optional[SocialSentimentObject]:
    """
    Retrieves social sentiment for a stock symbol. Since real social APIs require paid keys,
    this MVP implementation simulates realistic social data retrieval structure.
    """
    try:
        if "INVALID" in symbol:
            return None
            
        # Realistic synthesis for MVP without API keys
        post_count = random.randint(50, 2000)
        base_pos = random.uniform(0.2, 0.7)
        base_neg = random.uniform(0.1, 1.0 - base_pos)
        base_neu = 1.0 - (base_pos + base_neg)
        
        direction = random.choice(["up", "down", "flat"])
        delta = random.uniform(-25.0, 50.0)
        
        return SocialSentimentObject(
            platform="Aggregated",
            query_period=days_back,
            post_count=post_count,
            positive_pct=round(base_pos, 2),
            negative_pct=round(base_neg, 2),
            neutral_pct=round(base_neu, 2),
            top_posts=[
                TopPost(id="1", text=f"Retail sentiment shows strong interest in {symbol} heavily influenced by recent news.", engagement=random.randint(50, 500)),
                TopPost(id="2", text=f"Evaluating {symbol}'s recent market movements, lots of chatter.", engagement=random.randint(10, 100))
            ],
            trend_direction=direction,
            volume_delta_vs_prior_week=round(delta, 2)
        )
    except Exception as e:
        logger.error(f"Error fetching social sentiment for {symbol}: {e}")
        return None

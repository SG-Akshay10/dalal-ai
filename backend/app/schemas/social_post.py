"""SocialPost schema — Phase 1 data contract.

Produced by: social_listener.py
Consumed by: Phase 2 sentiment_agent.py (social sentiment scoring)
"""
from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field


class SocialPost(BaseModel):
    """Represents a social media post from Twitter/X or Reddit.

    Sourced from:
    - Twitter/X: cashtag mentions (e.g. $RELIANCE) via API v2
    - Reddit: r/IndiaInvestments and r/DalalStreet posts

    Engagement metadata (likes, comments) is used by Phase 2
    sentiment_agent to weight post importance in the CSS calculation.
    """

    platform: Literal["twitter", "reddit", "web"] = Field(
        description="Social platform the post was sourced from."
    )
    post_id: str = Field(
        description="Platform-native post ID (tweet ID or Reddit submission ID)."
    )
    content: str = Field(
        description="Full text content of the post or tweet."
    )
    author: str = Field(
        description="Username or display name of the author."
    )
    date: datetime = Field(
        description="Post creation date (UTC)."
    )
    likes: int = Field(
        default=0,
        ge=0,
        description="Number of likes/upvotes."
    )
    comments: int = Field(
        default=0,
        ge=0,
        description="Number of replies/comments."
    )
    url: str = Field(
        description="Direct URL to the post."
    )

    model_config = {"json_schema_extra": {"example": {
        "platform": "reddit",
        "post_id": "abc123",
        "content": "RELIANCE breaking out of long-term resistance. Expect 5-10% move. #DalalStreet",
        "author": "invest_guru_42",
        "date": "2024-01-20T14:30:00",
        "likes": 150,
        "comments": 34,
        "url": "https://reddit.com/r/DalalStreet/comments/abc123",
    }}}

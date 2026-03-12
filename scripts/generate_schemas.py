import json
import os
import sys

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))

from app.schemas.document_object import DocumentObject
from app.schemas.news_article import NewsArticle
from app.schemas.social_post import SocialPost

schemas = {
    "DocumentObject": DocumentObject.model_json_schema(),
    "NewsArticle": NewsArticle.model_json_schema(),
    "SocialPost": SocialPost.model_json_schema()
}

os.makedirs("docs/schemas", exist_ok=True)
with open("docs/schemas/phase1_contracts.json", "w") as f:
    json.dump(schemas, f, indent=2)
print("Schemas written to docs/schemas/phase1_contracts.json")

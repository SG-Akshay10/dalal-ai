import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def get_embedder():
    """Return the Gemini embedder as configured in .env."""
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set")
    
    model = os.getenv("GEMINI_EMBEDDING_MODEL", "models/embedding-001")
    return GoogleGenerativeAIEmbeddings(
        model=model,
        google_api_key=gemini_key
    )

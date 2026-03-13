import os
from langchain_huggingface import HuggingFaceEmbeddings

def get_embedder():
    """Return a local HuggingFace embedder."""
    # Use a fast, lightweight model by default
    model_name = os.getenv("LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    
    return HuggingFaceEmbeddings(
        model_name=model_name
    )

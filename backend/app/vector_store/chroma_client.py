import os
import chromadb
from chromadb.config import Settings

# Default DB Path if none is provided via env
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")

def get_chroma_client() -> chromadb.ClientAPI:
    """Initialize and return a persistent ChromaDB client."""
    return chromadb.PersistentClient(
        path=CHROMA_DB_PATH,
        settings=Settings(anonymized_telemetry=False)
    )

def get_or_create_collection(client: chromadb.ClientAPI, collection_name: str="dalalai_docs"):
    """Get or create the given Chroma collection. We use cosine similarity by default."""
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

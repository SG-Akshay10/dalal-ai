import os
import chromadb
from chromadb.config import Settings

# Anchor chroma_db path to the backend/ directory (2 levels up from this file),
# so it always resolves correctly regardless of the process working directory.
_BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_DEFAULT_CHROMA_PATH = os.path.join(_BACKEND_DIR, "chroma_db")

_raw_path = os.getenv("CHROMA_DB_PATH", _DEFAULT_CHROMA_PATH)
# If the env var is a relative path, resolve it relative to backend/
CHROMA_DB_PATH = os.path.abspath(
    _raw_path if os.path.isabs(_raw_path) else os.path.join(_BACKEND_DIR, _raw_path)
)

def get_chroma_client() -> chromadb.ClientAPI:
    """Initialize and return a persistent ChromaDB client."""
    os.makedirs(CHROMA_DB_PATH, exist_ok=True)
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

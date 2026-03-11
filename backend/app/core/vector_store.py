from typing import List
import chromadb
from chromadb.config import Settings
import logging

logger = logging.getLogger(__name__)

# Initialize ChromaDB locally for development/MVP
# In production, this would point to Pinecone or a hosted Chroma instance
try:
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection(name="dalalai_documents")
except Exception as e:
    logger.warning(f"Failed to initialize ChromaDB, falling back to ephemeral client: {e}")
    chroma_client = chromadb.EphemeralClient()
    collection = chroma_client.get_or_create_collection(name="dalalai_documents")

async def add_document_chunks(symbol: str, chunks: List[str]) -> bool:
    """
    Adds document text chunks to the ChromaDB collection for a specific symbol.
    Generates deterministic IDs based on symbol and chunk index.
    """
    try:
        if not chunks:
            return False
            
        ids = [f"{symbol}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"symbol": symbol} for _ in range(len(chunks))]
        
        collection.add(
            documents=chunks,
            metadatas=metadatas,
            ids=ids
        )
        return True
    except Exception as e:
        logger.error(f"Error adding chunks to ChromaDB for {symbol}: {e}")
        return False

async def search_similar_chunks(symbol: str, query: str, top_k: int = 3) -> List[str]:
    """
    Queries ChromaDB for chunks matching the query for a specific symbol.
    """
    try:
        results = collection.query(
            query_texts=[query],
            n_results=top_k,
            where={"symbol": symbol}
        )
        
        if results and "documents" in results and results["documents"]:
            return results["documents"][0]
        return []
    except Exception as e:
        logger.error(f"Error querying ChromaDB for {symbol}: {e}")
        return []

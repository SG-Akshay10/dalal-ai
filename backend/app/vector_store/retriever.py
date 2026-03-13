from typing import List, Dict, Any
from app.vector_store.chroma_client import get_chroma_client, get_or_create_collection
from app.vector_store.embedder import get_embedder

def retrieve_documents(query: str, ticker: str = None, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieve top_k documents for a given query and optionally filter by ticker.
    Returns a list of dicts with 'text' and 'metadata'.
    """
    client = get_chroma_client()
    collection = get_or_create_collection(client)
    embedder = get_embedder()
    
    # Generate embedding for the query
    query_embedding = embedder.embed_query(query)
    
    where_filter = {}
    if ticker:
        where_filter["ticker"] = ticker
        
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where_filter if where_filter else None,
        include=["documents", "metadatas"]
    )
    
    retrieved_docs = []
    if results["documents"] and len(results["documents"]) > 0:
        docs = results["documents"][0]
        metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
        
        for doc, meta in zip(docs, metas):
            retrieved_docs.append({
                "text": doc,
                "metadata": meta
            })
            
    return retrieved_docs

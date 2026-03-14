from typing import List, Dict, Any
from app.vector_store.supabase_client import get_supabase_client
from app.vector_store.embedder import get_embedder

def retrieve_documents(query: str, ticker: str = None, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieve top_k documents for a given query and optionally filter by ticker.
    Returns a list of dicts with 'text' and 'metadata'.
    """
    client = get_supabase_client()
    embedder = get_embedder()
    
    # Generate embedding for the query
    query_embedding = embedder.embed_query(query)
    
    params = {
        "query_embedding": query_embedding,
        "match_count": top_k,
    }
    if ticker:
        params["filter_ticker"] = ticker
        
    result = client.rpc("match_documents", params).execute()
    data = result.data
    
    retrieved_docs = []
    if data:
        for doc in data:
            retrieved_docs.append({
                "text": doc.get("content", ""),
                "metadata": doc.get("metadata", {})
            })
            
    return retrieved_docs

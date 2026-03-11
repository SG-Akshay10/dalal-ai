import pytest
from app.core.vector_store import add_document_chunks, search_similar_chunks
import asyncio

@pytest.mark.asyncio
async def test_vector_store_add_and_search():
    """Test adding text chunks to ChromaDB and searching for them."""
    symbol = "TEST_STOCK"
    chunks = [
        "In Q1 2024, the company recorded a 15% increase in revenue.",
        "The newly appointed CEO mentioned a strong focus on AI.",
        "Supply chain disruptions have caused a 5% drop in EBITDA margin."
    ]
    
    # Add to store
    success = await add_document_chunks(symbol, chunks)
    assert success is True
    
    # Search store
    results = await search_similar_chunks(symbol, "revenue growth", top_k=1)
    
    assert results is not None
    assert len(results) == 1
    assert "revenue" in results[0].lower()

@pytest.mark.asyncio
async def test_vector_store_search_empty():
    """Test searching when no documents exist."""
    results = await search_similar_chunks("UNKNOWN_STOCK", "anything", top_k=2)
    assert isinstance(results, list)
    assert len(results) == 0

import sys
import os
import argparse
from typing import List

# Setup path to import app modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))

from app.scrapers.document_fetcher import fetch_documents
from app.schemas.document_object import DocumentObject
from app.vector_store.chroma_client import get_chroma_client, get_or_create_collection
from app.vector_store.embedder import get_embedder
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb

def ingest_documents_for_ticker(ticker: str, days: int = 90):
    """Fetch documents, chunk them, and save them to ChromaDB."""
    print(f"Fetching documents for {ticker} for the last {days} days...")
    
    docs: List[DocumentObject] = fetch_documents(ticker=ticker, days=days)
    
    if not docs:
        print(f"No documents found for {ticker}.")
        return

    print(f"Fetched {len(docs)} documents. Chunking...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    
    client = get_chroma_client()
    collection = get_or_create_collection(client)
    embedder = get_embedder()
    
    total_chunks = 0
    for doc in docs:
        if not doc.text:
            continue
            
        chunks = text_splitter.split_text(doc.text)
        
        if not chunks:
            continue
            
        ids = [f"{ticker}_{doc.source}_{doc.date.isoformat()}_{i}" for i in range(len(chunks))]
        metadatas = [{
            "ticker": ticker,
            "source": doc.source,
            "doc_type": doc.doc_type,
            "url": doc.url,
            "date": doc.date.isoformat(),
            "chunk_index": i
        } for i in range(len(chunks))]
        
        # We process embedding ourselves since we want to use the unified embedder.
        # Alternatively, we could attach the embedding function directly to the collection,
        # but Langchain's embedder returns vectors we can add manually.
        embeddings = embedder.embed_documents(chunks)
        
        collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=chunks
        )
        total_chunks += len(chunks)
        
    print(f"Ingested {total_chunks} chunks from {len(docs)} documents into ChromaDB for {ticker}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ingest documents into ChromaDB')
    parser.add_argument('--ticker', type=str, default="RELIANCE", help='Stock ticker')
    parser.add_argument('--days', type=int, default=90, help='Number of days to look back')
    
    args = parser.parse_args()
    
    # Needs OPENAI_API_KEY
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY environment variable is not set. Ingestion might fail if you rely on OpenAI embeddings.")
        
    ingest_documents_for_ticker(args.ticker, args.days)

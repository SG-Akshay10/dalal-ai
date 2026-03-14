import argparse
import os

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.schemas.document_object import DocumentObject
from app.scrapers.document_fetcher import fetch_documents
from app.vector_store.embedder import get_embedder
from app.vector_store.supabase_client import get_supabase_client

load_dotenv()


def ingest_documents_for_ticker(ticker: str, days: int = 90):
    """Fetch documents, chunk them, and save them to Supabase."""
    print(f"Fetching documents for {ticker} for the last {days} days...")

    docs: list[DocumentObject] = fetch_documents(ticker=ticker, days=days)

    if not docs:
        print(f"No documents found for {ticker}.")
        return

    print(f"Fetched {len(docs)} documents. Chunking...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )

    client = get_supabase_client()
    embedder = get_embedder()

    total_chunks = 0
    for doc in docs:
        if not doc.text:
            continue

        chunks = text_splitter.split_text(doc.text)

        if not chunks:
            continue

        metadatas = [{
            "ticker": ticker,
            "source": doc.source,
            "doc_type": doc.doc_type,
            "url": doc.url,
            "date": doc.date.isoformat(),
            "chunk_index": i
        } for i in range(len(chunks))]

        # Embed the chunks
        embeddings = embedder.embed_documents(chunks)

        # Prepare records for Supabase insertion
        records = []
        for i, chunk in enumerate(chunks):
            records.append({
                "content": chunk,
                "metadata": metadatas[i],
                "embedding": embeddings[i]
            })

        # Insert to Supabase using the pgvector table dalalai_docs
        try:
            client.table("dalalai_docs").insert(records).execute()
        except Exception as e:
            print(f"Error inserting chunks for {doc.source}: {e}")
            continue

        total_chunks += len(chunks)

    print(f"Ingested {total_chunks} chunks from {len(docs)} documents into Supabase for {ticker}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ingest documents into Supabase')
    parser.add_argument('--ticker', type=str, default="RELIANCE", help='Stock ticker')
    parser.add_argument('--days', type=int, default=90, help='Number of days to look back')

    args = parser.parse_args()

    # Needs OPENAI_API_KEY
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY environment variable is not set. Ingestion might fail if you rely on OpenAI embeddings.")

    ingest_documents_for_ticker(args.ticker, args.days)

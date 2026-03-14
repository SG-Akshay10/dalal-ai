import sys
import os
import argparse
from dotenv import load_dotenv

env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))
load_dotenv(env_path)

# Setup path to import app modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.vector_store.supabase_client import get_supabase_client

def delete_documents_for_ticker(ticker: str):
    """Delete all document chunks for a specific ticker from Supabase."""
    print(f"Deleting documents for '{ticker}' from Supabase...")
    client = get_supabase_client()
    
    try:
        # Supabase filtering on jsonb column "metadata"
        # We delete rows where the 'ticker' key in metadata equals our ticker string.
        # Note: the syntax might require 'metadata->>ticker' depending on the Postgrest setup,
        # but eq("metadata->>ticker", ticker) is standard in postgrest-py.
        result = client.table("dalalai_docs").delete().eq("metadata->>ticker", ticker).execute()
        
        deleted_count = len(result.data) if result.data else 0
        print(f"Successfully deleted {deleted_count} document chunks for {ticker}.")
        
    except Exception as e:
        print(f"Error deleting documents for {ticker}: {e}")
        
def delete_all_documents():
    """Delete ALL document chunks from Supabase."""
    print(f"Deleting ALL documents from Supabase...")
    client = get_supabase_client()
    
    try:
        # To delete all rows, postgrest requires at least one filter, so we use neq id to 0.
        result = client.table("dalalai_docs").delete().neq("id", 0).execute()
        
        deleted_count = len(result.data) if result.data else 0
        print(f"Successfully deleted {deleted_count} total document chunks.")
        
    except Exception as e:
        print(f"Error deleting all documents: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Delete documents from Supabase')
    parser.add_argument('--ticker', type=str, help='Stock ticker to delete (e.g., RELIANCE)')
    parser.add_argument('--all', action='store_true', help='Delete ALL documents in the database')
    
    args = parser.parse_args()
    
    if args.all:
        confirmation = input("Are you sure you want to delete ALL documents? (y/n): ")
        if confirmation.lower() == 'y':
            delete_all_documents()
        else:
            print("Operation cancelled.")
    elif args.ticker:
        delete_documents_for_ticker(args.ticker)
    else:
        parser.print_help()

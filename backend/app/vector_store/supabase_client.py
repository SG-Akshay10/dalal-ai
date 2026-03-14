import os
from supabase import create_client, Client

_client = None

def get_supabase_client() -> Client:
    """Initialize and return a Supabase client."""
    global _client
    if _client is None:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_SECRET_KEY") or os.getenv("SUPABASE_KEY")
        
        if not url or not key:
            raise ValueError(
                "Missing Supabase credentials. Ensure SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are set in the .env file."
            )
            
        _client = create_client(url, key)
    return _client

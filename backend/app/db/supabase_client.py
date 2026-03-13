import os
from supabase import create_client, Client

# The Supabase client is a singleton-like connection factory
_supabase_client = None

def get_supabase() -> Client:
    """Return the initialized Supabase client."""
    global _supabase_client
    if _supabase_client is not None:
        return _supabase_client
        
    url: str = os.getenv("SUPABASE_URL", "")
    key: str = os.getenv("SUPABASE_ANON_KEY", "")
    
    if not url or not key:
        print("Warning: SUPABASE_URL or SUPABASE_ANON_KEY not set in environment.") 
        # For tests, return None or mock client, but in prod we want to raise
        # raise ValueError("Supabase credentials not found.")
        return None
        
    _supabase_client = create_client(url, key)
    return _supabase_client

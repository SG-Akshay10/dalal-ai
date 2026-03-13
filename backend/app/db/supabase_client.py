import os
from functools import lru_cache

from supabase import Client, create_client


def _get_supabase_credentials() -> tuple[str, str]:
    """Load Supabase credentials from env variables.

    Uses `SUPABASE_SERVICE_ROLE_KEY` for server-side writes when available,
    and falls back to `SUPABASE_ANON_KEY` for local/demo usage.
    """
    url = os.getenv("SUPABASE_URL", "").strip()
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip() or os.getenv(
        "SUPABASE_ANON_KEY", ""
    ).strip()

    if not url or not key:
        raise RuntimeError(
            "Supabase is not configured. Set SUPABASE_URL and "
            "SUPABASE_SERVICE_ROLE_KEY (or SUPABASE_ANON_KEY)."
        )
    return url, key


@lru_cache(maxsize=1)
def get_supabase() -> Client:
    """Return a cached Supabase client configured from environment variables."""
    url, key = _get_supabase_credentials()
    return create_client(url, key)

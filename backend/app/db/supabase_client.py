import os
from functools import lru_cache

from supabase import Client, create_client


def _get_supabase_credentials() -> tuple[str, str]:
    """Load Supabase credentials from environment variables.

    Supports multiple key names so local/dev/prod configs are easier to wire.
    Preferred order for server-side usage:
    1) SUPABASE_SERVICE_ROLE_KEY
    2) SUPABASE_SECRET_KEY
    3) SUPABASE_KEY
    4) SUPABASE_ANON_KEY (fallback for limited read/public use)
    """
    url = os.getenv("SUPABASE_URL", "").strip()

    key = (
        os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
        or os.getenv("SUPABASE_SECRET_KEY", "").strip()
        or os.getenv("SUPABASE_KEY", "").strip()
        or os.getenv("SUPABASE_ANON_KEY", "").strip()
    )

    if not url or not key:
        raise RuntimeError(
            "Supabase is not configured. Set SUPABASE_URL and one of "
            "SUPABASE_SERVICE_ROLE_KEY, SUPABASE_SECRET_KEY, SUPABASE_KEY, "
            "or SUPABASE_ANON_KEY."
        )

    return url, key


@lru_cache(maxsize=1)
def get_supabase() -> Client:
    """Return a cached Supabase client configured from env variables."""
    url, key = _get_supabase_credentials()
    return create_client(url, key)

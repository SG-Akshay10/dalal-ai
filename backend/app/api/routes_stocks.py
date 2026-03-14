from fastapi import APIRouter, HTTPException

from app.db.supabase_client import get_supabase

router = APIRouter()

@router.get("/api/stocks/search")
def search_stocks(q: str = ""):
    """Stock search returning matching items. Limits to 10."""
    try:
        supabase = get_supabase()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    resp = supabase.table("stocks").select("*").ilike("ticker", f"%{q}%").limit(10).execute()
    return {"results": resp.data}

@router.get("/api/stocks/{ticker}")
def get_stock(ticker: str):
    try:
        supabase = get_supabase()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    resp = supabase.table("stocks").select("*").eq("ticker", ticker.upper()).execute()
    if not resp.data:
        raise HTTPException(status_code=404, detail="Stock not found")
    return resp.data[0]

import uuid
import datetime
import logging
from app.db.supabase_client import get_supabase
from app.agents.orchestrator import run_pipeline

logger = logging.getLogger(__name__)

def start_report_generation(ticker: str, background_tasks) -> str:
    """Trigger the LangGraph workflow inside a BackgroundTask."""
    supabase = get_supabase()
    if not supabase:
        raise ValueError("Supabase client is not configured.")
        
    ticker = ticker.upper()
    
    # 1. Ensure stock exists
    stock_resp = supabase.table("stocks").select("*").eq("ticker", ticker).execute()
    if not stock_resp.data:
        # Create a basic record for the stock
        new_stock = supabase.table("stocks").insert({
            "ticker": ticker,
            "company_name": ticker
        }).execute()
        stock_id = new_stock.data[0]["id"]
    else:
        stock_id = stock_resp.data[0]["id"]
        
    # 2. Create a report job
    job_resp = supabase.table("report_jobs").insert({
        "stock_id": stock_id,
        "status": "pending"
    }).execute()
    job_id = job_resp.data[0]["id"]
    
    # 3. Add background task to run the LangGraph pipeline
    background_tasks.add_task(_run_report_task, job_id, stock_id, ticker)
    
    return job_id

def _run_report_task(job_id: str, stock_id: str, ticker: str):
    """The synchronous task function executed in the background."""
    supabase = get_supabase()
    try:
        # Set to analyzing
        supabase.table("report_jobs").update({"status": "analyzing"}).eq("id", job_id).execute()
        
        # Run orchestrator
        logger.info(f"Starting pipeline for job {job_id} / ticker {ticker}")
        final_markdown = run_pipeline(ticker)
        
        # Save the report with completed task state
        supabase.table("reports").insert({
            "job_id": job_id,
            "stock_id": stock_id,
            "markdown_content": final_markdown
        }).execute()
        
        supabase.table("report_jobs").update({
            "status": "completed", 
            "completed_at": datetime.datetime.utcnow().isoformat()
        }).eq("id", job_id).execute()

    except Exception as e:
        logger.error(f"Job {job_id} failed: {str(e)}")
        supabase.table("report_jobs").update({
            "status": "failed",
            "error_msg": str(e)
        }).eq("id", job_id).execute()

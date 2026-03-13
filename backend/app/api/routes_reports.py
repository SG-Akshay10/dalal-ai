from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from app.db.supabase_client import get_supabase
from app.services.report_service import start_report_generation

router = APIRouter()


class ReportRequest(BaseModel):
    ticker: str


@router.post("/api/reports")
def trigger_report(request: ReportRequest, background_tasks: BackgroundTasks):
    if not request.ticker:
        raise HTTPException(status_code=400, detail="Ticker is required")
    try:
        job_id = start_report_generation(request.ticker.upper(), background_tasks)
        return {"job_id": job_id, "status": "job_started"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/reports/{job_id}")
def get_report(job_id: str):
    try:
        supabase = get_supabase()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    job_resp = supabase.table("report_jobs").select("*").eq("id", job_id).execute()
    if not job_resp.data:
        raise HTTPException(status_code=404, detail="Job not found")

    job = job_resp.data[0]
    if job["status"] == "completed":
        report_resp = supabase.table("reports").select("*").eq("job_id", job_id).execute()
        if not report_resp.data:
            return {"job": job, "report": None}
        return {"job": job, "report": report_resp.data[0]}

    return {"job": job, "report": None}


@router.get("/api/jobs/{job_id}/status")
def get_job_status(job_id: str):
    try:
        supabase = get_supabase()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    job_resp = (
        supabase.table("report_jobs")
        .select("status, error_msg")
        .eq("id", job_id)
        .execute()
    )
    if not job_resp.data:
        raise HTTPException(status_code=404, detail="Job not found")
    return job_resp.data[0]

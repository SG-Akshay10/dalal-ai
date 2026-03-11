from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid

app = FastAPI(title="DalalAI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalyseRequest(BaseModel):
    query: str

@app.get("/api/health")
async def health_check():
    return {"status": "ok"}

@app.post("/api/analyse")
async def trigger_analyse(request: AnalyseRequest):
    # Foundational POST trigger to be built upon later
    job_id = str(uuid.uuid4())
    # TODO: Kick off Celery task here
    return {"job_id": job_id, "estimated_duration": "5 minutes"}

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes_reports import router as reports_router
from app.api.routes_stocks import router as stocks_router

app = FastAPI(title="StockSense AI API")

# Allow all CORS for development / decoupled frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In Phase 2, we just include REST endpoints. WebSockets are deferred correctly.
app.include_router(reports_router)
app.include_router(stocks_router)

@app.get("/health")
def health_check():
    return {"status": "ok", "version": "0.1.0"}

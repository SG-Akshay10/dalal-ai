import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from app.main import app
from app.agents.orchestrator import run_pipeline

client = TestClient(app)

# --- Endpoint Integration Tests ---

@patch("app.api.routes_reports.start_report_generation")
def test_trigger_report(mock_start):
    mock_start.return_value = "fake-job-uuid-1234"
    
    response = client.post("/api/reports", json={"ticker": "RELIANCE"})
    assert response.status_code == 200
    assert response.json() == {"job_id": "fake-job-uuid-1234", "status": "job_started"}
    mock_start.assert_called_once()

@patch("app.api.routes_reports.get_supabase")
def test_get_job_status(mock_get_supabase):
    mock_db = MagicMock()
    mock_get_supabase.return_value = mock_db
    
    # Mock chain: supabase.table("report_jobs").select("status, error_msg").eq("id", job_id).execute()
    mock_execute = mock_db.table.return_value.select.return_value.eq.return_value.execute
    mock_execute.return_value.data = [{"status": "completed", "error_msg": None}]
    
    response = client.get("/api/jobs/123/status")
    assert response.status_code == 200
    assert response.json() == {"status": "completed", "error_msg": None}

@patch("app.api.routes_stocks.get_supabase")
def test_search_stocks(mock_get_supabase):
    mock_db = MagicMock()
    mock_get_supabase.return_value = mock_db
    
    # Mock chain: supabase.table("stocks").select("*").ilike("ticker", f"%{q}%").limit(10).execute()
    mock_execute = mock_db.table.return_value.select.return_value.ilike.return_value.limit.return_value.execute
    mock_execute.return_value.data = [{"ticker": "TCS", "company_name": "Tata Consultancy Services"}]
    
    response = client.get("/api/stocks/search?q=TCS")
    assert response.status_code == 200
    assert response.json() == {"results": [{"ticker": "TCS", "company_name": "Tata Consultancy Services"}]}


# --- Pipeline Orchestrator Integration Tests ---

@patch("app.agents.orchestrator.fetch_documents")
@patch("app.agents.orchestrator.fetch_news")
@patch("app.agents.orchestrator.fetch_social")
@patch("app.agents.orchestrator.extract_fundamentals")
@patch("app.agents.orchestrator.analyze_sentiment")
@patch("app.agents.orchestrator.identify_competitors")
@patch("app.agents.orchestrator.analyze_sector")
@patch("app.agents.orchestrator.generate_report")
def test_pipeline_execution(
    mock_gen_report, mock_sector, mock_comp, mock_sent, mock_fund,
    mock_social, mock_news, mock_docs
):
    """Test that the full LangGraph DAG connects nodes and executes them successfully."""
    mock_docs.return_value = []
    mock_news.return_value = []
    mock_social.return_value = []
    mock_fund.return_value = None
    mock_sent.return_value = None
    mock_comp.return_value = None
    mock_sector.return_value = None
    mock_gen_report.return_value = "# Final Mocked Market Report"
    
    result = run_pipeline("HDFCBANK")
    assert result == "# Final Mocked Market Report"
    
    # Verify all nodes in the DAG were invoked
    mock_docs.assert_called_once()
    mock_news.assert_called_once()
    mock_social.assert_called_once()
    mock_fund.assert_called_once()
    mock_sent.assert_called_once()
    mock_comp.assert_called_once()
    mock_sector.assert_called_once()
    mock_gen_report.assert_called_once()

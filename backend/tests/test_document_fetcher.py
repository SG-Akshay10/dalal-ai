import pytest
from app.agents.document_fetcher import fetch_latest_annual_report

@pytest.mark.asyncio
async def test_fetch_latest_annual_report_success():
    """Test fetching the latest annual report for a valid NSE symbol."""
    symbol = "RELIANCE"
    document = await fetch_latest_annual_report(symbol)
    
    assert document is not None
    assert document.symbol == symbol
    assert document.doc_type == "Annual Report"
    assert len(document.text_chunks) > 0
    assert document.url is not None

@pytest.mark.asyncio
async def test_fetch_latest_annual_report_invalid_symbol():
    """Test fetching with an invalid symbol."""
    symbol = "INVALID_SYMBOL_123"
    document = await fetch_latest_annual_report(symbol)
    
    assert document is None

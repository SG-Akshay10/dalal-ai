"""DocumentObject schema — Phase 1 data contract.

Produced by: document_fetcher.py
Consumed by: Phase 2 fundamental_agent.py (via ChromaDB RAG)
"""
from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field, HttpUrl


class DocumentObject(BaseModel):
    """Represents a regulatory filing or announcement fetched from BSE, NSE, or SEBI.

    This is the canonical output of the document_fetcher scraper.
    All fields are required unless noted. Phase 2 agents rely on these
    field names and types — do not change them without a migration plan.
    """

    source: Literal["BSE", "NSE", "SEBI"] = Field(
        description="Exchange or regulator that published this document."
    )
    date: datetime = Field(
        description="Publication date of the document (UTC)."
    )
    doc_type: str = Field(
        description="Category of document: 'announcement', 'filing', 'prospectus', 'quarterly_result', etc."
    )
    text: str = Field(
        description="Full extracted text content of the document."
    )
    url: str = Field(
        description="Original URL of the document or PDF."
    )
    ocr_used: bool = Field(
        default=False,
        description="True if Tesseract OCR was used (indicates scanned/image PDF)."
    )
    parse_confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score for text extraction quality. 1.0 = digital PDF. <0.8 = OCR only."
    )

    model_config = {"json_schema_extra": {"example": {
        "source": "BSE",
        "date": "2024-01-15T00:00:00",
        "doc_type": "quarterly_result",
        "text": "Reliance Industries Q3FY24: Revenue grew 15% YoY to ₹2.28 lakh crore...",
        "url": "https://www.bseindia.com/xml-data/corpfiling/AttachLive/doc123.pdf",
        "ocr_used": False,
        "parse_confidence": 1.0,
    }}}

"""
Filings collector.
Scrapes regulatory filings and corporate announcements from NSE and BSE.
PDF text extraction via PyMuPDF (fitz).
"""

import io
import logging
from datetime import date, datetime, timezone
from typing import Optional

import fitz  # PyMuPDF
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://www.bseindia.com",
    "Referer": "https://www.bseindia.com/",
}
REQUEST_TIMEOUT = 30
MAX_FILING_TEXT_CHARS = 10_000
MAX_PDF_PAGES = 15


def collect(
    ticker: str,
    exchange: str,
    run_id: str,
) -> list[dict]:
    """
    Fetch filings for `ticker` from NSE and/or BSE.
    Returns a list of dicts ready for db.upsert_filings().
    """
    records: list[dict] = []

    if exchange in ("NSE", "AUTO"):
        records += _fetch_nse(ticker, run_id)

    if exchange in ("BSE", "AUTO"):
        records += _fetch_bse(ticker, run_id)

    # Deduplicate by URL within this batch
    seen: set[str] = set()
    unique = [r for r in records if not (r["document_url"] in seen or seen.add(r["document_url"]))]

    logger.info("Filings: collected %d filings for %s.", len(unique), ticker)
    return unique


# ── NSE ───────────────────────────────────────────────────────────────────────

def _fetch_nse(ticker: str, run_id: str) -> list[dict]:
    url = (
        "https://www.nseindia.com/api/annual-reports"
        f"?index=equities&symbol={ticker}"
    )
    session = requests.Session()
    # NSE requires a prior hit on the homepage to set cookies
    try:
        session.get("https://www.nseindia.com", headers=HEADERS, timeout=REQUEST_TIMEOUT)
        resp = session.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning("Filings/NSE: request failed for %s: %s", ticker, e)
        return []

    records = []
    for item in data if isinstance(data, list) else data.get("data", []):
        doc_url = item.get("fileName") or item.get("fileURL") or item.get("pdfLink", "")
        if not doc_url or not doc_url.startswith("http"):
            continue

        pub_raw = item.get("broadcastDate") or item.get("date", "")
        pub = _parse_date_loose(pub_raw)

        records.append({
            "ticker":           ticker,
            "exchange":         "NSE",
            "filing_type":      _classify_filing(item.get("subject", "")),
            "title":            (item.get("subject") or item.get("description", ""))[:500],
            "document_url":     doc_url,
            "published_at":     pub.isoformat() if pub else None,
            "period_end":       None,
            "extracted_text":   _extract_pdf_text(doc_url),
            "page_count":       None,
            "ingestion_run_id": run_id,
        })

    logger.debug("Filings/NSE: %d items for %s.", len(records), ticker)
    return records


# ── BSE ───────────────────────────────────────────────────────────────────────

def _fetch_bse(ticker: str, run_id: str) -> list[dict]:
    # BSE uses company code (scrip code); we search by ticker name via the API
    search_url = f"https://api.bseindia.com/BseIndiaAPI/api/ListofScripData/w?segment=Equity&status=Active&scripcode=&group=&industry=&isincode=&scrip_cd=&cname={ticker}"
    try:
        resp = requests.get(search_url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        results = resp.json()
    except Exception as e:
        logger.warning("Filings/BSE: company lookup failed for %s: %s", ticker, e)
        return []

    scrip_code = None
    for item in results:
        if item.get("short_name", "").upper() == ticker.upper():
            scrip_code = item.get("scripcode")
            break

    if not scrip_code:
        logger.info("Filings/BSE: no scrip code found for %s.", ticker)
        return []

    ann_url = (
        f"https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w"
        f"?strCat=-1&strPrevDate=&strScrip={scrip_code}&strSearch=P&strToDate=&strType=C&subcategory=-1"
    )
    try:
        resp = requests.get(ann_url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning("Filings/BSE: announcements failed for %s: %s", ticker, e)
        return []

    records = []
    for item in data.get("Table", []):
        doc_url = item.get("ATTACHMENTNAME", "")
        if not doc_url:
            continue
        if not doc_url.startswith("http"):
            doc_url = f"https://www.bseindia.com/xml-data/corpfiling/AttachLive/{doc_url}"

        pub = _parse_date_loose(item.get("NEWS_DT", ""))

        records.append({
            "ticker":           ticker,
            "exchange":         "BSE",
            "filing_type":      _classify_filing(item.get("NEWSSUB", "")),
            "title":            (item.get("NEWSSUB") or "")[:500],
            "document_url":     doc_url,
            "published_at":     pub.isoformat() if pub else None,
            "period_end":       None,
            "extracted_text":   _extract_pdf_text(doc_url),
            "page_count":       None,
            "ingestion_run_id": run_id,
        })

    logger.debug("Filings/BSE: %d items for %s.", len(records), ticker)
    return records


# ── PDF extraction ────────────────────────────────────────────────────────────

def _extract_pdf_text(url: str) -> Optional[str]:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT, stream=True)
        resp.raise_for_status()
        content = b"".join(resp.iter_content(chunk_size=8192))

        doc = fitz.open(stream=io.BytesIO(content), filetype="pdf")
        total_pages = len(doc)
        pages_to_read = min(total_pages, MAX_PDF_PAGES)

        text_parts = []
        for i in range(pages_to_read):
            text_parts.append(doc[i].get_text())
        doc.close()

        full_text = " ".join(text_parts)
        return full_text[:MAX_FILING_TEXT_CHARS] if full_text.strip() else None

    except Exception as e:
        logger.warning("Filings: PDF extraction failed for %s: %s", url, e)
        return None


# ── Helpers ───────────────────────────────────────────────────────────────────

FILING_KEYWORDS = {
    "ANNUAL_REPORT":      ["annual report", "annual-report"],
    "QUARTERLY_RESULT":   ["quarterly", "financial result", "q1", "q2", "q3", "q4"],
    "ANNOUNCEMENT":       ["announcement", "notice", "intimation", "disclosure"],
    "PRESENTATION":       ["presentation", "investor"],
}


def _classify_filing(subject: str) -> str:
    lower = subject.lower()
    for ftype, keywords in FILING_KEYWORDS.items():
        if any(k in lower for k in keywords):
            return ftype
    return "OTHER"


def _parse_date_loose(raw: str) -> Optional[datetime]:
    if not raw:
        return None
    formats = [
        "%d %b %Y", "%b %d, %Y", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d",
        "%d-%m-%Y", "%d/%m/%Y", "%Y%m%d",
    ]
    raw = raw.strip().split("T")[0] if "T" in raw else raw.strip()
    for fmt in formats:
        try:
            return datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None

"""Document Fetcher — fetches stock metadata and filings from NSE/BSE.

Data sources (validated via live diagnostics):
- NSE Quote API: /api/quote-equity — returns stock metadata, price info (WORKS)
- NSE Filings: /api/corporate-announcements — announcements with PDF links (WORKS with session)
- BSE API: blocked for JSON requests, so we scrape the BSE announcements HTML page as fallback

Output: list[DocumentObject] — locked Phase 2 contract.
"""
import logging
import re
import time
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple

import requests
from bs4 import BeautifulSoup

from app.schemas.document_object import DocumentObject
from app.scrapers.pdf_extractor import extract_text_from_pdf

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) "
        "Gecko/20100101 Firefox/120.0"
    ),
    "Accept-Language": "en-IN,en;q=0.9",
    "Accept": "application/json, text/plain, */*",
}
_REQUEST_DELAY_SECONDS = 1.5


def fetch_documents(ticker: str, days: int = 90) -> List[DocumentObject]:
    """Fetch regulatory documents and announcements for a given NSE/BSE ticker.

    Fetches from NSE corporate announcements API and BSE announcements page.
    PDF documents are extracted using pdf_extractor.

    Args:
        ticker: NSE or BSE ticker symbol (e.g. 'HDFCBANK', 'RELIANCE').
        days: Number of past days to fetch documents for (default 90).

    Returns:
        List of DocumentObject instances. Empty list if no documents found.
    """
    since = datetime.now(tz=timezone.utc) - timedelta(days=days)
    results: List[DocumentObject] = []

    # Fetch from each source independently
    for fetcher in (_fetch_nse_announcements, _fetch_bse_announcements_html):
        try:
            docs = fetcher(ticker, since)
            results.extend(docs)
            time.sleep(_REQUEST_DELAY_SECONDS)
        except Exception as exc:
            logger.warning("Fetcher %s failed for %s: %s", fetcher.__name__, ticker, exc)

    logger.info("fetch_documents(%s, days=%d) → %d documents", ticker, days, len(results))
    return results


def _get_nse_session() -> Optional[requests.Session]:
    """Create an NSE session with proper cookies.

    NSE blocks requests without a valid session cookie.
    We first hit the main NSE page to get cookies, then use them for API calls.
    """
    session = requests.Session()
    session.headers.update(_HEADERS)
    session.headers["Referer"] = "https://www.nseindia.com/"

    try:
        # Get initial cookies from NSE main page
        resp = session.get("https://www.nseindia.com", timeout=10)
        if resp.status_code == 200:
            return session

        # Sometimes NSE returns 403 on first try — retry once
        time.sleep(1)
        resp = session.get(
            "https://www.nseindia.com/get-quotes/equity?symbol=HDFCBANK",
            timeout=10,
        )
        if resp.status_code == 200:
            return session

        logger.warning("NSE session setup failed with status %d", resp.status_code)
        return session  # Return session anyway, API calls may still work

    except Exception as exc:
        logger.warning("NSE session setup error: %s", exc)
        return None


def _fetch_nse_announcements(ticker: str, since: datetime) -> List[DocumentObject]:
    """Fetch NSE corporate announcements for the given ticker.

    Uses the NSE /api/corporate-announcements endpoint which returns JSON
    with announcement details including PDF attachment links.
    """
    session = _get_nse_session()
    if not session:
        return []

    # NSE corporate announcements endpoint
    from_date = since.strftime("%d-%m-%Y")
    to_date = datetime.now(tz=timezone.utc).strftime("%d-%m-%Y")

    url = (
        f"https://www.nseindia.com/api/corporate-announcements"
        f"?index=equities&symbol={ticker}"
        f"&from_date={from_date}&to_date={to_date}"
    )

    docs: List[DocumentObject] = []
    try:
        resp = session.get(url, timeout=15)
        if resp.status_code != 200:
            logger.warning("NSE announcements returned %d for %s", resp.status_code, ticker)
            return []

        announcements = resp.json()
        if not isinstance(announcements, list):
            logger.warning("NSE announcements unexpected format for %s", ticker)
            return []

        for ann in announcements[:20]:
            doc = _parse_nse_announcement(ann)
            if doc:
                docs.append(doc)

    except Exception as exc:
        logger.warning("NSE announcements failed for %s: %s", ticker, exc)

    return docs


def _parse_nse_announcement(ann: dict) -> Optional[DocumentObject]:
    """Parse a single NSE announcement into a DocumentObject."""
    try:
        subject = ann.get("desc", "") or ann.get("attchmntText", "")
        pdf_url = ann.get("attchmntFile", "")
        date_str = ann.get("an_dt", "") or ann.get("dt", "")

        # Parse date
        doc_date = datetime.now(tz=timezone.utc)
        for fmt in ("%d-%b-%Y %H:%M:%S", "%d-%b-%Y", "%Y-%m-%dT%H:%M:%S"):
            try:
                doc_date = datetime.strptime(date_str.strip(), fmt).replace(tzinfo=timezone.utc)
                break
            except (ValueError, AttributeError):
                continue

        # If there's an attachment, try to extract text from it
        text = subject
        confidence = 0.5
        ocr_used = False
        full_url = ""

        if pdf_url:
            if pdf_url.startswith("http"):
                full_url = pdf_url
            else:
                full_url = f"https://nsearchives.nseindia.com/corporate/content/{pdf_url}"

            extracted = _safe_extract_pdf(full_url)
            if extracted[0]:
                text = f"{subject}\n\n{extracted[0]}"
                confidence = extracted[1]
                ocr_used = extracted[2]

        if not text or len(text.strip()) < 10:
            return None

        return DocumentObject(
            source="NSE",
            date=doc_date,
            doc_type=ann.get("smIndustry", "announcement").lower().replace(" ", "_"),
            text=text,
            url=full_url or f"https://www.nseindia.com/companies-listing/corporate-filings-announcements",
            ocr_used=ocr_used,
            parse_confidence=confidence,
        )

    except Exception as exc:
        logger.debug("Failed to parse NSE announcement: %s", exc)
        return None


def _fetch_bse_announcements_html(ticker: str, since: datetime) -> List[DocumentObject]:
    """Scrape BSE announcements HTML page for the given ticker.

    BSE's JSON API now returns HTML (rate-limited/blocked), so we
    scrape the announcements HTML page instead.

    Note: This requires the BSE scrip code, not the ticker symbol.
    We attempt a search first to find the scrip code.
    """
    # BSE uses scrip codes, not ticker symbols — try to look up
    scrip_code = _bse_lookup_scrip_code(ticker)
    if not scrip_code:
        logger.info("Could not find BSE scrip code for %s — skipping BSE", ticker)
        return []

    url = f"https://www.bseindia.com/corporates/ann.html?scrip_cd={scrip_code}"
    docs: List[DocumentObject] = []

    try:
        resp = requests.get(url, headers=_HEADERS, timeout=15)
        if resp.status_code != 200:
            logger.warning("BSE announcements page returned %d for %s", resp.status_code, ticker)
            return []

        soup = BeautifulSoup(resp.text, "lxml")

        # BSE announcements are typically in a table or list structure
        # The page is JS-rendered so we may only get the shell — this is a best-effort approach
        announcement_links = soup.select("a[href*='AttachLive']")

        for link in announcement_links[:15]:
            pdf_url = link.get("href", "")
            if not pdf_url.startswith("http"):
                pdf_url = f"https://www.bseindia.com{pdf_url}"

            title = link.get_text(strip=True) or "BSE Announcement"

            extracted = _safe_extract_pdf(pdf_url)
            text = title
            confidence = 0.3
            ocr_used = False

            if extracted[0]:
                text = f"{title}\n\n{extracted[0]}"
                confidence = extracted[1]
                ocr_used = extracted[2]

            docs.append(DocumentObject(
                source="BSE",
                date=datetime.now(tz=timezone.utc),
                doc_type="announcement",
                text=text,
                url=pdf_url,
                ocr_used=ocr_used,
                parse_confidence=confidence,
            ))

    except Exception as exc:
        logger.warning("BSE HTML scrape failed for %s: %s", ticker, exc)

    return docs


def _bse_lookup_scrip_code(ticker: str) -> Optional[str]:
    """Look up BSE scrip code from ticker symbol using BSE search API."""
    url = f"https://api.bseindia.com/BseIndiaAPI/api/Suggest/w?flag=sugg&val={ticker}"
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=10)
        if resp.status_code == 200:
            try:
                results = resp.json()
                if results:
                    # Results are like: [{"scrip_cd": "500180", "scrip_name": "HDFC Bank Ltd."}]
                    return str(results[0].get("scrip_cd", ""))
            except (ValueError, IndexError, TypeError):
                # JSON parse failed or empty results — try text parsing
                text = resp.text.strip()
                if text and text[0] in ('[', '{'):
                    pass  # JSON but malformed
                else:
                    # Sometimes BSE returns pipe-delimited: "500180|HDFCBANK|..."
                    parts = text.split("|")
                    if parts and parts[0].isdigit():
                        return parts[0]
    except Exception as exc:
        logger.debug("BSE scrip lookup failed for %s: %s", ticker, exc)

    # Hardcoded fallback for common tickers
    common_scrips = {
        "HDFCBANK": "500180", "RELIANCE": "500325", "TCS": "532540",
        "INFY": "500209", "ICICIBANK": "532174", "SBIN": "500112",
        "KOTAKBANK": "500247", "BHARTIARTL": "532454", "ITC": "500875",
        "HINDUNILVR": "500696", "AXISBANK": "532215", "WIPRO": "507685",
        "BAJFINANCE": "500034", "MARUTI": "532500", "LT": "500510",
        "HCLTECH": "532281", "SUNPHARMA": "524715", "TITAN": "500114",
        "ASIANPAINT": "500820", "ULTRACEMCO": "532538",
    }
    return common_scrips.get(ticker.upper())


def _safe_extract_pdf(pdf_url: str) -> Tuple[Optional[str], float, bool]:
    """Attempt PDF extraction; return (None, 0.0, False) on failure."""
    try:
        return extract_text_from_pdf(pdf_url)
    except Exception as exc:
        logger.debug("PDF extraction failed for %s: %s", pdf_url, exc)
        return None, 0.0, False

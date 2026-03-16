"""Stock Alias Generator — auto-discovers company name and popular aliases for any ticker.

Given just a ticker symbol, this module:
1. Looks up the official company name from NSE Quote API
2. Falls back to a hardcoded dictionary for popular Indian stocks

The output is a StockInfo dataclass used by all scrapers so the user only
needs to input the ticker symbol.

Example:
    info = get_stock_info("ETERNAL")
    # → StockInfo(
    #     ticker="ETERNAL",
    #     company_name="Eternal Limited",
    #     aliases=["Zomato", "ETERNAL", "Eternal Ltd"],
    #     search_query='"Zomato" OR "ETERNAL" OR "Eternal Limited"'
    #   )
"""
import json
import logging
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache

import requests

logger = logging.getLogger(__name__)

_NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0",
    "Accept-Language": "en-IN,en;q=0.9",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.nseindia.com/",
}


@dataclass
class StockInfo:
    """Complete stock identity resolved from a ticker symbol."""
    ticker: str
    company_name: str
    aliases: list[str] = field(default_factory=list)

    @property
    def search_query(self) -> str:
        """Build an OR-joined search query from all known names."""
        terms = set()
        terms.add(self.ticker)
        if self.company_name:
            terms.add(self.company_name)
        for alias in self.aliases:
            if alias:
                terms.add(alias)
        quoted = [f'"{t}"' for t in sorted(terms)]
        return " OR ".join(quoted)

    @property
    def all_names(self) -> list[str]:
        """All known names including ticker, company name, and aliases."""
        names = [self.ticker]
        if self.company_name:
            names.append(self.company_name)
        names.extend(a for a in self.aliases if a and a != self.ticker and a != self.company_name)
        # Deduplicate while preserving order
        return list(dict.fromkeys(names))


# ── Hardcoded aliases for popular Indian stocks ───────────────────────────────
# These are extremely common stocks where the popular name differs from
# the ticker or official company name.

_HARDCODED_ALIASES = {
    "ETERNAL": ["Zomato", "Zomato Ltd"],
    "SBIN": ["SBI", "State Bank of India", "State Bank"],
    "HDFCBANK": ["HDFC Bank", "HDFC"],
    "RELIANCE": ["Reliance Industries", "RIL", "Mukesh Ambani"],
    "TCS": ["Tata Consultancy Services", "Tata Consultancy"],
    "INFY": ["Infosys", "Infosys Ltd"],
    "ICICIBANK": ["ICICI Bank", "ICICI"],
    "KOTAKBANK": ["Kotak Mahindra Bank", "Kotak Bank", "Kotak"],
    "BHARTIARTL": ["Airtel", "Bharti Airtel", "Jio competitor"],
    "ITC": ["ITC Limited", "ITC Ltd"],
    "HINDUNILVR": ["HUL", "Hindustan Unilever"],
    "WIPRO": ["Wipro Ltd", "Wipro Technologies"],
    "BAJFINANCE": ["Bajaj Finance", "Bajaj Fin"],
    "MARUTI": ["Maruti Suzuki", "Maruti Suzuki India"],
    "LT": ["Larsen & Toubro", "L&T"],
    "HCLTECH": ["HCL Technologies", "HCL Tech"],
    "SUNPHARMA": ["Sun Pharma", "Sun Pharmaceutical"],
    "TITAN": ["Titan Company", "Titan Industries"],
    "ASIANPAINT": ["Asian Paints", "Asian Paint"],
    "ULTRACEMCO": ["UltraTech Cement", "UltraTech"],
    "AXISBANK": ["Axis Bank"],
    "TATAMOTORS": ["Tata Motors", "Tata Motor"],
    "TATASTEEL": ["Tata Steel"],
    "M&M": ["Mahindra & Mahindra", "M and M", "Mahindra"],
    "ADANIENT": ["Adani Enterprises", "Adani"],
    "ADANIPORTS": ["Adani Ports", "Adani Port"],
    "JSWSTEEL": ["JSW Steel"],
    "POWERGRID": ["Power Grid Corporation", "Power Grid"],
    "NTPC": ["NTPC Limited", "National Thermal Power"],
    "BPCL": ["Bharat Petroleum", "BPCL Ltd"],
    "ONGC": ["Oil and Natural Gas Corporation", "ONGC Ltd"],
    "COALINDIA": ["Coal India", "CIL"],
    "DRREDDY": ["Dr Reddy's", "Dr Reddys Laboratories"],
    "DIVISLAB": ["Divi's Laboratories", "Divis Lab"],
    "BAJAJFINSV": ["Bajaj Finserv"],
    "TECHM": ["Tech Mahindra"],
    "NESTLEIND": ["Nestle India", "Nestle"],
    "HEROMOTOCO": ["Hero MotoCorp", "Hero Honda"],
    "EICHERMOT": ["Eicher Motors", "Royal Enfield"],
    "APOLLOHOSP": ["Apollo Hospitals"],
    "ZOMATO": ["Zomato", "Eternal"],
    "PAYTM": ["One97 Communications", "Paytm", "One97"],
    "NYKAA": ["FSN E-Commerce", "Nykaa"],
    "POLICYBZR": ["PB Fintech", "PolicyBazaar"],
    "DELHIVERY": ["Delhivery Ltd"],
    "IRCTC": ["Indian Railway Catering", "IRCTC Ltd"],
}


@lru_cache(maxsize=100)
def get_stock_info(ticker: str) -> StockInfo:
    """Get complete stock identity from just a ticker symbol.

    Resolution order:
    1. NSE Quote API → official company name
    2. Hardcoded dictionary → fallback for popular stocks

    Results are cached (LRU, max 100 tickers).

    Args:
        ticker: NSE/BSE ticker symbol (e.g. 'SBIN', 'ETERNAL', 'HDFCBANK')

    Returns:
        StockInfo with ticker, company_name, aliases, and search_query
    """
    ticker = ticker.strip().upper()
    logger.info("Resolving stock info for ticker: %s", ticker)

    # Step 1: Get official company name from NSE
    company_name = _lookup_nse_company_name(ticker)
    logger.info("NSE company name for %s: %s", ticker, company_name or "not found")

    # Step 2: Merge with hardcoded aliases
    hardcoded = _HARDCODED_ALIASES.get(ticker, [])

    # Combine and deduplicate
    all_aliases = list(dict.fromkeys(hardcoded))

    # If we didn't get a company name from NSE, use the first alias
    if not company_name and all_aliases:
        company_name = all_aliases[0]

    info = StockInfo(
        ticker=ticker,
        company_name=company_name or ticker,
        aliases=all_aliases,
    )

    logger.info("Resolved %s → %s (aliases: %s)", ticker, info.company_name, info.aliases)
    return info


def _lookup_nse_company_name(ticker: str) -> str | None:
    """Look up company name from NSE Quote API."""
    session = requests.Session()
    session.headers.update(_NSE_HEADERS)

    try:
        # Get session cookies
        session.get("https://www.nseindia.com", timeout=5)
        # Quote API
        resp = session.get(
            f"https://www.nseindia.com/api/quote-equity?symbol={ticker}",
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            info = data.get("info", {})
            return info.get("companyName", "")
    except Exception as exc:
        logger.debug("NSE company lookup failed for %s: %s", ticker, exc)

    return None


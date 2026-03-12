#!/usr/bin/env python3
"""Live diagnostic: test all scrapers with HDFCBANK stock.

Run: cd backend && python tests/manual/test_live_scrapers.py

This script makes REAL HTTP calls to BSE, NSE, Reddit, etc.
No mocking — it's designed to diagnose what's working and what's broken.
"""
import os
import sys
import json
import logging
from datetime import datetime, timedelta, timezone

import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("diagnostic")

# Add parent to path so we can import app modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TICKER = "HDFCBANK"
COMPANY = "HDFC Bank"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0",
    "Accept-Language": "en-IN,en;q=0.9",
}


def sep(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


# ── 1. BSE API ──────────────────────────────────────────────────────────────

def test_bse():
    sep("BSE Announcements API")

    # Try BSE scrip code for HDFC Bank = 500180
    urls = [
        # Approach 1: BSE API with scrip code
        f"https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w"
        f"?scrip_cd=500180&Category=Company%20Update&subcategory=-1"
        f"&fromdate=20250101&todate=99991231&pageno=1&subcatgname=-1",
        # Approach 2: BSE announcements HTML page
        f"https://www.bseindia.com/corporates/ann.html?scrip_cd=500180",
        # Approach 3: BSE corporate filings JSON
        f"https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w"
        f"?scrip_cd=500180&Category=-1&subcategory=-1"
        f"&fromdate=20250101&todate=99991231&pageno=1&subcatgname=-1",
    ]

    for i, url in enumerate(urls, 1):
        print(f"[BSE Approach {i}] {url[:80]}...")
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            print(f"  Status: {resp.status_code}")
            print(f"  Content-Type: {resp.headers.get('content-type', 'N/A')}")
            print(f"  Body length: {len(resp.text)} chars")
            if resp.status_code == 200:
                try:
                    data = resp.json()
                    table = data.get("Table", [])
                    print(f"  JSON parsed: {len(table)} announcements")
                    if table:
                        print(f"  First entry keys: {list(table[0].keys())}")
                        print(f"  Sample: {json.dumps(table[0], indent=2)[:500]}")
                except:
                    print(f"  Not JSON. First 300 chars: {resp.text[:300]}")
            print(f"  ✅ Reachable" if resp.status_code == 200 else f"  ❌ HTTP {resp.status_code}")
        except Exception as e:
            print(f"  ❌ FAILED: {e}")


# ── 2. NSE API ──────────────────────────────────────────────────────────────

def test_nse():
    sep("NSE Corporate Filings API")

    # NSE requires session cookies from the main page first
    session = requests.Session()
    session.headers.update(HEADERS)

    # Step 1: Get session cookies
    print("[NSE Step 1] Getting session cookies from nseindia.com...")
    try:
        main_resp = session.get("https://www.nseindia.com", timeout=10)
        print(f"  Main page status: {main_resp.status_code}")
        print(f"  Cookies: {dict(session.cookies)}")
    except Exception as e:
        print(f"  ❌ Main page failed: {e}")
        return

    # Step 2: Try different API endpoints
    endpoints = [
        f"https://www.nseindia.com/api/corporates/searchCorpAction?index=equities&from_date=01-01-2025&to_date=13-03-2026&symbol={TICKER}",
        f"https://www.nseindia.com/api/corporates/corporateActions?index=equities&symbol={TICKER}",
        f"https://www.nseindia.com/api/quote-equity?symbol={TICKER}",
    ]

    for i, url in enumerate(endpoints, 1):
        print(f"\n[NSE Approach {i}] {url[:80]}...")
        try:
            resp = session.get(url, timeout=15)
            print(f"  Status: {resp.status_code}")
            if resp.status_code == 200:
                try:
                    data = resp.json()
                    print(f"  JSON keys: {list(data.keys()) if isinstance(data, dict) else f'array[{len(data)}]'}")
                    print(f"  Sample: {json.dumps(data, indent=2)[:500]}")
                except:
                    print(f"  Not JSON. First 200 chars: {resp.text[:200]}")
            print(f"  ✅ Success" if resp.status_code == 200 else f"  ❌ HTTP {resp.status_code}")
        except Exception as e:
            print(f"  ❌ FAILED: {e}")


# ── 3. Reddit Search (Direct Scraping) ──────────────────────────────────────

def test_reddit():
    sep("Reddit Direct Scraping (old.reddit.com)")

    subreddits = [
        "IndiaInvestments", "DalalStreet", "IndianStreetBets",
        "IndianStockMarket", "indiainvestments",
    ]

    for sub in subreddits:
        url = f"https://old.reddit.com/r/{sub}/search?q={TICKER}&sort=new&restrict_sr=on&t=month"
        print(f"[Reddit r/{sub}] Searching for '{TICKER}'...")
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, "lxml")
                results = soup.select("div.search-result-link")
                print(f"  ✅ {len(results)} posts found")
                for r in results[:3]:
                    title_el = r.select_one("a.search-title")
                    if title_el:
                        print(f"     → {title_el.get_text(strip=True)[:80]}")
            else:
                print(f"  ❌ HTTP {resp.status_code}")
        except Exception as e:
            print(f"  ❌ FAILED: {e}")


# ── 4. Google Search for Stock + Reddit ──────────────────────────────────────

def test_google_reddit():
    sep("Google Search (stock + reddit)")

    query = f"{COMPANY} {TICKER} stock reddit"
    # Note: Direct Google scraping may be rate-limited. This tests the approach.
    url = f"https://www.google.com/search?q={query}&tbs=qdr:m&num=10"
    print(f"[Google] Searching: '{query}'...")
    try:
        resp = requests.get(url, headers={
            **HEADERS,
            "Accept": "text/html,application/xhtml+xml",
        }, timeout=15)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "lxml")
            # Google results are in div.g or various containers
            results = soup.select("div.g")
            print(f"  Found {len(results)} result divs")
            for r in results[:5]:
                link = r.select_one("a")
                if link:
                    href = link.get("href", "")
                    text = link.get_text(strip=True)[:80]
                    print(f"     → {text}")
                    print(f"       {href[:100]}")
        elif resp.status_code == 429:
            print(f"  ⚠️ Rate limited (429) — Google blocks direct scraping")
        else:
            print(f"  ❌ HTTP {resp.status_code}")
    except Exception as e:
        print(f"  ❌ FAILED: {e}")


# ── 5. Google Search for Stock News (general) ────────────────────────────────

def test_google_news():
    sep("Google News Search (general)")

    query = f"{COMPANY} {TICKER} stock news India"
    url = f"https://www.google.com/search?q={query}&tbs=qdr:m&tbm=nws&num=10"
    print(f"[Google News] Searching: '{query}'...")
    try:
        resp = requests.get(url, headers={
            **HEADERS,
            "Accept": "text/html,application/xhtml+xml",
        }, timeout=15)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "lxml")
            articles = soup.select("div.SoaBEf") or soup.select("div.dbsr")
            print(f"  Found {len(articles)} news articles")
            # Also try regular link extraction
            links = soup.find_all("a", href=True)
            news_links = [l for l in links if "http" in l["href"] and "google" not in l["href"]][:5]
            for l in news_links:
                print(f"     → {l.get_text(strip=True)[:80]}")
        elif resp.status_code == 429:
            print(f"  ⚠️ Rate limited (429)")
        else:
            print(f"  ❌ HTTP {resp.status_code}")
    except Exception as e:
        print(f"  ❌ FAILED: {e}")


if __name__ == "__main__":
    print("🔍 StockSense AI — Live Scraper Diagnostics")
    print(f"   Ticker: {TICKER}  |  Company: {COMPANY}")
    print(f"   Time: {datetime.now()}")

    test_bse()
    test_nse()
    test_reddit()
    test_google_reddit()
    test_google_news()

    sep("SUMMARY")
    print("Review the output above to see which data sources are working.")
    print("Use these findings to fix and improve the scrapers.")

import logging
import requests
import json
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

# Headers to bypass basic bot protection on NSE
NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
}

class LiveDataService:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(NSE_HEADERS)
        self._initialized_cookies = False

    def _init_nse_cookies(self):
        """NSE requires cookies from the main page to authorize API requests."""
        if not self._initialized_cookies:
            try:
                self.session.get("https://www.nseindia.com", timeout=10)
                self._initialized_cookies = True
            except Exception as e:
                logger.error(f"Failed to fetch NSE cookies: {e}")

    def fetch_market_data(self, ticker: str) -> Dict[str, Any]:
        """Fetch current price, market cap, and basic multiples from yfinance."""
        yf_ticker = f"{ticker}.NS"
        stock = yf.Ticker(yf_ticker)
        
        try:
            info = stock.info
            current_price = info.get("currentPrice", info.get("regularMarketPrice"))
            market_cap = info.get("marketCap", 0) / 10000000  # Convert to Crores
            
            return {
                "current_price": current_price,
                "market_cap_cr": market_cap,
                "pe_ttm": info.get("trailingPE"),
                "pe_forward": info.get("forwardPE"),
                "pb_ratio": info.get("priceToBook"),
                "ev_ebitda": info.get("enterpriseToEbitda"),
                "price_to_sales": info.get("priceToSalesTrailing12Months"),
                "dividend_yield": info.get("dividendYield", 0) * 100 if info.get("dividendYield") else 0,
                "peg_ratio": info.get("pegRatio")
            }
        except Exception as e:
            logger.error(f"Error fetching market data for {ticker} via yfinance: {e}")
            return {
                "current_price": None,
                "market_cap_cr": None,
                "pe_ttm": None,
                "pe_forward": None,
                "pb_ratio": None,
                "ev_ebitda": None,
                "price_to_sales": None,
                "dividend_yield": None,
                "peg_ratio": None
            }

    def fetch_ohlcv(self, ticker: str, days: int = 90) -> pd.DataFrame:
        """Fetch daily OHLCV data for technical analysis."""
        yf_ticker = f"{ticker}.NS"
        stock = yf.Ticker(yf_ticker)
        
        try:
            # We fetch a bit more data to calculate 200-DMA properly (1 year ~ 252 trading days)
            fetch_period = f"{max(days, 252)}d"
            df = stock.history(period=fetch_period)
            return df
        except Exception as e:
            logger.error(f"Error fetching OHLCV for {ticker}: {e}")
            return pd.DataFrame()

    def fetch_corporate_events(self, ticker: str) -> Dict[str, List[Dict[str, Any]]]:
        """Fetch upcoming and recent corporate events."""
        # For a robust implementation, this would scrape from NSE / BSE directly or use yfinance calendar
        # yfinance provides limited event tracking, we'll try to get dividends, splits, earnings.
        
        yf_ticker = f"{ticker}.NS"
        stock = yf.Ticker(yf_ticker)
        
        upcoming = []
        recent = []
        
        try:
            calendar = stock.calendar
            if calendar:
                for k, v in calendar.items():
                    if 'Earnings Date' in str(k) and v:
                        upcoming.append({
                            "event_type": "Earnings Release",
                            "date": str(v[0]) if isinstance(v, list) and v else str(v),
                            "description": "Quarterly Earnings Announcement",
                            "investment_relevance": "High volatility event, potential guidance update."
                        })
        except Exception as e:
            logger.error(f"Error fetching calendar for {ticker}: {e}")

        # Currently we return mock or best-effort for others. True implementation would hit NSE corp-info API.
        return {
            "upcoming_events": upcoming,
            "recent_events": recent,
        }

    def fetch_shareholding_patterns(self, ticker: str) -> Dict[str, Any]:
        """Fetch promoter pledge and holding data."""
        # yfinance provides some institutional holder info but not specifically pledging.
        # This function stubs the capability for the Risk redflag agent.
        
        # Example API call to NSE if needed: self._init_nse_cookies() -> self.session.get("https://www.nseindia.com/api/quote-equity?symbol=RELIANCE&section=corp_info")
        return {
            "promoter_pledged_pct": None  # Set to None to signal missing data to Risk agent
        }

live_data_service = LiveDataService()

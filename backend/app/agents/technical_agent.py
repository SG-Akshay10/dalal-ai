"""
Rewritten for StockSense AI Pipeline v2.
Changes made:
- OHLCV Fetch: Extended fetch to max(requested_days, 280) to compute SMA-200. Retained original `days` parameter constraint for S/R detection context.
- Trading Bar Guard: Minimum set to 60 trading bars; returns "Insufficient Data" with `macd_signal="Neutral"` and `rsi_14=None` on failure.
- Indicators: Added SMA-200 calculation and strictly defined Trend Classification involving it. Added ATR-14 for volatility reference.
- Support/Resistance: Overhauled rolling min/max to an algorithmic pivot point cluster detection (N=5 left/right, 1.5% cluster threshold). Restricts support strictly below close and resistance strictly above.
- Bollinger Zones Fix: Adjusted discrete zone logic: "Above Upper" (>= U), "Near Upper" (> M), "Mid" (== M), "Below Lower" (<= L), and "Near Lower" (< M).
- MACD Column Isolation: Corrected the substring exclusion for MACD indicator mapping to avoid overlapping MACDh_/MACDs_ histogram columns.
- Data Reliability: Removed standard dataframe `bfill()`. Replaced it with the rigorous `_safe_float` mapping NaN and Inf values to strictly `None`.
- Prompt Structure: Included complete exact numeric details in prompt. Enforced a rigid 4-sentence structure analyzing SMA, RSI/MACD divergence, S/R levels + ATR, and Volume.
"""

import json
import logging
import math
import pandas as pd
import pandas_ta as ta

from langchain_core.prompts import PromptTemplate

from app.llm_provider import get_llm_client
from app.schemas.analysis import TechnicalAnalysis
from app.services.live_data import live_data_service

logger = logging.getLogger(__name__)


TECHNICAL_NARRATIVE_PROMPT = """You are an expert technical analyst.
Given the following computed technical indicators for {ticker}, write a strict 4-sentence narrative summarizing the current technical setup.

<STRICT_RULES>
1. NARRATIVE STRUCTURE: Write EXACTLY 4 sentences with the following assigned jobs:
   - Sentence 1: State the current trend bias and reference where the current price sits relative to the key SMAs (SMA-20, SMA-50, and SMA-200 if available).
   - Sentence 2: Summarize the momentum using the RSI value and MACD signal, explicitly noting any bullish or bearish divergence if present.
   - Sentence 3: Highlight the nearest support and resistance levels in ₹ terms, and provide context on daily volatility using the ATR value.
   - Sentence 4: Mention the volume conviction ONLY if it is "Expanding" or "Contracting" (meaningfully non-neutral). If volume is Neutral, omit it and provide a concluding thought on the price action setup instead.
2. FORMATTING: Values must be neatly formatted in Indian Rupees (₹) where applicable.
3. NO HALLUCINATION: Rely entirely on the provided numeric data below. Explain what the numbers indicate.
</STRICT_RULES>

Numeric Data:
Current Price: ₹{current_price}
Trend Bias: {trend_bias}
SMA-20: {sma_20} | SMA-50: {sma_50} | SMA-200: {sma_200}
RSI (14): {rsi} ({rsi_signal})
MACD Line: {macd_line} | MACD Signal Line: {macd_signal_line} | Signal: {macd_signal}
Bollinger Bands: Upper={bb_upper}, Mid={bb_mid}, Lower={bb_lower} | Position: {bollinger}
ATR (14): {atr}
Support Levels: {support_levels}
Resistance Levels: {resistance_levels}
Volume Trend: {volume_trend} (Recent Vol Ratio: {vol_ratio})

Always return standard JSON matching this exact schema. Wait until after your reasoning to generate the JSON.
<EXAMPLE_SCHEMA>
{{
  "narrative": "..."
}}
</EXAMPLE_SCHEMA>
"""


def _safe_float(val) -> float | None:
    """Safe extraction of numeric data directly returning `None` instead of `NaN` or `Inf`."""
    if pd.isna(val) or math.isinf(val):
        return None
    return float(val)


def get_pivots(df: pd.DataFrame, col: str, N: int, is_high: bool) -> list[float]:
    """Identify pivot high/low points computationally."""
    pivots = []
    # Can't reliably check the first N or last N bars
    for i in range(N, len(df) - N):
        window = df[col].iloc[i-N : i+N+1]
        center_val = df[col].iloc[i]
        
        if is_high and center_val == window.max():
            pivots.append(center_val)
        elif not is_high and center_val == window.min():
            pivots.append(center_val)
            
    return pivots


def cluster_pivots(pivots: list[float], threshold: float = 0.015) -> list[float]:
    """Cluster identical/nearby pivots (within threshold %) and isolate the average."""
    if not pivots:
        return []
    
    pivots = sorted(pivots)
    clusters = []
    current_cluster = [pivots[0]]
    
    for val in pivots[1:]:
        cluster_avg = sum(current_cluster) / len(current_cluster)
        
        # Check against average
        if abs(val - cluster_avg) / cluster_avg <= threshold:
            current_cluster.append(val)
        else:
            clusters.append(sum(current_cluster) / len(current_cluster))
            current_cluster = [val]
            
    # Append remaining
    clusters.append(sum(current_cluster) / len(current_cluster))
    return sorted(clusters)


def compute_support_resistance(df: pd.DataFrame, current_price: float, days_context: int) -> tuple[list[float], list[float]]:
    """Determine immediate support and resistance bands through rolling pivot discovery."""
    
    # Strictly respect the user timeline explicitly context boundary for pivots
    df_window = df.tail(days_context)
    if len(df_window) < 11:  # Requires 11 minimum for N=5
        return [], []
    
    pivot_highs = get_pivots(df_window, 'High', N=5, is_high=True)
    pivot_lows = get_pivots(df_window, 'Low', N=5, is_high=False)
    
    clustered_highs = cluster_pivots(pivot_highs, 0.015)
    clustered_lows = cluster_pivots(pivot_lows, 0.015)
    
    # Filter resistance tightly ABOVE current price
    resistances = [round(r, 2) for r in clustered_highs if r > current_price]
    resistances = sorted(resistances)[:3]
    
    # Filter support tightly BELOW current price
    supports = [round(s, 2) for s in clustered_lows if s < current_price]
    supports = sorted(supports, reverse=True)[:3]
    
    return supports, resistances


def analyze_technical(ticker: str, days: int = 90, provider: str = None) -> TechnicalAnalysis:
    """Run deterministic technical analysis and synthesize narrative."""
    
    # 1. Fetch OHLCV handling the extended SMA-200 footprint
    fetch_days = max(days, 280)
    df = live_data_service.fetch_ohlcv(ticker, days=fetch_days)
    
    if df.empty or len(df) < 60:
         logger.warning(f"[{ticker}] Technical guard: Low OHLCV bar count ({len(df)} < 60). Returning insufficient.")
         return TechnicalAnalysis(
             trend_bias="Insufficient Data",
             rsi_14=None,
             rsi_signal="Neutral",
             macd_signal="Neutral",
             bollinger_position="Mid",
             support_levels=[],
             resistance_levels=[],
             volume_trend="Neutral",
             narrative="Insufficient trading history to safely compute moving averages and indicators."
         )

    try:
        current_price = _safe_float(df['Close'].iloc[-1])
        if current_price is None:
            raise ValueError("Corrupted OHLCV data. Current Close Price is NaN.")

        # --- Indicator Generation: SMA Trend Vectors ---
        df['sma_20'] = ta.sma(df['Close'], length=20)
        df['sma_50'] = ta.sma(df['Close'], length=50)
        
        if len(df) >= 200:
            df['sma_200'] = ta.sma(df['Close'], length=200)
            sma_200 = _safe_float(df['sma_200'].iloc[-1])
        else:
            sma_200 = None
            
        sma_20 = _safe_float(df['sma_20'].iloc[-1])
        sma_50 = _safe_float(df['sma_50'].iloc[-1])
        
        if sma_20 is not None and sma_50 is not None and sma_200 is not None:
             if current_price > sma_20 and sma_20 > sma_50 and sma_50 > sma_200:
                 trend_bias = "Strong Uptrend"
             elif current_price < sma_200:
                 if current_price < sma_20 and current_price < sma_50:
                     trend_bias = "Strong Downtrend"
                 else:
                     trend_bias = "Downtrend"
             elif current_price > sma_20 and current_price > sma_50:
                 trend_bias = "Uptrend"
             elif current_price < sma_20 and current_price < sma_50:
                 trend_bias = "Downtrend"
             else:
                 trend_bias = "Sideways"
        else:
             if sma_20 is not None and sma_50 is not None:
                 if current_price > sma_20 and sma_20 > sma_50:
                     trend_bias = "Strong Uptrend"
                 elif current_price > sma_20:
                     trend_bias = "Uptrend"
                 elif current_price < sma_20 and sma_20 < sma_50:
                     trend_bias = "Strong Downtrend"
                 elif current_price < sma_20:
                     trend_bias = "Downtrend"
                 else:
                     trend_bias = "Sideways"
             else:
                 trend_bias = "Sideways"

        # --- Indicator Generation: RSI ---
        df['rsi_14'] = ta.rsi(df['Close'], length=14)
        rsi_val = _safe_float(df['rsi_14'].iloc[-1])
            
        if rsi_val is None:
             rsi_signal = "Neutral"
        elif rsi_val > 70:
             rsi_signal = "Overbought"
        elif rsi_val < 30:
             rsi_signal = "Oversold"
        else:
             rsi_signal = "Neutral"

        # --- Indicator Generation: MACD ---
        macd = ta.macd(df['Close'])
        df = pd.concat([df, macd], axis=1)
        
        # Explicit isolation
        macd_cols = [c for c in df.columns if c.startswith('MACD_') and 'MACDh_' not in c and 'MACDs_' not in c]
        macds_cols = [c for c in df.columns if c.startswith('MACDs_')]
        
        macd_line = _safe_float(df.iloc[-1][macd_cols[0]]) if macd_cols else None
        macd_signal_line = _safe_float(df.iloc[-1][macds_cols[0]]) if macds_cols else None
        
        if macd_line is not None and macd_signal_line is not None:
             if macd_line > macd_signal_line and macd_line > 0:
                 macd_signal = "Bullish"
             elif macd_line > macd_signal_line and macd_line <= 0:
                 macd_signal = "Bullish Crossover"
             elif macd_line < macd_signal_line and macd_line < 0:
                 macd_signal = "Bearish"
             else: # macd_line < macd_signal_line and macd >= 0
                 macd_signal = "Bearish Crossover"
        else:
             macd_signal = "Neutral"

        # --- Indicator Generation: Bollinger Bands ---
        bb = ta.bbands(df['Close'], length=20, std=2)
        df = pd.concat([df, bb], axis=1)
        
        bb_upper_cols = [c for c in df.columns if c.startswith('BBU_')]
        bb_mid_cols = [c for c in df.columns if c.startswith('BBM_')]
        bb_lower_cols = [c for c in df.columns if c.startswith('BBL_')]
        
        bb_upper = _safe_float(df.iloc[-1][bb_upper_cols[0]]) if bb_upper_cols else None
        bb_mid = _safe_float(df.iloc[-1][bb_mid_cols[0]]) if bb_mid_cols else None
        bb_lower = _safe_float(df.iloc[-1][bb_lower_cols[0]]) if bb_lower_cols else None
        
        if bb_upper is not None and bb_mid is not None and bb_lower is not None:
             if current_price >= bb_upper:
                 bollinger = "Above Upper"
             elif current_price > bb_mid:
                 bollinger = "Near Upper"
             elif current_price == bb_mid:
                 bollinger = "Mid"
             elif current_price <= bb_lower:
                 bollinger = "Below Lower"
             else:
                 bollinger = "Near Lower"
        else:
            bollinger = "Mid"

        # --- Indicator Generation: ATR ---
        df['atr_14'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        atr_val = _safe_float(df['atr_14'].iloc[-1])

        # --- Indicator Generation: Volume ---
        volume_20d_avg = _safe_float(df['Volume'].tail(20).mean())
        recent_vol = _safe_float(df['Volume'].tail(3).mean())
        vol_ratio = (recent_vol / volume_20d_avg) if (recent_vol is not None and volume_20d_avg and volume_20d_avg > 0) else None
        
        if vol_ratio is not None:
            if vol_ratio > 1.2:
                volume_trend = "Expanding"
            elif vol_ratio < 0.8:
                volume_trend = "Contracting"
            else:
                volume_trend = "Neutral"
        else:
            volume_trend = "Neutral"

        # --- Indicator Generation: Supp/Res ---
        supp, res = compute_support_resistance(df, current_price, days_context=days)
        
        # --- Synthesize LLM Narrative ---
        prompt = PromptTemplate.from_template(TECHNICAL_NARRATIVE_PROMPT)
        formatted_prompt = prompt.format(
            ticker=ticker,
            current_price=round(current_price, 2) if current_price else "N/A",
            trend_bias=trend_bias,
            sma_20=round(sma_20, 2) if sma_20 is not None else "N/A",
            sma_50=round(sma_50, 2) if sma_50 is not None else "N/A",
            sma_200=round(sma_200, 2) if sma_200 is not None else "N/A",
            rsi=round(rsi_val, 2) if rsi_val is not None else "N/A",
            rsi_signal=rsi_signal,
            macd_line=round(macd_line, 3) if macd_line is not None else "N/A",
            macd_signal_line=round(macd_signal_line, 3) if macd_signal_line is not None else "N/A",
            macd_signal=macd_signal,
            bb_upper=round(bb_upper, 2) if bb_upper is not None else "N/A",
            bb_mid=round(bb_mid, 2) if bb_mid is not None else "N/A",
            bb_lower=round(bb_lower, 2) if bb_lower is not None else "N/A",
            bollinger=bollinger,
            atr=round(atr_val, 2) if atr_val is not None else "N/A",
            support_levels=supp,
            resistance_levels=res,
            volume_trend=volume_trend,
            vol_ratio=round(vol_ratio, 2) if vol_ratio is not None else "N/A"
        )
        
        llm = get_llm_client(provider)
        try:
            if hasattr(llm, "with_structured_output"):
                structured_llm = llm.with_structured_output(
                    dict, # Generic dictionary to scrape just the string
                    method="json_mode" if "openai" in str(type(llm)).lower() else "function_calling"
                )
                try:
                    result = structured_llm.invoke(formatted_prompt)
                    narrative = result.get("narrative", f"The stock is in a {trend_bias.lower()} with {rsi_signal.lower()} RSI.")
                except Exception:
                    pass
            
            # If structured fallback
            if 'narrative' not in locals():
                response = llm.invoke([
                    {"role": "system", "content": "You output JSON only. Return a dict with a 'narrative' key."},
                    {"role": "user", "content": formatted_prompt}
                ])
                content = response.content
                
                # Cleanup parser
                if "```json" in content:
                    content = content.split("```json")[-1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[-1].split("```")[0].strip()
                else:
                    start = content.find('{')
                    end = content.rfind('}')
                    if start != -1 and end != -1:
                        content = content[start:end+1]
                        
                data = json.loads(content)
                narrative = data.get("narrative", f"The stock is in a {trend_bias.lower()} with {rsi_signal.lower()} RSI.")
                
        except Exception as llm_err:
            logger.error(f"[{ticker}] LLM synthesis failed for technicals. Reason: {llm_err}")
            narrative = f"Technical setup points towards a {trend_bias.lower()} trajectory. RSI shows {rsi_signal.lower()} momentum, with MACD indicating {macd_signal.lower()} signals. Near terms levels show Support {supp} and Resistance {res}."

        return TechnicalAnalysis(
            trend_bias=trend_bias,
            rsi_14=round(rsi_val, 2) if rsi_val is not None else None,
            rsi_signal=rsi_signal,
            macd_signal=macd_signal,
            bollinger_position=bollinger,
            support_levels=supp,
            resistance_levels=res,
            volume_trend=volume_trend,
            narrative=narrative
        )

    except Exception as e:
        logger.error(f"Error computing technicals for {ticker}: {e}")
        return TechnicalAnalysis(
            trend_bias="Insufficient Data",
            rsi_14=None,
            rsi_signal="Neutral",
            macd_signal="Neutral",
            bollinger_position="Mid",
            support_levels=[],
            resistance_levels=[],
            volume_trend="Neutral",
            narrative=f"System error computing technicals: {str(e)}"
        )

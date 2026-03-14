import json
import logging
import pandas as pd
import pandas_ta as ta

from langchain_core.prompts import PromptTemplate

from app.llm_provider import get_llm_client
from app.schemas.analysis import TechnicalAnalysis
from app.services.live_data import live_data_service

logger = logging.getLogger(__name__)

TECHNICAL_NARRATIVE_PROMPT = """You are an expert technical analyst.
Given the following computed technical indicators for {ticker}, write a very concise (2-3 sentences) narrative summarizing the current technical setup.

RSI (14): {rsi} ({rsi_signal})
MACD: {macd_signal}
Bollinger Band Position: {bollinger}
Trend Bias: {trend_bias}
Volume: {volume_trend}

Always return standard JSON matching this exact schema:
{{
  "narrative": "..."
}}
"""

def compute_support_resistance(df: pd.DataFrame) -> tuple[list[float], list[float]]:
    """Basic programmatic support/resistance from recent rolling window."""
    if len(df) < 20:
        return [], []
    recent = df.tail(60)
    # Extremely simplified pivot highs/lows for demo purposes
    supp = recent['Low'].min()
    res = recent['High'].max()
    return [round(supp, 2)], [round(res, 2)]

def analyze_technical(ticker: str, days: int = 90, provider: str = None) -> TechnicalAnalysis:
    """Run deterministic technical analysis and synthesize narrative."""
    # Fetch data
    df = live_data_service.fetch_ohlcv(ticker, days=days)
    
    if df.empty or len(df) < 30:
        return TechnicalAnalysis(
            trend_bias="Insufficient Data",
            rsi_14=0.0,
            rsi_signal="Neutral",
            macd_signal="Bullish", # placeholder
            bollinger_position="Mid",
            support_levels=[],
            resistance_levels=[],
            volume_trend="Neutral",
            narrative="Insufficient trading history to compute technical indicators."
        )

    try:
        # Calculate indicators
        # SMA for Trend
        df['sma_20'] = ta.sma(df['Close'], length=20)
        df['sma_50'] = ta.sma(df['Close'], length=50)
        df.bfill(inplace=True)
        
        last_close = df['Close'].iloc[-1]
        sma_20 = df['sma_20'].iloc[-1]
        sma_50 = df['sma_50'].iloc[-1]
        
        if last_close > sma_20 > sma_50:
            trend_bias = "Strong Uptrend"
        elif last_close > sma_20:
            trend_bias = "Uptrend"
        elif last_close < sma_20 < sma_50:
            trend_bias = "Strong Downtrend"
        elif last_close < sma_20:
            trend_bias = "Downtrend"
        else:
            trend_bias = "Sideways"

        # RSI
        df['rsi_14'] = ta.rsi(df['Close'], length=14)
        rsi_val = df['rsi_14'].iloc[-1]
        if pd.isna(rsi_val):
            rsi_val = 50.0
            
        if rsi_val > 70:
            rsi_signal = "Overbought"
        elif rsi_val < 30:
            rsi_signal = "Oversold"
        else:
            rsi_signal = "Neutral"

        # MACD
        macd = ta.macd(df['Close'])
        df = pd.concat([df, macd], axis=1)
        # MACD_12_26_9, MACDh_12_26_9 (histogram), MACDs_12_26_9 (signal)
        macd_line = df.iloc[-1][df.columns[df.columns.str.startswith('MACD_')][0]]
        macd_signal_line = df.iloc[-1][df.columns[df.columns.str.startswith('MACDs_')][0]]
        
        if macd_line > macd_signal_line and macd_line > 0:
            macd_signal = "Bullish"
        elif macd_line > macd_signal_line and macd_line <= 0:
            macd_signal = "Bullish Crossover"
        elif macd_line < macd_signal_line and macd_line < 0:
            macd_signal = "Bearish"
        else:
            macd_signal = "Bearish Crossover"

        # Bollinger Bands
        bb = ta.bbands(df['Close'], length=20, std=2)
        df = pd.concat([df, bb], axis=1)
        upper = df.iloc[-1][df.columns[df.columns.str.startswith('BBU_')][0]]
        lower = df.iloc[-1][df.columns[df.columns.str.startswith('BBL_')][0]]
        mid = df.iloc[-1][df.columns[df.columns.str.startswith('BBM_')][0]]
        
        if last_close >= upper:
            bollinger = "Above Upper"
        elif last_close > mid:
            bollinger = "Near Upper"
        elif last_close <= lower:
            bollinger = "Below Lower"
        else:
            bollinger = "Near Lower"

        volume_20d_avg = df['Volume'].tail(20).mean()
        recent_vol = df['Volume'].tail(3).mean()
        if recent_vol > volume_20d_avg * 1.2:
            volume_trend = "Expanding"
        elif recent_vol < volume_20d_avg * 0.8:
            volume_trend = "Contracting"
        else:
            volume_trend = "Neutral"

        supp, res = compute_support_resistance(df)
        
        # LLM Synthesis for Narrative
        prompt = PromptTemplate.from_template(TECHNICAL_NARRATIVE_PROMPT)
        formatted_prompt = prompt.format(
            ticker=ticker,
            rsi=round(rsi_val, 2),
            rsi_signal=rsi_signal,
            macd_signal=macd_signal,
            bollinger=bollinger,
            trend_bias=trend_bias,
            volume_trend=volume_trend
        )
        
        llm = get_llm_client(provider)
        try:
            response = llm.invoke([
                {"role": "system", "content": "You output JSON only."},
                {"role": "user", "content": formatted_prompt}
            ])
            content = response.content.replace("```json", "").replace("```", "").strip()
            data = json.loads(content)
            narrative = data.get("narrative", f"The stock is in a {trend_bias.lower()} with {rsi_signal.lower()} RSI.")
        except Exception:
            narrative = f"The stock is in a {trend_bias.lower()} with {rsi_signal.lower()} RSI."

        return TechnicalAnalysis(
            trend_bias=trend_bias,
            rsi_14=round(rsi_val, 2),
            rsi_signal=rsi_signal,
            macd_signal=macd_signal,
            bollinger_position=bollinger,
            support_levels=supp,
            resistance_levels=res,
            volume_trend=volume_trend,
            narrative=narrative
        )

    except Exception as e:
        logger.error(f"Error in technical analysis for {ticker}: {e}")
        return TechnicalAnalysis(
            trend_bias="Insufficient Data",
            rsi_14=0.0,
            rsi_signal="Neutral",
            macd_signal="Bullish",
            bollinger_position="Mid",
            support_levels=[],
            resistance_levels=[],
            volume_trend="Neutral",
            narrative=f"Error computing technicals: {str(e)}"
        )

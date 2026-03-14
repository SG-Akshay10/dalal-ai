import os
import sys
from time import time

# Add backend dir to path so absolute imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from app.agents.orchestrator import run_pipeline

if __name__ == "__main__":
    ticker = "RELIANCE"
    print(f"Testing pipeline for {ticker}...")
    start_time = time()
    
    # We use Google GenAI by default as it's faster and configured
    report = run_pipeline(ticker=ticker, days=30, preferred_provider="gemini")
    
    end_time = time()
    elapsed = end_time - start_time
    
    print(f"\nTime taken: {elapsed:.2f} seconds\n")
    print("=" * 50)
    print("FINAL REPORT:")
    print("=" * 50)
    print(report)

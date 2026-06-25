import sys
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.run_daily_analysis import build_telegram_message

fib_res = {
    "direction": "bearish",
    "cluster_valid": 1,
    "cluster_strength": 0.8,
    "cluster_upper": 65000.0,
    "cluster_lower": 62000.0,
    "scenario_a_price": 63000.0,
    "scenario_b_price": 62000.0,
    "invalidation": 66000.0,
    "target_a": 63000.0,
    "target_b": 62000.0,
}

# Fully supply all quantiles
tft_res = {}
for h in [7, 14, 30, 60]:
    tft_res[f"q10_{h}d"] = 60000.0 - h * 10
    tft_res[f"q50_{h}d"] = 62500.0 + h * 5
    tft_res[f"q90_{h}d"] = 65000.0 + h * 20

# Test 1D message format
print("Testing 1D telegram message formatting:")
msg = build_telegram_message(
    timeframe="1D",
    fib_result=fib_res,
    tft_result=tft_res,
    adj_strength=0.8,
    risk_flag="High-impact event in 2 days",
    current_price=63500.0
)

print(msg)

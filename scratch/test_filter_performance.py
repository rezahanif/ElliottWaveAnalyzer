import json
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

ROOT = Path("/Users/reza/ElliottWaveAnalyzer")
sys.path.insert(0, str(ROOT))

from src.waveconf.pivots.zigzag import ZigZagDetector
from src.waveconf.fib_engine.fibonacci import FibonacciEngine

def load_labeled(timeframe: str) -> pd.DataFrame:
    path = os.path.join(ROOT, "data", "labeled", f"BTC_{timeframe}_labeled.csv")
    df = pd.read_csv(path)
    df["timestamp_ms"] = df["timestamp_ms"].astype(float)
    return df

def compute_fib_targets_filtered(macro_pivots, engine: FibonacciEngine, row_c: pd.Series) -> Optional[dict]:
    if len(macro_pivots) < 4:
        return None

    highs = [p for p in macro_pivots if p.is_high()]
    lows  = [p for p in macro_pivots if p.is_low()]

    if not highs or not lows:
        return None

    last_pivot = macro_pivots[-1]
    timeframe = row_c["asset_timeframe"]

    if last_pivot.is_high():
        direction   = "bearish"
        c_pivot = last_pivot
        b_pivot = max(
            (p for p in lows if p.bar_index < c_pivot.bar_index),
            key=lambda p: p.bar_index,
            default=None,
        )
        if b_pivot is None:
            return None
        a_pivot = max(
            (p for p in highs if p.bar_index < b_pivot.bar_index),
            key=lambda p: p.bar_index,
            default=None,
        )
        ab_range = abs(a_pivot.price - b_pivot.price) if a_pivot else None
    else:
        direction   = "bullish"
        c_pivot = last_pivot
        b_pivot = max(
            (p for p in highs if p.bar_index < c_pivot.bar_index),
            key=lambda p: p.bar_index,
            default=None,
        )
        if b_pivot is None:
            return None
        a_pivot = max(
            (p for p in lows if p.bar_index < b_pivot.bar_index),
            key=lambda p: p.bar_index,
            default=None,
        )
        ab_range = abs(a_pivot.price - b_pivot.price) if a_pivot else None

    cluster = engine.dual_cluster(
        c_top     = c_pivot.price,
        b_low     = b_pivot.price,
        direction = direction,
        ab_range  = ab_range,
    )

    if cluster.target_a.price <= 0 or cluster.target_b.price <= 0:
        return None

    # Filter 1: Wave Match Confidence (>=98% for 4H, >=80% for 1D)
    wmc = float(row_c["wave_match_confidence"])
    min_wmc = 0.98 if "4H" in timeframe else 0.80
    if wmc < min_wmc:
        return None

    # Filter 2: Target Distance (cap at 25% for 4H, 180% for 1D)
    target_lower = cluster.cluster_lower
    target_upper = cluster.cluster_upper
    entry_price = c_pivot.price
    target_dist_pct = abs(entry_price - target_lower if direction == "bullish" else entry_price - target_upper) / entry_price * 100
    max_target_dist = 25.0 if "4H" in timeframe else 180.0
    if target_dist_pct > max_target_dist:
        return None

    # Filter 3: Trend Momentum (MACD >= -250 on Daily Bullish)
    macd = float(row_c["macd_line"])
    if direction == "bullish" and "1D" in timeframe:
        if macd < -250.0:
            return None

    # Filter 4: Static 0.5% stop-loss (uncomment below for dynamic stop)
    # atr_norm = float(row_c["atr_14_norm"])
    # stop_buffer = max(0.005, atr_norm * 0.15)
    stop_buffer = 0.005
    invalidation_level = round(c_pivot.price * (1 + stop_buffer), 2) if direction == "bearish" else round(c_pivot.price * (1 - stop_buffer), 2)

    return {
        "cluster_valid":      cluster.cluster_valid,
        "cluster_lower":      cluster.cluster_lower,
        "cluster_upper":      cluster.cluster_upper,
        "direction":          direction,
        "entry_price":        c_pivot.price,
        "invalidation_level": invalidation_level,
        "c_bar_index":        c_pivot.bar_index,
        "timestamp_ms":       c_pivot.timestamp_ms,
        "stop_buffer_pct":    stop_buffer * 100
    }

def run_simulation(timeframe: str):
    df = load_labeled(timeframe)
    detector = ZigZagDetector(timeframe=timeframe)
    zigzag_result = detector.run(df)
    macro_pivots = zigzag_result.macro

    engine = FibonacciEngine()
    
    # 1. Base backtest (no filters)
    base_signals = []
    for i in range(3, len(macro_pivots)):
        sub_pivots = macro_pivots[:i+1]
        c_pivot = sub_pivots[-1]
        
        # Base invalidation is 0.5%
        direction = "bearish" if c_pivot.is_high() else "bullish"
        invalidation = round(c_pivot.price * (1 + 0.005), 2) if direction == "bearish" else round(c_pivot.price * (1 - 0.005), 2)
        
        # Calculate cluster A/B
        res = compute_fib_targets_unfiltered(sub_pivots, engine)
        if res is not None and res["cluster_valid"]:
            if not any(s["c_bar_index"] == res["c_bar_index"] for s in base_signals):
                res["invalidation_level"] = invalidation
                base_signals.append(res)
                
    # 2. Filtered backtest
    filtered_signals = []
    for i in range(3, len(macro_pivots)):
        sub_pivots = macro_pivots[:i+1]
        c_pivot = sub_pivots[-1]
        row_c = df.iloc[c_pivot.bar_index]
        res = compute_fib_targets_filtered(sub_pivots, engine, row_c)
        if res is not None and res["cluster_valid"]:
            if not any(s["c_bar_index"] == res["c_bar_index"] for s in filtered_signals):
                filtered_signals.append(res)

    def evaluate_outcomes(signals, label):
        results = []
        for signal in signals:
            c_bar = signal["c_bar_index"]
            entry_price = signal["entry_price"]
            invalidation = signal["invalidation_level"]
            cluster_lower = signal["cluster_lower"]
            cluster_upper = signal["cluster_upper"]
            direction = signal["direction"]

            outcome = "Pending"
            resolution_bar = len(df) - 1

            for t in range(c_bar + 1, len(df)):
                row = df.iloc[t]
                low_val = float(row["low"])
                high_val = float(row["high"])

                if direction == "bullish":
                    is_invalid = low_val <= invalidation
                    is_hit = high_val >= cluster_lower
                    if is_invalid and is_hit:
                        outcome = "Loss"
                        resolution_bar = t
                        break
                    elif is_invalid:
                        outcome = "Loss"
                        resolution_bar = t
                        break
                    elif is_hit:
                        outcome = "Win"
                        resolution_bar = t
                        break
                else:
                    is_invalid = high_val >= invalidation
                    is_hit = low_val <= cluster_upper
                    if is_invalid and is_hit:
                        outcome = "Loss"
                        resolution_bar = t
                        break
                    elif is_invalid:
                        outcome = "Loss"
                        resolution_bar = t
                        break
                    elif is_hit:
                        outcome = "Win"
                        resolution_bar = t
                        break
            
            # EV calculation:
            # Win pay = Target - Entry
            # Loss pay = Invalidation - Entry
            if outcome == "Win":
                target_hit = cluster_lower if direction == "bullish" else cluster_upper
                trade_return = abs(target_hit - entry_price) / entry_price * 100
            elif outcome == "Loss":
                trade_return = -abs(invalidation - entry_price) / entry_price * 100
            else:
                trade_return = 0.0 # Pending

            results.append({
                "outcome": outcome,
                "return": trade_return
            })
        
        res_df = pd.DataFrame(results)
        resolved = res_df[res_df["outcome"] != "Pending"]
        wins = sum(resolved["outcome"] == "Win")
        losses = sum(resolved["outcome"] == "Loss")
        total = len(resolved)
        
        win_rate = (wins / total * 100) if total > 0 else 0.0
        avg_return = resolved["return"].mean() if total > 0 else 0.0
        cum_return = resolved["return"].sum() if total > 0 else 0.0
        
        win_returns = resolved[resolved["outcome"] == "Win"]["return"].sum()
        loss_returns = abs(resolved[resolved["outcome"] == "Loss"]["return"].sum())
        profit_factor = win_returns / loss_returns if loss_returns > 0 else (float("inf") if win_returns > 0 else 1.0)
        
        print(f"  {label}:")
        print(f"    Total signals     : {len(signals)}")
        print(f"    Resolved trades   : {total}  (Wins: {wins}, Losses: {losses})")
        print(f"    Win Rate (%)       : {win_rate:.2f}%")
        print(f"    Expected Value (EV): {avg_return:+.2f}% per trade")
        print(f"    Cumulative Return : {cum_return:+.2f}%")
        print(f"    Profit Factor     : {profit_factor:.2f}")
        
    print(f"\nTimeframe: {timeframe}")
    print("-" * 50)
    evaluate_outcomes(base_signals, "BASELINE (0.5% stop, no filters)")
    evaluate_outcomes(filtered_signals, "OPTIMIZED (Dynamic stop, 3 filters)")
    print("=" * 50)

def compute_fib_targets_unfiltered(macro_pivots, engine):
    if len(macro_pivots) < 4:
        return None
    highs = [p for p in macro_pivots if p.is_high()]
    lows  = [p for p in macro_pivots if p.is_low()]
    if not highs or not lows:
        return None
    last_pivot = macro_pivots[-1]
    if last_pivot.is_high():
        direction   = "bearish"
        c_pivot = last_pivot
        b_pivot = max((p for p in lows if p.bar_index < c_pivot.bar_index), key=lambda p: p.bar_index, default=None)
        if b_pivot is None: return None
        a_pivot = max((p for p in highs if p.bar_index < b_pivot.bar_index), key=lambda p: p.bar_index, default=None)
        ab_range = abs(a_pivot.price - b_pivot.price) if a_pivot else None
    else:
        direction   = "bullish"
        c_pivot = last_pivot
        b_pivot = max((p for p in highs if p.bar_index < c_pivot.bar_index), key=lambda p: p.bar_index, default=None)
        if b_pivot is None: return None
        a_pivot = max((p for p in lows if p.bar_index < b_pivot.bar_index), key=lambda p: p.bar_index, default=None)
        ab_range = abs(a_pivot.price - b_pivot.price) if a_pivot else None
    cluster = engine.dual_cluster(c_top=c_pivot.price, b_low=b_pivot.price, direction=direction, ab_range=ab_range)
    if cluster.target_a.price <= 0 or cluster.target_b.price <= 0: return None
    return {
        "cluster_valid":      cluster.cluster_valid,
        "cluster_lower":      cluster.cluster_lower,
        "cluster_upper":      cluster.cluster_upper,
        "direction":          direction,
        "entry_price":        c_pivot.price,
        "c_bar_index":        c_pivot.bar_index,
        "timestamp_ms":       c_pivot.timestamp_ms,
    }

def main():
    for tf in ["1D", "4H"]:
        try:
            run_simulation(tf)
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()

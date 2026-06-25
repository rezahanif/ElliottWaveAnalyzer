import json
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from scipy.stats import ttest_ind

ROOT = Path("/Users/reza/ElliottWaveAnalyzer")
sys.path.insert(0, str(ROOT))

from src.waveconf.pivots.zigzag import ZigZagDetector
from src.waveconf.fib_engine.fibonacci import FibonacciEngine

def load_layers(timeframe: str) -> pd.DataFrame:
    path = os.path.join(ROOT, "data", "pivots", f"BTC_{timeframe}_with_layers.json")
    with open(path, "r") as f:
        payload = json.load(f)
    df = pd.DataFrame(payload["data"], columns=payload["columns"])
    return df

def compute_fib_targets(macro_pivots, engine: FibonacciEngine) -> Optional[dict]:
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

    invalidation_level = round(c_pivot.price * (1 + engine.invalidation_buffer), 2) if direction == "bearish" else round(c_pivot.price * (1 - engine.invalidation_buffer), 2)

    return {
        "cluster_valid":      cluster.cluster_valid,
        "cluster_lower":      cluster.cluster_lower,
        "cluster_upper":      cluster.cluster_upper,
        "direction":          direction,
        "entry_price":        c_pivot.price,
        "invalidation_level": invalidation_level,
        "c_bar_index":        c_pivot.bar_index,
        "timestamp_ms":       c_pivot.timestamp_ms,
        "pivot_A": a_pivot.price if a_pivot else None,
        "pivot_B": b_pivot.price,
        "pivot_C": c_pivot.price
    }

def analyze_losses_for_tf(timeframe: str):
    df = load_layers(timeframe)
    detector = ZigZagDetector(timeframe=timeframe)
    zigzag_result = detector.run(df)
    macro_pivots = zigzag_result.macro

    engine = FibonacciEngine()
    signals = []

    # Get signals using 0.5% baseline
    for i in range(3, len(macro_pivots)):
        sub_pivots = macro_pivots[:i+1]
        res = compute_fib_targets(sub_pivots, engine)
        if res is not None and res["cluster_valid"]:
            if not any(s["c_bar_index"] == res["c_bar_index"] for s in signals):
                signals.append(res)

    trades = []

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

            elif direction == "bearish":
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

        if outcome == "Pending":
            continue # Only analyze resolved trades

        # Extract features at the bar where the pivot C occurred (c_bar)
        # Note: Pivot C is confirmed at some later bar, but the entry price is at c_bar.
        # Let's extract features at the confirmation bar as well as entry bar.
        # Let's check what columns exist in df
        features = {}
        c_row = df.iloc[c_bar]
        
        # Check standard indicators
        for col in ["rsi_14", "atr_14_norm", "bb_width", "mercury_retrograde", "lunar_phase_sin", "days_to_fomc", "days_to_nfp"]:
            if col in df.columns:
                features[col] = float(c_row[col])

        # Let's add wave sizes as percentage of entry price
        ab_size = abs(signal["pivot_A"] - signal["pivot_B"]) / entry_price * 100 if signal["pivot_A"] else 0.0
        bc_size = abs(signal["pivot_B"] - signal["pivot_C"]) / entry_price * 100
        
        features["ab_size_pct"] = ab_size
        features["bc_size_pct"] = bc_size
        features["stop_distance_pct"] = abs(entry_price - invalidation) / entry_price * 100
        features["target_distance_pct"] = abs(entry_price - cluster_lower if direction == "bullish" else entry_price - cluster_upper) / entry_price * 100

        trades.append({
            "outcome": outcome,
            "direction": direction,
            **features
        })

    trade_df = pd.DataFrame(trades)
    if len(trade_df) == 0:
        print(f"No resolved trades to analyze for {timeframe}")
        return

    print(f"\n============================================================")
    print(f"  LOSS ANALYSIS FOR {timeframe} | Total resolved: {len(trade_df)}")
    print(f"============================================================")
    
    wins = trade_df[trade_df["outcome"] == "Win"]
    losses = trade_df[trade_df["outcome"] == "Loss"]
    
    print(f"Wins: {len(wins)} | Losses: {len(losses)}")
    print("-" * 60)
    
    # Compare averages of all numerical columns
    numeric_cols = [c for c in trade_df.columns if c not in ["outcome", "direction"]]
    
    analysis_results = []
    
    for col in numeric_cols:
        win_mean = wins[col].mean() if len(wins) > 0 else np.nan
        loss_mean = losses[col].mean() if len(losses) > 0 else np.nan
        win_std = wins[col].std() if len(wins) > 1 else 0.0
        loss_std = losses[col].std() if len(losses) > 1 else 0.0
        
        # Perform t-test if both groups have enough data
        p_val = np.nan
        if len(wins) > 1 and len(losses) > 1:
            stat, p_val = ttest_ind(wins[col].dropna(), losses[col].dropna(), equal_var=False)
            
        analysis_results.append({
            "Feature": col,
            "Win Mean": round(win_mean, 4) if not np.isnan(win_mean) else "N/A",
            "Loss Mean": round(loss_mean, 4) if not np.isnan(loss_mean) else "N/A",
            "Win Std": round(win_std, 4),
            "Loss Std": round(loss_std, 4),
            "P-value": round(p_val, 4) if not np.isnan(p_val) else "N/A"
        })
        
    analysis_df = pd.DataFrame(analysis_results)
    # Sort by p-value to find most statistically significant differences first
    analysis_df = analysis_df.sort_values(by="P-value", key=lambda x: x.fillna(1.0))
    print(analysis_df.to_string(index=False))
    print("-" * 60)

    # Let's check direction bias
    direction_counts = trade_df.groupby(["direction", "outcome"]).size().unstack(fill_value=0)
    print("\nDirection Bias:")
    print(direction_counts)
    print("=" * 60)

def main():
    for tf in ["1D", "4H"]:
        try:
            analyze_losses_for_tf(tf)
        except Exception as e:
            print(f"Error analyzing {tf}: {e}")

if __name__ == "__main__":
    main()

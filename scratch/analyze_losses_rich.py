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

def load_labeled_csv(timeframe: str) -> pd.DataFrame:
    path = os.path.join(ROOT, "data", "labeled", f"BTC_{timeframe}_labeled.csv")
    df = pd.read_csv(path)
    # Ensure timestamp_ms is float and standard
    df["timestamp_ms"] = df["timestamp_ms"].astype(float)
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

def analyze_losses_rich(timeframe: str):
    df_layers = load_layers(timeframe)
    df_labeled = load_labeled_csv(timeframe)
    
    # ZigZag Detector runs on layers
    detector = ZigZagDetector(timeframe=timeframe)
    zigzag_result = detector.run(df_layers)
    macro_pivots = zigzag_result.macro

    engine = FibonacciEngine()
    signals = []

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
        timestamp_ms = signal["timestamp_ms"]

        outcome = "Pending"
        resolution_bar = len(df_layers) - 1

        for t in range(c_bar + 1, len(df_layers)):
            row = df_layers.iloc[t]
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
            continue

        # Look up timestamp in df_labeled to get rich indicators
        # Make timestamp_ms standard float comparison
        # Let's match by nearest timestamp or exact
        match_idx = (df_labeled["timestamp_ms"] - float(timestamp_ms)).abs().idxmin()
        labeled_row = df_labeled.loc[match_idx]
        
        # Extract features
        features = {}
        ignore_cols = ["timestamp_ms", "date", "open", "high", "low", "close", "volume", 
                       "open_norm", "high_norm", "low_norm", "close_norm", "volume_norm", 
                       "asset_timeframe", "bar_index", "time_idx"]
                       
        for col in df_labeled.columns:
            if col not in ignore_cols:
                try:
                    val = float(labeled_row[col])
                    if not np.isnan(val):
                        features[col] = val
                except ValueError:
                    pass

        # Wave details
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
    print(f"  RICH LOSS ANALYSIS FOR {timeframe} | Total resolved: {len(trade_df)}")
    print(f"============================================================")
    
    wins = trade_df[trade_df["outcome"] == "Win"]
    losses = trade_df[trade_df["outcome"] == "Loss"]
    
    print(f"Wins: {len(wins)} | Losses: {len(losses)}")
    print("-" * 60)
    
    numeric_cols = [c for c in trade_df.columns if c not in ["outcome", "direction"]]
    
    analysis_results = []
    
    for col in numeric_cols:
        win_mean = wins[col].mean() if len(wins) > 0 else np.nan
        loss_mean = losses[col].mean() if len(losses) > 0 else np.nan
        win_std = wins[col].std() if len(wins) > 1 else 0.0
        loss_std = losses[col].std() if len(losses) > 1 else 0.0
        
        p_val = np.nan
        if len(wins) > 1 and len(losses) > 1:
            # Drop nan values
            w_vals = wins[col].dropna()
            l_vals = losses[col].dropna()
            if len(w_vals) > 1 and len(l_vals) > 1 and w_vals.std() > 0 and l_vals.std() > 0:
                stat, p_val = ttest_ind(w_vals, l_vals, equal_var=False)
            
        analysis_results.append({
            "Feature": col,
            "Win Mean": win_mean,
            "Loss Mean": loss_mean,
            "Win Std": win_std,
            "Loss Std": loss_std,
            "P-value": p_val
        })
        
    analysis_df = pd.DataFrame(analysis_results)
    # Filter out columns where P-value is NaN
    analysis_df = analysis_df.dropna(subset=["P-value"])
    # Sort by p-value
    analysis_df = analysis_df.sort_values(by="P-value")
    
    # Format for display
    display_df = analysis_df.copy()
    for col in ["Win Mean", "Loss Mean", "Win Std", "Loss Std", "P-value"]:
        display_df[col] = display_df[col].apply(lambda x: round(x, 4) if pd.notnull(x) else "N/A")

    # Print the top 25 features
    print(display_df.head(25).to_string(index=False))
    print("-" * 60)

def main():
    for tf in ["1D", "4H"]:
        try:
            analyze_losses_rich(tf)
        except Exception as e:
            print(f"Error analyzing {tf}: {e}")

if __name__ == "__main__":
    main()

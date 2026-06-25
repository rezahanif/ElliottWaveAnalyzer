import json
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
ROOT = Path("/Users/reza/ElliottWaveAnalyzer")
sys.path.insert(0, str(ROOT))

from scripts.visualize_backtest import get_trades_for_timeframe

def compute_roi(trade: Dict) -> float:
    outcome = trade["outcome"]
    direction = trade["direction"].lower()
    entry_price = trade["entry_price"]
    invalidation = trade["invalidation_level"]
    target_lower = trade["target_lower"]
    target_upper = trade["target_upper"]

    if outcome == "Win":
        if direction == "bullish":
            return (target_lower - entry_price) / entry_price * 100
        else:
            return (entry_price - target_upper) / entry_price * 100
    elif outcome == "Loss":
        if direction == "bullish":
            return (invalidation - entry_price) / entry_price * 100
        else:
            return (entry_price - invalidation) / entry_price * 100
    return 0.0

def analyze_trades(trades: List[Dict], start_year_filter: Optional[int] = None) -> Dict:
    filtered_trades = []
    for t in trades:
        # Date format in start_date is '%Y-%m-%d %H:%M'
        date_str = t["start_date"]
        year = int(date_str.split("-")[0])
        if start_year_filter is not None and year < start_year_filter:
            continue
        filtered_trades.append(t)

    # Calculate metrics
    resolved_trades = [t for t in filtered_trades if t["outcome"] in ("Win", "Loss")]
    wins = [t for t in resolved_trades if t["outcome"] == "Win"]
    losses = [t for t in resolved_trades if t["outcome"] == "Loss"]
    
    total_signals = len(filtered_trades)
    resolved_count = len(resolved_trades)
    win_count = len(wins)
    loss_count = len(losses)
    
    win_rate = (win_count / resolved_count * 100) if resolved_count > 0 else 0.0
    
    rois = [compute_roi(t) for t in resolved_trades]
    avg_roi = np.mean(rois) if rois else 0.0
    sum_roi = np.sum(rois) if rois else 0.0
    
    # Calculate compounded return assuming 10% allocation per trade
    equity = 1.0
    for roi in rois:
        trade_return = (roi / 100.0) * 0.10
        equity *= (1.0 + trade_return)
    compounded_return = (equity - 1.0) * 100
    
    # Average MFE/MAE are calculated over resolved trades
    mfes = [t["mfe"] for t in resolved_trades]
    maes = [t["mae"] for t in resolved_trades]
    avg_mfe = np.mean(mfes) if mfes else 0.0
    avg_mae = np.mean(maes) if maes else 0.0
    
    # Calculate profit factor
    gross_profits = sum(r for r in rois if r > 0)
    gross_losses = abs(sum(r for r in rois if r < 0))
    profit_factor = (gross_profits / gross_losses) if gross_losses > 0 else (float('inf') if gross_profits > 0 else 1.0)

    # Tier breakdown
    sel_trades = [t for t in resolved_trades if t.get("signal_tier") == "selective"]
    agg_trades = [t for t in resolved_trades if t.get("signal_tier") == "aggressive"]
    
    sel_wins = sum(1 for t in sel_trades if t["outcome"] == "Win")
    agg_wins = sum(1 for t in agg_trades if t["outcome"] == "Win")
    
    sel_win_rate = (sel_wins / len(sel_trades) * 100) if sel_trades else 0.0
    agg_win_rate = (agg_wins / len(agg_trades) * 100) if agg_trades else 0.0

    # Max/Min MFE and MAE
    max_mfe = np.max(mfes) if mfes else 0.0
    min_mfe = np.min(mfes) if mfes else 0.0
    max_mae = np.max(maes) if maes else 0.0
    min_mae = np.min(maes) if maes else 0.0

    return {
        "total_signals": total_signals,
        "resolved_signals": resolved_count,
        "wins": win_count,
        "losses": loss_count,
        "win_rate": win_rate,
        "avg_roi": avg_roi,
        "sum_roi": sum_roi,
        "compounded_return": compounded_return,
        "avg_mfe": avg_mfe,
        "max_mfe": max_mfe,
        "min_mfe": min_mfe,
        "avg_mae": avg_mae,
        "max_mae": max_mae,
        "min_mae": min_mae,
        "profit_factor": profit_factor,
        "selective_count": len(sel_trades),
        "selective_win_rate": sel_win_rate,
        "aggressive_count": len(agg_trades),
        "aggressive_win_rate": agg_win_rate,
    }

def main():
    timeframes = ["1D", "4H"]
    configs = ["baseline", "hybrid"]
    versions = ["v1_buggy", "v2_linear", "v3_gated", "v4_log", "v5_relaxed"]
    periods = [("All-Time", None), ("2022-Present", 2022)]

    all_results = {}

    for tf in timeframes:
        for config in configs:
            for ver in versions:
                # Load trades from file/computation
                try:
                    trades = get_trades_for_timeframe(tf, config, ver)
                    for period_name, year_filter in periods:
                        key = (tf, config, ver, period_name)
                        stats = analyze_trades(trades, start_year_filter=year_filter)
                        all_results[key] = stats
                except Exception as e:
                    print(f"Error processing {tf} {config} {ver}: {e}")

    # Output tables in Markdown format
    print("# BACKTEST COMPARISON REPORT: v1, v2, v3, v4")
    for tf in timeframes:
        for config in configs:
            print(f"\n## Timeframe: {tf} | Configuration: {config.upper()}")
            for period_name, _ in periods:
                print(f"\n### Period: {period_name}")
                print("| Version | Total / Resolved | Win Rate | Avg ROI | Sum ROI | Comp. Return (10%) | Avg MFE | Max MFE | Min MFE | Avg MAE | Max MAE | Min MAE | Profit Factor | Sel (WR%) | Agg (WR%) |")
                print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
                for ver in versions:
                    key = (tf, config, ver, period_name)
                    if key not in all_results:
                        continue
                    r = all_results[key]
                    pf_str = f"{r['profit_factor']:.2f}" if r['profit_factor'] != float('inf') else "∞"
                    print(f"| **{ver}** | {r['total_signals']} / {r['resolved_signals']} | {r['win_rate']:.1f}% | {r['avg_roi']:+.2f}% | {r['sum_roi']:+.1f}% | {r['compounded_return']:+.1f}% | {r['avg_mfe']:.1f}% | {r['max_mfe']:.1f}% | {r['min_mfe']:.1f}% | {r['avg_mae']:.1f}% | {r['max_mae']:.1f}% | {r['min_mae']:.1f}% | {pf_str} | {r['selective_count']} ({r['selective_win_rate']:.1f}%) | {r['aggressive_count']} ({r['aggressive_win_rate']:.1f}%) |")

if __name__ == "__main__":
    main()

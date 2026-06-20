import pytest
import pandas as pd
from src.waveconf.pivots.zigzag import ZigZagDetector
from src.waveconf.pivots.pivot_schema import SwingType

def test_synthetic_zigzag():
    """
    Synthetic test: 10 bars up, 10 bars down.
    Tests if the ZigZagDetector correctly identifies the peak.
    """
    data = []
    
    # 10 bars up (bar_index 0 to 9)
    # High goes from 101.0 to 110.0
    for i in range(10):
        data.append({
            "timestamp_ms": i * 1000,
            "open": 100.0 + i,
            "high": 101.0 + i,
            "low": 99.0 + i,
            "close": 100.5 + i,
            "volume": 100,
            "wall_street_threshold_pct": 5.0, # 5% macro threshold
            "behavioral_threshold_pct": 2.0   # 2% micro threshold
        })
        
    # 10 bars down (bar_index 10 to 19)
    # High goes from 109.0 down to 100.0
    for i in range(10):
        data.append({
            "timestamp_ms": (10 + i) * 1000,
            "open": 110.0 - (i + 2),
            "high": 111.0 - (i + 2),
            "low": 109.0 - (i + 2),
            "close": 109.5 - (i + 2),
            "volume": 100,
            "wall_street_threshold_pct": 5.0,
            "behavioral_threshold_pct": 2.0
        })
        
    df = pd.DataFrame(data)
    
    # Run detector
    detector = ZigZagDetector(timeframe='1D', min_bars_between_pivots=1)
    result = detector.run(df)
    
    # Extract macro pivots
    macro_pivots = result.macro
    
    assert len(macro_pivots) > 0, "Should detect at least one macro pivot"
    
    # Find the high pivot
    high_pivots = [p for p in macro_pivots if p.swing_type == SwingType.HIGH]
    assert len(high_pivots) == 1, "Should detect exactly one macro HIGH pivot"
    
    peak = high_pivots[0]
    assert peak.bar_index == 9, f"Expected peak at bar 9, got {peak.bar_index}"
    assert peak.price == 110.0, f"Expected peak price 110.0, got {peak.price}"
    
    # Let's also check micro pivots
    micro_pivots = result.micro
    micro_high_pivots = [p for p in micro_pivots if p.swing_type == SwingType.HIGH]
    assert len(micro_high_pivots) >= 1
    assert micro_high_pivots[0].bar_index == 9

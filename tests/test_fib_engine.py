import pytest
from src.waveconf.fib_engine.fibonacci import FibonacciEngine, FibTarget, ClusterResult
from src.waveconf.fib_engine.invalidation import is_invalidated

def test_fibonacci_engine_init():
    engine = FibonacciEngine()
    assert engine.cluster_threshold_pct == 2.0
    assert engine.invalidation_buffer == 0.005
    assert engine.wave_ratio_tolerance == 0.05
    assert engine.tool_a_ratio == 2.618
    assert engine.tool_b_ratio == 1.618
    assert engine.completion_rates["ascending_broadening_wedge"] == 0.70
    assert engine.completion_rates["rising_wedge"] == 0.85

def test_generic_retracement_and_extension():
    engine = FibonacciEngine()
    # Bullish retracement (rising wave from 100 to 200)
    res = engine.retracement(100.0, 200.0, 0.618)
    assert res.price == 138.2
    assert res.direction == "bearish"  # correction direction of a bullish wave is bearish

    # Bearish retracement (falling wave from 200 to 100)
    res = engine.retracement(200.0, 100.0, 0.618)
    assert res.price == 161.8
    assert res.direction == "bullish"

    # Extension
    res = engine.extension(200.0, 100.0, 1.618, "bearish", "C_top")
    assert res.price == 38.2  # 200 - 161.8 = 38.2
    assert res.direction == "bearish"

    res = engine.extension(100.0, 100.0, 1.618, "bullish", "B_low")
    assert res.price == 261.8  # 100 + 161.8 = 261.8
    assert res.direction == "bullish"

def test_impulse_targets():
    engine = FibonacciEngine()
    # Bullish impulse W1: 100 -> 200
    res = engine.impulse_targets(100.0, 200.0, bullish=True)
    # W2 retrace zone expects 38.2% to 78.6% retrace of 100 (range)
    # 200 - 78.6 = 121.4 (lo) and 200 - 38.2 = 161.8 (hi)
    assert res.w2_retrace_zone == (121.4, 161.8)
    assert res.w3_min_target == 300.0  # 200 + 100
    assert res.w3_typical_target == 361.8  # 200 + 161.8
    # W4 retrace of W3 typical (361.8 - 121.4 = 240.4 range? Wait, W3 range = 1.618 * 100 = 161.8. 
    # Wait, W4 retrace of W3 range 161.8 is:
    # 361.8 - 161.8 * 0.50 = 280.9 (lo)
    # 361.8 - 161.8 * 0.236 = 323.62 (hi))
    assert res.w4_retrace_zone == (280.9, 323.62)
    assert res.invalidation_w2 == 100.0
    assert res.invalidation_w4 == 200.0

def test_correction_targets():
    engine = FibonacciEngine()
    # Downward correction (bearish), wave A: 200 -> 100 (range = 100)
    res = engine.correction_targets("expanded_flat", 200.0, 100.0)
    # Expanded flat: Wave B retraces 100% to 138.2% of A. Since downward, B goes up:
    # 100 + 100 = 200 (lo) and 100 + 138.2 = 238.2 (hi).
    assert res.b_zone == (200.0, 238.2)
    # Wave C: 123.6% to 161.8% of A. Downward C goes down from B:
    # 100 - 123.6 = -23.6 (lo) and 100 - 161.8 = -61.8 (hi) -> Wait!
    # In expanded flat, _c uses a_end - r_A * ratio.
    # a_end is 100. r_A = 100. 
    # So 100 - 100 * 1.618 = -61.8 (C max) and 100 - 100 * 1.236 = -23.6 (C min).
    # Since C goes down, the range is (-61.8, -23.6). Wait, round(_c(c_lo)) is round(100 - 100*1.236) = -23.6.
    # And round(_c(c_hi)) is round(100 - 100*1.618) = -61.8.
    # So c_zone is (-23.6, -61.8).
    assert res.c_zone == (-23.6, -61.8)
    assert res.b_breach_expected
    assert res.b_breach_price == 200.0

def test_measured_move():
    engine = FibonacciEngine()
    # Bearish breakout from ascending broadening wedge
    res = engine.measured_move(
        pattern_type="ascending_broadening_wedge",
        top_price=100.0,
        support_price=80.0,
        breakout_price=80.0,
        direction="bearish"
    )
    # Height = 20. rate = 0.70. scaled_move = 14.
    # Bearish target = 80 - 14 = 66.
    assert res.price == 66.0
    assert res.ratio == 0.70

def test_dual_cluster():
    engine = FibonacciEngine()
    # Bearish dual cluster
    # c_top = 200, b_low = 100. Range = 100.
    res = engine.dual_cluster(200.0, 100.0, direction="bearish")
    # Tool A: 2.618 from C top -> 200 - 261.8 = -61.8
    # Tool B: 1.618 from B low -> 100 - 161.8 = -61.8
    assert res.target_a.price == -61.8
    assert res.target_b.price == -61.8
    assert res.cluster_valid
    assert res.proximity_pct == 0.0
    assert res.cluster_strength == 1.0

def test_nearest_fib_level():
    engine = FibonacciEngine()
    # Retracements check
    # Wave 100 to 200. Price is 138.2 (0.618 retracement)
    assert engine.nearest_fib_level(138.2, 100.0, 200.0) == 0.618
    # Price is 151.0 (near 0.50)
    assert engine.nearest_fib_level(151.0, 100.0, 200.0) == 0.500
    # Price is 180.0 (no match under default 5% tolerance)
    assert engine.nearest_fib_level(180.0, 100.0, 200.0) is None

def test_invalidation():
    # Bearish case: price exceeds level
    assert is_invalidated(100.5, 100.0, "bearish") is True
    assert is_invalidated(99.5, 100.0, "bearish") is False

    # Bullish case: price drops below level
    assert is_invalidated(99.5, 100.0, "bullish") is True
    assert is_invalidated(100.5, 100.0, "bullish") is False

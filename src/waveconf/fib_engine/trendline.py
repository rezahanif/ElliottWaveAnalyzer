from dataclasses import dataclass

@dataclass
class Trendline:
    """
    Represents a fitted trendline over a set of pivot points.
    Typically generated using linear regression.
    """
    start_price: float
    end_price: float
    slope_pct_per_bar: float
    r_squared: float
    pivot_count: int

    def is_flat(self, threshold: float) -> bool:
        return abs(self.slope_pct_per_bar) <= threshold

    def is_rising(self, threshold: float) -> bool:
        return self.slope_pct_per_bar > threshold

    def is_falling(self, threshold: float) -> bool:
        return self.slope_pct_per_bar < -threshold

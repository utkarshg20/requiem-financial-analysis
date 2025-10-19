# workers/engine/signals.py
import pandas as pd

def signal_rank_top_frac(series: pd.Series, top_frac=0.1) -> pd.Series:
    """Long when series is in top X% percentile"""
    return (series.rank(pct=True) >= (1-top_frac)).astype(int)

def signal_threshold(series: pd.Series, lower=None, upper=None) -> pd.Series:
    """Long when lower < series < upper"""
    s = pd.Series(0, index=series.index)
    if lower is not None: 
        s = s.where(series <= lower, 1)  # Set to 1 where series > lower
    if upper is not None: 
        s = s.where(series >= upper, s)  # Keep current value where series < upper, set to 0 where series >= upper
    return s

def signal_crossover(fast: pd.Series, slow: pd.Series) -> pd.Series:
    """Long when fast MA > slow MA (golden cross)"""
    return (fast > slow).astype(int)

def signal_band(series: pd.Series, lower: float) -> pd.Series:
    """Mean-reversion: long when series < lower (contrarian)"""
    return (series < lower).astype(int)

def signal_tool_based(tool_values: pd.Series, action: str, comparison: str, threshold: float) -> pd.Series:
    """Generate signal based on tool values and natural language rules"""
    if comparison == "less_than":
        if action == "buy":
            return (tool_values < threshold).astype(int)
        else:  # sell
            return (tool_values < threshold).astype(int)
    else:  # greater_than
        if action == "buy":
            return (tool_values > threshold).astype(int)
        else:  # sell
            return (tool_values > threshold).astype(int)

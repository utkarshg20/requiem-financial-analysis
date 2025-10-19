from __future__ import annotations
import pandas as pd
from workers.adapters.calendar import trading_days

def last_trading_days(start: str, end: str) -> list[str]:
    days = pd.to_datetime(trading_days(start, end))
    df = pd.DataFrame(index=days)
    groups = df.index.to_period("M")
    last_days = df.groupby(groups).apply(lambda x: x.index.max()).tolist()
    return [d.date().isoformat() for d in last_days]

def monthly_rebalance_signal_from_rank(rank_series: pd.Series, top_frac=0.1) -> pd.Series:
    # only allow signal changes on last trading day of each month
    idx = rank_series.index
    ltds = last_trading_days(idx.min().date().isoformat(), idx.max().date().isoformat())
    ltds = set(pd.to_datetime(ltds))
    # keep rank values on LTDs, forward-fill between them
    mask = pd.Index(idx).isin(ltds)
    month_end_ranks = rank_series.where(mask)
    month_end_ranks = month_end_ranks.ffill()
    by_month = month_end_ranks.groupby([month_end_ranks.index.year, month_end_ranks.index.month])
    thresh = by_month.transform(lambda s: s.quantile(1 - top_frac))
    return (month_end_ranks >= thresh).astype(int)

def last_trading_day_of_week(start: str, end: str) -> list[str]:
    """Get the last trading day of each week in the date range"""
    days = pd.to_datetime(trading_days(start, end))
    df = pd.DataFrame(index=days)
    groups = df.index.to_period("W")  # Group by week
    last_days = df.groupby(groups).apply(lambda x: x.index.max()).tolist()
    return [d.date().isoformat() for d in last_days]

def gate_signal_to_schedule(signal: pd.Series, allowed_dates: set) -> pd.Series:
    # only allow changes on allowed_dates; forward-fill otherwise
    return signal.where(signal.index.isin(allowed_dates)).ffill()
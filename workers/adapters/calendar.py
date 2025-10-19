# workers/adapters/calendar.py
from __future__ import annotations

import pandas as pd
import pandas_market_calendars as mcal


def trading_days(start: str, end: str) -> list[str]:
    """
    Return ISO date strings for all NYSE trading days in [start, end].
    start/end format: "YYYY-MM-DD"
    """
    cal = mcal.get_calendar("XNYS")
    sched = cal.schedule(start_date=start, end_date=end)
    # sched index is tz-aware timestamps; convert to date ISO strings
    return [pd.Timestamp(ts).date().isoformat() for ts in sched.index]


def nearest_trading_day_utc(d: str) -> str:
    """
    Snap the given calendar date (YYYY-MM-DD) back to the most recent NYSE
    trading day (including the same day if it’s a trading day).
    Returns ISO date string.
    """
    cal = mcal.get_calendar("XNYS")
    dt = pd.Timestamp(d)

    # Loop backwards until we hit a valid trading day
    # This is robust to weekends and holidays.
    for _ in range(14):  # safety bound
        sched = cal.schedule(
            start_date=dt.date().isoformat(),
            end_date=dt.date().isoformat(),
        )
        if not sched.empty:
            return dt.date().isoformat()
        dt = dt - pd.Timedelta(days=1)

    # Fallback (should never hit with 14-day window)
    return dt.date().isoformat()

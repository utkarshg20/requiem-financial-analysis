import os, time, datetime as dt
from typing import List, Literal, Optional
import requests
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

BASE = "https://api.polygon.io"
API_KEY = os.getenv("POLYGON_API_KEY")

def _get(url: str, params: dict) -> dict:
    params = {**params, "apiKey": API_KEY}
    for i in range(5):
        r = requests.get(url, params=params, timeout=15)
        if r.status_code == 429:  # rate limit
            time.sleep(1.5 * (i + 1)); continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError("Polygon rate limit/backoff exhausted")

def get_prices_agg(
    ticker: str,
    start: str,  # "YYYY-MM-DD"
    end: str,    # "YYYY-MM-DD"
    timespan: Literal["minute","hour","day"] = "day",
    multiplier: int = 1,
) -> pd.DataFrame:
    """Daily OHLCV aggregates (split/CA-adjusted by Polygon)."""
    url = f"{BASE}/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{start}/{end}"
    data = _get(url, {})
    if data.get("status") not in ["OK", "DELAYED"] or "results" not in data:
        return pd.DataFrame(columns=["date","open","high","low","close","volume"])
    rows = []
    for r in data["results"]:
        rows.append({
            "date": dt.datetime.fromtimestamp(r["t"]/1000, tz=dt.timezone.utc).date(),
            "open": r.get("o"), "high": r.get("h"),
            "low": r.get("l"), "close": r.get("c"),
            "volume": r.get("v")
        })
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return df


def validate_ticker(ticker: str) -> bool:
    """
    Quick validation that a ticker exists in Polygon.
    Returns True if ticker is valid, False otherwise.
    """
    try:
        # Try to get 1 day of data from a known date
        df = get_prices_agg(ticker, "2024-01-05", "2024-01-05")
        return not df.empty
    except:
        return False


def get_intraday_price(
    ticker: str,
    datetime_str: str,  # "YYYY-MM-DD HH:MM" in ET
    window_minutes: int = 5,
) -> dict:
    """
    Get the closest minute bar to the specified datetime (in ET).
    Returns a single price point with timestamp.
    
    Args:
        ticker: Stock ticker symbol
        datetime_str: DateTime string in format "YYYY-MM-DD HH:MM" (ET timezone)
        window_minutes: How many minutes before/after to search (default 5)
    
    Returns:
        dict with keys: timestamp, date, time, open, high, low, close, volume
    """
    from datetime import datetime as pdt, timedelta
    import pytz
    
    # Parse the datetime string as ET
    et_tz = pytz.timezone('America/New_York')
    target_dt = pdt.strptime(datetime_str, "%Y-%m-%d %H:%M")
    target_dt = et_tz.localize(target_dt)
    
    # Search window: +/- window_minutes
    start_dt = target_dt - timedelta(minutes=window_minutes)
    end_dt = target_dt + timedelta(minutes=window_minutes)
    
    # Convert to date strings for API call
    start_date = start_dt.strftime("%Y-%m-%d")
    end_date = end_dt.strftime("%Y-%m-%d")
    
    # Fetch minute-level data
    url = f"{BASE}/v2/aggs/ticker/{ticker}/range/1/minute/{start_date}/{end_date}"
    data = _get(url, {"adjusted": "true", "sort": "asc"})
    
    if data.get("status") != "OK" or "results" not in data:
        return None
    
    # Find the closest minute bar to target time
    results = data["results"]
    if not results:
        return None
    
    target_ts = target_dt.timestamp() * 1000  # Convert to milliseconds
    
    # Find closest bar
    closest_bar = min(results, key=lambda r: abs(r["t"] - target_ts))
    
    # Convert timestamp to ET datetime
    bar_dt = pdt.fromtimestamp(closest_bar["t"] / 1000, tz=pytz.UTC)
    bar_dt_et = bar_dt.astimezone(et_tz)
    
    return {
        "timestamp": closest_bar["t"],
        "date": bar_dt_et.strftime("%Y-%m-%d"),
        "time": bar_dt_et.strftime("%H:%M:%S"),
        "datetime": bar_dt_et.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "open": closest_bar.get("o"),
        "high": closest_bar.get("h"),
        "low": closest_bar.get("l"),
        "close": closest_bar.get("c"),
        "volume": closest_bar.get("v"),
        "requested_time": datetime_str,
        "time_diff_seconds": abs((closest_bar["t"] / 1000) - (target_ts / 1000))
    }



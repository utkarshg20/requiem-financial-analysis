import os, time, datetime as dt
from typing import List, Optional, Dict
import requests
import pandas as pd

BASE = "https://api.polygon.io"
API_KEY = os.getenv("POLYGON_API_KEY")

def _get(url: str, params: dict) -> dict:
    params = {**params, "apiKey": API_KEY}
    for i in range(5):
        r = requests.get(url, params=params, timeout=15)
        if r.status_code == 429:
            time.sleep(1.5 * (i + 1)); continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError("Polygon rate limit/backoff exhausted")

def list_option_contracts(
    underlying: str,           # e.g., "IWM"
    expiry: Optional[str]=None # "YYYY-MM-DD"
) -> pd.DataFrame:
    """Reference contracts list (not quotes)."""
    url = f"{BASE}/v3/reference/options/contracts"
    params = {"underlying_ticker": underlying, "limit": 1000}
    if expiry: params["expiration_date"] = expiry
    data = _get(url, params)
    if data.get("status") != "OK" or "results" not in data: 
        return pd.DataFrame()
    rows = []
    for c in data["results"]:
        rows.append({
            "option_symbol": c.get("ticker"),
            "underlying": c.get("underlying_ticker"),
            "type": c.get("contract_type"),          # "call"/"put"
            "strike": c.get("strike_price"),
            "expiry": c.get("expiration_date"),
            "shares_per_contract": c.get("shares_per_contract")
        })
    df = pd.DataFrame(rows)
    return df

def previous_close_quote(option_symbol: str) -> Optional[Dict]:
    """Use previous close when real-time quotes not available on your plan."""
    url = f"{BASE}/v2/aggs/ticker/{option_symbol}/prev"
    data = _get(url, {})
    try:
        r = data["results"][0]
        return {"close": r.get("c"), "volume": r.get("v")}
    except Exception:
        return None

def build_chain_with_prev_close(underlying: str, expiry: str) -> pd.DataFrame:
    chain = list_option_contracts(underlying, expiry)
    if chain.empty: return chain
    quotes = []
    for sym in chain["option_symbol"].head(500):  # bound requests
        q = previous_close_quote(sym)
        quotes.append(q if q else {"close": None, "volume": None})
    chain = chain.iloc[:len(quotes)].copy()
    chain["prev_close"] = [q["close"] if q else None for q in quotes]
    chain["prev_volume"] = [q["volume"] if q else None for q in quotes]
    return chain

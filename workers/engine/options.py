# workers/engine/options.py
from __future__ import annotations
import math
from typing import Iterable, Optional, Tuple

# --- Normal helpers ---
def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def _norm_pdf(x: float) -> float:
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)

# --- Black–Scholes core (equity/index approximation) ---
def bs_d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    # Guard small/invalid inputs
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return 0.0
    return (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))

def call_delta(S: float, K: float, T: float, r: float, sigma: float) -> float:
    return _norm_cdf(bs_d1(S, K, T, r, sigma))

def put_delta(S: float, K: float, T: float, r: float, sigma: float) -> float:
    return call_delta(S, K, T, r, sigma) - 1.0

def call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        # intrinsic value at expiry fallback
        return max(S - K, 0.0)
    d1 = bs_d1(S, K, T, r, sigma)
    d2 = d1 - sigma * math.sqrt(T)
    return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)

def put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return max(K - S, 0.0)
    d1 = bs_d1(S, K, T, r, sigma)
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)

def vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = bs_d1(S, K, T, r, sigma)
    return S * _norm_pdf(d1) * math.sqrt(T)

# --- Strangle helpers you’re missing ---
def select_strikes_by_delta(
    S: float,
    T: float,
    r: float,
    sigma: float,
    target_call_delta: float = 0.16,
    target_put_delta: float = -0.16,
    strikes: Optional[Iterable[float]] = None,
) -> Tuple[float, float]:
    """
    Pick put/call strikes whose deltas are closest to target deltas.
    If no strikes provided, generate a coarse symmetric grid around spot.
    """
    if not strikes:
        # +/- 50% grid in 1% steps around S
        strikes = [round(S * (1 + x / 100.0), 2) for x in range(-50, 51, 1)]
    strikes = list(sorted(set(strikes)))
    best_call = min(strikes, key=lambda K: abs(call_delta(S, K, T, r, sigma) - target_call_delta))
    best_put  = min(strikes, key=lambda K: abs(put_delta(S, K, T, r, sigma)  - target_put_delta))
    return float(best_put), float(best_call)

def strangle_credit(S: float, Kp: float, Kc: float, T: float, r: float, sigma: float) -> float:
    """
    Premium received for selling a short strangle (sell 1 put @ Kp + sell 1 call @ Kc).
    """
    return call_price(S, Kc, T, r, sigma) + put_price(S, Kp, T, r, sigma)

def strangle_payoff_expiry(S_T: float, Kp: float, Kc: float) -> float:
    """
    Seller payoff at expiry, excluding premium (negative in tails, 0 between strikes).
    """
    if S_T < Kp:
        return -(Kp - S_T)
    if S_T > Kc:
        return -(S_T - Kc)
    return 0.0

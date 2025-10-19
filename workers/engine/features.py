# workers/engine/features.py
import pandas as pd
import numpy as np

def feat_momentum_12m_skip_1m(close: pd.Series) -> pd.Series:
    """12-month momentum skipping most recent month (classic factor)"""
    return (close.shift(21)/close.shift(252)) - 1.0

def feat_zscore(close: pd.Series, window: int = 20) -> pd.Series:
    """Z-score of returns over rolling window"""
    ret = close.pct_change(fill_method=None)
    mu = ret.rolling(window).mean()
    sd = ret.rolling(window).std()
    return (ret - mu) / (sd + 1e-12)

def feat_sma(close: pd.Series, window: int = 20) -> dict:
    """Simple Moving Average with price"""
    sma = close.rolling(window).mean()
    return {
        'sma': sma,
        'price': close
    }

def feat_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index (0-100)"""
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / (loss + 1e-12)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def feat_realized_vol(close: pd.Series, window: int = 20) -> pd.Series:
    """Realized volatility (annualized)"""
    ret = close.pct_change(fill_method=None)
    return ret.rolling(window).std() * np.sqrt(252)

def feat_aroon(close: pd.Series, window: int = 14) -> pd.Series:
    """Aroon indicator for trend detection (0-100)"""
    high = close.rolling(window).max()
    low = close.rolling(window).min()
    
    # Aroon Up: percentage of time since highest high
    aroon_up = ((window - close.rolling(window).apply(lambda x: window - 1 - x.argmax(), raw=True)) / window) * 100
    
    # Aroon Down: percentage of time since lowest low  
    aroon_down = ((window - close.rolling(window).apply(lambda x: window - 1 - x.argmin(), raw=True)) / window) * 100
    
    # Return Aroon Oscillator (Up - Down)
    return aroon_up - aroon_down

def feat_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """MACD oscillator"""
    ema_fast = close.ewm(span=fast).mean()
    ema_slow = close.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    return macd_line

def feat_bollinger(close: pd.Series, window: int = 20, std_dev: float = 2.0) -> dict:
    """Bollinger Bands with upper, middle, lower bands and price"""
    sma = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    bandwidth = (upper_band - lower_band) / sma
    
    return {
        'upper_band': upper_band,
        'middle_band': sma,
        'lower_band': lower_band,
        'price': close,
        'bandwidth': bandwidth
    }

def feat_williams_r(close: pd.Series, window: int = 14) -> pd.Series:
    """Williams %R momentum oscillator (-100 to 0)"""
    high = close.rolling(window).max()
    low = close.rolling(window).min()
    wr = ((high - close) / (high - low)) * -100
    return wr

def feat_stochastic(close: pd.Series, k_window: int = 14, d_window: int = 3) -> pd.Series:
    """Stochastic oscillator %K (0-100)"""
    low = close.rolling(k_window).min()
    high = close.rolling(k_window).max()
    k_percent = ((close - low) / (high - low)) * 100
    return k_percent

FEATURES = {
    "momentum_12m_skip_1m": feat_momentum_12m_skip_1m,
    "zscore_20d": lambda close: feat_zscore(close, 20),
    "sma_20": lambda close: feat_sma(close, 20),
    "sma_50": lambda close: feat_sma(close, 50),
    "rsi_14": lambda close: feat_rsi(close, 14),
}

# Tool Registry - Maps tool names to their functions
# Note: Technical analysis tools (RSI, MACD, SMA, etc.) are now handled by TA-Lib
# Only keeping non-technical analysis tools here
REQUIEM_TOOLS = {
    "zscore": {
        "function": feat_zscore,
        "description": "Z-Score - measures how many standard deviations a price is from its mean, useful for identifying extreme price movements and mean reversion opportunities",
        "parameters": {"window": {"type": "int", "default": 20, "description": "Rolling window for calculation"}}
    },
    "realized_vol": {
        "function": feat_realized_vol,
        "description": "Realized Volatility - measures the actual price volatility over a specified period, calculated as the standard deviation of returns annualized",
        "parameters": {"window": {"type": "int", "default": 20, "description": "Rolling window for volatility calculation"}}
    },
    "momentum": {
        "function": feat_momentum_12m_skip_1m,
        "description": "Price Momentum - measures the rate of change in price over a 12-month period, skipping the most recent month to avoid look-ahead bias",
        "parameters": {}
    }
}

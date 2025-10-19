"""
Vercel-compatible TA-Lib wrapper
Provides fallback implementations when TA-Lib is not available
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger("requiem.vercel_talib")

# Try to import TA-Lib, fallback to mock if not available
try:
    import talib
    TALIB_AVAILABLE = True
    logger.info("TA-Lib is available")
except ImportError:
    TALIB_AVAILABLE = False
    logger.warning("TA-Lib not available, using fallback implementations")

class VercelTALibWrapper:
    """
    Vercel-compatible TA-Lib wrapper with fallback implementations
    """
    
    def __init__(self):
        self.available = TALIB_AVAILABLE
    
    def RSI(self, prices: np.ndarray, timeperiod: int = 14) -> np.ndarray:
        """Relative Strength Index"""
        if TALIB_AVAILABLE:
            return talib.RSI(prices, timeperiod=timeperiod)
        else:
            return self._fallback_RSI(prices, timeperiod)
    
    def SMA(self, prices: np.ndarray, timeperiod: int = 30) -> np.ndarray:
        """Simple Moving Average"""
        if TALIB_AVAILABLE:
            return talib.SMA(prices, timeperiod=timeperiod)
        else:
            return self._fallback_SMA(prices, timeperiod)
    
    def EMA(self, prices: np.ndarray, timeperiod: int = 30) -> np.ndarray:
        """Exponential Moving Average"""
        if TALIB_AVAILABLE:
            return talib.EMA(prices, timeperiod=timeperiod)
        else:
            return self._fallback_EMA(prices, timeperiod)
    
    def MACD(self, prices: np.ndarray, fastperiod: int = 12, slowperiod: int = 26, signalperiod: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """MACD"""
        if TALIB_AVAILABLE:
            return talib.MACD(prices, fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)
        else:
            return self._fallback_MACD(prices, fastperiod, slowperiod, signalperiod)
    
    def BBANDS(self, prices: np.ndarray, timeperiod: int = 20, nbdevup: float = 2, nbdevdn: float = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bollinger Bands"""
        if TALIB_AVAILABLE:
            return talib.BBANDS(prices, timeperiod=timeperiod, nbdevup=nbdevup, nbdevdn=nbdevdn)
        else:
            return self._fallback_BBANDS(prices, timeperiod, nbdevup, nbdevdn)
    
    def STOCH(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, fastk_period: int = 5, slowk_period: int = 3, slowd_period: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """Stochastic Oscillator"""
        if TALIB_AVAILABLE:
            return talib.STOCH(high, low, close, fastk_period=fastk_period, slowk_period=slowk_period, slowd_period=slowd_period)
        else:
            return self._fallback_STOCH(high, low, close, fastk_period, slowk_period, slowd_period)
    
    def ATR(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, timeperiod: int = 14) -> np.ndarray:
        """Average True Range"""
        if TALIB_AVAILABLE:
            return talib.ATR(high, low, close, timeperiod=timeperiod)
        else:
            return self._fallback_ATR(high, low, close, timeperiod)
    
    # Fallback implementations
    def _fallback_RSI(self, prices: np.ndarray, timeperiod: int = 14) -> np.ndarray:
        """Fallback RSI implementation"""
        try:
            delta = np.diff(prices)
            gain = np.where(delta > 0, delta, 0)
            loss = np.where(delta < 0, -delta, 0)
            
            avg_gain = pd.Series(gain).rolling(window=timeperiod).mean().values
            avg_loss = pd.Series(loss).rolling(window=timeperiod).mean().values
            
            rs = avg_gain / (avg_loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))
            
            # Pad with NaN values to match TA-Lib output
            result = np.full(len(prices), np.nan)
            result[timeperiod:] = rsi[timeperiod-1:]
            return result
        except Exception as e:
            logger.error(f"Error in fallback RSI: {e}")
            return np.full(len(prices), np.nan)
    
    def _fallback_SMA(self, prices: np.ndarray, timeperiod: int = 30) -> np.ndarray:
        """Fallback SMA implementation"""
        try:
            return pd.Series(prices).rolling(window=timeperiod).mean().values
        except Exception as e:
            logger.error(f"Error in fallback SMA: {e}")
            return np.full(len(prices), np.nan)
    
    def _fallback_EMA(self, prices: np.ndarray, timeperiod: int = 30) -> np.ndarray:
        """Fallback EMA implementation"""
        try:
            return pd.Series(prices).ewm(span=timeperiod).mean().values
        except Exception as e:
            logger.error(f"Error in fallback EMA: {e}")
            return np.full(len(prices), np.nan)
    
    def _fallback_MACD(self, prices: np.ndarray, fastperiod: int = 12, slowperiod: int = 26, signalperiod: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fallback MACD implementation"""
        try:
            ema_fast = self._fallback_EMA(prices, fastperiod)
            ema_slow = self._fallback_EMA(prices, slowperiod)
            macd = ema_fast - ema_slow
            signal = self._fallback_EMA(macd, signalperiod)
            histogram = macd - signal
            return macd, signal, histogram
        except Exception as e:
            logger.error(f"Error in fallback MACD: {e}")
            nan_array = np.full(len(prices), np.nan)
            return nan_array, nan_array, nan_array
    
    def _fallback_BBANDS(self, prices: np.ndarray, timeperiod: int = 20, nbdevup: float = 2, nbdevdn: float = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fallback Bollinger Bands implementation"""
        try:
            sma = self._fallback_SMA(prices, timeperiod)
            std = pd.Series(prices).rolling(window=timeperiod).std().values
            upper = sma + (std * nbdevup)
            lower = sma - (std * nbdevdn)
            return upper, sma, lower
        except Exception as e:
            logger.error(f"Error in fallback BBANDS: {e}")
            nan_array = np.full(len(prices), np.nan)
            return nan_array, nan_array, nan_array
    
    def _fallback_STOCH(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, fastk_period: int = 5, slowk_period: int = 3, slowd_period: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """Fallback Stochastic implementation"""
        try:
            lowest_low = pd.Series(low).rolling(window=fastk_period).min().values
            highest_high = pd.Series(high).rolling(window=fastk_period).max().values
            
            k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low + 1e-10))
            k_percent = pd.Series(k_percent).rolling(window=slowk_period).mean().values
            d_percent = pd.Series(k_percent).rolling(window=slowd_period).mean().values
            
            return k_percent, d_percent
        except Exception as e:
            logger.error(f"Error in fallback STOCH: {e}")
            nan_array = np.full(len(prices), np.nan)
            return nan_array, nan_array
    
    def _fallback_ATR(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, timeperiod: int = 14) -> np.ndarray:
        """Fallback ATR implementation"""
        try:
            high_low = high - low
            high_close = np.abs(high - np.roll(close, 1))
            low_close = np.abs(low - np.roll(close, 1))
            
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            atr = pd.Series(true_range).rolling(window=timeperiod).mean().values
            
            # Pad with NaN values
            result = np.full(len(close), np.nan)
            result[timeperiod:] = atr[timeperiod-1:]
            return result
        except Exception as e:
            logger.error(f"Error in fallback ATR: {e}")
            return np.full(len(close), np.nan)

"""
TA-Lib Technical Indicators Integration
Comprehensive technical analysis using TA-Lib library
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging

# Use Vercel-compatible wrapper
from .vercel_talib_wrapper import VercelTALibWrapper

logger = logging.getLogger("requiem.talib")

class TALibIndicators:
    """Comprehensive technical indicators using TA-Lib"""
    
    def __init__(self):
        self.talib = VercelTALibWrapper()
        self.indicators = {
            # Moving Averages
            'sma': self._sma,
            'ema': self._ema,
            'wma': self._wma,
            'dema': self._dema,
            'tema': self._tema,
            'trima': self._trima,
            'kama': self._kama,
            'mama': self._mama,
            't3': self._t3,
            
            # Momentum Oscillators
            'rsi': self._rsi,
            'stoch': self._stoch,
            'stochf': self._stochf,
            'stochrsi': self._stochrsi,
            'willr': self._willr,
            'cci': self._cci,
            'cmo': self._cmo,
            'roc': self._roc,
            'rocp': self._rocp,
            'rocr': self._rocr,
            'rocr100': self._rocr100,
            'mom': self._mom,
            'dx': self._dx,
            'adx': self._adx,
            'adxr': self._adxr,
            'aroon': self._aroon,
            'aroonosc': self._aroonosc,
            'bop': self._bop,
            'trix': self._trix,
            'ultosc': self._ultosc,
            'mfi': self._mfi,
            'ppo': self._ppo,
            'macd': self._macd,
            'macdext': self._macdext,
            'macdfix': self._macdfix,
            
            # Volatility Indicators
            'bbands': self._bbands,
            'natr': self._natr,
            'trange': self._trange,
            'atr': self._atr,
            'ad': self._ad,
            'adosc': self._adosc,
            'obv': self._obv,
            'ht_dcperiod': self._ht_dcperiod,
            'ht_dcphase': self._ht_dcphase,
            'ht_phasor': self._ht_phasor,
            'ht_sine': self._ht_sine,
            'ht_trendmode': self._ht_trendmode,
            
            # Volume Indicators
            'adx': self._adx,
            'adxr': self._adxr,
            'apo': self._apo,
            'aroon': self._aroon,
            'aroonosc': self._aroonosc,
            'bop': self._bop,
            'cci': self._cci,
            'cmo': self._cmo,
            'dx': self._dx,
            'macd': self._macd,
            'macdext': self._macdext,
            'macdfix': self._macdfix,
            'mfi': self._mfi,
            'minus_di': self._minus_di,
            'minus_dm': self._minus_dm,
            'mom': self._mom,
            'plus_di': self._plus_di,
            'plus_dm': self._plus_dm,
            'ppo': self._ppo,
            'roc': self._roc,
            'rocp': self._rocp,
            'rocr': self._rocr,
            'rocr100': self._rocr100,
            'rsi': self._rsi,
            'stoch': self._stoch,
            'stochf': self._stochf,
            'stochrsi': self._stochrsi,
            'trix': self._trix,
            'ultosc': self._ultosc,
            'willr': self._willr,
        }
    
    def calculate_indicator(self, indicator_name: str, data: pd.DataFrame, 
                          params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Calculate a technical indicator using TA-Lib"""
        try:
            if indicator_name not in self.indicators:
                raise ValueError(f"Unknown indicator: {indicator_name}")
            
            if params is None:
                params = {}
            
            return self.indicators[indicator_name](data, params)
            
        except Exception as e:
            logger.error(f"Error calculating {indicator_name}: {e}")
            return {"error": str(e)}
    
    def _prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare OHLCV data for TA-Lib"""
        high = data['high'].values.astype(np.float64)
        low = data['low'].values.astype(np.float64)
        close = data['close'].values.astype(np.float64)
        volume = data['volume'].values.astype(np.float64) if 'volume' in data.columns else np.zeros_like(close)
        return high, low, close, volume
    
    # Moving Averages
    def _sma(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Simple Moving Average"""
        timeperiod = params.get('timeperiod', 30)
        high, low, close, volume = self._prepare_data(data)
        
        sma = self.talib.SMA(close, timeperiod=timeperiod)
        
        return {
            'indicator': 'SMA',
            'values': sma,
            'price': close,
            'parameters': {'timeperiod': timeperiod},
            'description': f'Simple Moving Average ({timeperiod} periods)',
            'chart_type': 'line',
            'series': {
                'sma': sma,
                'price': close
            }
        }
    
    def _ema(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Exponential Moving Average"""
        timeperiod = params.get('timeperiod', 30)
        high, low, close, volume = self._prepare_data(data)
        
        ema = self.talib.EMA(close, timeperiod=timeperiod)
        
        return {
            'indicator': 'EMA',
            'values': ema,
            'price': close,
            'parameters': {'timeperiod': timeperiod},
            'description': f'Exponential Moving Average ({timeperiod} periods)',
            'chart_type': 'line',
            'series': {
                'ema': ema,
                'price': close
            }
        }
    
    def _rsi(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Relative Strength Index"""
        timeperiod = params.get('timeperiod', 14)
        high, low, close, volume = self._prepare_data(data)
        
        rsi = self.talib.RSI(close, timeperiod=timeperiod)
        
        return {
            'indicator': 'RSI',
            'values': rsi,
            'price': close,
            'parameters': {'timeperiod': timeperiod},
            'description': f'Relative Strength Index ({timeperiod} periods)',
            'chart_type': 'oscillator',
            'series': {
                'rsi': rsi,
                'price': close
            },
            'levels': {
                'overbought': 70,
                'oversold': 30,
                'neutral': 50
            }
        }
    
    def _bbands(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Bollinger Bands"""
        timeperiod = params.get('timeperiod', 20)
        nbdevup = params.get('nbdevup', 2)
        nbdevdn = params.get('nbdevdn', 2)
        matype = params.get('matype', 0)
        
        high, low, close, volume = self._prepare_data(data)
        
        upper, middle, lower = self.talib.BBANDS(close, timeperiod=timeperiod, 
                                          nbdevup=nbdevup, nbdevdn=nbdevdn)
        
        return {
            'indicator': 'Bollinger Bands',
            'values': middle,
            'price': close,
            'parameters': {'timeperiod': timeperiod, 'nbdevup': nbdevup, 'nbdevdn': nbdevdn},
            'description': f'Bollinger Bands ({timeperiod} periods, {nbdevup}σ)',
            'chart_type': 'bands',
            'series': {
                'upper_band': upper,
                'middle_band': middle,
                'lower_band': lower,
                'price': close
            }
        }
    
    def _macd(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """MACD (Moving Average Convergence Divergence)"""
        fastperiod = params.get('fastperiod', 12)
        slowperiod = params.get('slowperiod', 26)
        signalperiod = params.get('signalperiod', 9)
        
        high, low, close, volume = self._prepare_data(data)
        
        macd, macdsignal, macdhist = self.talib.MACD(close, fastperiod=fastperiod, 
                                              slowperiod=slowperiod, 
                                              signalperiod=signalperiod)
        
        return {
            'indicator': 'MACD',
            'values': macd,
            'price': close,
            'parameters': {'fastperiod': fastperiod, 'slowperiod': slowperiod, 'signalperiod': signalperiod},
            'description': f'MACD ({fastperiod}, {slowperiod}, {signalperiod})',
            'chart_type': 'macd',
            'series': {
                'macd': macd,
                'signal': macdsignal,
                'histogram': macdhist,
                'price': close
            }
        }
    
    def _stoch(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Stochastic Oscillator"""
        fastk_period = params.get('fastk_period', 5)
        slowk_period = params.get('slowk_period', 3)
        slowd_period = params.get('slowd_period', 3)
        
        high, low, close, volume = self._prepare_data(data)
        
        slowk, slowd = self.talib.STOCH(high, low, close, 
                                  fastk_period=fastk_period,
                                  slowk_period=slowk_period,
                                  slowd_period=slowd_period)
        
        return {
            'indicator': 'Stochastic',
            'values': slowk,
            'price': close,
            'parameters': {'fastk_period': fastk_period, 'slowk_period': slowk_period, 'slowd_period': slowd_period},
            'description': f'Stochastic Oscillator ({fastk_period}, {slowk_period}, {slowd_period})',
            'chart_type': 'oscillator',
            'series': {
                'stoch_k': slowk,
                'stoch_d': slowd,
                'price': close
            },
            'levels': {
                'overbought': 80,
                'oversold': 20
            }
        }
    
    def _willr(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Williams %R"""
        timeperiod = params.get('timeperiod', 14)
        high, low, close, volume = self._prepare_data(data)
        
        willr = talib.WILLR(high, low, close, timeperiod=timeperiod)
        
        return {
            'indicator': 'Williams %R',
            'values': willr,
            'price': close,
            'parameters': {'timeperiod': timeperiod},
            'description': f'Williams %R ({timeperiod} periods)',
            'chart_type': 'oscillator',
            'series': {
                'willr': willr,
                'price': close
            },
            'levels': {
                'overbought': -20,
                'oversold': -80
            }
        }
    
    def _adx(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Average Directional Movement Index"""
        timeperiod = params.get('timeperiod', 14)
        high, low, close, volume = self._prepare_data(data)
        
        adx = talib.ADX(high, low, close, timeperiod=timeperiod)
        
        return {
            'indicator': 'ADX',
            'values': adx,
            'price': close,
            'parameters': {'timeperiod': timeperiod},
            'description': f'Average Directional Movement Index ({timeperiod} periods)',
            'chart_type': 'oscillator',
            'series': {
                'adx': adx,
                'price': close
            },
            'levels': {
                'strong_trend': 25,
                'weak_trend': 20
            }
        }
    
    def _cci(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Commodity Channel Index"""
        timeperiod = params.get('timeperiod', 14)
        high, low, close, volume = self._prepare_data(data)
        
        cci = talib.CCI(high, low, close, timeperiod=timeperiod)
        
        return {
            'indicator': 'CCI',
            'values': cci,
            'price': close,
            'parameters': {'timeperiod': timeperiod},
            'description': f'Commodity Channel Index ({timeperiod} periods)',
            'chart_type': 'oscillator',
            'series': {
                'cci': cci,
                'price': close
            },
            'levels': {
                'overbought': 100,
                'oversold': -100
            }
        }
    
    def _obv(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """On Balance Volume"""
        high, low, close, volume = self._prepare_data(data)
        
        obv = talib.OBV(close, volume)
        
        return {
            'indicator': 'OBV',
            'values': obv,
            'price': close,
            'parameters': {},
            'description': 'On Balance Volume',
            'chart_type': 'line',
            'series': {
                'obv': obv,
                'price': close
            }
        }
    
    def _atr(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Average True Range"""
        timeperiod = params.get('timeperiod', 14)
        high, low, close, volume = self._prepare_data(data)
        
        atr = self.talib.ATR(high, low, close, timeperiod=timeperiod)
        
        return {
            'indicator': 'ATR',
            'values': atr,
            'price': close,
            'parameters': {'timeperiod': timeperiod},
            'description': f'Average True Range ({timeperiod} periods)',
            'chart_type': 'line',
            'series': {
                'atr': atr,
                'price': close
            }
        }
    
    # Additional indicator methods would go here...
    # For brevity, I'll add a few more key ones
    
    def _wma(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Weighted Moving Average"""
        timeperiod = params.get('timeperiod', 30)
        high, low, close, volume = self._prepare_data(data)
        
        wma = talib.WMA(close, timeperiod=timeperiod)
        
        return {
            'indicator': 'WMA',
            'values': wma,
            'price': close,
            'parameters': {'timeperiod': timeperiod},
            'description': f'Weighted Moving Average ({timeperiod} periods)',
            'chart_type': 'line',
            'series': {
                'wma': wma,
                'price': close
            }
        }
    
    def _dema(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Double Exponential Moving Average"""
        timeperiod = params.get('timeperiod', 30)
        high, low, close, volume = self._prepare_data(data)
        
        dema = talib.DEMA(close, timeperiod=timeperiod)
        
        return {
            'indicator': 'DEMA',
            'values': dema,
            'price': close,
            'parameters': {'timeperiod': timeperiod},
            'description': f'Double Exponential Moving Average ({timeperiod} periods)',
            'chart_type': 'line',
            'series': {
                'dema': dema,
                'price': close
            }
        }
    
    def _tema(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Triple Exponential Moving Average"""
        timeperiod = params.get('timeperiod', 30)
        high, low, close, volume = self._prepare_data(data)
        
        tema = talib.TEMA(close, timeperiod=timeperiod)
        
        return {
            'indicator': 'TEMA',
            'values': tema,
            'price': close,
            'parameters': {'timeperiod': timeperiod},
            'description': f'Triple Exponential Moving Average ({timeperiod} periods)',
            'chart_type': 'line',
            'series': {
                'tema': tema,
                'price': close
            }
        }
    
    def _trima(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Triangular Moving Average"""
        timeperiod = params.get('timeperiod', 30)
        high, low, close, volume = self._prepare_data(data)
        
        trima = talib.TRIMA(close, timeperiod=timeperiod)
        
        return {
            'indicator': 'TRIMA',
            'values': trima,
            'price': close,
            'parameters': {'timeperiod': timeperiod},
            'description': f'Triangular Moving Average ({timeperiod} periods)',
            'chart_type': 'line',
            'series': {
                'trima': trima,
                'price': close
            }
        }
    
    def _kama(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Kaufman Adaptive Moving Average"""
        timeperiod = params.get('timeperiod', 30)
        high, low, close, volume = self._prepare_data(data)
        
        kama = talib.KAMA(close, timeperiod=timeperiod)
        
        return {
            'indicator': 'KAMA',
            'values': kama,
            'price': close,
            'parameters': {'timeperiod': timeperiod},
            'description': f'Kaufman Adaptive Moving Average ({timeperiod} periods)',
            'chart_type': 'line',
            'series': {
                'kama': kama,
                'price': close
            }
        }
    
    def _mama(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """MESA Adaptive Moving Average"""
        fastlimit = params.get('fastlimit', 0.5)
        slowlimit = params.get('slowlimit', 0.05)
        high, low, close, volume = self._prepare_data(data)
        
        mama, fama = talib.MAMA(close, fastlimit=fastlimit, slowlimit=slowlimit)
        
        return {
            'indicator': 'MAMA',
            'values': mama,
            'price': close,
            'parameters': {'fastlimit': fastlimit, 'slowlimit': slowlimit},
            'description': f'MESA Adaptive Moving Average ({fastlimit}, {slowlimit})',
            'chart_type': 'line',
            'series': {
                'mama': mama,
                'fama': fama,
                'price': close
            }
        }
    
    def _t3(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """T3 Moving Average"""
        timeperiod = params.get('timeperiod', 30)
        vfactor = params.get('vfactor', 0.7)
        high, low, close, volume = self._prepare_data(data)
        
        t3 = talib.T3(close, timeperiod=timeperiod, vfactor=vfactor)
        
        return {
            'indicator': 'T3',
            'values': t3,
            'price': close,
            'parameters': {'timeperiod': timeperiod, 'vfactor': vfactor},
            'description': f'T3 Moving Average ({timeperiod} periods, {vfactor})',
            'chart_type': 'line',
            'series': {
                't3': t3,
                'price': close
            }
        }
    
    def _stochf(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Stochastic Fast"""
        fastk_period = params.get('fastk_period', 5)
        fastd_period = params.get('fastd_period', 3)
        high, low, close, volume = self._prepare_data(data)
        
        fastk, fastd = talib.STOCHF(high, low, close, 
                                   fastk_period=fastk_period,
                                   fastd_period=fastd_period)
        
        return {
            'indicator': 'Stochastic Fast',
            'values': fastk,
            'price': close,
            'parameters': {'fastk_period': fastk_period, 'fastd_period': fastd_period},
            'description': f'Stochastic Fast ({fastk_period}, {fastd_period})',
            'chart_type': 'oscillator',
            'series': {
                'stoch_k': fastk,
                'stoch_d': fastd,
                'price': close
            },
            'levels': {
                'overbought': 80,
                'oversold': 20
            }
        }
    
    def _stochrsi(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Stochastic RSI"""
        timeperiod = params.get('timeperiod', 14)
        fastk_period = params.get('fastk_period', 3)
        fastd_period = params.get('fastd_period', 3)
        high, low, close, volume = self._prepare_data(data)
        
        fastk, fastd = talib.STOCHRSI(close, timeperiod=timeperiod,
                                     fastk_period=fastk_period,
                                     fastd_period=fastd_period)
        
        return {
            'indicator': 'Stochastic RSI',
            'values': fastk,
            'price': close,
            'parameters': {'timeperiod': timeperiod, 'fastk_period': fastk_period, 'fastd_period': fastd_period},
            'description': f'Stochastic RSI ({timeperiod}, {fastk_period}, {fastd_period})',
            'chart_type': 'oscillator',
            'series': {
                'stoch_rsi_k': fastk,
                'stoch_rsi_d': fastd,
                'price': close
            },
            'levels': {
                'overbought': 80,
                'oversold': 20
            }
        }
    
    def _cmo(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Chande Momentum Oscillator"""
        timeperiod = params.get('timeperiod', 14)
        high, low, close, volume = self._prepare_data(data)
        
        cmo = talib.CMO(close, timeperiod=timeperiod)
        
        return {
            'indicator': 'CMO',
            'values': cmo,
            'price': close,
            'parameters': {'timeperiod': timeperiod},
            'description': f'Chande Momentum Oscillator ({timeperiod} periods)',
            'chart_type': 'oscillator',
            'series': {
                'cmo': cmo,
                'price': close
            },
            'levels': {
                'overbought': 50,
                'oversold': -50
            }
        }
    
    def _roc(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Rate of Change"""
        timeperiod = params.get('timeperiod', 10)
        high, low, close, volume = self._prepare_data(data)
        
        roc = talib.ROC(close, timeperiod=timeperiod)
        
        return {
            'indicator': 'ROC',
            'values': roc,
            'price': close,
            'parameters': {'timeperiod': timeperiod},
            'description': f'Rate of Change ({timeperiod} periods)',
            'chart_type': 'oscillator',
            'series': {
                'roc': roc,
                'price': close
            }
        }
    
    def _mom(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Momentum"""
        timeperiod = params.get('timeperiod', 10)
        high, low, close, volume = self._prepare_data(data)
        
        mom = talib.MOM(close, timeperiod=timeperiod)
        
        return {
            'indicator': 'Momentum',
            'values': mom,
            'price': close,
            'parameters': {'timeperiod': timeperiod},
            'description': f'Momentum ({timeperiod} periods)',
            'chart_type': 'oscillator',
            'series': {
                'momentum': mom,
                'price': close
            }
        }
    
    def _dx(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Directional Movement Index"""
        timeperiod = params.get('timeperiod', 14)
        high, low, close, volume = self._prepare_data(data)
        
        dx = talib.DX(high, low, close, timeperiod=timeperiod)
        
        return {
            'indicator': 'DX',
            'values': dx,
            'price': close,
            'parameters': {'timeperiod': timeperiod},
            'description': f'Directional Movement Index ({timeperiod} periods)',
            'chart_type': 'oscillator',
            'series': {
                'dx': dx,
                'price': close
            }
        }
    
    def _adxr(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Average Directional Movement Index Rating"""
        timeperiod = params.get('timeperiod', 14)
        high, low, close, volume = self._prepare_data(data)
        
        adxr = talib.ADXR(high, low, close, timeperiod=timeperiod)
        
        return {
            'indicator': 'ADXR',
            'values': adxr,
            'price': close,
            'parameters': {'timeperiod': timeperiod},
            'description': f'Average Directional Movement Index Rating ({timeperiod} periods)',
            'chart_type': 'oscillator',
            'series': {
                'adxr': adxr,
                'price': close
            }
        }
    
    def _aroon(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Aroon"""
        timeperiod = params.get('timeperiod', 14)
        high, low, close, volume = self._prepare_data(data)
        
        aroondown, aroonup = talib.AROON(high, low, timeperiod=timeperiod)
        
        return {
            'indicator': 'Aroon',
            'values': aroonup,
            'price': close,
            'parameters': {'timeperiod': timeperiod},
            'description': f'Aroon ({timeperiod} periods)',
            'chart_type': 'oscillator',
            'series': {
                'aroon_up': aroonup,
                'aroon_down': aroondown,
                'price': close
            },
            'levels': {
                'overbought': 70,
                'oversold': 30
            }
        }
    
    def _aroonosc(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Aroon Oscillator"""
        timeperiod = params.get('timeperiod', 14)
        high, low, close, volume = self._prepare_data(data)
        
        aroonosc = talib.AROONOSC(high, low, timeperiod=timeperiod)
        
        return {
            'indicator': 'Aroon Oscillator',
            'values': aroonosc,
            'price': close,
            'parameters': {'timeperiod': timeperiod},
            'description': f'Aroon Oscillator ({timeperiod} periods)',
            'chart_type': 'oscillator',
            'series': {
                'aroon_osc': aroonosc,
                'price': close
            }
        }
    
    def _bop(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Balance of Power"""
        high, low, close, volume = self._prepare_data(data)
        
        bop = talib.BOP(open=data['open'].values.astype(np.float64), high=high, low=low, close=close)
        
        return {
            'indicator': 'BOP',
            'values': bop,
            'price': close,
            'parameters': {},
            'description': 'Balance of Power',
            'chart_type': 'oscillator',
            'series': {
                'bop': bop,
                'price': close
            }
        }
    
    def _trix(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """TRIX"""
        timeperiod = params.get('timeperiod', 30)
        high, low, close, volume = self._prepare_data(data)
        
        trix = talib.TRIX(close, timeperiod=timeperiod)
        
        return {
            'indicator': 'TRIX',
            'values': trix,
            'price': close,
            'parameters': {'timeperiod': timeperiod},
            'description': f'TRIX ({timeperiod} periods)',
            'chart_type': 'oscillator',
            'series': {
                'trix': trix,
                'price': close
            }
        }
    
    def _ultosc(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Ultimate Oscillator"""
        timeperiod1 = params.get('timeperiod1', 7)
        timeperiod2 = params.get('timeperiod2', 14)
        timeperiod3 = params.get('timeperiod3', 28)
        high, low, close, volume = self._prepare_data(data)
        
        ultosc = talib.ULTOSC(high, low, close, 
                             timeperiod1=timeperiod1,
                             timeperiod2=timeperiod2,
                             timeperiod3=timeperiod3)
        
        return {
            'indicator': 'Ultimate Oscillator',
            'values': ultosc,
            'price': close,
            'parameters': {'timeperiod1': timeperiod1, 'timeperiod2': timeperiod2, 'timeperiod3': timeperiod3},
            'description': f'Ultimate Oscillator ({timeperiod1}, {timeperiod2}, {timeperiod3})',
            'chart_type': 'oscillator',
            'series': {
                'ultosc': ultosc,
                'price': close
            },
            'levels': {
                'overbought': 70,
                'oversold': 30
            }
        }
    
    def _mfi(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Money Flow Index"""
        timeperiod = params.get('timeperiod', 14)
        high, low, close, volume = self._prepare_data(data)
        
        mfi = talib.MFI(high, low, close, volume, timeperiod=timeperiod)
        
        return {
            'indicator': 'MFI',
            'values': mfi,
            'price': close,
            'parameters': {'timeperiod': timeperiod},
            'description': f'Money Flow Index ({timeperiod} periods)',
            'chart_type': 'oscillator',
            'series': {
                'mfi': mfi,
                'price': close
            },
            'levels': {
                'overbought': 80,
                'oversold': 20
            }
        }
    
    def _ppo(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Percentage Price Oscillator"""
        fastperiod = params.get('fastperiod', 12)
        slowperiod = params.get('slowperiod', 26)
        high, low, close, volume = self._prepare_data(data)
        
        ppo = talib.PPO(close, fastperiod=fastperiod, slowperiod=slowperiod)
        
        return {
            'indicator': 'PPO',
            'values': ppo,
            'price': close,
            'parameters': {'fastperiod': fastperiod, 'slowperiod': slowperiod},
            'description': f'Percentage Price Oscillator ({fastperiod}, {slowperiod})',
            'chart_type': 'oscillator',
            'series': {
                'ppo': ppo,
                'price': close
            }
        }
    
    def _macdext(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """MACD with controllable MA type"""
        fastperiod = params.get('fastperiod', 12)
        slowperiod = params.get('slowperiod', 26)
        signalperiod = params.get('signalperiod', 9)
        fastmatype = params.get('fastmatype', 0)
        slowmatype = params.get('slowmatype', 0)
        signalmatype = params.get('signalmatype', 0)
        high, low, close, volume = self._prepare_data(data)
        
        macd, macdsignal, macdhist = talib.MACDEXT(close, 
                                                 fastperiod=fastperiod,
                                                 slowperiod=slowperiod,
                                                 signalperiod=signalperiod,
                                                 fastmatype=fastmatype,
                                                 slowmatype=slowmatype,
                                                 signalmatype=signalmatype)
        
        return {
            'indicator': 'MACD Extended',
            'values': macd,
            'price': close,
            'parameters': {'fastperiod': fastperiod, 'slowperiod': slowperiod, 'signalperiod': signalperiod},
            'description': f'MACD Extended ({fastperiod}, {slowperiod}, {signalperiod})',
            'chart_type': 'macd',
            'series': {
                'macd': macd,
                'signal': macdsignal,
                'histogram': macdhist,
                'price': close
            }
        }
    
    def _macdfix(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """MACD Fix"""
        signalperiod = params.get('signalperiod', 9)
        high, low, close, volume = self._prepare_data(data)
        
        macd, macdsignal, macdhist = talib.MACDFIX(close, signalperiod=signalperiod)
        
        return {
            'indicator': 'MACD Fix',
            'values': macd,
            'price': close,
            'parameters': {'signalperiod': signalperiod},
            'description': f'MACD Fix ({signalperiod})',
            'chart_type': 'macd',
            'series': {
                'macd': macd,
                'signal': macdsignal,
                'histogram': macdhist,
                'price': close
            }
        }
    
    def _minus_di(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Minus Directional Indicator"""
        timeperiod = params.get('timeperiod', 14)
        high, low, close, volume = self._prepare_data(data)
        
        minus_di = talib.MINUS_DI(high, low, close, timeperiod=timeperiod)
        
        return {
            'indicator': 'Minus DI',
            'values': minus_di,
            'price': close,
            'parameters': {'timeperiod': timeperiod},
            'description': f'Minus Directional Indicator ({timeperiod} periods)',
            'chart_type': 'oscillator',
            'series': {
                'minus_di': minus_di,
                'price': close
            }
        }
    
    def _minus_dm(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Minus Directional Movement"""
        timeperiod = params.get('timeperiod', 14)
        high, low, close, volume = self._prepare_data(data)
        
        minus_dm = talib.MINUS_DM(high, low, timeperiod=timeperiod)
        
        return {
            'indicator': 'Minus DM',
            'values': minus_dm,
            'price': close,
            'parameters': {'timeperiod': timeperiod},
            'description': f'Minus Directional Movement ({timeperiod} periods)',
            'chart_type': 'oscillator',
            'series': {
                'minus_dm': minus_dm,
                'price': close
            }
        }
    
    def _plus_di(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Plus Directional Indicator"""
        timeperiod = params.get('timeperiod', 14)
        high, low, close, volume = self._prepare_data(data)
        
        plus_di = talib.PLUS_DI(high, low, close, timeperiod=timeperiod)
        
        return {
            'indicator': 'Plus DI',
            'values': plus_di,
            'price': close,
            'parameters': {'timeperiod': timeperiod},
            'description': f'Plus Directional Indicator ({timeperiod} periods)',
            'chart_type': 'oscillator',
            'series': {
                'plus_di': plus_di,
                'price': close
            }
        }
    
    def _plus_dm(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Plus Directional Movement"""
        timeperiod = params.get('timeperiod', 14)
        high, low, close, volume = self._prepare_data(data)
        
        plus_dm = talib.PLUS_DM(high, low, timeperiod=timeperiod)
        
        return {
            'indicator': 'Plus DM',
            'values': plus_dm,
            'price': close,
            'parameters': {'timeperiod': timeperiod},
            'description': f'Plus Directional Movement ({timeperiod} periods)',
            'chart_type': 'oscillator',
            'series': {
                'plus_dm': plus_dm,
                'price': close
            }
        }
    
    def _rocp(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Rate of Change Percentage"""
        timeperiod = params.get('timeperiod', 10)
        high, low, close, volume = self._prepare_data(data)
        
        rocp = talib.ROCP(close, timeperiod=timeperiod)
        
        return {
            'indicator': 'ROCP',
            'values': rocp,
            'price': close,
            'parameters': {'timeperiod': timeperiod},
            'description': f'Rate of Change Percentage ({timeperiod} periods)',
            'chart_type': 'oscillator',
            'series': {
                'rocp': rocp,
                'price': close
            }
        }
    
    def _rocr(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Rate of Change Ratio"""
        timeperiod = params.get('timeperiod', 10)
        high, low, close, volume = self._prepare_data(data)
        
        rocr = talib.ROCR(close, timeperiod=timeperiod)
        
        return {
            'indicator': 'ROCR',
            'values': rocr,
            'price': close,
            'parameters': {'timeperiod': timeperiod},
            'description': f'Rate of Change Ratio ({timeperiod} periods)',
            'chart_type': 'oscillator',
            'series': {
                'rocr': rocr,
                'price': close
            }
        }
    
    def _rocr100(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Rate of Change Ratio 100 Scale"""
        timeperiod = params.get('timeperiod', 10)
        high, low, close, volume = self._prepare_data(data)
        
        rocr100 = talib.ROCR100(close, timeperiod=timeperiod)
        
        return {
            'indicator': 'ROCR100',
            'values': rocr100,
            'price': close,
            'parameters': {'timeperiod': timeperiod},
            'description': f'Rate of Change Ratio 100 Scale ({timeperiod} periods)',
            'chart_type': 'oscillator',
            'series': {
                'rocr100': rocr100,
                'price': close
            }
        }
    
    def _natr(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Normalized Average True Range"""
        timeperiod = params.get('timeperiod', 14)
        high, low, close, volume = self._prepare_data(data)
        
        natr = talib.NATR(high, low, close, timeperiod=timeperiod)
        
        return {
            'indicator': 'NATR',
            'values': natr,
            'price': close,
            'parameters': {'timeperiod': timeperiod},
            'description': f'Normalized Average True Range ({timeperiod} periods)',
            'chart_type': 'line',
            'series': {
                'natr': natr,
                'price': close
            }
        }
    
    def _trange(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """True Range"""
        high, low, close, volume = self._prepare_data(data)
        
        trange = talib.TRANGE(high, low, close)
        
        return {
            'indicator': 'True Range',
            'values': trange,
            'price': close,
            'parameters': {},
            'description': 'True Range',
            'chart_type': 'line',
            'series': {
                'trange': trange,
                'price': close
            }
        }
    
    def _ad(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Accumulation/Distribution Line"""
        high, low, close, volume = self._prepare_data(data)
        
        ad = talib.AD(high, low, close, volume)
        
        return {
            'indicator': 'A/D Line',
            'values': ad,
            'price': close,
            'parameters': {},
            'description': 'Accumulation/Distribution Line',
            'chart_type': 'line',
            'series': {
                'ad': ad,
                'price': close
            }
        }
    
    def _adosc(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Accumulation/Distribution Oscillator"""
        fastperiod = params.get('fastperiod', 3)
        slowperiod = params.get('slowperiod', 10)
        high, low, close, volume = self._prepare_data(data)
        
        adosc = talib.ADOSC(high, low, close, volume, 
                           fastperiod=fastperiod, slowperiod=slowperiod)
        
        return {
            'indicator': 'A/D Oscillator',
            'values': adosc,
            'price': close,
            'parameters': {'fastperiod': fastperiod, 'slowperiod': slowperiod},
            'description': f'Accumulation/Distribution Oscillator ({fastperiod}, {slowperiod})',
            'chart_type': 'oscillator',
            'series': {
                'adosc': adosc,
                'price': close
            }
        }
    
    def _ht_dcperiod(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Hilbert Transform - Dominant Cycle Period"""
        high, low, close, volume = self._prepare_data(data)
        
        ht_dcperiod = talib.HT_DCPERIOD(close)
        
        return {
            'indicator': 'HT DC Period',
            'values': ht_dcperiod,
            'price': close,
            'parameters': {},
            'description': 'Hilbert Transform - Dominant Cycle Period',
            'chart_type': 'line',
            'series': {
                'ht_dcperiod': ht_dcperiod,
                'price': close
            }
        }
    
    def _ht_dcphase(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Hilbert Transform - Dominant Cycle Phase"""
        high, low, close, volume = self._prepare_data(data)
        
        ht_dcphase = talib.HT_DCPHASE(close)
        
        return {
            'indicator': 'HT DC Phase',
            'values': ht_dcphase,
            'price': close,
            'parameters': {},
            'description': 'Hilbert Transform - Dominant Cycle Phase',
            'chart_type': 'line',
            'series': {
                'ht_dcphase': ht_dcphase,
                'price': close
            }
        }
    
    def _ht_phasor(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Hilbert Transform - Phasor Components"""
        high, low, close, volume = self._prepare_data(data)
        
        inphase, quadrature = talib.HT_PHASOR(close)
        
        return {
            'indicator': 'HT Phasor',
            'values': inphase,
            'price': close,
            'parameters': {},
            'description': 'Hilbert Transform - Phasor Components',
            'chart_type': 'oscillator',
            'series': {
                'inphase': inphase,
                'quadrature': quadrature,
                'price': close
            }
        }
    
    def _ht_sine(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Hilbert Transform - SineWave"""
        high, low, close, volume = self._prepare_data(data)
        
        sine, leadsine = talib.HT_SINE(close)
        
        return {
            'indicator': 'HT Sine',
            'values': sine,
            'price': close,
            'parameters': {},
            'description': 'Hilbert Transform - SineWave',
            'chart_type': 'oscillator',
            'series': {
                'sine': sine,
                'leadsine': leadsine,
                'price': close
            }
        }
    
    def _ht_trendmode(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Hilbert Transform - Trend vs Cycle Mode"""
        high, low, close, volume = self._prepare_data(data)
        
        ht_trendmode = talib.HT_TRENDMODE(close)
        
        return {
            'indicator': 'HT Trend Mode',
            'values': ht_trendmode,
            'price': close,
            'parameters': {},
            'description': 'Hilbert Transform - Trend vs Cycle Mode',
            'chart_type': 'oscillator',
            'series': {
                'ht_trendmode': ht_trendmode,
                'price': close
            }
        }
    
    def _apo(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Absolute Price Oscillator"""
        fastperiod = params.get('fastperiod', 12)
        slowperiod = params.get('slowperiod', 26)
        high, low, close, volume = self._prepare_data(data)
        
        apo = talib.APO(close, fastperiod=fastperiod, slowperiod=slowperiod)
        
        return {
            'indicator': 'APO',
            'values': apo,
            'price': close,
            'parameters': {'fastperiod': fastperiod, 'slowperiod': slowperiod},
            'description': f'Absolute Price Oscillator ({fastperiod}, {slowperiod})',
            'chart_type': 'oscillator',
            'series': {
                'apo': apo,
                'price': close
            }
        }
    
    def get_available_indicators(self) -> List[str]:
        """Get list of available indicators"""
        return list(self.indicators.keys())
    
    def get_indicator_info(self, indicator_name: str) -> Dict[str, Any]:
        """Get information about a specific indicator"""
        if indicator_name not in self.indicators:
            return {"error": f"Unknown indicator: {indicator_name}"}
        
        # This would contain metadata about each indicator
        # For now, return basic info
        return {
            "name": indicator_name,
            "available": True,
            "description": f"TA-Lib {indicator_name} indicator"
        }

"""
TA-Lib Tool Executor
Handles execution of TA-Lib technical indicators with comprehensive charting support
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging
from .talib_indicators import TALibIndicators
from ..adapters.prices_polygon import get_prices_agg

logger = logging.getLogger("requiem.talib_executor")

class TALibToolExecutor:
    """Executes TA-Lib technical indicators with charting support"""
    
    def __init__(self):
        self.talib = TALibIndicators()
        self.available_indicators = self.talib.get_available_indicators()
        
        # Default parameters for common indicators
        self.default_params = {
            'sma': {'timeperiod': 20},
            'ema': {'timeperiod': 20},
            'rsi': {'timeperiod': 14},
            'bbands': {'timeperiod': 20, 'nbdevup': 2, 'nbdevdn': 2},
            'macd': {'fastperiod': 12, 'slowperiod': 26, 'signalperiod': 9},
            'stoch': {'fastk_period': 5, 'slowk_period': 3, 'slowd_period': 3},
            'willr': {'timeperiod': 14},
            'adx': {'timeperiod': 14},
            'cci': {'timeperiod': 14},
            'obv': {},
            'atr': {'timeperiod': 14},
            'wma': {'timeperiod': 20},
            'dema': {'timeperiod': 20},
            'tema': {'timeperiod': 20},
            'trima': {'timeperiod': 20},
            'kama': {'timeperiod': 20},
            'mama': {'fastlimit': 0.5, 'slowlimit': 0.05},
            't3': {'timeperiod': 20, 'vfactor': 0.7},
            'stochf': {'fastk_period': 5, 'fastd_period': 3},
            'stochrsi': {'timeperiod': 14, 'fastk_period': 3, 'fastd_period': 3},
            'cmo': {'timeperiod': 14},
            'roc': {'timeperiod': 10},
            'mom': {'timeperiod': 10},
            'dx': {'timeperiod': 14},
            'adxr': {'timeperiod': 14},
            'aroon': {'timeperiod': 14},
            'aroonosc': {'timeperiod': 14},
            'bop': {},
            'trix': {'timeperiod': 30},
            'ultosc': {'timeperiod1': 7, 'timeperiod2': 14, 'timeperiod3': 28},
            'mfi': {'timeperiod': 14},
            'ppo': {'fastperiod': 12, 'slowperiod': 26},
            'macdext': {'fastperiod': 12, 'slowperiod': 26, 'signalperiod': 9},
            'macdfix': {'signalperiod': 9},
            'minus_di': {'timeperiod': 14},
            'minus_dm': {'timeperiod': 14},
            'plus_di': {'timeperiod': 14},
            'plus_dm': {'timeperiod': 14},
            'rocp': {'timeperiod': 10},
            'rocr': {'timeperiod': 10},
            'rocr100': {'timeperiod': 10},
            'natr': {'timeperiod': 14},
            'trange': {},
            'ad': {},
            'adosc': {'fastperiod': 3, 'slowperiod': 10},
            'ht_dcperiod': {},
            'ht_dcphase': {},
            'ht_phasor': {},
            'ht_sine': {},
            'ht_trendmode': {}
        }
    
    def execute_indicator(self, indicator_name: str, ticker: str, 
                         start_date: str, end_date: str, 
                         params: Dict[str, Any] = None,
                         original_start_date: str = None, 
                         original_end_date: str = None) -> Dict[str, Any]:
        """Execute a TA-Lib indicator with data fetching and charting"""
        try:
            # Get default parameters if none provided
            if params is None:
                params = self.default_params.get(indicator_name, {})
            
            # Fetch price data
            df = get_prices_agg(ticker, start_date, end_date)
            if df.empty:
                return {"error": f"No data available for {ticker}"}
            
            # Set date column as index for proper datetime handling
            df = df.set_index('date')
            
            # Ensure we have OHLCV data
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_columns):
                return {"error": "Missing required OHLCV data"}
            
            # Calculate the indicator
            result = self.talib.calculate_indicator(indicator_name, df, params)
            
            if "error" in result:
                return result
            
            # Filter results to original requested period if specified
            if original_start_date and original_end_date:
                logger.info(f"Filtering to original period: {original_start_date} to {original_end_date}")
                result, df = self._filter_to_original_period(result, df, original_start_date, original_end_date)
                display_period = f"{original_start_date} to {original_end_date}"
            else:
                display_period = f"{start_date} to {end_date}"
            
            # Prepare chart data using filtered dataframe and filtered result
            chart_data = self._prepare_chart_data(result, df, indicator_name)
            
            # Generate insights using OpenAI (will be done in the API layer)
            insights = ""
            
            # Format response
            return {
                "tool_name": indicator_name,
                "ticker": ticker,
                "period": display_period,
                "latest_value": self._get_latest_value(result),
                "mean_value": self._get_mean_value(result),
                "min_value": self._get_min_value(result),
                "max_value": self._get_max_value(result),
                "parameters_used": result.get('parameters', {}),
                "description": result.get('description', ''),
                "data_points": len(df),
                "chart_data": chart_data,
                "insights": insights,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error executing {indicator_name} for {ticker}: {e}")
            return {"error": str(e), "success": False}
    
    def _filter_to_original_period(self, result: Dict[str, Any], df: pd.DataFrame, 
                                 original_start_date: str, original_end_date: str) -> Dict[str, Any]:
        """Filter indicator results to only include the original requested period with valid values"""
        try:
            from datetime import datetime, date
            
            # Convert dates to datetime objects
            start_dt = datetime.strptime(original_start_date, "%Y-%m-%d").date()
            end_dt = datetime.strptime(original_end_date, "%Y-%m-%d").date()
            
            # Find the first valid data point in the main indicator
            series = result.get('series', {})
            first_valid_idx = 0
            
            # Find first valid data point in the main series
            main_series_name = None
            if 'macd' in series:
                main_series_name = 'macd'
            elif 'rsi' in series:
                main_series_name = 'rsi'
            elif 'middle_band' in series:
                main_series_name = 'middle_band'
            elif 'values' in result:
                main_series_name = 'values'
            
            if main_series_name:
                main_data = series.get(main_series_name, result.get('values', []))
                if hasattr(main_data, 'tolist'):
                    main_data = main_data.tolist()
                
                for i, val in enumerate(main_data):
                    if val is not None and not pd.isna(val):
                        first_valid_idx = i
                        break
                
                logger.info(f"First valid data point found at index {first_valid_idx}")
            
            # Create mask for the original period, but only from the first valid data point onwards
            mask = (df.index.date >= start_dt) & (df.index.date <= end_dt)
            
            # If the first valid data point is after the original start date, 
            # adjust the start date to the first valid data point
            first_valid_date = df.index.date[first_valid_idx] if first_valid_idx < len(df) else start_dt
            if first_valid_date > start_dt:
                logger.info(f"First valid data point is on {first_valid_date}, adjusting start date from {start_dt}")
                # Update the mask to start from the first valid data point
                mask = (df.index.date >= first_valid_date) & (df.index.date <= end_dt)
            else:
                logger.info(f"First valid data point is on {first_valid_date}, which is before or equal to start date {start_dt}")
            
            logger.info(f"Created mask: {mask.sum()} out of {len(mask)} dates match the filter")
            logger.info(f"Date range: {df.index.date[0]} to {df.index.date[-1]}")
            logger.info(f"Filter range: {first_valid_date} to {end_dt}")
            
            # Filter the dataframe
            filtered_df = df[mask]
            logger.info(f"Filtered dataframe has {len(filtered_df)} rows")
            
            # Filter the result values to match the filtered dataframe
            filtered_result = result.copy()
            filtered_series = {}
            
            for series_name, series_data in series.items():
                if isinstance(series_data, pd.Series):
                    # Filter the series to match the filtered dataframe dates
                    filtered_series[series_name] = series_data[mask]
                elif isinstance(series_data, list):
                    # For list data, filter based on the mask
                    if len(series_data) == len(df):
                        filtered_series[series_name] = [series_data[i] for i in range(len(series_data)) if mask[i]]
                    else:
                        # If the series length doesn't match, we need to find the correct alignment
                        # This happens when we have more data than the filtered period
                        # Find the start index that corresponds to the first valid data point
                        first_valid_idx = 0
                        for i, val in enumerate(series_data):
                            if val is not None and not pd.isna(val):
                                first_valid_idx = i
                                break
                        
                        # Calculate how many values we need to skip to align with the filtered period
                        # The series should start from the first valid data point
                        start_idx = first_valid_idx
                        end_idx = start_idx + len(filtered_df)
                        
                        if end_idx <= len(series_data):
                            filtered_series[series_name] = series_data[start_idx:end_idx]
                        else:
                            # Fallback: take the last N values
                            filtered_series[series_name] = series_data[-len(filtered_df):] if len(series_data) >= len(filtered_df) else series_data
                else:
                    filtered_series[series_name] = series_data
            
            filtered_result['series'] = filtered_series
            
            # Update values list if it exists
            if 'values' in result:
                values = result['values']
                if isinstance(values, list):
                    if len(values) == len(df):
                        filtered_result['values'] = [values[i] for i in range(len(values)) if mask[i]]
                    else:
                        filtered_result['values'] = values[-len(filtered_df):] if len(values) >= len(filtered_df) else values
                elif isinstance(values, pd.Series):
                    filtered_result['values'] = values[mask]
            
            # Also filter the chart data if it exists
            if 'chart_data' in result:
                chart_data = result['chart_data'].copy()
                if 'dates' in chart_data:
                    # Filter dates to match the filtered period
                    chart_dates = chart_data['dates']
                    
                    # Find indices that fall within the original period
                    filtered_indices = []
                    for i, date_str in enumerate(chart_dates):
                        try:
                            chart_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                            if first_valid_date <= chart_date <= end_dt:
                                filtered_indices.append(i)
                        except:
                            continue
                    
                    # Filter all chart data arrays
                    chart_data['dates'] = [chart_dates[i] for i in filtered_indices]
                    
                    # Filter other arrays in chart_data
                    for key, values in chart_data.items():
                        if key != 'dates' and isinstance(values, list) and len(values) == len(chart_dates):
                            chart_data[key] = [values[i] for i in filtered_indices]
                    
                    filtered_result['chart_data'] = chart_data
            
            return filtered_result, filtered_df
            
        except Exception as e:
            logger.error(f"Error filtering to original period: {e}")
            return result, df  # Return original result if filtering fails
    
    def _prepare_chart_data(self, result: Dict[str, Any], df: pd.DataFrame, 
                           indicator_name: str) -> Dict[str, Any]:
        """Prepare chart data for frontend rendering"""
        try:
            # Use the filtered dataframe for dates
            dates = [d.strftime("%Y-%m-%d") for d in df.index]
            series = result.get('series', {})
            
            logger.info(f"Chart data preparation: {len(dates)} dates, series keys: {list(series.keys())}")
            for key, values in series.items():
                if hasattr(values, '__len__'):
                    logger.info(f"Series {key}: {len(values)} values")
            
            chart_data = {
                "dates": dates,
                "indicator": result.get('indicator', indicator_name),
                "chart_type": result.get('chart_type', 'line'),
                "levels": result.get('levels', {}),
                "parameters": result.get('parameters', {})
            }
            
            # Log the filtering status
            logger.info(f"Preparing chart data with {len(dates)} dates from {dates[0] if dates else 'N/A'} to {dates[-1] if dates else 'N/A'}")
            logger.info(f"Series data keys: {list(series.keys()) if series else 'None'}")
            if 'rsi' in series:
                logger.info(f"RSI series length: {len(series['rsi'])}")
            if 'values' in series:
                logger.info(f"Values series length: {len(series['values'])}")
            
            # Add series data based on chart type
            if result.get('chart_type') == 'bands':
                # Bollinger Bands - apply null value filtering
                upper_band = series.get('upper_band', []).tolist() if hasattr(series.get('upper_band', []), 'tolist') else series.get('upper_band', [])
                middle_band = series.get('middle_band', []).tolist() if hasattr(series.get('middle_band', []), 'tolist') else series.get('middle_band', [])
                lower_band = series.get('lower_band', []).tolist() if hasattr(series.get('lower_band', []), 'tolist') else series.get('lower_band', [])
                price_values = series.get('price', []).tolist() if hasattr(series.get('price', []), 'tolist') else series.get('price', [])
                
                # Find the first non-null value index for middle band (SMA)
                first_valid_idx = 0
                for i, val in enumerate(middle_band):
                    if val is not None and not pd.isna(val):
                        first_valid_idx = i
                        break
                
                # Filter all data to start from the first valid data point
                filtered_dates = dates[first_valid_idx:]
                filtered_upper = upper_band[first_valid_idx:]
                filtered_middle = middle_band[first_valid_idx:]
                filtered_lower = lower_band[first_valid_idx:]
                filtered_price = price_values[first_valid_idx:] if len(price_values) > first_valid_idx else price_values
                
                chart_data.update({
                    "dates": filtered_dates,
                    "upper_band": filtered_upper,
                    "middle_band": filtered_middle,
                    "lower_band": filtered_lower,
                    "price": filtered_price
                })
            elif result.get('chart_type') == 'macd':
                # MACD - don't filter null values here since we want to show the full requested period
                chart_data.update({
                    "macd": series.get('macd', []).tolist() if hasattr(series.get('macd', []), 'tolist') else series.get('macd', []),
                    "signal": series.get('signal', []).tolist() if hasattr(series.get('signal', []), 'tolist') else series.get('signal', []),
                    "histogram": series.get('histogram', []).tolist() if hasattr(series.get('histogram', []), 'tolist') else series.get('histogram', []),
                    "price": series.get('price', []).tolist() if hasattr(series.get('price', []), 'tolist') else series.get('price', [])
                })
            elif result.get('chart_type') == 'oscillator':
                # Oscillators (RSI, Stochastic, etc.)
                main_series = list(series.keys())[0] if series else 'values'
                main_values = series.get(main_series, [])
                price_values = series.get('price', [])
                
                # Convert to lists if they're pandas Series
                main_list = main_values.tolist() if hasattr(main_values, 'tolist') else main_values
                price_list = price_values.tolist() if hasattr(price_values, 'tolist') else price_values
                
                # Find the first non-null value index for the main series
                first_valid_idx = 0
                for i, val in enumerate(main_list):
                    if val is not None and not pd.isna(val):
                        first_valid_idx = i
                        break
                
                # Filter all data to start from the first valid data point
                filtered_dates = dates[first_valid_idx:]
                filtered_main = main_list[first_valid_idx:]
                filtered_price = price_list[first_valid_idx:] if len(price_list) > first_valid_idx else price_list
                
                chart_data.update({
                    "dates": filtered_dates,
                    "values": filtered_main,
                    "price": filtered_price
                })
                
                # Add additional series if available
                for key, values in series.items():
                    if key != 'price':
                        values_list = values.tolist() if hasattr(values, 'tolist') else values
                        chart_data[key] = values_list[first_valid_idx:] if len(values_list) > first_valid_idx else values_list
            else:
                # Line charts (moving averages, etc.)
                main_series = list(series.keys())[0] if series else 'values'
                main_values = series.get(main_series, [])
                price_values = series.get('price', [])
                
                # Convert to lists if they're pandas Series
                main_list = main_values.tolist() if hasattr(main_values, 'tolist') else main_values
                price_list = price_values.tolist() if hasattr(price_values, 'tolist') else price_values
                
                # Find the first non-null value index for the main series
                first_valid_idx = 0
                for i, val in enumerate(main_list):
                    if val is not None and not pd.isna(val):
                        first_valid_idx = i
                        break
                
                # Filter all data to start from the first valid data point
                filtered_dates = dates[first_valid_idx:]
                filtered_main = main_list[first_valid_idx:]
                filtered_price = price_list[first_valid_idx:] if len(price_list) > first_valid_idx else price_list
                
                chart_data.update({
                    "dates": filtered_dates,
                    "values": filtered_main,
                    "price": filtered_price
                })
                
                # Add additional series if available
                for key, values in series.items():
                    if key != 'price':
                        values_list = values.tolist() if hasattr(values, 'tolist') else values
                        chart_data[key] = values_list[first_valid_idx:] if len(values_list) > first_valid_idx else values_list
            
            return chart_data
            
        except Exception as e:
            logger.error(f"Error preparing chart data: {e}")
            return {"error": str(e)}
    
    def _generate_insights(self, result: Dict[str, Any], indicator_name: str, ticker: str) -> str:
        """Generate AI-powered insights for the indicator"""
        try:
            values = result.get('values', [])
            if len(values) == 0:
                return "No data available for analysis."
            
            # Remove NaN values
            clean_values = [v for v in values if not np.isnan(v)]
            if len(clean_values) == 0:
                return "No valid data points for analysis."
            
            latest_value = clean_values[-1]
            mean_value = np.mean(clean_values)
            std_value = np.std(clean_values)
            
            # Generate insights based on indicator type
            if indicator_name in ['rsi', 'stoch', 'stochf', 'stochrsi', 'willr', 'cci', 'mfi', 'ultosc']:
                return self._generate_oscillator_insights(indicator_name, latest_value, mean_value, result)
            elif indicator_name in ['sma', 'ema', 'wma', 'dema', 'tema', 'trima', 'kama', 'mama', 't3']:
                return self._generate_moving_average_insights(indicator_name, latest_value, mean_value, result)
            elif indicator_name == 'bbands':
                return self._generate_bollinger_insights(latest_value, result)
            elif indicator_name in ['macd', 'macdext', 'macdfix']:
                return self._generate_macd_insights(latest_value, result)
            elif indicator_name in ['adx', 'adxr', 'dx']:
                return self._generate_trend_insights(indicator_name, latest_value, result)
            elif indicator_name in ['atr', 'natr', 'trange']:
                return self._generate_volatility_insights(indicator_name, latest_value, mean_value, result)
            elif indicator_name in ['obv', 'ad', 'adosc']:
                return self._generate_volume_insights(indicator_name, latest_value, result)
            else:
                return self._generate_general_insights(indicator_name, latest_value, mean_value, std_value)
                
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return "Unable to generate insights at this time."
    
    def _generate_oscillator_insights(self, indicator_name: str, latest_value: float, 
                                    mean_value: float, result: Dict[str, Any]) -> str:
        """Generate insights for oscillator indicators"""
        levels = result.get('levels', {})
        overbought = levels.get('overbought', 70)
        oversold = levels.get('oversold', 30)
        
        if latest_value > overbought:
            signal = "overbought"
            recommendation = "Consider selling or taking profits"
        elif latest_value < oversold:
            signal = "oversold"
            recommendation = "Consider buying or entering long positions"
        else:
            signal = "neutral"
            recommendation = "Wait for clearer signals"
        
        return f"**{result.get('indicator', indicator_name)} Analysis:**\n\n" \
               f"Current Value: {latest_value:.2f}\n" \
               f"Average: {mean_value:.2f}\n" \
               f"Signal: {signal.upper()}\n" \
               f"Recommendation: {recommendation}\n\n" \
               f"Overbought Level: {overbought}\n" \
               f"Oversold Level: {oversold}"
    
    def _generate_moving_average_insights(self, indicator_name: str, latest_value: float, 
                                        mean_value: float, result: Dict[str, Any]) -> str:
        """Generate insights for moving average indicators"""
        series = result.get('series', {})
        price = series.get('price', [])
        
        if len(price) > 0:
            current_price = price[-1]
            if latest_value > current_price:
                signal = "bullish"
                recommendation = "Price above moving average - uptrend"
            else:
                signal = "bearish"
                recommendation = "Price below moving average - downtrend"
        else:
            signal = "neutral"
            recommendation = "Insufficient data for trend analysis"
        
        return f"**{result.get('indicator', indicator_name)} Analysis:**\n\n" \
               f"Current Value: {latest_value:.2f}\n" \
               f"Average: {mean_value:.2f}\n" \
               f"Signal: {signal.upper()}\n" \
               f"Recommendation: {recommendation}"
    
    def _generate_bollinger_insights(self, latest_value: float, result: Dict[str, Any]) -> str:
        """Generate insights for Bollinger Bands"""
        series = result.get('series', {})
        upper_band = series.get('upper_band', [])
        lower_band = series.get('lower_band', [])
        price = series.get('price', [])
        
        if len(upper_band) > 0 and len(lower_band) > 0 and len(price) > 0:
            current_price = price[-1]
            current_upper = upper_band[-1]
            current_lower = lower_band[-1]
            
            if current_price > current_upper:
                signal = "overbought"
                recommendation = "Price above upper band - potential reversal"
            elif current_price < current_lower:
                signal = "oversold"
                recommendation = "Price below lower band - potential bounce"
            else:
                signal = "neutral"
                recommendation = "Price within bands - normal volatility"
        else:
            signal = "neutral"
            recommendation = "Insufficient data for analysis"
        
        return f"**Bollinger Bands Analysis:**\n\n" \
               f"Current Price: {current_price:.2f}\n" \
               f"Upper Band: {current_upper:.2f}\n" \
               f"Lower Band: {current_lower:.2f}\n" \
               f"Signal: {signal.upper()}\n" \
               f"Recommendation: {recommendation}"
    
    def _generate_macd_insights(self, latest_value: float, result: Dict[str, Any]) -> str:
        """Generate insights for MACD indicators"""
        series = result.get('series', {})
        macd = series.get('macd', [])
        signal = series.get('signal', [])
        histogram = series.get('histogram', [])
        
        if len(macd) > 1 and len(signal) > 1 and len(histogram) > 1:
            current_macd = macd[-1]
            current_signal = signal[-1]
            current_histogram = histogram[-1]
            prev_histogram = histogram[-2]
            
            if current_macd > current_signal and current_histogram > prev_histogram:
                signal_type = "bullish"
                recommendation = "MACD above signal line and rising - buy signal"
            elif current_macd < current_signal and current_histogram < prev_histogram:
                signal_type = "bearish"
                recommendation = "MACD below signal line and falling - sell signal"
            else:
                signal_type = "neutral"
                recommendation = "MACD signals mixed - wait for confirmation"
        else:
            signal_type = "neutral"
            recommendation = "Insufficient data for MACD analysis"
        
        return f"**MACD Analysis:**\n\n" \
               f"MACD: {current_macd:.4f}\n" \
               f"Signal: {current_signal:.4f}\n" \
               f"Histogram: {current_histogram:.4f}\n" \
               f"Signal: {signal_type.upper()}\n" \
               f"Recommendation: {recommendation}"
    
    def _generate_trend_insights(self, indicator_name: str, latest_value: float, result: Dict[str, Any]) -> str:
        """Generate insights for trend indicators"""
        levels = result.get('levels', {})
        strong_trend = levels.get('strong_trend', 25)
        weak_trend = levels.get('weak_trend', 20)
        
        if latest_value > strong_trend:
            trend_strength = "strong"
            recommendation = "Strong trend detected - consider trend-following strategies"
        elif latest_value > weak_trend:
            trend_strength = "moderate"
            recommendation = "Moderate trend - use caution with position sizing"
        else:
            trend_strength = "weak"
            recommendation = "Weak trend - avoid trend-following strategies"
        
        return f"**{result.get('indicator', indicator_name)} Analysis:**\n\n" \
               f"Current Value: {latest_value:.2f}\n" \
               f"Trend Strength: {trend_strength.upper()}\n" \
               f"Recommendation: {recommendation}\n\n" \
               f"Strong Trend Level: {strong_trend}\n" \
               f"Weak Trend Level: {weak_trend}"
    
    def _generate_volatility_insights(self, indicator_name: str, latest_value: float, 
                                    mean_value: float, result: Dict[str, Any]) -> str:
        """Generate insights for volatility indicators"""
        if latest_value > mean_value * 1.5:
            volatility = "high"
            recommendation = "High volatility - consider reducing position size"
        elif latest_value < mean_value * 0.5:
            volatility = "low"
            recommendation = "Low volatility - potential for breakout"
        else:
            volatility = "normal"
            recommendation = "Normal volatility - standard position sizing"
        
        return f"**{result.get('indicator', indicator_name)} Analysis:**\n\n" \
               f"Current Value: {latest_value:.4f}\n" \
               f"Average: {mean_value:.4f}\n" \
               f"Volatility: {volatility.upper()}\n" \
               f"Recommendation: {recommendation}"
    
    def _generate_volume_insights(self, indicator_name: str, latest_value: float, result: Dict[str, Any]) -> str:
        """Generate insights for volume indicators"""
        if latest_value > 0:
            signal = "positive"
            recommendation = "Positive volume flow - bullish signal"
        elif latest_value < 0:
            signal = "negative"
            recommendation = "Negative volume flow - bearish signal"
        else:
            signal = "neutral"
            recommendation = "Neutral volume flow - no clear signal"
        
        return f"**{result.get('indicator', indicator_name)} Analysis:**\n\n" \
               f"Current Value: {latest_value:.2f}\n" \
               f"Signal: {signal.upper()}\n" \
               f"Recommendation: {recommendation}"
    
    def _generate_general_insights(self, indicator_name: str, latest_value: float, 
                                 mean_value: float, std_value: float) -> str:
        """Generate general insights for other indicators"""
        if latest_value > mean_value + std_value:
            signal = "above average"
            recommendation = "Value significantly above average"
        elif latest_value < mean_value - std_value:
            signal = "below average"
            recommendation = "Value significantly below average"
        else:
            signal = "normal"
            recommendation = "Value within normal range"
        
        return f"**{indicator_name.upper()} Analysis:**\n\n" \
               f"Current Value: {latest_value:.4f}\n" \
               f"Average: {mean_value:.4f}\n" \
               f"Standard Deviation: {std_value:.4f}\n" \
               f"Signal: {signal.upper()}\n" \
               f"Recommendation: {recommendation}"
    
    def _get_latest_value(self, result: Dict[str, Any]) -> float:
        """Get the latest non-NaN value"""
        values = result.get('values', [])
        if len(values) == 0:
            return 0.0
        
        # Find the last non-NaN value
        for value in reversed(values):
            if not np.isnan(value):
                return float(value)
        return 0.0
    
    def _get_mean_value(self, result: Dict[str, Any]) -> float:
        """Get the mean of non-NaN values"""
        values = result.get('values', [])
        clean_values = [v for v in values if not np.isnan(v)]
        return float(np.mean(clean_values)) if len(clean_values) > 0 else 0.0
    
    def _get_min_value(self, result: Dict[str, Any]) -> float:
        """Get the minimum of non-NaN values"""
        values = result.get('values', [])
        clean_values = [v for v in values if not np.isnan(v)]
        return float(np.min(clean_values)) if len(clean_values) > 0 else 0.0
    
    def _get_max_value(self, result: Dict[str, Any]) -> float:
        """Get the maximum of non-NaN values"""
        values = result.get('values', [])
        clean_values = [v for v in values if not np.isnan(v)]
        return float(np.max(clean_values)) if len(clean_values) > 0 else 0.0
    
    def get_available_indicators(self) -> List[str]:
        """Get list of available indicators"""
        return self.available_indicators
    
    def get_indicator_parameters(self, indicator_name: str) -> Dict[str, Any]:
        """Get default parameters for an indicator"""
        return self.default_params.get(indicator_name, {})

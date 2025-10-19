# workers/engine/tool_signals.py
import pandas as pd
import re
import logging
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta
from workers.adapters.prices_polygon import get_prices_agg
from .tool_executor import get_tool_executor_for_user
from workers.utils.date_utils import get_realistic_date_range

logger = logging.getLogger("requiem.tool_signals")

class ToolSignalGenerator:
    """Generate trading signals based on tool outputs and natural language rules"""
    
    def __init__(self, selected_tools: List[str]):
        self.selected_tools = selected_tools
        self.tool_executor = get_tool_executor_for_user(selected_tools)
    
    def generate_signals_from_query(self, query: str, ticker: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Parse a natural language query and generate trading signals
        
        Example: "backtest rsi tool over last 1 year on aapl, buy when rsi < 30 and sell when rsi > 70"
        """
        try:
            # Parse the query to extract signal rules
            signal_rules = self._parse_signal_rules(query)
            
            if not signal_rules:
                raise ValueError("No valid signal rules found in query")
            
            # Get price data
            price_data = self._get_price_data(ticker, start_date, end_date)
            
            # Execute tools and generate signals
            signals = self._generate_tool_based_signals(signal_rules, ticker, price_data)
            
            return {
                "ticker": ticker,
                "signal_rules": signal_rules,
                "signals": signals,
                "price_data": price_data,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error generating tool-based signals: {str(e)}")
            return {
                "error": str(e),
                "success": False
            }
    
    def _parse_signal_rules(self, query: str) -> List[Dict[str, Any]]:
        """Parse natural language signal rules from query"""
        signal_rules = []
        query_lower = query.lower()
        
        # Pattern for tool-based signals: "buy when tool < value" or "sell when tool > value"
        patterns = [
            # Buy signals
            (r'buy\s+when\s+(\w+)\s*[<≤]\s*(\d+(?:\.\d+)?)', 'buy', 'less_than'),
            (r'buy\s+when\s+(\w+)\s*[>≥]\s*(\d+(?:\.\d+)?)', 'buy', 'greater_than'),
            (r'long\s+when\s+(\w+)\s*[<≤]\s*(\d+(?:\.\d+)?)', 'buy', 'less_than'),
            (r'long\s+when\s+(\w+)\s*[>≥]\s*(\d+(?:\.\d+)?)', 'buy', 'greater_than'),
            
            # Sell signals
            (r'sell\s+when\s+(\w+)\s*[<≤]\s*(\d+(?:\.\d+)?)', 'sell', 'less_than'),
            (r'sell\s+when\s+(\w+)\s*[>≥]\s*(\d+(?:\.\d+)?)', 'sell', 'greater_than'),
            (r'short\s+when\s+(\w+)\s*[<≤]\s*(\d+(?:\.\d+)?)', 'sell', 'less_than'),
            (r'short\s+when\s+(\w+)\s*[>≥]\s*(\d+(?:\.\d+)?)', 'sell', 'greater_than'),
        ]
        
        for pattern, action, comparison in patterns:
            matches = re.finditer(pattern, query_lower)
            for match in matches:
                tool_name = match.group(1)
                threshold = float(match.group(2))
                
                # Map common tool names
                tool_name = self._normalize_tool_name(tool_name)
                
                signal_rules.append({
                    "tool_name": tool_name,
                    "action": action,
                    "comparison": comparison,
                    "threshold": threshold,
                    "original_text": match.group(0)
                })
        
        return signal_rules
    
    def _normalize_tool_name(self, tool_name: str) -> str:
        """Normalize tool names to match available tools"""
        tool_mapping = {
            'rsi': 'rsi',
            'sma': 'sma',
            'macd': 'macd',
            'bollinger': 'bollinger',
            'williams': 'williams_r',
            'williams_r': 'williams_r',
            'stochastic': 'stochastic',
            'momentum': 'momentum',
            'volatility': 'realized_vol',
            'vol': 'realized_vol',
            'zscore': 'zscore',
            'aroon': 'aroon'
        }
        
        return tool_mapping.get(tool_name.lower(), tool_name.lower())
    
    def _get_price_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get price data for the specified period"""
        # Extend start date to ensure we have enough data for tool calculations
        extended_start = (datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=365)).strftime("%Y-%m-%d")
        
        df = get_prices_agg(ticker, extended_start, end_date)
        
        if df.empty:
            raise ValueError(f"No price data available for {ticker}")
        
        # Ensure proper DatetimeIndex
        if 'date' in df.columns:
            df = df.set_index('date')
        df.index = pd.to_datetime(df.index)
        
        return df
    
    def _generate_tool_based_signals(self, signal_rules: List[Dict[str, Any]], ticker: str, price_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate signals based on tool outputs and rules"""
        signals = {}
        
        # Calculate start and end dates for tool execution
        start_date = price_data.index[0].strftime("%Y-%m-%d")
        end_date = price_data.index[-1].strftime("%Y-%m-%d")
        
        # Execute each tool and generate signals
        for rule in signal_rules:
            tool_name = rule["tool_name"]
            
            try:
                # Execute the tool
                tool_result = self.tool_executor.execute_tool(
                    tool_name=tool_name,
                    ticker=ticker,
                    start_date=start_date,
                    end_date=end_date
                )
                
                if not tool_result.get("success", False):
                    logger.warning(f"Tool {tool_name} execution failed: {tool_result.get('error', 'Unknown error')}")
                    continue
                
                # Extract the tool values
                tool_values = self._extract_tool_values(tool_result, price_data.index)
                
                if tool_values is None or tool_values.empty:
                    logger.warning(f"No valid values from tool {tool_name}")
                    continue
                
                # Generate signal based on rule
                signal = self._apply_signal_rule(tool_values, rule)
                
                signals[tool_name] = {
                    "tool_result": tool_result,
                    "tool_values": tool_values,
                    "signal": signal,
                    "rule": rule
                }
                
            except Exception as e:
                logger.error(f"Error processing tool {tool_name}: {str(e)}")
                continue
        
        return signals
    
    def _extract_tool_values(self, tool_result: Dict[str, Any], price_index: pd.DatetimeIndex) -> Optional[pd.Series]:
        """Extract time series values from tool result"""
        try:
            # Try to get series data first
            if "series_data" in tool_result and tool_result["series_data"]:
                series_data = tool_result["series_data"]
                if "dates" in series_data and "values" in series_data:
                    dates = pd.to_datetime(series_data["dates"])
                    values = series_data["values"]
                    return pd.Series(values, index=dates)
            
            # Fallback: create a series with the latest value
            if "latest_value" in tool_result and tool_result["latest_value"] is not None:
                latest_value = tool_result["latest_value"]
                # Create a series with the latest value for all dates
                return pd.Series([latest_value] * len(price_index), index=price_index)
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting tool values: {str(e)}")
            return None
    
    def _apply_signal_rule(self, tool_values: pd.Series, rule: Dict[str, Any]) -> pd.Series:
        """Apply a signal rule to tool values"""
        action = rule["action"]
        comparison = rule["comparison"]
        threshold = rule["threshold"]
        
        # Generate signal based on comparison
        if comparison == "less_than":
            if action == "buy":
                signal = (tool_values < threshold).astype(int)
            else:  # sell
                signal = (tool_values < threshold).astype(int)
        else:  # greater_than
            if action == "buy":
                signal = (tool_values > threshold).astype(int)
            else:  # sell
                signal = (tool_values > threshold).astype(int)
        
        return signal
    
    def combine_signals(self, signals: Dict[str, Any], combination_logic: str = "and") -> pd.Series:
        """Combine multiple signals using AND/OR logic"""
        if not signals:
            return pd.Series()
        
        signal_series = []
        for tool_name, signal_data in signals.items():
            if "signal" in signal_data:
                signal_series.append(signal_data["signal"])
        
        if not signal_series:
            return pd.Series()
        
        # Align all signals to the same index
        aligned_signals = []
        base_index = signal_series[0].index
        
        for signal in signal_series:
            aligned_signal = signal.reindex(base_index, fill_value=0)
            aligned_signals.append(aligned_signal)
        
        # Combine signals
        if combination_logic.lower() == "and":
            # All signals must be 1 for final signal to be 1
            combined = pd.Series(1, index=base_index)
            for signal in aligned_signals:
                combined = combined & signal
        else:  # or
            # Any signal being 1 makes final signal 1
            combined = pd.Series(0, index=base_index)
            for signal in aligned_signals:
                combined = combined | signal
        
        return combined.astype(int)

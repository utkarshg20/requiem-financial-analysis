# workers/engine/tool_executor.py
import pandas as pd
import re
from typing import Dict, Any, List, Optional
from .features import REQUIEM_TOOLS
from ..adapters.prices_polygon import get_prices_agg

class ToolExecutor:
    """Executes selected tools for data analysis and calculations"""
    
    def __init__(self, selected_tools: List[str]):
        """Initialize with list of selected tool names"""
        self.selected_tools = selected_tools
        self.available_tools = {name: tool for name, tool in REQUIEM_TOOLS.items() if name in selected_tools}
    
    def execute_tool(self, tool_name: str, ticker: str, start_date: str, end_date: str, **params) -> Dict[str, Any]:
        """Execute a specific tool with given parameters"""
        if tool_name not in self.available_tools:
            return {"error": f"Tool '{tool_name}' is not available or not selected"}
        
        try:
            # Get tool function and parameters first to determine window size
            tool_info = self.available_tools[tool_name]
            tool_func = tool_info["function"]
            default_params = {param: info["default"] for param, info in tool_info["parameters"].items()}
            final_params = {**default_params, **params}
            
            # For indicators that need a window period, fetch extra data
            window_size = final_params.get('window', 14)  # Default window size
            if tool_name in ['zscore', 'realized_vol']:  # Only non-technical analysis tools need extra data
                # Calculate how many extra days we need
                from datetime import datetime, timedelta
                start_dt = datetime.strptime(start_date, "%Y-%m-%d")
                extra_days = window_size + 10  # Add buffer for weekends/holidays
                extended_start = (start_dt - timedelta(days=extra_days)).strftime("%Y-%m-%d")
                
                # Get extended price data
                df = get_prices_agg(ticker, extended_start, end_date)
            else:
                # Get price data for the requested period only
                df = get_prices_agg(ticker, start_date, end_date)
            
            if 'date' in df.columns:
                df = df.set_index('date')
            df.index = pd.to_datetime(df.index)
            
            # Execute the tool
            result = tool_func(df["close"], **final_params)
            
            # Handle multi-series tools (Bollinger Bands, SMA) that return dict
            if tool_name in ['bollinger', 'sma'] and isinstance(result, dict):
                from datetime import datetime
                start_dt = datetime.strptime(start_date, "%Y-%m-%d")
                end_dt = datetime.strptime(end_date, "%Y-%m-%d")
                
                # Filter each series to the requested period
                filtered_data = {}
                for key, series in result.items():
                    series_clean = series.dropna()
                    filtered_series = series_clean[
                        (series_clean.index.date >= start_dt.date()) & 
                        (series_clean.index.date <= end_dt.date())
                    ]
                    filtered_data[key] = filtered_series
                
                # Get latest values for each band
                latest_values = {key: series.iloc[-1] if len(series) > 0 else None 
                               for key, series in filtered_data.items()}
                mean_values = {key: series.mean() if len(series) > 0 else None 
                             for key, series in filtered_data.items()}
                
                # Prepare multi-series data for charting
                series_data = {
                    "dates": [d.strftime("%Y-%m-%d") for d in filtered_data['price'].index],
                    "price": [float(v) for v in filtered_data['price'].values]
                }
                
                # Add tool-specific series
                if tool_name == 'bollinger':
                    series_data.update({
                        "upper_band": [float(v) for v in filtered_data['upper_band'].values],
                        "middle_band": [float(v) for v in filtered_data['middle_band'].values],
                        "lower_band": [float(v) for v in filtered_data['lower_band'].values]
                    })
                elif tool_name == 'sma':
                    series_data.update({
                        "sma": [float(v) for v in filtered_data['sma'].values]
                    })
                
                return {
                    "tool_name": tool_name,
                    "ticker": ticker,
                    "period": f"{start_date} to {end_date}",
                    "latest_value": latest_values['price'],
                    "mean_value": mean_values['price'],
                    "min_value": filtered_data['price'].min() if len(filtered_data['price']) > 0 else None,
                    "max_value": filtered_data['price'].max() if len(filtered_data['price']) > 0 else None,
                    "parameters_used": final_params,
                    "description": self.available_tools[tool_name]["description"],
                    "data_points": len(filtered_data['price']),
                    "series_data": series_data,
                    "bollinger_data": {
                        "upper_band_latest": latest_values['upper_band'],
                        "middle_band_latest": latest_values['middle_band'],
                        "lower_band_latest": latest_values['lower_band'],
                        "price_latest": latest_values['price'],
                        "bandwidth_latest": latest_values['bandwidth']
                    } if tool_name == 'bollinger' else None,
                    "sma_data": {
                        "sma_latest": latest_values['sma'],
                        "price_latest": latest_values['price']
                    } if tool_name == 'sma' else None,
                    "success": True
                }
            
            # Convert result to a more usable format
            elif isinstance(result, pd.Series):
                result_clean = result.dropna()
                
                # Filter to only the requested period for indicators that need extra data
                if tool_name in ['rsi', 'sma', 'ema', 'macd', 'williams_r', 'stochastic']:
                    from datetime import datetime
                    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
                    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
                    
                    # Filter result to only include the requested period
                    result_clean = result_clean[
                        (result_clean.index.date >= start_dt.date()) & 
                        (result_clean.index.date <= end_dt.date())
                    ]
                
                latest_value = result_clean.iloc[-1] if len(result_clean) > 0 else None
                mean_value = result_clean.mean() if len(result_clean) > 0 else None
                
                # Include time series data for charting
                series_data = {
                    "dates": [d.strftime("%Y-%m-%d") for d in result_clean.index],
                    "values": [float(v) for v in result_clean.values]
                }
                
                return {
                    "tool_name": tool_name,
                    "ticker": ticker,
                    "period": f"{start_date} to {end_date}",
                    "latest_value": float(latest_value) if latest_value is not None else None,
                    "mean_value": float(mean_value) if mean_value is not None else None,
                    "min_value": float(result_clean.min()) if len(result_clean) > 0 else None,
                    "max_value": float(result_clean.max()) if len(result_clean) > 0 else None,
                    "parameters_used": final_params,
                    "description": tool_info["description"],
                    "data_points": len(result_clean),
                    "series_data": series_data,
                    "success": True
                }
            else:
                return {"error": f"Tool '{tool_name}' returned unexpected result type"}
                
        except Exception as e:
            return {"error": f"Error executing tool '{tool_name}': {str(e)}"}
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get list of available tools with their descriptions and parameters"""
        return [
            {
                "name": name,
                "description": tool["description"],
                "parameters": tool["parameters"]
            }
            for name, tool in self.available_tools.items()
        ]
    
    def execute_multiple_tools(self, ticker: str, start_date: str, end_date: str, 
                             tool_requests: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute multiple tools in one request"""
        results = {}
        
        for request in tool_requests:
            tool_name = request.get("tool")
            params = {k: v for k, v in request.items() if k != "tool"}
            
            result = self.execute_tool(tool_name, ticker, start_date, end_date, **params)
            results[tool_name] = result
        
        return results

def get_tool_executor_for_user(selected_tools: List[str]) -> ToolExecutor:
    """Factory function to create ToolExecutor with user's selected tools"""
    return ToolExecutor(selected_tools)

def parse_tool_request(query: str, available_tools: List[str]) -> Optional[Dict[str, Any]]:
    """Parse a natural language query to extract tool execution request"""
    query_lower = query.lower()
    
    # Check for user uploaded tools (quoted tool names)
    user_tool_pattern = r"'([^']+)'"
    user_tool_match = re.search(user_tool_pattern, query_lower)
    if user_tool_match:
        quoted_tool = user_tool_match.group(1).strip()
        # Check if this quoted tool is in available tools
        if quoted_tool in available_tools:
            return {
                "tool_name": quoted_tool,
                "ticker": None,
                "start_date": None,
                "end_date": None,
                "params": {}
            }
    
    # Enhanced tool detection with natural language descriptions
    tool_patterns = {
        'sma': [
            r'\bsma\b',
            r'\bsimple\s+moving\s+average\b',
            r'\bmoving\s+average\b(?!\s+(?:convergence|divergence))',
            r'\baverage\s+of\s+(?:the\s+)?(?:close\s+)?prices?\b',
            r'\b(?:close\s+)?price\s+average\b',
            r'\bsma\s+\d+\b'
        ],
        'macd': [
            r'\bmacd\b',
            r'\bmoving\s+average\s+convergence\s+divergence\b',
            r'\bconvergence\s+divergence\b',
            r'\bmac\s+d\b'
        ],
        'rsi': [
            r'\brsi\b',
            r'\brelative\s+strength\s+index\b',
            r'\bstrength\s+index\b',
            r'\bmomentum\s+oscillator\b'
        ],
        'bollinger': [
            r'\bbollinger\s+bands?\b',
            r'\bbollinger\b',
            r'\bbands?\b',
            r'\bbollinger\s+band\b'
        ],
        'williams_r': [
            r'\bwilliams?\s+r\b',
            r'\bwilliams?\s+%r\b',
            r'\bwilliams\s+oscillator\b'
        ],
        'stochastic': [
            r'\bstochastic\b',
            r'\bstoch\b',
            r'\bstochastic\s+oscillator\b'
        ],
        'momentum': [
            r'\bmomentum\b',
            r'\bprice\s+momentum\b',
            r'\bchange\s+in\s+price\b'
        ],
        'zscore': [
            r'\bzscore\b',
            r'\bz\s+score\b',
            r'\bz\s+value\b',
            r'\bstandard\s+score\b',
            r'\bmean\s+reversion\s+score\b'
        ],
        'realized_vol': [
            r'\brealized\s+volatility\b',
            r'\brealized\s+vol\b',
            r'\bvolatility\b',
            r'\bhistorical\s+volatility\b'
        ],
        'aroon': [
            r'\baroon\b',
            r'\baroon\s+indicator\b',
            r'\btrend\s+indicator\b'
        ]
    }
    
    # Check for enhanced patterns first
    for tool_name, patterns in tool_patterns.items():
        if tool_name in available_tools:
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    # Extract ticker
                    ticker = None
                    ticker_patterns = [
                        r'for\s+([A-Za-z]{1,5})\b',
                        r'on\s+([A-Za-z]{1,5})\b',
                        r'\$([A-Za-z]{1,5})\b',
                        r'\b([A-Za-z]{2,5})\b',  # Any 2-5 letter ticker
                    ]
                    for ticker_pattern in ticker_patterns:
                        match = re.search(ticker_pattern, query_lower)
                        if match:
                            ticker = match.group(1).upper()
                            break
                    
                    if not ticker:
                        ticker = "SPY"  # Default ticker
                    
                    return {
                        "tool": tool_name,
                        "ticker": ticker,
                        "params": {}
                    }
    
    # Fallback to original logic for tools not in enhanced patterns
    sorted_tools = sorted(available_tools, key=len, reverse=True)
    
    for tool_name in sorted_tools:
        # Skip if already handled by enhanced patterns
        if tool_name in tool_patterns:
            continue
            
        # Handle tool name variations (underscores vs spaces)
        tool_name_variants = [tool_name, tool_name.replace('_', ' '), tool_name.replace('_', '')]
        
        tool_found = False
        matched_variant = None
        
        for variant in tool_name_variants:
            # Use word boundary regex to avoid partial matches
            pattern = r'\b' + re.escape(variant) + r'\b'
            if re.search(pattern, query_lower):
                tool_found = True
                matched_variant = variant
                break
        
        if tool_found:
            # Extract parameters based on tool type
            params = {}
            
            # Pattern 1: "rsi 30", "sma 50", "zscore 20" (direct number after tool name)
            direct_param_pattern = rf'{re.escape(matched_variant)}\s+(\d+)'
            direct_match = re.search(direct_param_pattern, query_lower)
            
            # Pattern 2: "30 day rsi", "50 day sma" (number + day + tool)
            day_param_pattern = r'(\d+)\s*day\s+' + re.escape(matched_variant)
            day_match = re.search(day_param_pattern, query_lower)
            
            if direct_match:
                params["window"] = int(direct_match.group(1))
            elif day_match:
                params["window"] = int(day_match.group(1))
            
            # Special handling for multi-parameter tools
            if tool_name == "bollinger":
                # Look for "bollinger 20 2.5" or "bollinger bands 20 2.5"
                bollinger_match = re.search(rf'{re.escape(matched_variant)}(?:\s+bands)?\s+(\d+)(?:\s+(\d+(?:\.\d+)?))?', query_lower)
                if bollinger_match:
                    params["window"] = int(bollinger_match.group(1))
                    if bollinger_match.group(2):
                        params["std_dev"] = float(bollinger_match.group(2))
                    # Remove any window parameter that might have been set by the generic pattern
                    # (bollinger uses window correctly, so no need to remove it)
            
            elif tool_name == "macd":
                # Look for "macd 12 26 9" or "macd 12,26,9"
                macd_match = re.search(rf'{re.escape(matched_variant)}\s+(\d+)(?:[\s,]+(\d+))?(?:[\s,]+(\d+))?', query_lower)
                if macd_match:
                    params["fast"] = int(macd_match.group(1))
                    if macd_match.group(2):
                        params["slow"] = int(macd_match.group(2))
                    if macd_match.group(3):
                        params["signal"] = int(macd_match.group(3))
                    # Remove any window parameter that might have been set by the generic pattern
                    if "window" in params:
                        del params["window"]
            
            elif tool_name == "stochastic":
                # Look for "stochastic 14 3" or "stoch 14,3"
                stoch_match = re.search(rf'{re.escape(matched_variant)}(?:\s+oscillator)?\s+(\d+)(?:[\s,]+(\d+))?', query_lower)
                if stoch_match:
                    params["k_window"] = int(stoch_match.group(1))
                    if stoch_match.group(2):
                        params["d_window"] = int(stoch_match.group(2))
                    # Remove any window parameter that might have been set by the generic pattern
                    if "window" in params:
                        del params["window"]
            
            # Extract ticker
            ticker = None
            ticker_patterns = [
                r'on\s+([A-Za-z]{1,5})\b',
                r'ticker[:\s]+([A-Za-z]{1,5})\b',
                r'\$([A-Za-z]{1,5})\b',
                r'\b([A-Za-z]{2,5})\b',  # Any 2-5 letter ticker
            ]
            for pattern in ticker_patterns:
                match = re.search(pattern, query_lower)
                if match:
                    ticker = match.group(1).upper()
                    break
            
            if not ticker:
                ticker = "SPY"  # Default ticker
            
            return {
                "tool": tool_name,
                "ticker": ticker,
                "params": params
            }
    
    return None

"""
Time-aware utilities for market data fetching
"""
from datetime import datetime, time
import pytz
from typing import Tuple, Optional


def get_market_time_aware_date() -> Tuple[str, str]:
    """
    Get the appropriate date and time context based on current ET market time.
    
    Returns:
        Tuple of (date_str, time_context) where:
        - date_str: The date to fetch data for (YYYY-MM-DD)
        - time_context: "pre_market", "market_hours", "after_hours", or "closed"
    """
    # Get current time in ET
    et_tz = pytz.timezone('US/Eastern')
    now_et = datetime.now(et_tz)
    
    # Market hours
    market_open = time(9, 30)  # 9:30 AM ET
    market_close = time(16, 0)  # 4:00 PM ET
    
    current_time = now_et.time()
    current_date = now_et.date()
    
    # Determine time context
    if current_time < market_open:
        # Before market open - use previous trading day's close
        time_context = "pre_market"
        # For now, use yesterday (we'll enhance this with trading calendar later)
        from datetime import timedelta
        target_date = current_date - timedelta(days=1)
    elif market_open <= current_time <= market_close:
        # During market hours - use current real-time data
        time_context = "market_hours"
        target_date = current_date
    else:
        # After market close - use today's close
        time_context = "after_hours"
        target_date = current_date
    
    return target_date.strftime("%Y-%m-%d"), time_context


def should_fetch_realtime_price(time_context: str) -> bool:
    """
    Determine if we should fetch real-time price data.
    
    Args:
        time_context: Time context from get_market_time_aware_date()
        
    Returns:
        True if we should fetch real-time data, False for historical close
    """
    return time_context == "market_hours"


def get_price_data_message(time_context: str, date_str: str) -> str:
    """
    Get appropriate message explaining the price data source.
    
    Args:
        time_context: Time context from get_market_time_aware_date()
        date_str: Date string for the data
        
    Returns:
        Human-readable message about the price data
    """
    if time_context == "pre_market":
        return f"Market hasn't opened yet. Showing previous trading day's close price from {date_str}."
    elif time_context == "market_hours":
        return f"Market is currently open. Showing real-time price as of {datetime.now().strftime('%H:%M ET')}."
    elif time_context == "after_hours":
        return f"Market is closed. Showing today's close price from {date_str}."
    else:
        return f"Showing price data from {date_str}."

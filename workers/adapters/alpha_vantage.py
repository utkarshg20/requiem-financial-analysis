# workers/adapters/alpha_vantage.py
import requests
import os
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger("requiem.alpha_vantage")

class AlphaVantageAdapter:
    """Adapter for Alpha Vantage API to fetch fundamental data"""
    
    def __init__(self):
        self.api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        self.base_url = "https://www.alphavantage.co/query"
        if not self.api_key:
            logger.warning("ALPHA_VANTAGE_API_KEY not found in environment variables")
    
    def get_overview(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get company overview including P/E, P/B, P/S ratios"""
        if not self.api_key:
            return None
            
        params = {
            "function": "OVERVIEW",
            "symbol": symbol,
            "apikey": self.api_key
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if "Error Message" in data:
                logger.error(f"Alpha Vantage error for {symbol}: {data['Error Message']}")
                return None
                
            if "Note" in data:
                logger.warning(f"Alpha Vantage rate limit: {data['Note']}")
                return None
                
            return data
            
        except Exception as e:
            logger.error(f"Error fetching Alpha Vantage overview for {symbol}: {str(e)}")
            return None
    
    def get_earnings(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get earnings data"""
        if not self.api_key:
            return None
            
        params = {
            "function": "EARNINGS",
            "symbol": symbol,
            "apikey": self.api_key
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if "Error Message" in data:
                logger.error(f"Alpha Vantage earnings error for {symbol}: {data['Error Message']}")
                return None
                
            return data
            
        except Exception as e:
            logger.error(f"Error fetching Alpha Vantage earnings for {symbol}: {str(e)}")
            return None
    
    def get_income_statement(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get income statement data"""
        if not self.api_key:
            return None
            
        params = {
            "function": "INCOME_STATEMENT",
            "symbol": symbol,
            "apikey": self.api_key
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if "Error Message" in data:
                logger.error(f"Alpha Vantage income statement error for {symbol}: {data['Error Message']}")
                return None
                
            return data
            
        except Exception as e:
            logger.error(f"Error fetching Alpha Vantage income statement for {symbol}: {str(e)}")
            return None
    
    def get_balance_sheet(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get balance sheet data"""
        if not self.api_key:
            return None
            
        params = {
            "function": "BALANCE_SHEET",
            "symbol": symbol,
            "apikey": self.api_key
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if "Error Message" in data:
                logger.error(f"Alpha Vantage balance sheet error for {symbol}: {data['Error Message']}")
                return None
                
            return data
            
        except Exception as e:
            logger.error(f"Error fetching Alpha Vantage balance sheet for {symbol}: {str(e)}")
            return None

# Global instance
alpha_vantage = AlphaVantageAdapter()

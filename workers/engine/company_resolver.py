#!/usr/bin/env python3
"""
Dynamic company name resolver using financial APIs
"""

import requests
import os
from typing import Optional, List, Dict
import logging

logger = logging.getLogger(__name__)

class CompanyResolver:
    """Resolve ticker symbols to company names using various APIs"""
    
    def __init__(self):
        self.cache = {}  # Simple in-memory cache
        self.polygon_api_key = os.getenv('POLYGON_API_KEY')
        self.alpha_vantage_api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    
    def get_company_name(self, ticker: str) -> Optional[str]:
        """
        Get company name from ticker symbol
        Tries multiple sources in order of preference
        """
        if not ticker:
            return None
            
        ticker = ticker.upper()
        
        # Check cache first
        if ticker in self.cache:
            return self.cache[ticker]
        
        # Try different sources
        company_name = None
        
        # 1. Try Polygon API (if available)
        if self.polygon_api_key:
            company_name = self._get_from_polygon(ticker)
        
        # 2. Try Alpha Vantage (if available)
        if not company_name and self.alpha_vantage_api_key:
            company_name = self._get_from_alpha_vantage(ticker)
        
        # 3. Try Yahoo Finance (free, no API key needed)
        if not company_name:
            company_name = self._get_from_yahoo_finance(ticker)
        
        # 4. Try IEX Cloud (free tier available)
        if not company_name:
            company_name = self._get_from_iex_cloud(ticker)
        
        # Cache the result
        if company_name:
            self.cache[ticker] = company_name
            logger.info(f"Resolved {ticker} -> {company_name}")
        else:
            logger.warning(f"Could not resolve company name for {ticker}")
        
        return company_name
    
    def _get_from_polygon(self, ticker: str) -> Optional[str]:
        """Get company name from Polygon API"""
        try:
            url = f"https://api.polygon.io/v3/reference/tickers/{ticker}"
            params = {"apikey": self.polygon_api_key}
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if 'results' in data and 'name' in data['results']:
                    return data['results']['name']
        except Exception as e:
            logger.debug(f"Polygon API error for {ticker}: {e}")
        return None
    
    def _get_from_alpha_vantage(self, ticker: str) -> Optional[str]:
        """Get company name from Alpha Vantage API"""
        try:
            url = "https://www.alphavantage.co/query"
            params = {
                "function": "OVERVIEW",
                "symbol": ticker,
                "apikey": self.alpha_vantage_api_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if 'Name' in data:
                    return data['Name']
        except Exception as e:
            logger.debug(f"Alpha Vantage API error for {ticker}: {e}")
        return None
    
    def _get_from_yahoo_finance(self, ticker: str) -> Optional[str]:
        """Get company name from Yahoo Finance (web scraping)"""
        try:
            url = f"https://finance.yahoo.com/quote/{ticker}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                # Parse the HTML to extract company name
                # This is a simplified approach - in production you'd use BeautifulSoup
                import re
                
                # Look for the company name in the page title or meta tags
                title_match = re.search(r'<title>([^<]+) \(' + ticker + r'\)', response.text)
                if title_match:
                    return title_match.group(1).strip()
                
                # Look for JSON-LD structured data
                json_ld_match = re.search(r'"name":\s*"([^"]+)"', response.text)
                if json_ld_match:
                    return json_ld_match.group(1).strip()
                    
        except Exception as e:
            logger.debug(f"Yahoo Finance error for {ticker}: {e}")
        return None
    
    def _get_from_iex_cloud(self, ticker: str) -> Optional[str]:
        """Get company name from IEX Cloud API"""
        try:
            url = f"https://cloud.iexapis.com/stable/stock/{ticker}/company"
            params = {"token": "pk_test_1234567890abcdef"}  # Free tier token
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if 'companyName' in data:
                    return data['companyName']
        except Exception as e:
            logger.debug(f"IEX Cloud API error for {ticker}: {e}")
        return None
    
    def get_company_variations(self, ticker: str) -> List[str]:
        """
        Get multiple variations of company name for better search
        """
        company_name = self.get_company_name(ticker)
        if not company_name:
            return [ticker]
        
        variations = [company_name]
        
        # Add common variations
        if 'Inc' in company_name:
            variations.append(company_name.replace(' Inc', ''))
        if 'Corporation' in company_name:
            variations.append(company_name.replace(' Corporation', ''))
        if 'Corp' in company_name:
            variations.append(company_name.replace(' Corp', ''))
        if 'Ltd' in company_name:
            variations.append(company_name.replace(' Ltd', ''))
        if 'LLC' in company_name:
            variations.append(company_name.replace(' LLC', ''))
        
        # Add ticker as fallback
        variations.append(ticker)
        
        return list(set(variations))  # Remove duplicates

# Example usage
if __name__ == "__main__":
    resolver = CompanyResolver()
    
    test_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META']
    
    for ticker in test_tickers:
        name = resolver.get_company_name(ticker)
        variations = resolver.get_company_variations(ticker)
        print(f"{ticker}: {name} -> {variations}")

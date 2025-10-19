#!/usr/bin/env python3
"""
Company resolver using yfinance (free, no API key needed)
"""

import yfinance as yf
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)

class YFinanceCompanyResolver:
    """Resolve ticker symbols to company names using yfinance"""
    
    def __init__(self):
        self.cache = {}  # Simple in-memory cache
    
    def get_company_name(self, ticker: str) -> Optional[str]:
        """Get company name from ticker using yfinance"""
        if not ticker:
            return None
            
        ticker = ticker.upper()
        
        # Check cache first
        if ticker in self.cache:
            return self.cache[ticker]
        
        try:
            # Create ticker object
            stock = yf.Ticker(ticker)
            
            # Get company info
            info = stock.info
            
            # Try different possible name fields
            name_fields = ['longName', 'shortName', 'name', 'companyName']
            
            for field in name_fields:
                if field in info and info[field]:
                    company_name = info[field]
                    self.cache[ticker] = company_name
                    logger.info(f"Resolved {ticker} -> {company_name}")
                    return company_name
            
            logger.warning(f"Could not find company name for {ticker}")
            return None
            
        except Exception as e:
            logger.debug(f"yfinance error for {ticker}: {e}")
            return None
    
    def get_company_variations(self, ticker: str) -> List[str]:
        """Get multiple variations of company name"""
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
    resolver = YFinanceCompanyResolver()
    
    test_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'JPM', 'JNJ', 'PG']
    
    print("Testing yfinance company resolver:")
    print("=" * 50)
    
    for ticker in test_tickers:
        name = resolver.get_company_name(ticker)
        variations = resolver.get_company_variations(ticker)
        print(f"{ticker:6} -> {name}")
        print(f"       Variations: {variations[:3]}...")  # Show first 3 variations
        print()

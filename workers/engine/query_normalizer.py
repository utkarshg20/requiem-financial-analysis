"""
Query Normalization for Earnings Analysis
Converts user queries into optimized search terms for Perplexity API
"""

import re
from typing import Dict, Tuple, Optional, List
from datetime import datetime
from .yfinance_resolver import YFinanceCompanyResolver

class EarningsQueryNormalizer:
    """Normalizes earnings queries for better source discovery"""
    
    def __init__(self):
        # Initialize dynamic company resolver
        self.company_resolver = YFinanceCompanyResolver()
        
        # Keep some common mappings for fallback (ETFs, etc.)
        self.fallback_mappings = {
            'SPY': ['SPDR S&P 500 ETF Trust'],
            'QQQ': ['Invesco QQQ Trust'],
            'DIA': ['SPDR Dow Jones Industrial Average ETF Trust']
        }
        
        # Quarter mappings
        self.quarter_mappings = {
            'Q1': ['Q1', 'first quarter', 'Q1 2024', 'Q1 2023'],
            'Q2': ['Q2', 'second quarter', 'Q2 2024', 'Q2 2023'],
            'Q3': ['Q3', 'third quarter', 'Q3 2024', 'Q3 2023'],
            'Q4': ['Q4', 'fourth quarter', 'Q4 2024', 'Q4 2023']
        }
    
    def normalize_earnings_query(self, query: str) -> Dict[str, str]:
        """
        Normalize earnings query into optimized search terms
        
        Returns:
            Dict with normalized components for better search
        """
        query_lower = query.lower()
        
        # Extract ticker
        ticker = self._extract_ticker(query_lower)
        
        # Extract quarter and year
        quarter, year = self._extract_quarter_year(query_lower)
        
        # Get company name variations dynamically
        if ticker in self.fallback_mappings:
            company_names = self.fallback_mappings[ticker]
        else:
            company_names = self.company_resolver.get_company_variations(ticker)
        
        # Build optimized search queries
        search_queries = self._build_search_queries(ticker, quarter, year, company_names)
        
        return {
            'ticker': ticker,
            'quarter': quarter,
            'year': year,
            'company_names': company_names,
            'search_queries': search_queries,
            'original_query': query
        }
    
    def _extract_ticker(self, query: str) -> Optional[str]:
        """Extract ticker symbol from query"""
        query_lower = query.lower()
        
        # Common ticker patterns
        ticker_patterns = [
            r'\b([A-Z]{2,5})\b',  # 2-5 uppercase letters
            r'\$([A-Z]{2,5})\b',  # $TICKER format
        ]
        
        for pattern in ticker_patterns:
            matches = re.findall(pattern, query.upper())
            if matches:
                return matches[0]
        
        # Try fallback mappings first (for ETFs, etc.)
        for ticker, names in self.fallback_mappings.items():
            for name in names:
                if name.lower() in query_lower:
                    return ticker
        
        return None
    
    def _extract_quarter_year(self, query: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract quarter and year from query"""
        quarter = None
        year = None
        
        # Extract year
        year_patterns = [
            r'\b(20\d{2})\b',  # 2024, 2023, etc.
            r'\b(\d{4})\b'     # Any 4-digit year
        ]
        
        for pattern in year_patterns:
            matches = re.findall(pattern, query)
            if matches:
                year = matches[0]
                break
        
        # Default to current year if not found
        if not year:
            year = str(datetime.now().year)
        
        # Extract quarter
        quarter_patterns = [
            r'\b(q[1-4])\b',  # Q1, Q2, Q3, Q4
            r'\b([1-4]q)\b',  # 1Q, 2Q, 3Q, 4Q
            r'\b(quarter\s*[1-4])\b',  # quarter 1, quarter 2, etc.
            r'\b(first|second|third|fourth)\s+quarter\b',  # first quarter, etc.
            r'\b(1st|2nd|3rd|4th)\s+quarter\b'  # 1st quarter, etc.
        ]
        
        for pattern in quarter_patterns:
            matches = re.findall(pattern, query)
            if matches:
                quarter_text = matches[0].lower()
                if 'q1' in quarter_text or 'first' in quarter_text or '1st' in quarter_text or quarter_text == '1q':
                    quarter = 'Q1'
                elif 'q2' in quarter_text or 'second' in quarter_text or '2nd' in quarter_text or quarter_text == '2q':
                    quarter = 'Q2'
                elif 'q3' in quarter_text or 'third' in quarter_text or '3rd' in quarter_text or quarter_text == '3q':
                    quarter = 'Q3'
                elif 'q4' in quarter_text or 'fourth' in quarter_text or '4th' in quarter_text or quarter_text == '4q':
                    quarter = 'Q4'
                break
        
        return quarter, year
    
    def _build_search_queries(self, ticker: str, quarter: str, year: str, company_names: list) -> list:
        """Build multiple optimized search queries for Perplexity - Press Release Focused"""
        queries = []
        
        if not ticker or not quarter:
            return queries
        
        # Primary company name
        primary_company = company_names[0] if company_names else ticker
        
        # Query 1: Earnings report PDF transcript (BEST FORMAT - proven to work)
        queries.append(f"{primary_company} {year} {quarter} earnings report PDF transcript")
        
        # Query 2: Press release search
        queries.append(f"{primary_company} {year} {quarter} earnings press release investor relations")
        
        # Query 3: Specific quarter format
        queries.append(f"{primary_company} Q{quarter[1]} {year} earnings report PDF")
        
        # Query 4: Alternative company names
        if len(company_names) > 1:
            queries.append(f"{company_names[1]} {year} {quarter} earnings report PDF transcript")
        
        # Query 5: Conference call + press release
        queries.append(f"{primary_company} {year} {quarter} conference call press release earnings")
        
        # Query 6: Financial results search
        queries.append(f"{primary_company} {year} {quarter} financial results press release")
        
        return queries

# Example usage
if __name__ == "__main__":
    normalizer = EarningsQueryNormalizer()
    
    test_queries = [
        "NVDA Q3 2024 earnings",
        "Apple third quarter 2024 earnings report",
        "Microsoft Q2 2024 earnings call",
        "NVIDIA 3Q2024 earnings",
        "AAPL earnings report 2024 Q4"
    ]
    
    for query in test_queries:
        result = normalizer.normalize_earnings_query(query)
        print(f"Query: {query}")
        print(f"Normalized: {result}")
        print("---")

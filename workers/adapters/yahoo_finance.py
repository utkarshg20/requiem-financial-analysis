# workers/adapters/yahoo_finance.py
import requests
import json
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger("requiem.yahoo_finance")

class YahooFinanceAdapter:
    """Adapter for Yahoo Finance API to fetch additional fundamental data"""
    
    def __init__(self):
        self.base_url = "https://query1.finance.yahoo.com/v10/finance/quoteSummary"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def get_quote_summary(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive quote summary from Yahoo Finance"""
        params = {
            "modules": "summaryDetail,defaultKeyStatistics,financialData,majorHoldersBreakdown,calendarEvents,upgradeDowngradeHistory,recommendationTrend,earnings,price,quoteType"
        }
        
        try:
            response = requests.get(
                f"{self.base_url}/{symbol}",
                params=params,
                headers=self.headers,
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            if "quoteSummary" in data and "result" in data["quoteSummary"]:
                return data["quoteSummary"]["result"][0] if data["quoteSummary"]["result"] else None
            return None
            
        except Exception as e:
            logger.error(f"Error fetching Yahoo Finance data for {symbol}: {str(e)}")
            return None
    
    def get_key_statistics(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get key statistics including P/E, P/B, P/S ratios"""
        data = self.get_quote_summary(symbol)
        if not data:
            return None
            
        try:
            stats = {}
            
            # Extract from summaryDetail
            if "summaryDetail" in data:
                summary = data["summaryDetail"]
                stats.update({
                    "dividend_yield": self._safe_float(summary.get("dividendYield", {}).get("raw")),
                    "dividend_rate": self._safe_float(summary.get("dividendRate", {}).get("raw")),
                    "ex_dividend_date": summary.get("exDividendDate", {}).get("fmt"),
                    "payout_ratio": self._safe_float(summary.get("payoutRatio", {}).get("raw")),
                })
            
            # Extract from defaultKeyStatistics
            if "defaultKeyStatistics" in data:
                key_stats = data["defaultKeyStatistics"]
                stats.update({
                    "market_cap": self._safe_float(key_stats.get("marketCap", {}).get("raw")),
                    "enterprise_value": self._safe_float(key_stats.get("enterpriseValue", {}).get("raw")),
                    "trailing_pe": self._safe_float(key_stats.get("trailingPE", {}).get("raw")),
                    "forward_pe": self._safe_float(key_stats.get("forwardPE", {}).get("raw")),
                    "peg_ratio": self._safe_float(key_stats.get("pegRatio", {}).get("raw")),
                    "price_to_book": self._safe_float(key_stats.get("priceToBook", {}).get("raw")),
                    "price_to_sales_trailing_12_months": self._safe_float(key_stats.get("priceToSalesTrailing12Months", {}).get("raw")),
                    "enterprise_to_revenue": self._safe_float(key_stats.get("enterpriseToRevenue", {}).get("raw")),
                    "enterprise_to_ebitda": self._safe_float(key_stats.get("enterpriseToEbitda", {}).get("raw")),
                    "beta": self._safe_float(key_stats.get("beta", {}).get("raw")),
                    "52_week_change": self._safe_float(key_stats.get("52WeekChange", {}).get("raw")),
                    "shares_outstanding": self._safe_float(key_stats.get("sharesOutstanding", {}).get("raw")),
                    "float_shares": self._safe_float(key_stats.get("floatShares", {}).get("raw")),
                    "shares_short": self._safe_float(key_stats.get("sharesShort", {}).get("raw")),
                    "short_ratio": self._safe_float(key_stats.get("shortRatio", {}).get("raw")),
                    "short_percent_of_float": self._safe_float(key_stats.get("shortPercentOfFloat", {}).get("raw")),
                })
            
            # Extract from financialData
            if "financialData" in data:
                financial = data["financialData"]
                stats.update({
                    "current_price": self._safe_float(financial.get("currentPrice", {}).get("raw")),
                    "target_high_price": self._safe_float(financial.get("targetHighPrice", {}).get("raw")),
                    "target_low_price": self._safe_float(financial.get("targetLowPrice", {}).get("raw")),
                    "target_mean_price": self._safe_float(financial.get("targetMeanPrice", {}).get("raw")),
                    "recommendation_mean": financial.get("recommendationMean", {}).get("raw"),
                    "recommendation_key": financial.get("recommendationKey"),
                    "total_cash": self._safe_float(financial.get("totalCash", {}).get("raw")),
                    "total_cash_per_share": self._safe_float(financial.get("totalCashPerShare", {}).get("raw")),
                    "ebitda": self._safe_float(financial.get("ebitda", {}).get("raw")),
                    "total_debt": self._safe_float(financial.get("totalDebt", {}).get("raw")),
                    "debt_to_equity": self._safe_float(financial.get("debtToEquity", {}).get("raw")),
                    "return_on_assets": self._safe_float(financial.get("returnOnAssets", {}).get("raw")),
                    "return_on_equity": self._safe_float(financial.get("returnOnEquity", {}).get("raw")),
                    "gross_margins": self._safe_float(financial.get("grossMargins", {}).get("raw")),
                    "operating_margins": self._safe_float(financial.get("operatingMargins", {}).get("raw")),
                    "profit_margins": self._safe_float(financial.get("profitMargins", {}).get("raw")),
                })
            
            return stats
            
        except Exception as e:
            logger.error(f"Error parsing Yahoo Finance data for {symbol}: {str(e)}")
            return None
    
    def _safe_float(self, value) -> Optional[float]:
        """Safely convert value to float"""
        if value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    
    def get_analyst_recommendations(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get analyst recommendations and price targets"""
        data = self.get_quote_summary(symbol)
        if not data:
            return None
            
        try:
            recommendations = {}
            
            if "recommendationTrend" in data:
                trend = data["recommendationTrend"]
                recommendations.update({
                    "strong_buy": trend.get("strongBuy", {}).get("raw"),
                    "buy": trend.get("buy", {}).get("raw"),
                    "hold": trend.get("hold", {}).get("raw"),
                    "sell": trend.get("sell", {}).get("raw"),
                    "strong_sell": trend.get("strongSell", {}).get("raw"),
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error parsing analyst recommendations for {symbol}: {str(e)}")
            return None

# Global instance
yahoo_finance = YahooFinanceAdapter()

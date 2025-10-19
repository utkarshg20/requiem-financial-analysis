# workers/engine/valuation.py
from typing import Dict, Any, Optional, Tuple
import logging
from ..adapters.alpha_vantage import alpha_vantage
from ..adapters.yahoo_finance import yahoo_finance
from ..adapters.prices_polygon import get_prices_agg
from datetime import datetime, timedelta

logger = logging.getLogger("requiem.valuation")

class ValuationAnalyzer:
    """Analyzes stock valuation using multiple metrics and data sources"""
    
    def __init__(self):
        self.alpha_vantage = alpha_vantage
        self.yahoo_finance = yahoo_finance
    
    def analyze_valuation(self, symbol: str) -> Dict[str, Any]:
        """Comprehensive valuation analysis for a stock"""
        try:
            # Get data from multiple sources (prioritize Alpha Vantage)
            alpha_data = self.alpha_vantage.get_overview(symbol)
            yahoo_data = None
            
            # Only try Yahoo Finance if Alpha Vantage fails
            if not alpha_data:
                logger.info(f"Alpha Vantage failed for {symbol}, trying Yahoo Finance")
                yahoo_data = self.yahoo_finance.get_key_statistics(symbol)
            
            # Get current price
            current_price = self._get_current_price(symbol)
            logger.info(f"Current price for {symbol}: {current_price}")
            
            if not any([alpha_data, yahoo_data]) and not current_price:
                return {"error": "Unable to fetch fundamental data from any source"}
            
            # Combine data sources
            combined_data = self._combine_data_sources(alpha_data, yahoo_data, current_price)
            
            # Calculate valuation metrics
            valuation_metrics = self._calculate_valuation_metrics(combined_data)
            
            # Generate valuation assessment
            assessment = self._generate_valuation_assessment(valuation_metrics, combined_data)
            
            return {
                "symbol": symbol,
                "current_price": current_price,
                "data_sources": {
                    "alpha_vantage": bool(alpha_data),
                    "yahoo_finance": bool(yahoo_data),
                },
                "valuation_metrics": valuation_metrics,
                "assessment": assessment,
                "raw_data": combined_data
            }
            
        except Exception as e:
            logger.error(f"Error analyzing valuation for {symbol}: {str(e)}")
            return {"error": f"Error analyzing valuation: {str(e)}"}
    
    def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current stock price using time-aware logic"""
        try:
            # Use time-aware logic to get the most appropriate current price
            from ..utils.time_aware_utils import get_market_time_aware_date
            from ..adapters.calendar import nearest_trading_day_utc
            
            # Get time-aware date
            target_date, time_context = get_market_time_aware_date()
            
            # Snap to nearest trading day
            trading_date = nearest_trading_day_utc(target_date)
            
            logger.info(f"Fetching current price for {symbol} on {trading_date} (time_context: {time_context})")
            df = get_prices_agg(symbol, trading_date, trading_date)
            
            if not df.empty:
                latest_price = float(df['close'].iloc[-1])
                logger.info(f"Current price for {symbol}: {latest_price} (from {trading_date})")
                return latest_price
            else:
                # Fallback: try to get the most recent available data
                logger.warning(f"No price data found for {symbol} on {trading_date}, trying fallback...")
                
                # Try going back a few days to find available data
                from datetime import datetime, timedelta
                for days_back in range(1, 10):
                    fallback_date = (datetime.strptime(trading_date, "%Y-%m-%d") - timedelta(days=days_back)).strftime("%Y-%m-%d")
                    fallback_df = get_prices_agg(symbol, fallback_date, fallback_date)
                    if not fallback_df.empty:
                        latest_price = float(fallback_df['close'].iloc[-1])
                        logger.info(f"Fallback price for {symbol}: {latest_price} (from {fallback_date})")
                        return latest_price
                
                logger.warning(f"No price data found for {symbol} in the last 10 days")
                return None
            
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {str(e)}")
            return None
    
    def _combine_data_sources(self, alpha_data: Optional[Dict], yahoo_data: Optional[Dict], current_price: Optional[float]) -> Dict[str, Any]:
        """Combine data from multiple sources, preferring the most recent/accurate"""
        combined = {}
        
        # Alpha Vantage data
        if alpha_data:
            combined.update({
                "pe_ratio": self._safe_float(alpha_data.get("PERatio")),
                "peg_ratio": self._safe_float(alpha_data.get("PEGRatio")),
                "price_to_book": self._safe_float(alpha_data.get("PriceToBookRatio")),
                "price_to_sales": self._safe_float(alpha_data.get("PriceToSalesRatioTTM")),
                "ev_to_revenue": self._safe_float(alpha_data.get("EVToRevenue")),
                "ev_to_ebitda": self._safe_float(alpha_data.get("EVToEBITDA")),
                "dividend_yield": self._safe_float(alpha_data.get("DividendYield")),
                "dividend_per_share": self._safe_float(alpha_data.get("DividendPerShare")),
                "dividend_date": alpha_data.get("DividendDate"),
                "ex_dividend_date": alpha_data.get("ExDividendDate"),
                "market_cap": self._safe_float(alpha_data.get("MarketCapitalization")),
                "enterprise_value": self._safe_float(alpha_data.get("EnterpriseValue")),
                "beta": self._safe_float(alpha_data.get("Beta")),
                "52_week_high": self._safe_float(alpha_data.get("52WeekHigh")),
                "52_week_low": self._safe_float(alpha_data.get("52WeekLow")),
                "eps": self._safe_float(alpha_data.get("EPS")),
                "revenue_ttm": self._safe_float(alpha_data.get("RevenueTTM")),
                "gross_profit_ttm": self._safe_float(alpha_data.get("GrossProfitTTM")),
                "ebitda": self._safe_float(alpha_data.get("EBITDA")),
                "return_on_equity": self._safe_float(alpha_data.get("ReturnOnEquityTTM")),
                "return_on_assets": self._safe_float(alpha_data.get("ReturnOnAssetsTTM")),
                "profit_margin": self._safe_float(alpha_data.get("ProfitMargin")),
                "operating_margin": self._safe_float(alpha_data.get("OperatingMarginTTM")),
                "shares_outstanding": self._safe_float(alpha_data.get("SharesOutstanding")),
            })
        
        # Yahoo Finance data (overwrite with more recent data if available)
        if yahoo_data:
            # Only update if we don't have the data or Yahoo has a more recent value
            for key, value in yahoo_data.items():
                if value is not None:
                    combined[key] = value
        
        # Set current price last to ensure it's not overridden
        if current_price:
            combined["current_price"] = current_price
        
        return combined
    
    def _calculate_valuation_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate additional valuation metrics"""
        metrics = {}
        
        current_price = data.get("current_price")
        pe_ratio = data.get("pe_ratio") or data.get("trailing_pe")
        eps = data.get("eps")
        market_cap = data.get("market_cap")
        revenue_ttm = data.get("revenue_ttm")
        book_value = data.get("book_value")
        
        # P/E Analysis
        if pe_ratio:
            metrics["pe_ratio"] = pe_ratio
            metrics["pe_assessment"] = self._assess_pe_ratio(pe_ratio)
        
        # Price-to-Book Analysis
        pb_ratio = data.get("price_to_book")
        if pb_ratio:
            metrics["pb_ratio"] = pb_ratio
            metrics["pb_assessment"] = self._assess_pb_ratio(pb_ratio)
        
        # Price-to-Sales Analysis
        ps_ratio = data.get("price_to_sales") or data.get("price_to_sales_trailing_12_months")
        if ps_ratio:
            metrics["ps_ratio"] = ps_ratio
            metrics["ps_assessment"] = self._assess_ps_ratio(ps_ratio)
        
        # PEG Ratio Analysis
        peg_ratio = data.get("peg_ratio")
        if peg_ratio:
            metrics["peg_ratio"] = peg_ratio
            metrics["peg_assessment"] = self._assess_peg_ratio(peg_ratio)
        
        # Dividend Analysis
        dividend_yield = data.get("dividend_yield")
        if dividend_yield:
            metrics["dividend_yield"] = dividend_yield
            metrics["dividend_assessment"] = self._assess_dividend_yield(dividend_yield)
        
        # Market Cap Analysis
        if market_cap:
            metrics["market_cap"] = market_cap
            metrics["market_cap_category"] = self._categorize_market_cap(market_cap)
        
        # Fair Value Estimate (simplified)
        if current_price and pe_ratio and eps:
            # Simple fair value based on industry average P/E (simplified)
            fair_value = self._estimate_fair_value(data)
            if fair_value:
                metrics["estimated_fair_value"] = fair_value
                metrics["fair_value_premium_discount"] = ((current_price - fair_value) / fair_value) * 100
        
        return metrics
    
    def _generate_valuation_assessment(self, metrics: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall valuation assessment"""
        assessments = []
        scores = []
        
        # Collect individual assessments
        if "pe_assessment" in metrics:
            assessments.append(metrics["pe_assessment"])
            scores.append(metrics["pe_assessment"]["score"])
        
        if "pb_assessment" in metrics:
            assessments.append(metrics["pb_assessment"])
            scores.append(metrics["pb_assessment"]["score"])
        
        if "ps_assessment" in metrics:
            assessments.append(metrics["ps_assessment"])
            scores.append(metrics["ps_assessment"]["score"])
        
        if "peg_assessment" in metrics:
            assessments.append(metrics["peg_assessment"])
            scores.append(metrics["peg_assessment"]["score"])
        
        if not scores:
            return {
                "overall_rating": "Unknown",
                "confidence": "Low",
                "reasoning": "Insufficient data for valuation analysis"
            }
        
        # Calculate overall score
        avg_score = sum(scores) / len(scores)
        
        # Determine overall rating
        if avg_score >= 7:
            rating = "Undervalued"
        elif avg_score >= 5:
            rating = "Fair Value"
        elif avg_score >= 3:
            rating = "Slightly Overvalued"
        else:
            rating = "Overvalued"
        
        # Determine confidence
        if len(scores) >= 4:
            confidence = "High"
        elif len(scores) >= 2:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        # Generate reasoning
        reasoning_parts = []
        for assessment in assessments:
            if assessment["score"] <= 3:
                reasoning_parts.append(f"{assessment['metric']} suggests {assessment['assessment'].lower()}")
        
        reasoning = "; ".join(reasoning_parts) if reasoning_parts else f"Based on {len(scores)} valuation metrics"
        
        return {
            "overall_rating": rating,
            "confidence": confidence,
            "average_score": round(avg_score, 1),
            "reasoning": reasoning,
            "detailed_assessments": assessments
        }
    
    def _assess_pe_ratio(self, pe_ratio: float) -> Dict[str, Any]:
        """Assess P/E ratio"""
        if pe_ratio < 15:
            return {"metric": "P/E Ratio", "value": pe_ratio, "assessment": "Undervalued", "score": 8}
        elif pe_ratio < 20:
            return {"metric": "P/E Ratio", "value": pe_ratio, "assessment": "Fair Value", "score": 6}
        elif pe_ratio < 25:
            return {"metric": "P/E Ratio", "value": pe_ratio, "assessment": "Slightly Overvalued", "score": 4}
        else:
            return {"metric": "P/E Ratio", "value": pe_ratio, "assessment": "Overvalued", "score": 2}
    
    def _assess_pb_ratio(self, pb_ratio: float) -> Dict[str, Any]:
        """Assess Price-to-Book ratio"""
        if pb_ratio < 1:
            return {"metric": "P/B Ratio", "value": pb_ratio, "assessment": "Undervalued", "score": 8}
        elif pb_ratio < 2:
            return {"metric": "P/B Ratio", "value": pb_ratio, "assessment": "Fair Value", "score": 6}
        elif pb_ratio < 3:
            return {"metric": "P/B Ratio", "value": pb_ratio, "assessment": "Slightly Overvalued", "score": 4}
        else:
            return {"metric": "P/B Ratio", "value": pb_ratio, "assessment": "Overvalued", "score": 2}
    
    def _assess_ps_ratio(self, ps_ratio: float) -> Dict[str, Any]:
        """Assess Price-to-Sales ratio"""
        if ps_ratio < 2:
            return {"metric": "P/S Ratio", "value": ps_ratio, "assessment": "Undervalued", "score": 8}
        elif ps_ratio < 4:
            return {"metric": "P/S Ratio", "value": ps_ratio, "assessment": "Fair Value", "score": 6}
        elif ps_ratio < 6:
            return {"metric": "P/S Ratio", "value": ps_ratio, "assessment": "Slightly Overvalued", "score": 4}
        else:
            return {"metric": "P/S Ratio", "value": ps_ratio, "assessment": "Overvalued", "score": 2}
    
    def _assess_peg_ratio(self, peg_ratio: float) -> Dict[str, Any]:
        """Assess PEG ratio"""
        if peg_ratio < 1:
            return {"metric": "PEG Ratio", "value": peg_ratio, "assessment": "Undervalued", "score": 8}
        elif peg_ratio < 1.5:
            return {"metric": "PEG Ratio", "value": peg_ratio, "assessment": "Fair Value", "score": 6}
        elif peg_ratio < 2:
            return {"metric": "PEG Ratio", "value": peg_ratio, "assessment": "Slightly Overvalued", "score": 4}
        else:
            return {"metric": "PEG Ratio", "value": peg_ratio, "assessment": "Overvalued", "score": 2}
    
    def _assess_dividend_yield(self, dividend_yield: float) -> Dict[str, Any]:
        """Assess dividend yield"""
        if dividend_yield > 4:
            return {"metric": "Dividend Yield", "value": dividend_yield, "assessment": "High Yield", "score": 7}
        elif dividend_yield > 2:
            return {"metric": "Dividend Yield", "value": dividend_yield, "assessment": "Good Yield", "score": 6}
        elif dividend_yield > 1:
            return {"metric": "Dividend Yield", "value": dividend_yield, "assessment": "Modest Yield", "score": 5}
        else:
            return {"metric": "Dividend Yield", "value": dividend_yield, "assessment": "Low Yield", "score": 3}
    
    def _categorize_market_cap(self, market_cap: float) -> str:
        """Categorize market capitalization"""
        if market_cap >= 200e9:  # 200B+
            return "Mega Cap"
        elif market_cap >= 10e9:  # 10B+
            return "Large Cap"
        elif market_cap >= 2e9:   # 2B+
            return "Mid Cap"
        elif market_cap >= 300e6:  # 300M+
            return "Small Cap"
        else:
            return "Micro Cap"
    
    def _estimate_fair_value(self, data: Dict[str, Any]) -> Optional[float]:
        """Estimate fair value using multiple approaches"""
        # This is a simplified fair value estimation
        # In a real implementation, you'd use DCF, comparable company analysis, etc.
        
        eps = data.get("eps")
        pe_ratio = data.get("pe_ratio") or data.get("trailing_pe")
        
        if eps and pe_ratio:
            # Use a conservative P/E multiple (industry average approach)
            conservative_pe = min(pe_ratio * 0.8, 15)  # 20% discount or max 15
            return eps * conservative_pe
        
        return None
    
    def _safe_float(self, value) -> Optional[float]:
        """Safely convert value to float"""
        if value is None or value == "None":
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

# Global instance
valuation_analyzer = ValuationAnalyzer()

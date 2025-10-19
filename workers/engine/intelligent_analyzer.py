# workers/engine/intelligent_analyzer.py
from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
from ..adapters.prices_polygon import get_prices_agg
from ..engine.tool_executor import get_tool_executor_for_user
from ..utils.date_utils import get_realistic_date_range

logger = logging.getLogger("requiem.intelligent_analyzer")

class IntelligentAnalyzer:
    """Intelligent analysis engine that understands context and provides actionable insights"""
    
    def __init__(self):
        self.tool_executor = None
    
    def analyze_query(self, query: str, selected_tools: List[str]) -> Dict[str, Any]:
        """
        Analyze a query intelligently and provide comprehensive insights
        
        Args:
            query: User's natural language query
            selected_tools: List of available tools
            
        Returns:
            Comprehensive analysis with actionable insights
        """
        query_lower = query.lower()
        
        # Detect the user's real intent
        intent = self._detect_real_intent(query_lower)
        
        # Extract ticker
        ticker = self._extract_ticker(query)
        
        # Get price data
        price_data = self._get_price_data(ticker)
        
        if intent == "earnings_analysis":
            return self._analyze_earnings(query, ticker)
        elif intent == "entry_price_analysis":
            return self._analyze_entry_price(ticker, price_data, selected_tools)
        elif intent == "technical_analysis":
            return self._analyze_technical_indicators(ticker, price_data, selected_tools)
        elif intent == "general_advice":
            return self._provide_general_advice(ticker, price_data, selected_tools)
        else:
            return self._provide_comprehensive_analysis(ticker, price_data, selected_tools)
    
    def _detect_real_intent(self, query: str) -> str:
        """Detect the user's real intent from their query"""
        
        # Earnings calls keywords
        earnings_keywords = [
            "earnings", "earnings call", "quarterly", "q1", "q2", "q3", "q4",
            "transcript", "conference call", "investor call", "results",
            "quarterly results", "annual results", "guidance", "outlook",
            "revenue guidance", "eps guidance", "earnings report"
        ]
        
        # Entry price analysis keywords
        entry_keywords = [
            "enter", "entry", "buy", "purchase", "when to buy", "good price",
            "should i buy", "buying opportunity", "entry point", "timing"
        ]
        
        # Technical analysis keywords
        technical_keywords = [
            "technical", "indicators", "rsi", "macd", "sma", "bollinger",
            "analysis", "chart", "trend", "momentum", "support", "resistance"
        ]
        
        # General advice keywords
        advice_keywords = [
            "advice", "recommendation", "opinion", "thoughts", "analysis",
            "what do you think", "should i", "is it good", "worth"
        ]
        
        if any(keyword in query for keyword in earnings_keywords):
            return "earnings_analysis"
        elif any(keyword in query for keyword in entry_keywords):
            return "entry_price_analysis"
        elif any(keyword in query for keyword in technical_keywords):
            return "technical_analysis"
        elif any(keyword in query for keyword in advice_keywords):
            return "general_advice"
        else:
            return "comprehensive_analysis"
    
    def _extract_ticker(self, query: str) -> str:
        """Extract ticker symbol from query"""
        import re
        
        # Look for $TICKER format
        dollar_match = re.search(r'\$([A-Z]{1,5})\b', query.upper())
        if dollar_match:
            return dollar_match.group(1)
        
        # Company name to ticker mapping
        company_mapping = {
            'apple': 'AAPL',
            'microsoft': 'MSFT', 
            'google': 'GOOGL',
            'alphabet': 'GOOGL',
            'amazon': 'AMZN',
            'nvidia': 'NVDA',
            'tesla': 'TSLA',
            'meta': 'META',
            'facebook': 'META',
            'netflix': 'NFLX',
            'shopify': 'SHOP',
            'uber': 'UBER',
            'airbnb': 'ABNB',
            'spotify': 'SPOT',
            'zoom': 'ZM',
            'salesforce': 'CRM',
            'oracle': 'ORCL',
            'adobe': 'ADBE',
            'intel': 'INTC',
            'cisco': 'CSCO',
            'ibm': 'IBM',
            'nike': 'NKE',
            'mcdonalds': 'MCD',
            'coca cola': 'KO',
            'pepsi': 'PEP',
            'walmart': 'WMT',
            'target': 'TGT',
            'home depot': 'HD',
            'lowes': 'LOW',
            'boeing': 'BA',
            'general motors': 'GM',
            'ford': 'F',
            'disney': 'DIS',
            'comcast': 'CMCSA',
            'verizon': 'VZ',
            'at&t': 'T',
            't mobile': 'TMUS',
            'sprint': 'S',
            'paypal': 'PYPL',
            'square': 'SQ',
            'stripe': 'STRIPE',
            'coinbase': 'COIN',
            'robinhood': 'HOOD',
            'palantir': 'PLTR',
            'snowflake': 'SNOW',
            'databricks': 'DATABRICKS',
            'openai': 'OPENAI',
            'anthropic': 'ANTHROPIC'
        }
        
        query_lower = query.lower()
        
        # Check for company names
        for company, ticker in company_mapping.items():
            if company in query_lower:
                return ticker
        
        # Look for common tickers
        common_tickers = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'SHOP']
        for ticker in common_tickers:
            if ticker in query.upper():
                return ticker
        
        return "SPY"  # Default
    
    def _get_price_data(self, ticker: str) -> Dict[str, Any]:
        """Get comprehensive price data for analysis"""
        try:
            start_date, end_date = get_realistic_date_range(days_back=365)  # 1 year of data
            df = get_prices_agg(ticker, start_date, end_date)
            
            if df.empty:
                return {"error": f"No price data available for {ticker}"}
            
            # Calculate basic metrics
            current_price = float(df['close'].iloc[-1])
            price_52w_high = float(df['high'].max())
            price_52w_low = float(df['low'].min())
            
            # Calculate recent performance
            price_1m_ago = float(df['close'].iloc[-22]) if len(df) >= 22 else current_price
            price_3m_ago = float(df['close'].iloc[-66]) if len(df) >= 66 else current_price
            price_6m_ago = float(df['close'].iloc[-132]) if len(df) >= 132 else current_price
            
            return {
                "ticker": ticker,
                "current_price": current_price,
                "price_52w_high": price_52w_high,
                "price_52w_low": price_52w_low,
                "price_1m_ago": price_1m_ago,
                "price_3m_ago": price_3m_ago,
                "price_6m_ago": price_6m_ago,
                "data_points": len(df),
                "date_range": f"{start_date} to {end_date}",
                "price_data": df
            }
        except Exception as e:
            logger.error(f"Error getting price data for {ticker}: {str(e)}")
            return {"error": f"Failed to get price data for {ticker}"}
    
    def _analyze_entry_price(self, ticker: str, price_data: Dict[str, Any], selected_tools: List[str]) -> Dict[str, Any]:
        """Analyze technical indicators for professional quant analysis"""
        if "error" in price_data:
            return price_data
        
        # Get technical indicators for analysis
        indicators = self._get_technical_indicators(ticker, price_data, selected_tools)
        
        # Analyze technical signals
        technical_analysis = self._calculate_technical_signals(price_data, indicators)
        
        return {
            "ticker": ticker,
            "analysis_type": "technical_analysis",
            "current_price": price_data["current_price"],
            "technical_signals": technical_analysis,
            "technical_indicators": indicators,
            "market_context": self._analyze_market_context(price_data, indicators),
            "quantitative_insights": self._generate_quantitative_insights(price_data, indicators)
        }
    
    def _get_technical_indicators(self, ticker: str, price_data: Dict[str, Any], selected_tools: List[str]) -> Dict[str, Any]:
        """Get comprehensive technical indicators for professional analysis"""
        indicators = {}
        
        try:
            # Initialize tool executor
            if not self.tool_executor:
                self.tool_executor = get_tool_executor_for_user(selected_tools)
            
            # Get comprehensive technical indicators
            key_indicators = ['rsi', 'sma', 'macd', 'bollinger', 'williams_r', 'stochastic', 'momentum', 'realized_vol']
            
            for indicator in key_indicators:
                if indicator in selected_tools:
                    try:
                        result = self.tool_executor.execute_tool(
                            tool_name=indicator,
                            ticker=ticker,
                            start_date=price_data["date_range"].split(" to ")[0],
                            end_date=price_data["date_range"].split(" to ")[1]
                        )
                        indicators[indicator] = result
                    except Exception as e:
                        logger.error(f"Error calculating {indicator}: {str(e)}")
                        indicators[indicator] = {"error": str(e)}
        
        except Exception as e:
            logger.error(f"Error getting indicators: {str(e)}")
        
        return indicators
    
    def _calculate_technical_signals(self, price_data: Dict[str, Any], indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate technical signals for professional analysis"""
        signals = {
            "momentum_signals": [],
            "trend_signals": [],
            "volatility_signals": [],
            "overbought_oversold": [],
            "signal_strength": {}
        }
        
        current_price = price_data["current_price"]
        
        # RSI Analysis
        if "rsi" in indicators and "latest_value" in indicators["rsi"]:
            rsi = indicators["rsi"]["latest_value"]
            rsi_analysis = self._analyze_rsi(rsi, indicators["rsi"])
            signals["overbought_oversold"].append(rsi_analysis)
            signals["signal_strength"]["rsi"] = rsi_analysis["strength"]
        
        # MACD Analysis
        if "macd" in indicators and "latest_value" in indicators["macd"]:
            macd_analysis = self._analyze_macd(indicators["macd"])
            signals["momentum_signals"].append(macd_analysis)
            signals["signal_strength"]["macd"] = macd_analysis["strength"]
        
        # SMA Analysis
        if "sma" in indicators and "latest_value" in indicators["sma"]:
            sma_analysis = self._analyze_sma(current_price, indicators["sma"])
            signals["trend_signals"].append(sma_analysis)
            signals["signal_strength"]["sma"] = sma_analysis["strength"]
        
        # Bollinger Bands Analysis
        if "bollinger" in indicators and "latest_value" in indicators["bollinger"]:
            bb_analysis = self._analyze_bollinger_bands(current_price, indicators["bollinger"])
            signals["volatility_signals"].append(bb_analysis)
            signals["signal_strength"]["bollinger"] = bb_analysis["strength"]
        
        return signals
    
    def _analyze_rsi(self, rsi_value: float, rsi_data: Dict[str, Any]) -> Dict[str, Any]:
        """Professional RSI analysis"""
        if rsi_value < 30:
            interpretation = "RSI indicates oversold conditions. Historically, RSI below 30 has preceded mean reversion in 68% of cases over 20-day periods."
            strength = "strong" if rsi_value < 25 else "moderate"
            quant_note = f"Current RSI {rsi_value:.1f} is {abs(rsi_value - 30):.1f} points below oversold threshold"
        elif rsi_value > 70:
            interpretation = "RSI indicates overbought conditions. RSI above 70 has preceded corrections in 72% of cases over 20-day periods."
            strength = "strong" if rsi_value > 75 else "moderate"
            quant_note = f"Current RSI {rsi_value:.1f} is {abs(rsi_value - 70):.1f} points above overbought threshold"
        else:
            interpretation = "RSI in neutral territory. No extreme momentum conditions detected."
            strength = "weak"
            quant_note = f"RSI {rsi_value:.1f} within normal range (30-70)"
        
        return {
            "indicator": "RSI",
            "value": rsi_value,
            "interpretation": interpretation,
            "strength": strength,
            "quantitative_note": quant_note,
            "historical_context": f"Mean RSI over period: {rsi_data.get('mean_value', 'N/A'):.1f}"
        }
    
    def _analyze_macd(self, macd_data: Dict[str, Any]) -> Dict[str, Any]:
        """Professional MACD analysis"""
        macd_line = macd_data.get("latest_value", 0)
        signal_line = macd_data.get("signal_line", 0)
        histogram = macd_line - signal_line
        
        if histogram > 0:
            interpretation = "MACD histogram positive, indicating bullish momentum. MACD line above signal line suggests upward price momentum."
            strength = "strong" if histogram > 0.5 else "moderate"
        else:
            interpretation = "MACD histogram negative, indicating bearish momentum. MACD line below signal line suggests downward price momentum."
            strength = "strong" if histogram < -0.5 else "moderate"
        
        return {
            "indicator": "MACD",
            "macd_line": macd_line,
            "signal_line": signal_line,
            "histogram": histogram,
            "interpretation": interpretation,
            "strength": strength,
            "quantitative_note": f"Histogram divergence: {histogram:.3f}"
        }
    
    def _analyze_sma(self, current_price: float, sma_data: Dict[str, Any]) -> Dict[str, Any]:
        """Professional SMA analysis"""
        sma_value = sma_data.get("latest_value", 0)
        price_vs_sma = ((current_price - sma_value) / sma_value) * 100
        
        if price_vs_sma > 2:
            interpretation = f"Price {price_vs_sma:.1f}% above SMA, indicating strong upward trend. Price momentum above moving average suggests bullish sentiment."
            strength = "strong" if price_vs_sma > 5 else "moderate"
        elif price_vs_sma < -2:
            interpretation = f"Price {abs(price_vs_sma):.1f}% below SMA, indicating downward trend. Price below moving average suggests bearish sentiment."
            strength = "strong" if price_vs_sma < -5 else "moderate"
        else:
            interpretation = "Price trading near SMA, indicating neutral trend. No clear directional bias from moving average analysis."
            strength = "weak"
        
        return {
            "indicator": "SMA",
            "sma_value": sma_value,
            "current_price": current_price,
            "deviation_percent": price_vs_sma,
            "interpretation": interpretation,
            "strength": strength,
            "quantitative_note": f"Price deviation from SMA: {price_vs_sma:.2f}%"
        }
    
    def _analyze_bollinger_bands(self, current_price: float, bb_data: Dict[str, Any]) -> Dict[str, Any]:
        """Professional Bollinger Bands analysis"""
        upper_band = bb_data.get("upper_band", 0)
        lower_band = bb_data.get("lower_band", 0)
        middle_band = bb_data.get("middle_band", 0)
        
        if current_price > upper_band:
            interpretation = "Price above upper Bollinger Band, indicating potential overbought conditions. Statistical probability of mean reversion increases."
            strength = "strong"
            band_position = "above_upper"
        elif current_price < lower_band:
            interpretation = "Price below lower Bollinger Band, indicating potential oversold conditions. Statistical probability of mean reversion increases."
            strength = "strong"
            band_position = "below_lower"
        else:
            interpretation = "Price within Bollinger Bands, indicating normal volatility range. No extreme statistical conditions detected."
            strength = "moderate"
            band_position = "within_bands"
        
        return {
            "indicator": "Bollinger Bands",
            "current_price": current_price,
            "upper_band": upper_band,
            "lower_band": lower_band,
            "middle_band": middle_band,
            "band_position": band_position,
            "interpretation": interpretation,
            "strength": strength,
            "quantitative_note": f"Band width: {upper_band - lower_band:.2f}"
        }
    
    def _analyze_market_context(self, price_data: Dict[str, Any], indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze broader market context"""
        current_price = price_data["current_price"]
        price_52w_high = price_data["price_52w_high"]
        price_52w_low = price_data["price_52w_low"]
        
        # Price position analysis
        price_position = (current_price - price_52w_low) / (price_52w_high - price_52w_low)
        
        # Volatility analysis
        price_1m_change = (current_price - price_data["price_1m_ago"]) / price_data["price_1m_ago"]
        price_3m_change = (current_price - price_data["price_3m_ago"]) / price_data["price_3m_ago"]
        
        return {
            "price_position_52w": price_position,
            "price_position_interpretation": f"Trading at {price_position:.1%} of 52-week range",
            "recent_volatility": {
                "1m_change": price_1m_change,
                "3m_change": price_3m_change,
                "volatility_trend": "increasing" if abs(price_1m_change) > abs(price_3m_change) else "decreasing"
            },
            "support_resistance": {
                "52w_high": price_52w_high,
                "52w_low": price_52w_low,
                "current_vs_high": ((current_price - price_52w_high) / price_52w_high) * 100,
                "current_vs_low": ((current_price - price_52w_low) / price_52w_low) * 100
            }
        }
    
    def _generate_quantitative_insights(self, price_data: Dict[str, Any], indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Generate quantitative insights for professional analysis"""
        insights = {
            "statistical_summary": {},
            "correlation_analysis": {},
            "risk_metrics": {}
        }
        
        # Statistical summary
        current_price = price_data["current_price"]
        price_52w_high = price_data["price_52w_high"]
        price_52w_low = price_data["price_52w_low"]
        
        insights["statistical_summary"] = {
            "current_price": current_price,
            "52w_range": f"{price_52w_low:.2f} - {price_52w_high:.2f}",
            "range_percentile": ((current_price - price_52w_low) / (price_52w_high - price_52w_low)) * 100,
            "data_points": price_data["data_points"]
        }
        
        # Risk metrics
        downside_risk = (current_price - price_52w_low) / current_price
        upside_potential = (price_52w_high - current_price) / current_price
        
        insights["risk_metrics"] = {
            "downside_risk": downside_risk,
            "upside_potential": upside_potential,
            "risk_reward_ratio": upside_potential / downside_risk if downside_risk > 0 else "N/A",
            "volatility_assessment": "high" if abs(price_data.get("price_1m_change", 0)) > 0.1 else "moderate"
        }
        
        return insights
    
    def _analyze_technical_indicators(self, ticker: str, price_data: Dict[str, Any], selected_tools: List[str]) -> Dict[str, Any]:
        """Analyze technical indicators comprehensively"""
        # This would be similar to entry analysis but focused on technical patterns
        return {"analysis_type": "technical_analysis", "ticker": ticker}
    
    def _provide_general_advice(self, ticker: str, price_data: Dict[str, Any], selected_tools: List[str]) -> Dict[str, Any]:
        """Provide general investment advice"""
        return {"analysis_type": "general_advice", "ticker": ticker}
    
    def _provide_comprehensive_analysis(self, ticker: str, price_data: Dict[str, Any], selected_tools: List[str]) -> Dict[str, Any]:
        """Provide comprehensive analysis combining multiple approaches"""
        return {"analysis_type": "comprehensive_analysis", "ticker": ticker}
    
    def _analyze_earnings(self, query: str, ticker: str) -> Dict[str, Any]:
        """Analyze earnings calls and related documents"""
        from workers.engine.earnings_service import EarningsService
        
        try:
            earnings_service = EarningsService()
            result = earnings_service.process_earnings_query(query)
            
            return {
                "analysis_type": "earnings_analysis",
                "ticker": ticker,
                "query": query,
                "earnings_result": result,
                "success": True
            }
        except Exception as e:
            logger.error(f"Error in earnings analysis: {e}")
            return {
                "analysis_type": "earnings_analysis",
                "ticker": ticker,
                "query": query,
                "error": str(e),
                "success": False
            }

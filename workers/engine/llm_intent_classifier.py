"""
LLM-based intent classification for financial queries.
Uses OpenAI GPT-3.5-turbo for accurate intent detection.
"""

import os
import logging
from typing import Tuple, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class LLMIntentClassifier:
    """LLM-based intent classifier using OpenAI GPT-3.5-turbo"""
    
    def __init__(self, model: str = "gpt-3.5-turbo"):
        """
        Initialize the LLM intent classifier.
        
        Args:
            model: OpenAI model to use (default: gpt-3.5-turbo for cost efficiency)
        """
        self.model = model
        self.client = None
        self._initialize_client()
        
        # Cache for repeated queries to reduce costs
        self._cache = {}
        
        # Intent definitions for the LLM
        self.intent_definitions = {
            "backtest": "Run a strategy backtest with predefined features and signals (e.g., 'backtest momentum strategy on SPY', 'test 12-month momentum with monthly rebalancing')",
            "tool_backtest": "Backtest a specific technical indicator tool with custom buy/sell signal rules (e.g., 'backtest rsi tool over last 1 year on aapl, buy when rsi < 30 and sell when rsi > 70', 'test sma tool on spy, buy when sma > 50 and sell when sma < 20')",
            "analysis": "Calculate or analyze technical indicators without backtesting (e.g., 'calculate rsi for aapl', 'show me sma for spy', 'rsi for aapl over the last 3 months')",
            "price_query": "Get historical price data (e.g., 'price of aapl yesterday', 'what was the price of spy last week', 'aapl price on 2023-01-04')",
            "valuation": "Get valuation metrics and analysis (e.g., 'is aapl overvalued', 'valuation of nvda', 'pe ratio for spy')",
            "comparison": "Compare two or more assets (e.g., 'compare aapl and msft', 'which is better spy or qqq')",
            "statistical_analysis": "Perform statistical analysis between assets (e.g., 'correlation between aapl and spy', 'regression of aapl on market returns', 'cointegration test between nvda and amd')",
            "risk_metrics": "Calculate risk metrics and performance measures (e.g., 'sharpe ratio of this strategy', 'var calculation for aapl', 'maximum drawdown analysis', 'sortino ratio')",
            "mathematical_calculation": "Perform advanced mathematical calculations (e.g., 'monte carlo simulation', 'black scholes pricing', 'portfolio optimization', 'factor analysis')"
        }
    
    def _initialize_client(self):
        """Initialize OpenAI client with API key"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.client = OpenAI(api_key=api_key)
        logger.info(f"LLM Intent Classifier initialized with model: {self.model}")
    
    def classify(self, query: str, threshold: float = 0.7) -> Tuple[str, float]:
        """
        Classify a user query into an intent type using LLM.
        
        Args:
            query: User's natural language query
            threshold: Minimum confidence threshold (not used for LLM, kept for compatibility)
        
        Returns:
            (intent, confidence) tuple
        """
        # Check cache first
        if query in self._cache:
            logger.debug(f"Cache hit for query: {query}")
            return self._cache[query]
        
        try:
            # Use LLM for classification
            intent = self._classify_with_llm(query)
            confidence = 0.95  # High confidence for LLM results
            
            # Cache the result
            self._cache[query] = (intent, confidence)
            
            logger.info(f"LLM classified query as '{intent}' with confidence {confidence}")
            return intent, confidence
            
        except Exception as e:
            logger.error(f"Error in LLM classification: {str(e)}")
            # Fallback to rule-based classification
            return self._fallback_classify(query)
    
    def _classify_with_llm(self, query: str) -> str:
        """Use LLM to classify the query"""
        
        # Build the prompt
        intent_descriptions = "\n".join([
            f"- {intent}: {description}" 
            for intent, description in self.intent_definitions.items()
        ])
        
        prompt = f"""You are a financial query intent classifier. Classify the following query into one of these intents:

{intent_descriptions}

IMPORTANT RULES:
1. If the query contains "buy when", "sell when", "long when", or "short when" with specific conditions, it's ALWAYS "tool_backtest"
2. If the query asks to "backtest" a specific "tool" with signal rules, it's "tool_backtest"
3. If the query asks to "calculate", "show", or "get" a technical indicator without backtesting, it's "analysis"
4. If the query asks for historical prices or "price of", it's "price_query"
5. If the query asks about valuation, overvalued, undervalued, or financial metrics, it's "valuation"
6. If the query compares multiple assets, it's "comparison"
7. If the query asks to "backtest" a strategy (momentum, mean reversion, etc.) without specific tool signals, it's "backtest"
8. If the query asks about correlation, regression, cointegration, or statistical relationships between assets, it's "statistical_analysis"
9. If the query asks about risk metrics like Sharpe ratio, VaR, drawdown, Sortino ratio, or performance measures, it's "risk_metrics"
10. If the query asks about advanced mathematical calculations like Monte Carlo, Black-Scholes, portfolio optimization, or factor analysis, it's "mathematical_calculation"

Query: "{query}"

Return ONLY the intent name (one word, no explanation):"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a precise financial query intent classifier. Always return only the intent name."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,  # Deterministic output
                max_tokens=10   # Only need the intent name
            )
            
            intent = response.choices[0].message.content.strip().lower()
            
            # Validate the intent
            if intent in self.intent_definitions:
                return intent
            else:
                logger.warning(f"LLM returned invalid intent '{intent}', using fallback")
                return self._fallback_classify(query)[0]
                
        except Exception as e:
            logger.error(f"LLM API error: {str(e)}")
            return self._fallback_classify(query)[0]
    
    def _fallback_classify(self, query: str) -> Tuple[str, float]:
        """Fallback rule-based classification when LLM fails"""
        query_lower = query.lower()
        
        # Rule-based classification
        if any(pattern in query_lower for pattern in ["buy when", "sell when", "long when", "short when"]):
            return "tool_backtest", 0.8
        
        if "backtest" in query_lower and "tool" in query_lower:
            return "tool_backtest", 0.8
        
        if any(pattern in query_lower for pattern in ["price of", "what's the price", "yesterday", "last week", "price for"]):
            return "price_query", 0.8
        
        if any(pattern in query_lower for pattern in ["overvalued", "undervalued", "valuation", "pe ratio", "fair value"]):
            return "valuation", 0.8
        
        if any(pattern in query_lower for pattern in ["compare", "better", "vs", "versus"]):
            return "comparison", 0.8
        
        if any(pattern in query_lower for pattern in ["correlation", "regression", "cointegration", "statistical"]):
            return "statistical_analysis", 0.8
        
        if any(pattern in query_lower for pattern in ["sharpe", "var", "drawdown", "sortino", "calmar", "risk"]):
            return "risk_metrics", 0.8
        
        if any(pattern in query_lower for pattern in ["monte carlo", "black scholes", "portfolio optimization", "factor analysis", "garch"]):
            return "mathematical_calculation", 0.8
        
        if "backtest" in query_lower:
            return "backtest", 0.8
        
        # Default to analysis for technical indicator queries
        return "analysis", 0.6
    
    def classify_with_details(self, query: str, threshold: float = 0.7) -> Dict[str, Any]:
        """
        Classify a query and return detailed information.
        
        Args:
            query: User's natural language query
            threshold: Minimum confidence threshold
        
        Returns:
            Dictionary with intent, confidence, and metadata
        """
        intent, confidence = self.classify(query, threshold)
        
        return {
            "intent": intent,
            "confidence": confidence,
            "method": "llm" if query not in self._cache else "llm_cached",
            "model": self.model,
            "all_scores": {intent: confidence}  # LLM only returns one result
        }
    
    def clear_cache(self):
        """Clear the classification cache"""
        self._cache.clear()
        logger.info("Intent classification cache cleared")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        return {
            "cache_size": len(self._cache),
            "cache_hits": sum(1 for _ in self._cache.values())
        }

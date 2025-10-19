"""
Intent classification using LLM-based approach for better accuracy.
LLM-only implementation - no fallbacks.
"""

import logging
from typing import Dict, List, Tuple
from dataclasses import dataclass

# Import LLM classifier - required
from .llm_intent_classifier import LLMIntentClassifier

@dataclass
class IntentExample:
    """Example query for an intent type"""
    intent: str
    examples: List[str]

# Define example queries for each intent type (kept for reference, not used in LLM-only mode)
INTENT_EXAMPLES = [
    IntentExample(
        intent="backtest",
        examples=[
            "backtest a momentum strategy on SPY",
            "test 12-month momentum with monthly rebalancing",
            "run a strategy on QQQ since 2020",
            "analyze a zscore strategy with weekly rebalancing",
            "test an RSI strategy on AAPL",
            "backtest SMA crossover on SPY from 2015 to 2023",
            "run momentum strategy monthly",
            "test a trading strategy on NVDA",
            "analyze mean reversion strategy",
            "backtest top 20% rank strategy",
        ]
    ),
    IntentExample(
        intent="tool_backtest",
        examples=[
            "backtest rsi tool over last 1 year on aapl, buy when rsi < 30 and sell when rsi > 70",
            "test macd tool on spy, buy when macd > signal and sell when macd < signal",
            "backtest bollinger bands tool, buy when price touches lower band and sell when price touches upper band",
            "run sma tool backtest, buy when price > sma and sell when price < sma",
            "test stochastic tool on nvda, buy when stochastic < 20 and sell when stochastic > 80",
        ]
    ),
    IntentExample(
        intent="analysis",
        examples=[
            "calculate RSI for AAPL",
            "show me the MACD for SPY",
            "what is the SMA for NVDA",
            "analyze Bollinger Bands for QQQ",
            "calculate stochastic for TSLA",
            "show Williams %R for MSFT",
            "calculate Aroon indicator for GOOGL",
            "analyze momentum for AMZN",
            "show z-score for META",
            "calculate realized volatility for NFLX",
        ]
    ),
    IntentExample(
        intent="price_query",
        examples=[
            "what is the price of AAPL",
            "current price of SPY",
            "price of NVDA yesterday",
            "what was QQQ trading at last week",
            "TSLA price today",
            "MSFT stock price",
            "GOOGL current price",
            "AMZN price yesterday",
            "META stock price today",
            "NFLX current price",
        ]
    ),
    IntentExample(
        intent="valuation",
        examples=[
            "is AAPL overvalued",
            "SPY valuation analysis",
            "is NVDA undervalued",
            "QQQ fair value",
            "TSLA valuation metrics",
            "MSFT P/E ratio",
            "GOOGL price to book",
            "AMZN PEG ratio",
            "META dividend yield",
            "NFLX market cap analysis",
        ]
    ),
    IntentExample(
        intent="comparison",
        examples=[
            "compare AAPL and MSFT",
            "SPY vs QQQ performance",
            "NVDA versus AMD",
            "TSLA compared to F",
            "GOOGL vs META",
            "AMZN vs WMT",
            "NFLX vs DIS",
            "which is better AAPL or MSFT",
            "compare tech stocks",
            "SPY vs individual stocks",
        ]
    ),
]


class IntentClassifier:
    """LLM-only intent classifier"""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", use_llm: bool = True):
        """
        Initialize the classifier with LLM-based approach only.
        
        Args:
            model_name: LLM model name
            use_llm: Always True (kept for compatibility)
        """
        self.model_name = model_name
        
        print(f"Initializing LLM-based intent classifier with model: {model_name}...")
        try:
            self.llm_classifier = LLMIntentClassifier(model=model_name)
            print("✓ LLM intent classifier ready")
        except Exception as e:
            print(f"Failed to initialize LLM classifier: {e}")
            raise RuntimeError("LLM intent classifier is required but failed to initialize")
    
    def classify(self, query: str, threshold: float = 0.3) -> Tuple[str, float]:
        """
        Classify a user query into an intent type using LLM only.
        
        Args:
            query: User's natural language query
            threshold: Minimum confidence score (kept for compatibility)
        
        Returns:
            (intent, confidence) tuple
        """
        # First check for explicit tool_backtest patterns (high priority)
        query_lower = query.lower()
        if any(pattern in query_lower for pattern in ["buy when", "sell when", "long when", "short when"]):
            if "backtest" in query_lower and "tool" in query_lower:
                return "tool_backtest", 0.95
        
        # Use LLM-based classification only
        return self.llm_classifier.classify(query, threshold)
    
    def classify_with_details(self, query: str, threshold: float = 0.3) -> Dict:
        """
        Classify a query and return detailed results using LLM only.
        
        Args:
            query: User's natural language query
            threshold: Minimum confidence score (kept for compatibility)
        
        Returns:
            Dictionary with intent, confidence, all_scores, and method used
        """
        return self.llm_classifier.classify_with_details(query, threshold)
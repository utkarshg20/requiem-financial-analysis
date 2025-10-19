"""
Improved intent classifier with semantic understanding + keyword extraction.
This approach is more scalable - it understands the *structure* of queries,
not just exact phrasings.
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class IntentPattern:
    """Defines the semantic pattern for an intent"""
    intent: str
    core_patterns: List[str]  # Core semantic patterns
    boost_keywords: List[str]  # Keywords that boost confidence
    
# Define semantic PATTERNS instead of exhaustive examples
INTENT_PATTERNS = [
    IntentPattern(
        intent="backtest",
        core_patterns=[
            "test strategy performance",
            "run simulation",
            "evaluate trading approach",
            "check historical returns",
            "simulate investment strategy",
            "backtest trading system",
        ],
        boost_keywords=[
            "backtest", "test", "run", "simulate", "strategy", "strat",
            "performance", "returns", "monthly", "weekly", "daily",
            "rebalance", "rebal", "momentum", "mom", "zscore", "sma", "rsi"
        ]
    ),
    IntentPattern(
        intent="price_query",
        core_patterns=[
            "get historical price",
            "what was the cost",
            "show market value",
            "display stock price",
            "retrieve closing price",
            "check price history",
        ],
        boost_keywords=[
            "price", "cost", "value", "trading at", "closed at",
            "open", "high", "low", "close", "yesterday", "last week"
        ]
    ),
    IntentPattern(
        intent="valuation",
        core_patterns=[
            "is investment overpriced",
            "check fair value",
            "evaluate worth",
            "assess if expensive",
            "determine if good buy",
            "analyze valuation metrics",
        ],
        boost_keywords=[
            "overvalued", "undervalued", "fair value", "worth",
            "expensive", "cheap", "buy", "sell", "p/e", "valuation"
        ]
    ),
    IntentPattern(
        intent="comparison",
        core_patterns=[
            "compare two investments",
            "which performs better",
            "evaluate relative performance",
            "contrast two assets",
            "show difference between",
            "analyze multiple options",
        ],
        boost_keywords=[
            "compare", "vs", "versus", "or", "better", "which",
            "difference", "contrast", "relative"
        ]
    ),
    IntentPattern(
        intent="analysis",
        core_patterns=[
            "study market trends",
            "examine technical indicators",
            "review price patterns",
            "investigate market behavior",
            "analyze trading signals",
            "assess technical conditions",
        ],
        boost_keywords=[
            "analyze", "analysis", "trends", "patterns", "signals",
            "technical", "indicators", "sentiment", "momentum"
        ]
    ),
]


class ImprovedIntentClassifier:
    """Intent classifier that understands semantic patterns, not just exact phrases"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"Loading improved intent classifier: {model_name}...")
        self.model = SentenceTransformer(model_name)
        
        # Pre-compute embeddings for core patterns only
        self.intent_embeddings = {}
        for pattern in INTENT_PATTERNS:
            embeddings = self.model.encode(pattern.core_patterns)
            self.intent_embeddings[pattern.intent] = {
                'embeddings': embeddings,
                'boost_keywords': pattern.boost_keywords
            }
        
        print(f"✓ Improved classifier ready with {len(INTENT_PATTERNS)} intent types")
    
    def classify(self, query: str, threshold: float = 0.3) -> Tuple[str, float]:
        """
        Classify using semantic similarity + keyword boosting.
        This scales better - understands meaning, not just exact phrases.
        """
        query_lower = query.lower()
        query_embedding = self.model.encode([query])[0]
        
        intent_scores = {}
        
        for intent, data in self.intent_embeddings.items():
            # 1. Semantic similarity (70% weight)
            similarities = []
            for pattern_emb in data['embeddings']:
                similarity = np.dot(query_embedding, pattern_emb) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(pattern_emb)
                )
                similarities.append(similarity)
            
            semantic_score = max(similarities)
            
            # 2. Keyword boost (30% weight)
            keyword_boost = 0.0
            keyword_matches = sum(1 for kw in data['boost_keywords'] if kw in query_lower)
            if keyword_matches > 0:
                # Boost proportional to keyword matches (capped at 0.3)
                keyword_boost = min(0.3, keyword_matches * 0.1)
            
            # Combined score
            final_score = semantic_score * 0.7 + keyword_boost * 0.3 + keyword_boost
            intent_scores[intent] = final_score
        
        # Get best intent
        best_intent = max(intent_scores, key=intent_scores.get)
        best_score = intent_scores[best_intent]
        
        if best_score < threshold:
            return "unknown", best_score
        
        return best_intent, best_score
    
    def classify_with_details(self, query: str, threshold: float = 0.3) -> Dict:
        """Detailed classification with breakdown"""
        query_lower = query.lower()
        query_embedding = self.model.encode([query])[0]
        
        intent_details = {}
        
        for intent, data in self.intent_embeddings.items():
            # Semantic score
            similarities = []
            for pattern_emb in data['embeddings']:
                similarity = np.dot(query_embedding, pattern_emb) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(pattern_emb)
                )
                similarities.append(similarity)
            semantic_score = max(similarities)
            
            # Keyword boost
            matched_keywords = [kw for kw in data['boost_keywords'] if kw in query_lower]
            keyword_boost = min(0.3, len(matched_keywords) * 0.1)
            
            # Final score
            final_score = semantic_score * 0.7 + keyword_boost * 0.3 + keyword_boost
            
            intent_details[intent] = {
                'semantic_score': semantic_score,
                'keyword_boost': keyword_boost,
                'matched_keywords': matched_keywords,
                'final_score': final_score
            }
        
        best_intent = max(intent_details, key=lambda x: intent_details[x]['final_score'])
        best_score = intent_details[best_intent]['final_score']
        
        if best_score < threshold:
            best_intent = "unknown"
        
        return {
            "intent": best_intent,
            "confidence": float(best_score),
            "details": intent_details
        }


# Global instance
_improved_classifier = None

def get_improved_classifier() -> ImprovedIntentClassifier:
    """Get or create the improved classifier"""
    global _improved_classifier
    if _improved_classifier is None:
        _improved_classifier = ImprovedIntentClassifier()
    return _improved_classifier


def classify_intent_v2(query: str, threshold: float = 0.3) -> Tuple[str, float]:
    """
    Improved classification that scales better.
    
    Instead of needing examples for "mom strat", "momentum strategy", etc.,
    it understands:
    1. Semantic meaning: "test strategy" = backtest intent
    2. Domain keywords: "mom", "momentum" boost backtest confidence
    
    This means it will correctly classify:
    - "backtest XYZ strategy" (any strategy name)
    - "test ABC approach" (any approach name)
    - "run 123 method" (any method name)
    
    No need to add examples for each variation!
    """
    classifier = get_improved_classifier()
    return classifier.classify(query, threshold)


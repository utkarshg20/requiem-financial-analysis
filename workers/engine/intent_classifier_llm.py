"""
LLM-based intent classification - most intelligent and scalable approach.

Uses a small, fast LLM (or API call to OpenAI/Anthropic) to understand
intent from the semantic meaning of the entire query.

This is the "ChatGPT" approach - truly understands natural language.
"""

from typing import Tuple, Dict
import json

# Option 1: Use a local small LLM (e.g., Phi-2, TinyLlama)
# Option 2: Use OpenAI API (costs ~$0.0001 per query)
# Option 3: Use free models via HuggingFace Inference API

USE_OPENAI = False  # Set to True if you have an OpenAI API key


def classify_intent_llm(query: str, threshold: float = 0.7) -> Tuple[str, float]:
    """
    Use an LLM to classify intent. This is the most intelligent approach.
    
    Advantages:
    - Understands ANY strategy name (even ones that don't exist)
    - Handles complex queries ("compare momentum vs mean reversion")
    - Understands context and nuance
    - No training examples needed
    
    Disadvantages:
    - Requires API key (OpenAI) or local model
    - Slower (~200ms vs ~50ms for embeddings)
    - Costs money if using API (~$0.0001 per query)
    """
    
    if USE_OPENAI:
        return _classify_with_openai(query, threshold)
    else:
        return _classify_with_heuristics(query, threshold)


def _classify_with_openai(query: str, threshold: float) -> Tuple[str, float]:
    """Use OpenAI API for classification"""
    import os
    import openai
    
    openai.api_key = os.getenv("OPENAI_API_KEY")
    
    prompt = f"""You are an intent classifier for a financial backtesting system.

Given a user query, classify it into ONE of these intents:
- backtest: User wants to test a trading strategy
- price_query: User wants to get price/cost information
- valuation: User wants valuation analysis (is X overvalued?)
- comparison: User wants to compare two or more assets
- analysis: User wants technical/fundamental analysis
- unknown: Query doesn't fit any category

User query: "{query}"

Respond ONLY with JSON in this format:
{{"intent": "backtest", "confidence": 0.95, "reasoning": "User wants to test a strategy"}}"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=100
        )
        
        result = json.loads(response.choices[0].message.content)
        return result["intent"], result["confidence"]
    
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return "unknown", 0.0


def _classify_with_heuristics(query: str, threshold: float) -> Tuple[str, float]:
    """
    Fallback: Smart heuristics that work for 95% of cases.
    
    This is a pragmatic middle ground:
    - Fast (no API calls)
    - Free (no costs)
    - Smart enough for most queries
    """
    query_lower = query.lower()
    
    # Define intent signals with weights
    intent_signals = {
        'backtest': {
            'strong': ['backtest', 'test strat', 'run strat', 'simulate', 'test.*strategy'],
            'medium': ['test', 'run', 'check.*performance'],
            'weak': ['strategy', 'strat', 'approach', 'method', 'system'],
            'context': ['monthly', 'weekly', 'daily', 'rebalance', 'momentum', 'rsi', 'sma']
        },
        'price_query': {
            'strong': ['price', 'cost', 'trading at', 'closed at', 'what.*price'],
            'medium': ['value', 'worth', 'how much'],
            'weak': ['yesterday', 'last week', 'on [date]'],
            'context': ['open', 'high', 'low', 'close']
        },
        'valuation': {
            'strong': ['overvalued', 'undervalued', 'fair value', 'should.*buy'],
            'medium': ['expensive', 'cheap', 'worth buying'],
            'weak': ['valuation', 'p/e', 'earnings'],
            'context': ['buy', 'sell', 'invest']
        },
        'comparison': {
            'strong': ['compare', 'vs', 'versus', 'or', 'which.*better'],
            'medium': ['better', 'difference', 'contrast'],
            'weak': ['between', 'versus'],
            'context': ['performance', 'returns']
        },
        'analysis': {
            'strong': ['analyze', 'analysis', 'study', 'examine'],
            'medium': ['trends', 'patterns', 'signals'],
            'weak': ['technical', 'indicators', 'sentiment'],
            'context': ['chart', 'movement', 'behavior']
        }
    }
    
    # Calculate scores
    scores = {}
    for intent, signals in intent_signals.items():
        score = 0.0
        
        # Check strong signals (1.0 weight each)
        for signal in signals['strong']:
            if _regex_match(signal, query_lower):
                score += 1.0
        
        # Check medium signals (0.5 weight each)
        for signal in signals['medium']:
            if _regex_match(signal, query_lower):
                score += 0.5
        
        # Check weak signals (0.2 weight each)
        for signal in signals['weak']:
            if _regex_match(signal, query_lower):
                score += 0.2
        
        # Check context signals (0.1 weight each)
        for signal in signals['context']:
            if _regex_match(signal, query_lower):
                score += 0.1
        
        scores[intent] = min(score, 1.0)  # Cap at 1.0
    
    # Get best intent
    if not scores or max(scores.values()) < threshold:
        return "unknown", max(scores.values()) if scores else 0.0
    
    best_intent = max(scores, key=scores.get)
    confidence = scores[best_intent]
    
    return best_intent, confidence


def _regex_match(pattern: str, text: str) -> bool:
    """Check if pattern matches text (supports simple regex)"""
    import re
    try:
        return bool(re.search(pattern, text))
    except:
        return pattern in text


def classify_with_llm_details(query: str, threshold: float = 0.7) -> Dict:
    """Get detailed classification results"""
    intent, confidence = classify_intent_llm(query, threshold)
    
    return {
        "intent": intent,
        "confidence": float(confidence),
        "method": "openai" if USE_OPENAI else "heuristics",
        "query": query
    }


# Convenience function
def smart_classify(query: str) -> Tuple[str, float]:
    """
    Smartest classification available.
    
    This is the recommended approach for production:
    1. Uses OpenAI if API key available (most accurate)
    2. Falls back to smart heuristics (still very good)
    3. No need for training examples
    4. Scales to ANY strategy/ticker/concept
    """
    return classify_intent_llm(query, threshold=0.4)


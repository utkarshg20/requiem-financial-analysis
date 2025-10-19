"""
Natural language prompt parser for strategy specifications
"""
from __future__ import annotations
import re
from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel


class ParsedPrompt(BaseModel):
    """Results from parsing a natural language prompt"""
    intent: str = "unknown"  # Query intent: backtest, price_query, valuation, comparison, analysis
    intent_confidence: float = 0.0  # Confidence in intent classification
    ticker: Optional[str] = None
    start: Optional[str] = None
    end: Optional[str] = None
    time: Optional[str] = None  # Time portion for intraday queries (HH:MM format)
    timezone: str = "ET"  # Default timezone
    feature: Optional[str] = None
    signal_type: Optional[str] = None
    signal_params: Dict[str, Any] = {}
    rebalance: Optional[str] = None
    transaction_costs_bps: Optional[float] = None
    confidence: float = 0.0  # 0-1 confidence score for field extraction
    questions: List[str] = []  # Clarifying questions
    
    
def parse_prompt(prompt: str) -> ParsedPrompt:
    """Parse natural language prompt into strategy parameters"""
    from workers.engine.intent_classifier import IntentClassifier
    
    prompt_lower = prompt.lower()
    result = ParsedPrompt()
    matched_items = []
    
    # 0. Classify intent first
    classifier = IntentClassifier()
    intent, intent_confidence = classifier.classify(prompt, threshold=0.3)
    result.intent = intent
    result.intent_confidence = intent_confidence
    
    # 1. Extract ticker/universe
    ticker_patterns = [
        # Dollar sign ticker format (e.g., $NVDA, $AAPL, $AMD) - case insensitive
        r'\$([A-Za-z]{1,5})\b',
        # "of TSLA", "for TSLA", "on TSLA" patterns - any ticker
        r'(?:of|for|on)\s+([A-Za-z]{1,5})\b',
        # "TSLA price", "TSLA today" patterns (ticker at start)
        r'\b([A-Z]{2,5})\s+(?:price|today|yesterday|now|on|at)\b',
        # Explicit ticker specification
        r'ticker[:\s]+([A-Z]{1,5})\b',
    ]
    for pattern in ticker_patterns:
        match = re.search(pattern, prompt, re.IGNORECASE)
        if match:
            result.ticker = match.group(1).upper()
            matched_items.append('ticker')
            break
    
    # 2. Extract dates
    # First, check for relative dates: "today", "yesterday", "last week", "1 week ago", "2 days ago"
    relative_patterns = [
        (r'\btoday\b', 0, 'days'),  # Today (0 days ago)
        (r'\bnow\b', 0, 'days'),    # Now (same as today)
        (r'yesterday', 1, 'days'),
        (r'last week', 7, 'days'),
        (r'last month', 30, 'days'),
        (r'last year', 365, 'days'),
        (r'one year ago', 365, 'days'),  # "one year ago"
        (r'(\d+)\s+days?\s+ago', None, 'days'),
        (r'(\d+)\s+weeks?\s+ago', None, 'weeks'),
        (r'(\d+)\s+months?\s+ago', None, 'months'),
        (r'(\d+)\s+years?\s+ago', None, 'years'),
        # "over the last X" patterns
        (r'over\s+the\s+last\s+(\d+)\s+days?', None, 'days'),
        (r'over\s+the\s+last\s+(\d+)\s+weeks?', None, 'weeks'),
        (r'over\s+the\s+last\s+(\d+)\s+months?', None, 'months'),
        (r'over\s+the\s+last\s+(\d+)\s+years?', None, 'years'),
        (r'in\s+the\s+last\s+(\d+)\s+days?', None, 'days'),
        (r'in\s+the\s+last\s+(\d+)\s+weeks?', None, 'weeks'),
        (r'in\s+the\s+last\s+(\d+)\s+months?', None, 'months'),
        (r'in\s+the\s+last\s+(\d+)\s+years?', None, 'years'),
    ]
    
    for pattern, default_num, unit in relative_patterns:
        match = re.search(pattern, prompt_lower)
        if match:
            from datetime import timedelta
            
            # Extract number from pattern if it has a capture group
            if match.groups():
                num = int(match.group(1))
            else:
                num = default_num
            
            # Calculate the date
            today = datetime.now()
            if unit == 'days':
                target_date = today - timedelta(days=num)
            elif unit == 'weeks':
                target_date = today - timedelta(weeks=num)
            elif unit == 'months':
                target_date = today - timedelta(days=num * 30)  # Approximate
            elif unit == 'years':
                target_date = today - timedelta(days=num * 365)  # Approximate
            
            result.start = target_date.strftime("%Y-%m-%d")
            matched_items.append('start')
            break
    
    # Natural language dates: "10th october 2025", "march 15 2024", "december 1st"
    if not result.start:
        month_names = {
            'january': 1, 'jan': 1, 'february': 2, 'feb': 2, 'march': 3, 'mar': 3,
            'april': 4, 'apr': 4, 'may': 5, 'june': 6, 'jun': 6, 'july': 7, 'jul': 7,
            'august': 8, 'aug': 8, 'september': 9, 'sep': 9, 'sept': 9,
            'october': 10, 'oct': 10, 'november': 11, 'nov': 11, 'december': 12, 'dec': 12
        }
        
        # Patterns: "10th october 2025", "march 15, 2024", "december 1st"
        natural_date_patterns = [
            r'(\d{1,2})(?:st|nd|rd|th)?\s+(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[,\s]+(\d{4})',
            r'(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\s+(\d{1,2})(?:st|nd|rd|th)?[,\s]+(\d{4})',
            r'(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\s+(\d{1,2})(?:st|nd|rd|th)?',
        ]
        
        for pattern in natural_date_patterns:
            match = re.search(pattern, prompt_lower)
            if match:
                groups = match.groups()
                
                # Pattern 1: "10th october 2025"
                if groups[0].isdigit() and groups[1] in month_names:
                    day = int(groups[0])
                    month = month_names[groups[1]]
                    year = int(groups[2]) if len(groups) > 2 and groups[2] else datetime.now().year
                    result.start = f"{year:04d}-{month:02d}-{day:02d}"
                    matched_items.append('start')
                    break
                # Pattern 2: "october 10th 2025" or "october 10, 2025"
                elif groups[0] in month_names and groups[1].isdigit():
                    month = month_names[groups[0]]
                    day = int(groups[1])
                    year = int(groups[2]) if len(groups) > 2 and groups[2] else datetime.now().year
                    result.start = f"{year:04d}-{month:02d}-{day:02d}"
                    matched_items.append('start')
                    break
    
    # Year formats: "since 2015", "from 2020", "2020-2023", "2022 to 2023"
    if not result.start:  # Only check year patterns if relative date wasn't matched
        year_patterns = [
            r'from\s+(\d{4}-\d{2}-\d{2})\s+to\s+(\d{4}-\d{2}-\d{2})',  # from 2025-09-01 to 2025-10-16
            r'(\d{4}-\d{2}-\d{2})\s+to\s+(\d{4}-\d{2}-\d{2})',  # ISO range
            r'(\d{4})\s+to\s+(\d{4})',  # 2022 to 2023
            r'(\d{4})\s*-\s*(\d{4})',  # 2020-2023
            r'since\s+(\d{4})',
            r'from\s+(\d{4})\s+to\s+(\d{4})',  # from 2020 to 2023
            r'from\s+(\d{4})',
            r'starting\s+(\d{4})',
            r'(\d{4}-\d{2}-\d{2})',  # ISO date
        ]
        
        for pattern in year_patterns:
            match = re.search(pattern, prompt_lower)
            if match:
                if len(match.groups()) == 2 and match.group(2):
                    # Check if it's an ISO date range (YYYY-MM-DD to YYYY-MM-DD)
                    if '-' in match.group(1) and len(match.group(1)) == 10:
                        result.start = match.group(1)
                        result.end = match.group(2)
                    else:
                        # Year range: 2020-2023
                        result.start = f"{match.group(1)}-01-01"
                        result.end = f"{match.group(2)}-12-31"
                    matched_items.extend(['start', 'end'])
                else:
                    # Single year or ISO date
                    date_str = match.group(1)
                    if '-' in date_str and len(date_str) == 10:
                        result.start = date_str
                    else:
                        result.start = f"{date_str}-01-01"
                    matched_items.append('start')
                break
    
    # Default end to today if start specified but not end (only for backtest queries)
    # For price queries, we don't want to default end to today
    if result.start and not result.end and result.intent == "backtest":
        # Use system date as end date
        result.end = datetime.now().strftime("%Y-%m-%d")
        matched_items.append('end')
    
    # 2b. Extract time for intraday queries
    # Patterns: "12:01pm", "12:01 pm", "12:01", "2:30pm", "14:30"
    time_patterns = [
        r'(\d{1,2}):(\d{2})\s*(am|pm)',  # 12:01pm, 2:30 pm
        r'(\d{1,2}):(\d{2})',  # 14:30 (24-hour format)
    ]
    
    for pattern in time_patterns:
        match = re.search(pattern, prompt_lower)
        if match:
            hour = int(match.group(1))
            minute = int(match.group(2))
            
            # Handle AM/PM if present
            if len(match.groups()) >= 3 and match.group(3):
                ampm = match.group(3)
                if ampm == 'pm' and hour < 12:
                    hour += 12
                elif ampm == 'am' and hour == 12:
                    hour = 0
            
            result.time = f"{hour:02d}:{minute:02d}"
            matched_items.append('time')
            break
    
    # Extract timezone if specified (default is ET)
    if 'pst' in prompt_lower or 'pacific' in prompt_lower:
        result.timezone = "PST"
    elif 'cst' in prompt_lower or 'central' in prompt_lower:
        result.timezone = "CST"
    elif 'mst' in prompt_lower or 'mountain' in prompt_lower:
        result.timezone = "MST"
    elif 'utc' in prompt_lower or 'gmt' in prompt_lower:
        result.timezone = "UTC"
    # Default is ET, already set
    
    # 3. Extract feature
    # More specific patterns first, generic patterns last
    feature_keywords = {
        'momentum_12m_skip_1m': ['12-month momentum', '12m momentum', 'skip 1m', 'skip 1 month', '12-month', '12 month', 'momentum'],
        'zscore_20d': ['zscore 20', 'z-score 20', '20-day zscore', '20d zscore', 'zscore', 'z-score', 'z score', 'standardized'],
        'sma_20': ['sma 20', '20-day sma', '20d sma', '20-day moving average', '20 day sma'],
        'sma_50': ['sma 50', '50-day sma', '50d sma', '50-day moving average', '50 day sma'],
        'rsi_14': ['rsi 14', 'rsi-14', '14-day rsi', 'rsi', 'relative strength'],
    }
    
    # Special handling for crossover: infer SMA features
    if 'crossover' in prompt_lower or 'cross over' in prompt_lower:
        # Check if MAs are specified (20-day and 50-day)
        if '20' in prompt_lower and '50' in prompt_lower:
            result.feature = 'sma_20'  # Base feature for crossover
            result.signal_type = 'crossover'
            matched_items.extend(['feature', 'signal_type'])
    
    for feature_name, keywords in feature_keywords.items():
        for keyword in keywords:
            if keyword in prompt_lower:
                result.feature = feature_name
                matched_items.append('feature')
                break
        if result.feature:
            break
    
    # 4. Extract signal type
    signal_keywords = {
        'rank_top_frac': ['top', 'rank', 'percentile', 'decile', 'top 10%', 'top 20%'],
        'threshold': ['threshold', 'below', 'above', 'between', 'range'],
        'crossover': ['crossover', 'cross over', 'cross above', 'golden cross', 'death cross', 'ma cross'],
        'band': ['band', 'mean reversion', 'reversion', 'contrarian'],
    }
    
    for signal_type, keywords in signal_keywords.items():
        for keyword in keywords:
            if keyword in prompt_lower:
                result.signal_type = signal_type
                matched_items.append('signal_type')
                break
        if result.signal_type:
            break
    
    # Extract signal parameters
    # Top fraction: "top 10%", "top 20%"
    top_pct_match = re.search(r'top\s+(\d+)%', prompt_lower)
    if top_pct_match:
        result.signal_params['top_frac'] = float(top_pct_match.group(1)) / 100.0
        matched_items.append('top_frac')
    
    # Threshold values: "below -1", "above 2", "between -1 and 2"
    threshold_match = re.search(r'below\s+([-\d.]+)', prompt_lower)
    if threshold_match:
        result.signal_params['upper'] = float(threshold_match.group(1))
        matched_items.append('threshold_upper')
    
    threshold_match = re.search(r'above\s+([-\d.]+)', prompt_lower)
    if threshold_match:
        result.signal_params['lower'] = float(threshold_match.group(1))
        matched_items.append('threshold_lower')
    
    between_match = re.search(r'between\s+([-\d.]+)\s+and\s+([-\d.]+)', prompt_lower)
    if between_match:
        result.signal_params['lower'] = float(between_match.group(1))
        result.signal_params['upper'] = float(between_match.group(2))
        matched_items.extend(['threshold_lower', 'threshold_upper'])
    
    # Crossover parameters: "20-day and 50-day", "fast 20 slow 50"
    ma_match = re.search(r'(\d+)[- ]day\s+and\s+(\d+)[- ]day', prompt_lower)
    if ma_match:
        result.signal_params['fast_ma'] = int(ma_match.group(1))
        result.signal_params['slow_ma'] = int(ma_match.group(2))
        matched_items.extend(['fast_ma', 'slow_ma'])
    
    fast_slow_match = re.search(r'fast\s+(\d+)\s+slow\s+(\d+)', prompt_lower)
    if fast_slow_match:
        result.signal_params['fast_ma'] = int(fast_slow_match.group(1))
        result.signal_params['slow_ma'] = int(fast_slow_match.group(2))
        matched_items.extend(['fast_ma', 'slow_ma'])
    
    # 5. Extract rebalance frequency
    rebalance_keywords = {
        'daily': ['daily', 'every day', 'each day'],
        'weekly': ['weekly', 'every week', 'each week', 'week'],
        'monthly': ['monthly', 'every month', 'each month', 'month end', 'end of month'],
    }
    
    for rebalance_freq, keywords in rebalance_keywords.items():
        for keyword in keywords:
            if keyword in prompt_lower:
                result.rebalance = rebalance_freq
                matched_items.append('rebalance')
                break
        if result.rebalance:
            break
    
    # 6. Extract transaction costs
    tc_patterns = [
        r'(\d+)\s*bps',
        r'(\d+)\s*basis\s+points',
        r'tc[:\s]+(\d+)',
        r'transaction\s+costs?[:\s]+(\d+)',
    ]
    
    for pattern in tc_patterns:
        match = re.search(pattern, prompt_lower)
        if match:
            result.transaction_costs_bps = float(match.group(1))
            matched_items.append('tc_bps')
            break
    
    # 7. Calculate confidence and generate questions
    required_fields = ['ticker', 'start', 'feature']
    optional_fields = ['end', 'signal_type', 'rebalance', 'tc_bps']
    
    required_matched = sum(1 for field in required_fields if field in matched_items)
    optional_matched = sum(1 for field in optional_fields if field in matched_items)
    total_matched = required_matched + optional_matched
    total_fields = len(required_fields) + len(optional_fields)
    
    result.confidence = total_matched / total_fields
    
    # Generate clarifying questions for missing fields
    if not result.ticker:
        result.questions.append("Which ticker/symbol would you like to test? (e.g., SPY, QQQ)")
    
    if not result.start:
        result.questions.append("What time period should we backtest? (e.g., 'since 2020' or '2020-2023')")
    
    if not result.feature:
        result.questions.append("Which feature should we use? Options: momentum_12m_skip_1m, zscore_20d, sma_20, sma_50, rsi_14")
    
    if not result.signal_type and result.feature:
        # Suggest appropriate signal types based on feature
        if 'momentum' in (result.feature or ''):
            result.questions.append("How should we generate signals? Suggest: 'rank top 20%' for momentum")
        elif 'zscore' in (result.feature or ''):
            result.questions.append("How should we generate signals? Suggest: 'below -1' for mean reversion")
        elif 'sma' in (result.feature or '') or 'crossover' in prompt_lower:
            result.questions.append("Which moving averages for crossover? (e.g., '20-day and 50-day')")
        else:
            result.questions.append("How should we generate signals? Options: rank_top_frac, threshold, crossover, band")
    
    if not result.rebalance:
        result.questions.append("How often should we rebalance? Options: daily, weekly, monthly")
    
    if result.transaction_costs_bps is None:
        result.questions.append("What transaction costs should we assume? (default: 5 bps)")
    
    return result


def prompt_to_spec_skeleton(parsed: ParsedPrompt) -> Dict[str, Any]:
    """Convert parsed prompt to StrategySpec skeleton"""
    spec_skeleton = {
        "spec_id": f"prompt_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "domain": "equities_factor",
    }
    
    # Add confirmed fields with defaults
    spec_skeleton["ticker"] = parsed.ticker or "SPY"
    from ..utils.date_utils import get_default_backtest_dates
    default_start, default_end = get_default_backtest_dates()
    
    spec_skeleton["start"] = parsed.start or default_start
    spec_skeleton["end"] = parsed.end or default_end
    
    # Custom config
    custom_config = {}
    
    # Feature (required)
    custom_config["feature"] = parsed.feature or "momentum_12m_skip_1m"
    
    # Signal rule with defaults
    if parsed.signal_type:
        signal_rule = {"type": parsed.signal_type}
        signal_rule.update(parsed.signal_params)
        custom_config["signal_rule"] = signal_rule
    else:
        # Default signal rule for momentum strategies
        custom_config["signal_rule"] = {"type": "rank_top_frac", "top_frac": 0.2}
    
    if custom_config:
        spec_skeleton["custom"] = custom_config
    
    # Trading rules with defaults
    trading_rules = {
        "rebalance": parsed.rebalance or "monthly",
        "transaction_costs_bps": parsed.transaction_costs_bps or 5.0
    }
    
    spec_skeleton["trading_rules"] = trading_rules
    
    return spec_skeleton


def validate_feature_signal_combo(feature: str, signal_type: str) -> tuple[bool, Optional[str]]:
    """Validate that feature and signal type are compatible"""
    
    # Crossover requires SMA features
    if signal_type == "crossover":
        if not feature.startswith("sma_"):
            return False, "Crossover signals require SMA features (sma_20, sma_50). Use fast/slow MA parameters instead."
    
    # Band signal works best with mean-reverting features
    if signal_type == "band":
        if feature in ["momentum_12m_skip_1m"]:
            return False, "Band (mean-reversion) signals work poorly with momentum features. Consider threshold or rank_top_frac instead."
    
    # Rank works best with directional features
    if signal_type == "rank_top_frac":
        if feature in ["zscore_20d"]:
            return False, "Rank signals work poorly with z-scores. Consider threshold signals for mean-reversion."
    
    return True, None


def validate_spec_skeleton(spec_skeleton: Dict[str, Any]) -> tuple[bool, List[str]]:
    """Validate spec skeleton and return any error messages"""
    errors = []
    
    # Check required fields
    if "ticker" not in spec_skeleton:
        errors.append("Missing required field: ticker")
    
    if "start" not in spec_skeleton:
        errors.append("Missing required field: start date")
    
    # Validate custom config
    if "custom" in spec_skeleton:
        custom = spec_skeleton["custom"]
        
        if "feature" not in custom:
            errors.append("Missing required field: feature")
        
        if "signal_rule" not in custom:
            errors.append("Missing required field: signal_rule")
        else:
            signal_rule = custom["signal_rule"]
            
            if "type" not in signal_rule:
                errors.append("Missing required field: signal_rule.type")
            else:
                # Validate signal type parameters
                signal_type = signal_rule["type"]
                
                if signal_type == "rank_top_frac" and "top_frac" not in signal_rule:
                    errors.append("rank_top_frac signal requires 'top_frac' parameter (e.g., 0.1 for top 10%)")
                
                if signal_type == "threshold" and "lower" not in signal_rule and "upper" not in signal_rule:
                    errors.append("threshold signal requires at least 'lower' or 'upper' parameter")
                
                if signal_type == "crossover" and ("fast_ma" not in signal_rule or "slow_ma" not in signal_rule):
                    errors.append("crossover signal requires both 'fast_ma' and 'slow_ma' parameters")
                
                if signal_type == "band" and "lower" not in signal_rule:
                    errors.append("band signal requires 'lower' parameter for mean-reversion threshold")
                
                # Validate feature-signal combo
                if "feature" in custom:
                    valid, msg = validate_feature_signal_combo(custom["feature"], signal_type)
                    if not valid:
                        errors.append(f"Invalid feature-signal combination: {msg}")
    
    return len(errors) == 0, errors


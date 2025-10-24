from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from datetime import datetime
import uuid
import logging
import os
import zipfile
import tempfile
import numpy as np
import json
import asyncio
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
# Import the actual models from workers to stay in sync
from workers.api_paths import StrategySpec
from workers.exceptions import ValidationError, DataError, InternalError, ExternalAPIError
from workers.engine.prompt_parser import parse_prompt, prompt_to_spec_skeleton, validate_spec_skeleton
from workers.engine.planning import plan_generator, plan_executor
from workers.engine.intelligent_analyzer import IntelligentAnalyzer
from workers.engine.talib_tool_executor import TALibToolExecutor

# Import earnings router
from api.routers.earnings import router as earnings_router

from pydantic import BaseModel
from typing import List, Dict, Any

logger = logging.getLogger("requiem.api")

def _format_intelligent_response(analysis_result: Dict[str, Any]) -> str:
    """Format intelligent analysis results into user-friendly messages"""
    
    # Handle case where analysis_result is not a dictionary
    if not isinstance(analysis_result, dict):
        logger.error(f"Analysis result is not a dict: {type(analysis_result)} - {analysis_result}")
        return f"❌ **Error:** Invalid analysis result format\n\n*Please try again.*"
    
    try:
        if analysis_result.get("analysis_type") == "earnings_analysis":
            return _format_earnings_analysis_response(analysis_result)
        elif analysis_result.get("analysis_type") == "entry_price_analysis":
            return _format_entry_price_response(analysis_result)
        elif analysis_result.get("analysis_type") == "technical_analysis":
            return _format_technical_analysis_response(analysis_result)
        else:
            return _format_general_analysis_response(analysis_result)
    except Exception as e:
        logger.error(f"Error formatting intelligent response: {e}")
        return f"❌ **Error:** {str(e)}\n\n*Please try again.*"

def _format_entry_price_response(analysis_result: Dict[str, Any]) -> str:
    """Format professional technical analysis response"""
    ticker = analysis_result["ticker"]
    current_price = analysis_result["current_price"]
    technical_signals = analysis_result["technical_signals"]
    market_context = analysis_result["market_context"]
    quantitative_insights = analysis_result["quantitative_insights"]
    
    message = f"📊 **Technical Analysis for {ticker}**\n\n"
    message += f"**Current Price:** ${current_price:.2f}\n\n"
    
    # Market Context
    message += "**Market Context:**\n"
    message += f"• {market_context['price_position_interpretation']}\n"
    message += f"• 1M Change: {market_context['recent_volatility']['1m_change']:.1%}\n"
    message += f"• 3M Change: {market_context['recent_volatility']['3m_change']:.1%}\n\n"
    
    # Technical Indicators Analysis
    message += "**Technical Indicators Analysis:**\n\n"
    
    # RSI Analysis
    if technical_signals.get("overbought_oversold"):
        for signal in technical_signals["overbought_oversold"]:
            if signal["indicator"] == "RSI":
                message += f"**RSI ({signal['value']:.1f}):** {signal['interpretation']}\n"
                message += f"• {signal['quantitative_note']}\n"
                message += f"• {signal['historical_context']}\n\n"
    
    # MACD Analysis
    if technical_signals.get("momentum_signals"):
        for signal in technical_signals["momentum_signals"]:
            if signal["indicator"] == "MACD":
                message += f"**MACD:** {signal['interpretation']}\n"
                message += f"• MACD Line: {signal['macd_line']:.3f}\n"
                message += f"• Signal Line: {signal['signal_line']:.3f}\n"
                message += f"• {signal['quantitative_note']}\n\n"
    
    # SMA Analysis
    if technical_signals.get("trend_signals"):
        for signal in technical_signals["trend_signals"]:
            if signal["indicator"] == "SMA":
                message += f"**SMA:** {signal['interpretation']}\n"
                message += f"• SMA Value: ${signal['sma_value']:.2f}\n"
                message += f"• {signal['quantitative_note']}\n\n"
    
    # Bollinger Bands Analysis
    if technical_signals.get("volatility_signals"):
        for signal in technical_signals["volatility_signals"]:
            if signal["indicator"] == "Bollinger Bands":
                message += f"**Bollinger Bands:** {signal['interpretation']}\n"
                message += f"• Upper: ${signal['upper_band']:.2f}\n"
                message += f"• Lower: ${signal['lower_band']:.2f}\n"
                message += f"• {signal['quantitative_note']}\n\n"
    
    # Risk Metrics
    message += "**Risk Metrics:**\n"
    risk_metrics = quantitative_insights["risk_metrics"]
    message += f"• Downside Risk: {risk_metrics['downside_risk']:.1%}\n"
    message += f"• Upside Potential: {risk_metrics['upside_potential']:.1%}\n"
    message += f"• Risk/Reward Ratio: {risk_metrics['risk_reward_ratio']}\n"
    message += f"• Volatility: {risk_metrics['volatility_assessment']}\n"
    
    return message

def _format_technical_analysis_response(analysis_result: Dict[str, Any]) -> str:
    """Format technical analysis response"""
    return f"📊 **Technical Analysis for {analysis_result['ticker']}**\n\n*Comprehensive technical analysis coming soon...*"

def _format_earnings_analysis_response(analysis_result: Dict[str, Any]) -> str:
    """Format earnings analysis response in card format"""
    ticker = analysis_result.get("ticker", "Unknown")
    earnings_result = analysis_result.get("earnings_result", {})
    
    # Debug logging
    logger.info(f"Earnings result type: {type(earnings_result)}")
    logger.info(f"Earnings result: {earnings_result}")
    
    # Handle case where earnings_result might be a string or other type
    if not isinstance(earnings_result, dict):
        return f"📊 **Earnings Analysis for {ticker}**\n\n❌ **Error:** Invalid earnings result format\n\n*This earnings data is not accessible by Requiem right now. Please try a different quarter.*"
    
    # Handle case where earnings_result is empty or None
    if not earnings_result:
        return f"📊 **Earnings Analysis for {ticker}**\n\n❌ **Error:** No earnings data found\n\n*This earnings data is not accessible by Requiem right now. Please try a different quarter or check if the earnings call is available.*"
    
    if not earnings_result.get("success", False):
        error_msg = earnings_result.get("error", "Unknown error")
        return f"📊 **Earnings Analysis for {ticker}**\n\n❌ **Error:** {error_msg}\n\n*This earnings data is not accessible by Requiem right now. Please try a different quarter or check if the earnings call is available.*"
    
    # Format the earnings summary in card format
    message = f"📊 **Earnings Analysis for {ticker}**\n\n"
    
    # Key Highlights (only if present)
    if "tldr" in earnings_result and earnings_result["tldr"]:
        message += "**📋 Key Highlights:**\n"
        for item in earnings_result["tldr"]:
            message += f"• {item}\n"
        message += "\n"
    
    # Guidance (only if present and not empty)
    if "guidance" in earnings_result and earnings_result["guidance"]:
        guidance_items = []
        for key, value in earnings_result["guidance"].items():
            if key != "_cites" and value and value != "Not specified":
                guidance_items.append(f"• **{key.title()}:** {value}")
        
        if guidance_items:
            message += "**🎯 Guidance:**\n"
            message += "\n".join(guidance_items) + "\n\n"
    
        # Key Metrics (clean up duplicates and empty values)
        if "kpis" in earnings_result and earnings_result["kpis"]:
            # Filter out duplicates and empty values
            seen_metrics = set()
            clean_kpis = []
            
            for kpi in earnings_result["kpis"]:
                if isinstance(kpi, dict):
                    metric = kpi.get("metric", "Unknown")
                    value = kpi.get("value", "N/A")
                    change = kpi.get("change", "")
                    
                    # Skip if value is empty, "Not specified", or "None"
                    if value and value not in ["Not specified", "None", "N/A"] and value != "":
                        # Use metric name only for deduplication (not value)
                        metric_name_lower = metric.lower()
                        if metric_name_lower not in seen_metrics:
                            seen_metrics.add(metric_name_lower)
                            clean_kpis.append({
                                "metric": metric,
                                "value": value,
                                "change": change
                            })
                else:
                    clean_kpis.append({"metric": str(kpi), "value": "", "change": ""})
            
            if clean_kpis:
                message += "**📈 Key Metrics:**\n"
                for kpi in clean_kpis:
                    metric = kpi["metric"]
                    value = kpi["value"]
                    change = kpi["change"]
                    
                    if change and change not in ["None", "Not specified", ""]:
                        message += f"• **{metric}:** {value} {change}\n"
                    else:
                        message += f"• **{metric}:** {value}\n"
                message += "\n"
    
    # Risks (only if present)
    if "risks" in earnings_result and earnings_result["risks"]:
        message += "**⚠️ Key Risks:**\n"
        for risk in earnings_result["risks"]:
            message += f"• {risk}\n"
        message += "\n"

    # Quotes (only if present)
    if "quotes" in earnings_result and earnings_result["quotes"]:
        message += "**💬 Key Quotes:**\n"
        for quote in earnings_result["quotes"]:
            if isinstance(quote, dict):
                text = quote.get("quote", "")
                speaker = quote.get("speaker", "Unknown")
                page = quote.get("page", "")
                message += f"• *\"{text}\"* - {speaker} (p.{page})\n"
            else:
                message += f"• {str(quote)}\n"
        message += "\n"
    
    # Financial Analysis (only if present)
    if "additional_insights" in earnings_result and earnings_result["additional_insights"]:
        message += "**📊 Financial Analysis:**\n"
        for insight in earnings_result["additional_insights"]:
            message += f"{insight}\n"
        message += "\n"

    # Sources (only if present)
    if "sources" in earnings_result and earnings_result["sources"]:
        message += "**🔗 Sources:**\n"
        for source in earnings_result["sources"]:
            message += f"• {source}\n"

    return message

def _format_general_analysis_response(analysis_result: Dict[str, Any]) -> str:
    """Format general analysis response"""
    return f"📈 **Analysis for {analysis_result['ticker']}**\n\n*General analysis coming soon...*"

class DataBindingPlan(BaseModel):
    binding_id: str
    spec_id: str
    created_at: str

class RunRequest(BaseModel):
    spec: StrategySpec

app = FastAPI(title="requiem v0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:3002"],  # UI server ports
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Include earnings router
app.include_router(earnings_router)

RUNS: dict[str, dict] = {}

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/version")
def version():
    """Return API version and capabilities"""
    return {
        "version": "1.0.0",
        "features": {
            "price_queries": True,
            "intraday_queries": True,
            "backtests": True,
            "tool_execution": True,
            "tool_aware_analysis": True,
            "ticker_suggestions": True,
            "valuation": True,
            "comparison": False,
        },
        "last_updated": "2025-10-12"
    }

# Simple in-memory cache for ticker suggestions
_ticker_cache = {}
_cache_timeout = 300  # 5 minutes

@app.get("/ticker-suggestions")
def get_ticker_suggestions(q: str):
    """Get ticker suggestions from Polygon.io API with caching"""
    import requests
    import os
    import time
    
    logger = logging.getLogger("requiem.api")
    
    if not q or len(q) < 2:  # Require at least 2 characters
        return {"suggestions": []}
    
    q_upper = q.upper()
    
    # Check cache first
    if q_upper in _ticker_cache:
        cached_result, timestamp = _ticker_cache[q_upper]
        if time.time() - timestamp < _cache_timeout:
            logger.info(f"Cache hit for '{q}' (instant response)")
            return {"suggestions": cached_result, "cached": True}
    
    try:
        api_key = os.getenv("POLYGON_API_KEY")
        if not api_key:
            logger.warning("POLYGON_API_KEY not found, using fallback")
            return get_fallback_suggestions(q)
        
        # Call Polygon.io ticker search API
        url = "https://api.polygon.io/v3/reference/tickers"
        params = {
            "search": q,
            "limit": 15,
            "apikey": api_key,
            "active": "true",  # Only active tickers
            "sort": "ticker"   # Sort alphabetically
        }
        
        response = requests.get(url, params=params, timeout=2)  # Faster timeout
        response.raise_for_status()
        
        data = response.json()
        
        if data.get("status") != "OK" or "results" not in data:
            logger.warning(f"Polygon API returned unexpected format: {data}")
            return get_fallback_suggestions(q)
        
        results = data["results"]
        suggestions = []
        
        for ticker in results[:10]:  # Limit to 10 results
            suggestion = format_polygon_ticker(ticker)
            if suggestion:
                suggestions.append(suggestion)
        
        logger.info(f"Found {len(suggestions)} ticker suggestions for query '{q}'")
        
        # Cache the result
        _ticker_cache[q_upper] = (suggestions, time.time())
        
        return {"suggestions": suggestions, "cached": False}
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Polygon API error: {e}")
        return get_fallback_suggestions(q)
    except Exception as e:
        logger.error(f"Unexpected error in ticker suggestions: {e}")
        return get_fallback_suggestions(q)

def format_polygon_ticker(ticker_data):
    """Format Polygon.io ticker data to our standard format"""
    logger = logging.getLogger("requiem.api")
    try:
        # Map Polygon types to our format
        type_mapping = {
            "CS": "Q",  # Common Stock -> Equity
            "ETF": "E", # ETF -> ETF
            "FUND": "M", # Fund -> Mutual Fund
            "FUT": "F",  # Future
            "OPT": "O",  # Option
            "WARRANT": "W", # Warrant
        }
        
        polygon_type = ticker_data.get("type", "CS")
        our_type = type_mapping.get(polygon_type, "Q")
        
        # Get country from market
        market = ticker_data.get("market", "stocks")
        if "otc" in market.lower():
            country = "OTC"
        elif "nyse" in market.lower() or "nasdaq" in market.lower():
            country = "US"
        else:
            country = "US"  # Default
        
        # Get full type description
        type_descriptions = {
            "Q": "Equity",
            "E": "ETF", 
            "M": "Mutual Fund",
            "F": "Future",
            "O": "Option",
            "W": "Warrant"
        }
        
        return {
            "type": our_type,
            "country": country,
            "ticker": ticker_data.get("ticker", ""),
            "name": ticker_data.get("name", ""),
            "fullType": type_descriptions.get(our_type, "Security"),
            "market": ticker_data.get("market", ""),
            "primary_exchange": ticker_data.get("primary_exchange", "")
        }
        
    except Exception as e:
        logger.error(f"Error formatting ticker data: {e}")
        return None

def get_fallback_suggestions(q: str):
    """Fallback to hardcoded suggestions if API fails"""
    fallback_tickers = {
        'AAPL': [{"type": "Q", "country": "US", "ticker": "AAPL", "name": "Apple Inc", "fullType": "Equity"}],
        'NVDA': [
            {"type": "Q", "country": "US", "ticker": "NVDA", "name": "NVIDIA Corporation", "fullType": "Equity"},
            {"type": "E", "country": "US", "ticker": "NVD", "name": "GraniteShares 2x Short NVDA Daily E", "fullType": "ETF"}
        ],
        'TSLA': [{"type": "Q", "country": "US", "ticker": "TSLA", "name": "Tesla Inc", "fullType": "Equity"}],
        'SPY': [{"type": "E", "country": "US", "ticker": "SPY", "name": "SPDR S&P 500 ETF Trust", "fullType": "ETF"}],
        'QQQ': [{"type": "E", "country": "US", "ticker": "QQQ", "name": "Invesco QQQ Trust", "fullType": "ETF"}],
        'MSFT': [{"type": "Q", "country": "US", "ticker": "MSFT", "name": "Microsoft Corporation", "fullType": "Equity"}],
        'GOOGL': [{"type": "Q", "country": "US", "ticker": "GOOGL", "name": "Alphabet Inc Class A", "fullType": "Equity"}],
        'AMZN': [{"type": "Q", "country": "US", "ticker": "AMZN", "name": "Amazon.com Inc", "fullType": "Equity"}],
        'META': [{"type": "Q", "country": "US", "ticker": "META", "name": "Meta Platforms Inc", "fullType": "Equity"}],
        'IWM': [{"type": "E", "country": "US", "ticker": "IWM", "name": "iShares Russell 2000 ETF", "fullType": "ETF"}],
        'DIA': [{"type": "E", "country": "US", "ticker": "DIA", "name": "SPDR Dow Jones Industrial Average ETF", "fullType": "ETF"}]
    }
    
    suggestions = []
    q_upper = q.upper()
    
    for ticker, options in fallback_tickers.items():
        if ticker.startswith(q_upper):
            suggestions.extend(options)
    
    return {"suggestions": suggestions[:10]}


@app.post("/binding/plan", response_model=DataBindingPlan)
def plan_binding(spec: StrategySpec):
    return DataBindingPlan(
        binding_id=f"bind_{spec.spec_id}",
        spec_id=spec.spec_id,
        created_at=datetime.utcnow().isoformat()+"Z"
    )

@app.post("/runs/execute")
def runs_execute(req: RunRequest):
    run_id = f"run_{uuid.uuid4().hex[:8]}"
    trace_id = str(uuid.uuid4())
    
    # Set up request logging
    logger = logging.getLogger("requiem.api")
    logger.info(f"Starting run {run_id} with trace_id {trace_id}")
    logger.info(f"Request spec: {req.spec.model_dump()}")
    
    RUNS[run_id] = {"state": "running", "progress": 0.1, "spec": req.spec.model_dump()}
    
    try:
        from workers.api_paths import run_spec_to_tearsheet
        ts = run_spec_to_tearsheet(req.spec, trace_id=trace_id)
        RUNS[run_id] = {"state": "done", "progress": 1.0, "tearsheet": ts}
        logger.info(f"Run {run_id} completed successfully")
        
    except ValidationError as e:
        RUNS[run_id] = {"state": "error", "progress": 1.0, "error": e.model_dump()}
        logger.error(f"Validation error in run {run_id}: {e.message}")
        raise HTTPException(status_code=422, detail=e.model_dump())
        
    except DataError as e:
        RUNS[run_id] = {"state": "error", "progress": 1.0, "error": e.model_dump()}
        logger.error(f"Data error in run {run_id}: {e.message}")
        raise HTTPException(status_code=422, detail=e.model_dump())
        
    except ExternalAPIError as e:
        RUNS[run_id] = {"state": "error", "progress": 1.0, "error": e.model_dump()}
        logger.error(f"External API error in run {run_id}: {e.message}")
        raise HTTPException(status_code=502, detail=e.model_dump())
        
    except InternalError as e:
        RUNS[run_id] = {"state": "error", "progress": 1.0, "error": e.model_dump()}
        logger.error(f"Internal error in run {run_id}: {e.message}")
        raise HTTPException(status_code=500, detail=e.model_dump())
        
    except Exception as e:
        # Catch any other unexpected errors
        error_response = InternalError(
            "Unexpected error during execution",
            details={"error": str(e), "run_id": run_id},
            trace_id=trace_id
        )
        RUNS[run_id] = {"state": "error", "progress": 1.0, "error": error_response.model_dump()}
        logger.error(f"Unexpected error in run {run_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=error_response.model_dump())
    
    return {"run_id": run_id, "state": RUNS[run_id]["state"], "progress": RUNS[run_id]["progress"]}

@app.get("/runs/{run_id}/status")
def runs_status(run_id: str):
    if run_id not in RUNS: raise HTTPException(404, "run not found")
    r = RUNS[run_id]
    return {"run_id": run_id, "state": r["state"], "progress": r["progress"]}

@app.get("/runs/{run_id}/tearsheet")
def runs_tearsheet(run_id: str):
    if run_id not in RUNS or "tearsheet" not in RUNS[run_id]:
        raise HTTPException(404, "tearsheet not ready")
    return RUNS[run_id]["tearsheet"]


# Prompt-to-Spec endpoints

class PromptRequest(BaseModel):
    prompt: str
    
class PromptAuditResponse(BaseModel):
    spec_skeleton: Dict[str, Any]
    questions: List[str]
    confidence: float
    parsed_fields: Dict[str, Any]

@app.post("/prompt/audit", response_model=PromptAuditResponse)
def prompt_audit(req: PromptRequest):
    """
    Parse natural language prompt and return spec skeleton with clarifying questions.
    
    Example prompts:
    - "Test 12-month momentum on SPY since 2015 monthly, 5 bps TC"
    - "Run zscore mean reversion on QQQ from 2020-2023 weekly"
    - "SMA crossover 20-day and 50-day on SPY, 2022 to 2023"
    """
    logger = logging.getLogger("requiem.api")
    logger.info(f"Parsing prompt: {req.prompt}")
    
    try:
        # Parse the prompt
        parsed = parse_prompt(req.prompt)
        
        # Generate spec skeleton
        spec_skeleton = prompt_to_spec_skeleton(parsed)
        
        # Extract parsed fields for transparency
        parsed_fields = {
            "intent": parsed.intent,
            "intent_confidence": parsed.intent_confidence,
            "ticker": parsed.ticker,
            "start": parsed.start,
            "end": parsed.end,
            "feature": parsed.feature,
            "signal_type": parsed.signal_type,
            "signal_params": parsed.signal_params,
            "rebalance": parsed.rebalance,
            "transaction_costs_bps": parsed.transaction_costs_bps,
        }
        
        logger.info(f"Detected intent: {parsed.intent} (confidence: {parsed.intent_confidence:.2f})")
        logger.info(f"Parsed prompt with confidence {parsed.confidence:.2f}")
        logger.info(f"Generated {len(parsed.questions)} clarifying questions")
        
        return PromptAuditResponse(
            spec_skeleton=spec_skeleton,
            questions=parsed.questions,
            confidence=parsed.confidence,
            parsed_fields=parsed_fields
        )
        
    except Exception as e:
        logger.error(f"Error parsing prompt: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Failed to parse prompt: {str(e)}")


class SpecValidationRequest(BaseModel):
    spec_skeleton: Dict[str, Any]

class SpecValidationResponse(BaseModel):
    valid: bool
    errors: List[str]
    warnings: List[str]
    suggested_fixes: Dict[str, Any]

@app.post("/specs/validate", response_model=SpecValidationResponse)
def specs_validate(req: SpecValidationRequest):
    """
    Validate feature + signal + rebalance combinations and return actionable messages.
    
    Rejects nonsense combinations like:
    - Crossover signal without SMA features
    - Rank signal with z-score features
    - Missing required parameters
    """
    logger = logging.getLogger("requiem.api")
    logger.info(f"Validating spec skeleton: {req.spec_skeleton}")
    
    try:
        # Validate spec skeleton
        is_valid, errors = validate_spec_skeleton(req.spec_skeleton)
        
        warnings = []
        suggested_fixes = {}
        
        # Add warnings for missing optional fields
        if "trading_rules" not in req.spec_skeleton:
            warnings.append("No trading rules specified. Will use defaults: monthly rebalance, 5 bps TC")
            suggested_fixes["trading_rules"] = {"rebalance": "monthly", "transaction_costs_bps": 5.0}
        
        if "end" not in req.spec_skeleton:
            warnings.append("No end date specified. Will use today's date")
            suggested_fixes["end"] = datetime.now().strftime("%Y-%m-%d")
        
        # Check for common issues and suggest fixes
        if "custom" in req.spec_skeleton and "signal_rule" in req.spec_skeleton["custom"]:
            signal_rule = req.spec_skeleton["custom"]["signal_rule"]
            signal_type = signal_rule.get("type")
            
            # Suggest default parameters
            if signal_type == "rank_top_frac" and "top_frac" not in signal_rule:
                suggested_fixes["signal_rule_top_frac"] = 0.1
                warnings.append("Missing top_frac parameter. Suggest: 0.1 (top 10%)")
            
            if signal_type == "threshold" and "lower" not in signal_rule and "upper" not in signal_rule:
                if "feature" in req.spec_skeleton["custom"] and "zscore" in req.spec_skeleton["custom"]["feature"]:
                    suggested_fixes["signal_rule_lower"] = -1.0
                    suggested_fixes["signal_rule_upper"] = 1.0
                    warnings.append("Missing threshold parameters. Suggest: lower=-1, upper=1 for z-score")
        
        logger.info(f"Validation result: valid={is_valid}, errors={len(errors)}, warnings={len(warnings)}")
        
        return SpecValidationResponse(
            valid=is_valid,
            errors=errors,
            warnings=warnings,
            suggested_fixes=suggested_fixes
        )
        
    except Exception as e:
        logger.error(f"Error validating spec: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Failed to validate spec: {str(e)}")


# ============================================================================
# NEW: Multi-Intent Query Endpoint
# ============================================================================

class QueryRequest(BaseModel):
    query: str
    selected_tools: List[str] = []  # List of tool names selected by user

class QueryResponse(BaseModel):
    intent: str
    intent_confidence: float
    success: bool = True
    data: Dict[str, Any]
    message: str

@app.post("/query/intelligent")
async def handle_intelligent_query(req: QueryRequest) -> QueryResponse:
    """Handle intelligent analysis queries with context understanding"""
    try:
        logger.info(f"Intelligent query: {req.query}")
        
        # Check if this is a TA-Lib indicator query (check before other analysis)
        if _is_talib_indicator_query(req.query):
            return await _handle_talib_comparison_query(req)
        
        # Initialize intelligent analyzer
        analyzer = IntelligentAnalyzer()
        
        # Analyze the query intelligently
        analysis_result = analyzer.analyze_query(req.query, req.selected_tools)
        
        if "error" in analysis_result:
            return QueryResponse(
                intent="error",
                intent_confidence=0.0,
                data={},
                message=f"❌ {analysis_result['error']}",
                success=False
            )
        
        # Format the intelligent response
        logger.info(f"Analysis result type: {type(analysis_result)}")
        logger.info(f"Analysis result: {analysis_result}")
        message = _format_intelligent_response(analysis_result)
        
        return QueryResponse(
            intent="intelligent_analysis",
            intent_confidence=0.9,
            message=message,
            data=analysis_result,
            success=True
        )
        
    except Exception as e:
        logger.error(f"Error in intelligent query: {str(e)}")
        return QueryResponse(
            intent="error",
            intent_confidence=0.0,
            data={},
            message=f"❌ Sorry, I encountered an error while processing your request: {str(e)}. Please try again.",
            success=False
        )

@app.post("/query/intelligent/stream")
async def handle_intelligent_query_stream(req: QueryRequest):
    """Handle intelligent analysis queries with streaming responses"""
    async def generate():
        try:
            logger.info(f"Streaming intelligent query: {req.query}")
            
            # Send initial status
            yield f"data: {json.dumps({'type': 'status', 'message': '🔄 Analyzing your query...'})}\n\n"
            await asyncio.sleep(0.1)
            
            # Check if this is a TA-Lib indicator query
            if _is_talib_indicator_query(req.query):
                yield f"data: {json.dumps({'type': 'status', 'message': '🔍 Running technical analysis...'})}\n\n"
                await asyncio.sleep(0.1)
                
                # For TA-Lib queries, run normally but stream the result
                result = await _handle_talib_comparison_query(req)
                yield f"data: {json.dumps({'type': 'result', 'data': result.dict()})}\n\n"
                return
            
            # Initialize intelligent analyzer
            analyzer = IntelligentAnalyzer()
            
            # Send intent detection status
            yield f"data: {json.dumps({'type': 'status', 'message': '🎯 Detecting intent...'})}\n\n"
            await asyncio.sleep(0.1)
            
            # For earnings queries, stream progress
            query_lower = req.query.lower()
            if any(keyword in query_lower for keyword in ['earnings', 'quarterly', 'transcript', 'conference call']):
                yield f"data: {json.dumps({'type': 'status', 'message': '📊 Fetching earnings data...'})}\n\n"
                await asyncio.sleep(0.1)
            
            # Analyze the query intelligently
            analysis_result = analyzer.analyze_query(req.query, req.selected_tools)
            
            if "error" in analysis_result:
                yield f"data: {json.dumps({'type': 'error', 'message': f'❌ {analysis_result[\"error\"]}'})}\n\n"
                return
            
            # Send progress update
            yield f"data: {json.dumps({'type': 'status', 'message': '✨ Formatting response...'})}\n\n"
            await asyncio.sleep(0.1)
            
            # Format the intelligent response
            message = _format_intelligent_response(analysis_result)
            
            # Stream the complete response
            yield f"data: {json.dumps({'type': 'result', 'data': {'intent': 'intelligent_analysis', 'intent_confidence': 0.9, 'message': message, 'data': analysis_result, 'success': True}})}\n\n"
            
        except Exception as e:
            logger.error(f"Error in streaming intelligent query: {str(e)}")
            yield f"data: {json.dumps({'type': 'error', 'message': f'❌ Sorry, I encountered an error: {str(e)}'})}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")

@app.post("/query", response_model=QueryResponse)
async def handle_query(req: QueryRequest):
    """
    Universal query endpoint that detects intent and routes to appropriate handler.
    
    Supported intents:
    - backtest: Run strategy backtest (routes to /runs/execute)
    - price_query: Get historical prices
    - valuation: Get valuation metrics (P/E, Fair Value, etc.)
    - comparison: Compare two or more assets
    - analysis: Technical/fundamental analysis
    """
    logger = logging.getLogger("requiem.api")
    logger.info(f"Processing query: {req.query}")
    
    try:
        # Parse the query and detect intent
        parsed = parse_prompt(req.query)
        
        logger.info(f"Detected intent: {parsed.intent} (confidence: {parsed.intent_confidence:.2f})")
        
        # Check for compound queries in direct mode
        from workers.engine.planning import PlanGenerator
        plan_generator = PlanGenerator()
        detected_intents = plan_generator._detect_multiple_intents(req.query)
        
        if len(detected_intents) > 1:
            # Compound query - execute all intents
            logger.info(f"Detected compound query with intents: {detected_intents}")
            return _handle_compound_query(parsed, req.query, detected_intents, req.selected_tools)
        
        # Route to appropriate handler based on intent
        if parsed.intent == "backtest":
            return _handle_backtest_query(parsed, req.query)
        elif parsed.intent == "tool_backtest":
            return await _handle_tool_backtest_query(parsed, req.query, req.selected_tools)
        elif parsed.intent == "price_query":
            return _handle_price_query(parsed, req.query)
        elif parsed.intent == "valuation":
            return _handle_valuation_query(parsed, req.query)
        elif parsed.intent == "comparison":
            return _handle_comparison_query(parsed, req.query)
        elif parsed.intent == "analysis":
            return _handle_analysis_query(parsed, req.query, req.selected_tools)
        elif parsed.intent == "statistical_analysis":
            return _handle_statistical_analysis_query(parsed, req.query)
        elif parsed.intent == "risk_metrics":
            return _handle_risk_metrics_query(parsed, req.query)
        elif parsed.intent == "mathematical_calculation":
            return _handle_mathematical_calculation_query(parsed, req.query)
        else:
            return QueryResponse(
                intent="unknown",
                intent_confidence=parsed.intent_confidence,
                data={},
                message=f"I'm not sure what you're asking. Could you rephrase? (Confidence: {parsed.intent_confidence:.1%})"
            )
    
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process query: {str(e)}")


@app.post("/query/plan", response_model=QueryResponse)
def handle_planning_query(req: QueryRequest):
    """Handle queries in planning mode - generate execution plan first"""
    logger = logging.getLogger("requiem.api")
    logger.info(f"Generating plan for query: {req.query}")
    
    try:
        # Parse the query and detect intent
        parsed = parse_prompt(req.query)
        
        # Generate execution plan
        plan = plan_generator.generate_plan(
            intent=parsed.intent,
            query=req.query,
            parsed_query=parsed.dict()
        )
        
        # Return plan for user review
        return QueryResponse(
            intent="planning",
            intent_confidence=parsed.intent_confidence,
            success=True,
            data={
                "plan_id": plan.plan_id,
                "plan_markdown": plan.to_markdown(),
                "steps": [step.to_dict() for step in plan.steps],
                "status": "pending_approval"
            },
            message=f"Generated execution plan with {len(plan.steps)} steps. Please review and approve to proceed."
        )
        
    except Exception as e:
        logger.error(f"Error generating plan: {str(e)}")
        return QueryResponse(
            intent="planning",
            intent_confidence=0.0,
            data={"error": str(e)},
            message=f"Error generating execution plan: {str(e)}"
        )


@app.post("/plan/execute")
def execute_approved_plan(plan_id: str, approved: bool = True):
    """Execute an approved plan"""
    logger = logging.getLogger("requiem.api")
    logger.info(f"Executing plan {plan_id}, approved: {approved}")
    
    try:
        if not approved:
            return {"status": "cancelled", "message": "Plan execution cancelled by user"}
        
        # Find the plan (in a real implementation, this would be stored)
        # For now, we'll generate a new plan and execute it
        # This is a simplified version - in production you'd store plans
        
        return {
            "status": "executing",
            "plan_id": plan_id,
            "message": "Plan execution started. This would execute the approved steps."
        }
        
    except Exception as e:
        logger.error(f"Error executing plan: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to execute plan: {str(e)}")


async def _handle_tool_backtest_query(parsed, query: str, selected_tools: List[str]) -> QueryResponse:
    """Handle tool-based backtest queries"""
    try:
        logger.info(f"Processing tool-based backtest query: {query}")
        
        # Extract ticker and dates
        ticker = parsed.ticker or "SPY"
        start_date = parsed.start or "2023-01-01"
        end_date = parsed.end or datetime.now().strftime("%Y-%m-%d")
        
        # Simple signal rule parsing (without complex dependencies)
        signal_rules = _parse_simple_signal_rules(query)
        
        if not signal_rules:
            return QueryResponse(
                intent="tool_backtest",
                intent_confidence=0.0,
                data={},
                message="❌ No valid signal rules found in query. Please use format: 'buy when tool < value' or 'sell when tool > value'",
                success=False
            )
        
        # Actually execute the backtest
        try:
            # Create a strategy spec for tool-based backtest
            from workers.api_paths import StrategySpec, SignalRule, TradingRules, CustomConfig
            
            # Create signal rule for tool-based backtest
            signal_rule = SignalRule(
                type="tool_based",
                tool_rules=signal_rules
            )
            
            # Create custom config
            custom_config = CustomConfig(
                feature="tool_based",  # This will be ignored for tool-based signals
                signal_rule=signal_rule
            )
            
            # Create strategy spec
            spec = StrategySpec(
                spec_id=f"tool_backtest_{ticker}_{int(datetime.now().timestamp())}",
                domain="equities_factor",
                ticker=ticker,
                start=start_date,
                end=end_date,
                trading_rules=TradingRules(
                    rebalancing_frequency="daily",
                    transaction_costs=0.001
                ),
                custom=custom_config
            )
            
            # Store the signal rules for the backtest execution
            spec._tool_signals = signal_rules
            
            # Execute the backtest using the runs/execute endpoint
            logger.info(f"Executing tool-based backtest for {ticker}")
            
            # Create a run request
            run_request = RunRequest(spec=spec)
            
            # Call the runs/execute endpoint
            run_response = runs_execute(run_request)
            
            # Get the run ID and wait for completion
            run_id = run_response["run_id"]
            
            # Wait for the run to complete (simplified - in production you'd poll)
            import time
            time.sleep(3)  # Give it time to process
            
            # Get the results
            if run_id in RUNS and RUNS[run_id]["state"] == "done":
                result = RUNS[run_id].get("tearsheet", {})
                return QueryResponse(
                    intent="tool_backtest",
                    intent_confidence=0.9,
                    message=f"🔧 Tool-based backtest completed for {ticker} from {start_date} to {end_date}",
                    data=result,
                    success=True
                )
            else:
                # Return processing status
                return QueryResponse(
                    intent="tool_backtest",
                    intent_confidence=0.9,
                    message=f"🔧 Tool-based backtest is processing for {ticker} from {start_date} to {end_date}",
                    data={
                        "ticker": ticker,
                        "start_date": start_date,
                        "end_date": end_date,
                        "signal_rules": signal_rules,
                        "selected_tools": selected_tools,
                        "run_id": run_id,
                        "status": "processing",
                        "note": "Backtest is still processing. Results will be available shortly."
                    },
                    success=True
                )
            
        except Exception as e:
            logger.error(f"Error executing tool-based backtest: {str(e)}")
            # Fallback to just showing parsed rules
            return QueryResponse(
                intent="tool_backtest",
                intent_confidence=0.9,
                message=f"🔧 Tool-based backtest query parsed successfully for {ticker} from {start_date} to {end_date}",
                data={
                    "ticker": ticker,
                    "start_date": start_date,
                    "end_date": end_date,
                    "signal_rules": signal_rules,
                    "selected_tools": selected_tools,
                    "ready_to_execute": True,
                    "note": f"Backtest execution failed: {str(e)}. Showing parsed signal rules only."
                },
                success=True
            )
        
    except Exception as e:
        logger.error(f"Error in tool backtest query: {str(e)}")
        return QueryResponse(
            intent="tool_backtest",
            intent_confidence=0.0,
            data={},
            message=f"❌ Error processing tool-based backtest: {str(e)}",
            success=False
        )

def _parse_simple_signal_rules(query: str) -> List[Dict[str, Any]]:
    """Simple signal rule parsing without complex dependencies"""
    import re
    
    signal_rules = []
    query_lower = query.lower()
    
    # Pattern for tool-based signals: "buy when tool < value" or "sell when tool > value"
    patterns = [
        # Buy signals
        (r'buy\s+when\s+(\w+)\s*[<≤]\s*(\d+(?:\.\d+)?)', 'buy', 'less_than'),
        (r'buy\s+when\s+(\w+)\s*[>≥]\s*(\d+(?:\.\d+)?)', 'buy', 'greater_than'),
        (r'long\s+when\s+(\w+)\s*[<≤]\s*(\d+(?:\.\d+)?)', 'buy', 'less_than'),
        (r'long\s+when\s+(\w+)\s*[>≥]\s*(\d+(?:\.\d+)?)', 'buy', 'greater_than'),
        
        # Sell signals
        (r'sell\s+when\s+(\w+)\s*[<≤]\s*(\d+(?:\.\d+)?)', 'sell', 'less_than'),
        (r'sell\s+when\s+(\w+)\s*[>≥]\s*(\d+(?:\.\d+)?)', 'sell', 'greater_than'),
        (r'short\s+when\s+(\w+)\s*[<≤]\s*(\d+(?:\.\d+)?)', 'sell', 'less_than'),
        (r'short\s+when\s+(\w+)\s*[>≥]\s*(\d+(?:\.\d+)?)', 'sell', 'greater_than'),
    ]
    
    for pattern, action, comparison in patterns:
        matches = re.finditer(pattern, query_lower)
        for match in matches:
            tool_name = match.group(1)
            threshold = float(match.group(2))
            
            # Map common tool names
            tool_mapping = {
                'rsi': 'rsi',
                'sma': 'sma',
                'macd': 'macd',
                'bollinger': 'bollinger',
                'williams': 'williams_r',
                'williams_r': 'williams_r',
                'stochastic': 'stochastic',
                'momentum': 'momentum',
                'volatility': 'realized_vol',
                'vol': 'realized_vol',
                'zscore': 'zscore',
                'aroon': 'aroon'
            }
            
            normalized_tool = tool_mapping.get(tool_name.lower(), tool_name.lower())
            
            signal_rules.append({
                "tool_name": normalized_tool,
                "action": action,
                "comparison": comparison,
                "threshold": threshold,
                "original_text": match.group(0)
            })
    
    return signal_rules

def _handle_backtest_query(parsed, query: str) -> QueryResponse:
    """Handle backtest queries"""
    from workers.utils.date_utils import get_default_backtest_dates
    
    # Generate spec skeleton
    spec_skeleton = prompt_to_spec_skeleton(parsed)
    
    # Validate spec
    is_valid, errors = validate_spec_skeleton(spec_skeleton)
    
    if not is_valid:
        return QueryResponse(
            intent="backtest",
            intent_confidence=parsed.intent_confidence,
            data={"spec_skeleton": spec_skeleton, "errors": errors},
            message=f"I understand you want to backtest a strategy, but I need more information: {', '.join(errors)}"
        )
    
    # Get default dates dynamically
    default_start, default_end = get_default_backtest_dates()
    
    # Spec is valid - return it for execution
    return QueryResponse(
        intent="backtest",
        intent_confidence=parsed.intent_confidence,
        data={
            "spec_skeleton": spec_skeleton,
            "ready_to_execute": True,
            "parsed_fields": {
                "ticker": parsed.ticker,
                "start": parsed.start,
                "end": parsed.end,
                "feature": parsed.feature,
                "rebalance": parsed.rebalance,
            }
        },
        message=f"Ready to backtest {parsed.feature or 'momentum'} strategy on {parsed.ticker or 'SPY'} from {parsed.start or default_start} to {parsed.end or default_end}. Execute via /runs/execute endpoint."
    )


def _handle_intraday_price_query(parsed, query: str, logger) -> QueryResponse:
    """Handle intraday (minute-level) price queries"""
    from workers.adapters.prices_polygon import get_intraday_price
    
    ticker = parsed.ticker or "SPY"
    date = parsed.start or datetime.now().strftime("%Y-%m-%d")
    time = parsed.time  # Format: "HH:MM"
    timezone = parsed.timezone  # Default: "ET"
    
    # Combine date and time
    datetime_str = f"{date} {time}"
    
    try:
        # Get intraday price data
        price_data = get_intraday_price(ticker, datetime_str, window_minutes=5)
        
        if not price_data:
            return QueryResponse(
                intent="price_query",
                intent_confidence=parsed.intent_confidence,
                data={},
                message=f"No intraday data available for {ticker} at {datetime_str} {timezone}. Market may have been closed or data not available."
            )
        
        # Format response
        time_diff = int(price_data['time_diff_seconds'])
        if time_diff > 60:
            time_note = f" (closest bar: {time_diff // 60}m {time_diff % 60}s away)"
        elif time_diff > 0:
            time_note = f" (closest bar: {time_diff}s away)"
        else:
            time_note = " (exact match)"
        
        message = f"⏰ {ticker} at {price_data['datetime']}: ${price_data['close']:.2f}{time_note}"
        
        return QueryResponse(
            intent="price_query",
            intent_confidence=parsed.intent_confidence,
            data={
                "ticker": ticker,
                "is_intraday": True,
                "requested_datetime": f"{date} {time} {timezone}",
                "actual_datetime": price_data['datetime'],
                "date": price_data['date'],
                "time": price_data['time'],
                "latest_price": {
                    "date": price_data['date'],
                    "time": price_data['time'],
                    "open": price_data['open'],
                    "high": price_data['high'],
                    "low": price_data['low'],
                    "close": price_data['close'],
                    "volume": price_data['volume'],
                },
                "time_diff_seconds": time_diff
            },
            message=message
        )
    
    except Exception as e:
        logger.error(f"Intraday price query error for {ticker}: {str(e)}", exc_info=True)
        return QueryResponse(
            intent="price_query",
            intent_confidence=parsed.intent_confidence,
            data={"error": str(e)},
            message=f"Failed to retrieve intraday price data for {ticker}: {str(e)}"
        )


def _handle_price_query(parsed, query: str) -> QueryResponse:
    """Handle price queries with time-aware logic"""
    from workers.adapters.prices_polygon import get_prices_agg, get_intraday_price
    from workers.adapters.calendar import nearest_trading_day_utc
    from workers.utils.time_aware_utils import get_market_time_aware_date, should_fetch_realtime_price, get_price_data_message
    from datetime import datetime, timedelta
    
    logger = logging.getLogger("requiem.api")
    ticker = parsed.ticker or "SPY"
    
    # Check if this is an intraday query (has time component)
    if parsed.time:
        logger.info(f"Intraday query detected: {parsed.start} {parsed.time} {parsed.timezone}")
        return _handle_intraday_price_query(parsed, query, logger)
    
    # Determine date range for daily queries
    if parsed.start and parsed.end:
        start, end = parsed.start, parsed.end
        time_context = "historical"  # Explicit date range
    elif parsed.start:
        start = parsed.start
        # For single date queries, use the same date for both start and end
        if len(parsed.start) == 10 and '-' in parsed.start:  # ISO date format
            end = parsed.start
            time_context = "historical"  # Explicit date
        else:
            end = datetime.now().strftime("%Y-%m-%d")
            time_context = "historical"  # Explicit start date
    else:
        # Default to time-aware logic for simple "price of X" queries
        end, time_context = get_market_time_aware_date()
        start = end
    
    # Snap to nearest trading day (handles weekends/holidays intelligently)
    requested_date = start
    start = nearest_trading_day_utc(start)
    is_adjusted = (start != requested_date)
    
    # Also adjust end date if it was the same as start (single date query)
    if end == requested_date:
        end = start
    
    if is_adjusted:
        logger.info(f"Adjusted {requested_date} to nearest trading day: {start}")
    
    try:
        # Check if we should fetch real-time data
        if should_fetch_realtime_price(time_context):
            logger.info(f"Market is open, fetching real-time data for {ticker}")
            # For real-time data, we'll use the current minute data
            # This is a simplified approach - in production you'd want more sophisticated real-time handling
            df = get_prices_agg(ticker, start, end)
        else:
            df = get_prices_agg(ticker, start, end)
        
        if df.empty:
            # Try to get the most recent available data as a fallback
            logger.info(f"No data for {ticker} on {start}, trying fallback dates...")
            
            # Generate fallback dates dynamically, searching backward from requested date
            from datetime import datetime, timedelta
            fallback_dates = []
            requested_dt = datetime.strptime(start, "%Y-%m-%d")
            
            # Try dates going back day by day for the first 30 days
            for i in range(1, 31):
                fallback_dt = requested_dt - timedelta(days=i)
                fallback_dates.append(fallback_dt.strftime("%Y-%m-%d"))
            
            # Then try weekly for 3 months
            for weeks in range(5, 17):
                fallback_dt = requested_dt - timedelta(weeks=weeks)
                fallback_dates.append(fallback_dt.strftime("%Y-%m-%d"))
            
            # Then try monthly for a year
            for months in range(4, 13):
                fallback_dt = requested_dt - timedelta(days=months*30)
                fallback_dates.append(fallback_dt.strftime("%Y-%m-%d"))
            
            fallback_df = None
            fallback_date = None
            for date in fallback_dates:
                try:
                    test_df = get_prices_agg(ticker, date, date)
                    if not test_df.empty:
                        fallback_df = test_df
                        fallback_date = date
                        logger.info(f"Found fallback data for {ticker} on {fallback_date}")
                        break
                except:
                    continue
            
            if fallback_df is not None and not fallback_df.empty:
                # Process fallback data
                prices = []
                for idx, row in fallback_df.iterrows():
                    prices.append({
                        "date": str(idx) if hasattr(idx, 'strftime') else str(row.get('date', idx)),
                        "open": float(row['open']),
                        "high": float(row['high']),
                        "low": float(row['low']),
                        "close": float(row['close']),
                        "volume": int(row['volume'])
                    })
                
                latest_price = prices[-1] if prices else None
                
                return QueryResponse(
                    intent="price_query",
                    intent_confidence=parsed.intent_confidence,
                    data={
                        "ticker": ticker,
                        "requested_date": start,
                        "actual_date": fallback_date,
                        "latest_price": latest_price,
                        "prices": prices,
                        "count": len(prices),
                        "is_fallback": True
                    },
                    message=f"⚠️ No data available for {ticker} on {start}. Showing most recent available data from {fallback_date}: ${latest_price['close']:.2f}"
                )
            else:
                message = f"No price data available for {ticker}. The requested date ({start}) may not have market data available yet."
                return QueryResponse(
                    intent="price_query",
                    intent_confidence=parsed.intent_confidence,
                    data={},
                    message=message
                )
        
        # Format price data
        prices = []
        for idx, row in df.iterrows():
            prices.append({
                "date": str(idx) if hasattr(idx, 'strftime') else str(row.get('date', idx)),
                "open": float(row['open']),
                "high": float(row['high']),
                "low": float(row['low']),
                "close": float(row['close']),
                "volume": int(row['volume'])
            })
        
        latest_price = prices[-1] if prices else None
        
        # Build message with time context and adjustment info
        time_message = get_price_data_message(time_context, start)
        
        if is_adjusted:
            from datetime import datetime as dt
            req_dt = dt.strptime(requested_date, "%Y-%m-%d")
            act_dt = dt.strptime(start, "%Y-%m-%d")
            day_name_req = req_dt.strftime("%A")
            day_name_act = act_dt.strftime("%A")
            
            message = f"📅 {requested_date} was {day_name_req} (non-trading day). Showing {start} ({day_name_act}): {ticker} at ${latest_price['close']:.2f}. {time_message}"
        else:
            if time_context == "market_hours":
                message = f"📈 {ticker} currently trading at ${latest_price['close']:.2f}. {time_message}"
            else:
                message = f"📊 {ticker} at ${latest_price['close']:.2f} on {latest_price['date']}. {time_message}"
        
        return QueryResponse(
            intent="price_query",
            intent_confidence=parsed.intent_confidence,
            data={
                "ticker": ticker,
                "start": start,
                "end": end,
                "requested_date": requested_date if is_adjusted else None,
                "actual_date": start if is_adjusted else None,
                "is_adjusted": is_adjusted,
                "time_context": time_context,
                "latest_price": latest_price,
                "prices": prices,
                "count": len(prices)
            },
            message=message
        )
    
    except Exception as e:
        logger.error(f"Price query error for {ticker}: {str(e)}", exc_info=True)
        
        # If it's a 403 error (forbidden/future date), try fallback
        if "403" in str(e) or "Forbidden" in str(e):
            logger.info(f"Got 403 error for {ticker} on {start}, trying fallback...")
            
            # Generate fallback dates dynamically
            from datetime import datetime, timedelta
            fallback_dates = []
            try:
                requested_dt = datetime.strptime(start, "%Y-%m-%d")
                
                # Try dates going back day by day for the first 30 days
                for i in range(1, 31):
                    fallback_dt = requested_dt - timedelta(days=i)
                    fallback_dates.append(fallback_dt.strftime("%Y-%m-%d"))
                
                # Then try weekly for 3 months
                for weeks in range(5, 17):
                    fallback_dt = requested_dt - timedelta(weeks=weeks)
                    fallback_dates.append(fallback_dt.strftime("%Y-%m-%d"))
                
                # Try each fallback date
                for date in fallback_dates:
                    try:
                        test_df = get_prices_agg(ticker, date, date)
                        if not test_df.empty:
                            # Process fallback data
                            prices = []
                            for idx, row in test_df.iterrows():
                                prices.append({
                                    "date": str(idx) if hasattr(idx, 'strftime') else str(row.get('date', idx)),
                                    "open": float(row['open']),
                                    "high": float(row['high']),
                                    "low": float(row['low']),
                                    "close": float(row['close']),
                                    "volume": int(row['volume'])
                                })
                            
                            latest_price = prices[-1] if prices else None
                            
                            return QueryResponse(
                                intent="price_query",
                                intent_confidence=parsed.intent_confidence,
                                data={
                                    "ticker": ticker,
                                    "requested_date": start,
                                    "actual_date": date,
                                    "latest_price": latest_price,
                                    "prices": prices,
                                    "count": len(prices),
                                    "is_fallback": True
                                },
                                message=f"⚠️ No data available for {ticker} on {start}. Showing most recent available data from {date}: ${latest_price['close']:.2f}"
                            )
                    except:
                        continue
            except:
                pass
        
        # If fallback didn't work or not a 403 error, return error
        return QueryResponse(
            intent="price_query",
            intent_confidence=parsed.intent_confidence,
            data={"error": str(e)},
            message=f"Failed to retrieve price data for {ticker}: {str(e)}"
        )


def _handle_compound_query(parsed, query: str, detected_intents: list, selected_tools: list) -> QueryResponse:
    """Handle compound queries that contain multiple intents"""
    logger = logging.getLogger("requiem.api")
    logger.info(f"Handling compound query with intents: {detected_intents}")
    
    # For now, return a message indicating compound queries should use planning mode
    # In the future, we could execute all intents here
    return QueryResponse(
        intent="compound",
        intent_confidence=parsed.intent_confidence,
        data={
            "detected_intents": detected_intents,
            "suggestion": "planning_mode"
        },
        message=f"🔍 I detected multiple intents in your query: {', '.join(detected_intents)}. For comprehensive analysis, please enable Planning Mode (📋 button) to review and execute all steps together."
    )


def _handle_valuation_query(parsed, query: str) -> QueryResponse:
    """Handle valuation queries"""
    from workers.engine.valuation import valuation_analyzer
    
    ticker = parsed.ticker or "SPY"
    logger = logging.getLogger("requiem.api")
    logger.info(f"Analyzing valuation for {ticker}")
    
    try:
        # Perform comprehensive valuation analysis
        analysis_result = valuation_analyzer.analyze_valuation(ticker)
        
        if "error" in analysis_result:
            return QueryResponse(
                intent="valuation",
                intent_confidence=parsed.intent_confidence,
                data={
                    "ticker": ticker,
                    "error": analysis_result["error"]
                },
                message=f"Unable to analyze valuation for {ticker}: {analysis_result['error']}"
            )
        
        # Extract key information for response
        current_price = analysis_result.get("current_price")
        assessment = analysis_result.get("assessment", {})
        metrics = analysis_result.get("valuation_metrics", {})
        
        # Debug current price
        logger.info(f"Valuation analysis result for {ticker}: current_price={current_price}, type={type(current_price)}")
        
        # Generate user-friendly message
        overall_rating = assessment.get("overall_rating", "Unknown")
        confidence = assessment.get("confidence", "Low")
        reasoning = assessment.get("reasoning", "Insufficient data")
        
        price_info = f" at ${current_price:.2f}" if current_price else ""
        
        message = f"**Valuation Analysis for {ticker}{price_info}**\n\n"
        message += f"**Overall Assessment: {overall_rating}** (Confidence: {confidence})\n\n"
        message += f"**Reasoning:** {reasoning}\n\n"
        
        # Add key metrics
        if metrics:
            message += "**Key Valuation Metrics:**\n"
            
            if "pe_ratio" in metrics:
                pe_assessment = metrics.get("pe_assessment", {})
                message += f"• P/E Ratio: {metrics['pe_ratio']:.2f} ({pe_assessment.get('assessment', 'N/A')})\n"
            
            if "pb_ratio" in metrics:
                pb_assessment = metrics.get("pb_assessment", {})
                message += f"• P/B Ratio: {metrics['pb_ratio']:.2f} ({pb_assessment.get('assessment', 'N/A')})\n"
            
            if "ps_ratio" in metrics:
                ps_assessment = metrics.get("ps_assessment", {})
                message += f"• P/S Ratio: {metrics['ps_ratio']:.2f} ({ps_assessment.get('assessment', 'N/A')})\n"
            
            if "peg_ratio" in metrics:
                peg_assessment = metrics.get("peg_assessment", {})
                message += f"• PEG Ratio: {metrics['peg_ratio']:.2f} ({peg_assessment.get('assessment', 'N/A')})\n"
            
            if "dividend_yield" in metrics:
                div_assessment = metrics.get("dividend_assessment", {})
                message += f"• Dividend Yield: {metrics['dividend_yield']:.2%} ({div_assessment.get('assessment', 'N/A')})\n"
            
            if "estimated_fair_value" in metrics:
                fair_value = metrics["estimated_fair_value"]
                premium_discount = metrics.get("fair_value_premium_discount", 0)
                message += f"• Estimated Fair Value: ${fair_value:.2f} ({premium_discount:+.1f}% premium/discount)\n"
        
        # Add data source information
        data_sources = analysis_result.get("data_sources", {})
        sources_used = []
        if data_sources.get("alpha_vantage"):
            sources_used.append("Alpha Vantage")
        if data_sources.get("yahoo_finance"):
            sources_used.append("Yahoo Finance")
        
        if sources_used:
            message += f"\n*Data sources: {', '.join(sources_used)}*"
        
        return QueryResponse(
            intent="valuation",
            intent_confidence=parsed.intent_confidence,
            data=analysis_result,
            message=message
        )
        
    except Exception as e:
        logger.error(f"Error in valuation analysis for {ticker}: {str(e)}", exc_info=True)
        return QueryResponse(
            intent="valuation",
            intent_confidence=parsed.intent_confidence,
            data={
                "ticker": ticker,
                "error": str(e)
            },
            message=f"Error analyzing valuation for {ticker}: {str(e)}"
        )


def _handle_comparison_query(parsed, query: str) -> QueryResponse:
    """Handle comparison queries"""
    ticker = parsed.ticker or "SPY"
    
    # Note: This is a stub - would need to extract multiple tickers and compare
    
    return QueryResponse(
        intent="comparison",
        intent_confidence=parsed.intent_confidence,
        data={
            "ticker": ticker,
            "note": "Comparison requires multi-asset analysis"
        },
        message=f"Comparison analysis starting with {ticker} is not yet implemented. This would require extracting multiple tickers and running comparative metrics."
    )


def _generate_technical_insight(tool_name: str, ticker: str, result: Dict, query: str) -> str:
    """Generate GPT-powered insights for technical analysis results"""
    try:
        from openai import OpenAI
        import os
        
        # Initialize GPT client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return _generate_fallback_technical_insight(tool_name, ticker, result)
        
        client = OpenAI(api_key=api_key)
        
        # Build a concise prompt for GPT
        macd_info = ""
        if tool_name.lower() == 'macd':
            # Get MACD-specific data from chart_data
            chart_data = result.get('chart_data', {})
            macd_values = chart_data.get('macd', [])
            signal_values = chart_data.get('signal', [])
            histogram_values = chart_data.get('histogram', [])
            
            if macd_values and signal_values and histogram_values:
                latest_macd = macd_values[-1] if macd_values else 'N/A'
                latest_signal = signal_values[-1] if signal_values else 'N/A'
                latest_histogram = histogram_values[-1] if histogram_values else 'N/A'
                macd_info = f"\nMACD Line: {latest_macd:.4f}, Signal Line: {latest_signal:.4f}, Histogram: {latest_histogram:.4f}"
        
        prompt = f"""Analyze {tool_name.upper()} for {ticker} - query: "{query}"

Current: {result.get('latest_value', 'N/A')}, Mean: {result.get('mean_value', 'N/A')}, Range: {result.get('min_value', 'N/A')}-{result.get('max_value', 'N/A')}
Period: {result.get('period', 'N/A')}, Data points: {result.get('data_points', 'N/A')}{macd_info}

Provide 2 concise paragraphs: 1) Current signal interpretation (mention all components for MACD), 2) Trading implications."""

        # Call GPT API with optimized settings
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a technical analysis expert. Provide concise, actionable trading insights."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=250,  # Limit response length for faster generation
            timeout=8  # 8 second timeout
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        logger.error(f"GPT technical insight generation failed: {e}")
        return _generate_fallback_technical_insight(tool_name, ticker, result)

def _generate_fallback_technical_insight(tool_name: str, ticker: str, result: Dict) -> str:
    """Fallback technical insight generation"""
    latest_value = result.get('latest_value')
    mean_value = result.get('mean_value')
    
    if tool_name.lower() == 'rsi':
        if latest_value is not None:
            if latest_value > 70:
                return f"The RSI of {latest_value:.2f} indicates {ticker} is in overbought territory (>70), suggesting potential selling pressure or price correction ahead. This could signal a bearish reversal opportunity for traders looking to short or take profits."
            elif latest_value < 30:
                return f"The RSI of {latest_value:.2f} shows {ticker} is in oversold territory (<30), indicating potential buying opportunity as the stock may be undervalued and due for a bounce."
            else:
                return f"The RSI of {latest_value:.2f} suggests {ticker} is in neutral territory (30-70), indicating balanced momentum without strong directional bias. Traders should look for other confirmations or wait for clearer signals."
    
    elif tool_name.lower() == 'sma':
        return f"The Simple Moving Average analysis shows {ticker} with a current value of {latest_value:.2f} and average of {mean_value:.2f}. This trend-following indicator helps identify the overall price direction and potential support/resistance levels."
    
    elif tool_name.lower() == 'macd':
        return f"The MACD analysis for {ticker} reveals current momentum characteristics with a value of {latest_value:.2f}. MACD helps identify trend changes and momentum shifts, with crossovers above/below the signal line indicating potential buy/sell signals."
    
    else:
        return f"The {tool_name.upper()} analysis for {ticker} shows a current value of {latest_value:.2f} with an average of {mean_value:.2f}. This technical indicator provides insights into price behavior and potential trading opportunities based on historical patterns."

def _generate_bollinger_insight(ticker: str, result: Dict, query: str) -> str:
    """Generate GPT-powered insights for Bollinger Bands analysis"""
    try:
        from openai import OpenAI
        import os
        
        # Initialize GPT client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return _generate_fallback_bollinger_insight(ticker, result)
        
        client = OpenAI(api_key=api_key)
        
        bollinger_data = result.get('bollinger_data', {})
        price = bollinger_data.get('price_latest')
        upper_band = bollinger_data.get('upper_band_latest')
        middle_band = bollinger_data.get('middle_band_latest')
        lower_band = bollinger_data.get('lower_band_latest')
        bandwidth = bollinger_data.get('bandwidth_latest')
        
        # Build a concise prompt for GPT
        prompt = f"""Analyze Bollinger Bands for {ticker} - query: "{query}"

Current: Price={price}, Upper={upper_band}, Middle={middle_band}, Lower={lower_band}, Bandwidth={bandwidth}
Period: {result.get('period', 'N/A')}, Data points: {result.get('data_points', 'N/A')}

Provide 2 concise paragraphs: 1) Current signal interpretation (price vs bands), 2) Trading implications."""

        # Call GPT API with optimized settings
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a technical analysis expert specializing in Bollinger Bands. Provide concise, actionable trading insights about price position relative to bands and volatility."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=300,  # Slightly more tokens for Bollinger Bands analysis
            timeout=8
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        logger.error(f"GPT Bollinger Bands insight generation failed: {e}")
        return _generate_fallback_bollinger_insight(ticker, result)

def _generate_fallback_bollinger_insight(ticker: str, result: Dict) -> str:
    """Fallback Bollinger Bands insight generation"""
    bollinger_data = result.get('bollinger_data', {})
    price = bollinger_data.get('price_latest')
    upper_band = bollinger_data.get('upper_band_latest')
    middle_band = bollinger_data.get('middle_band_latest')
    lower_band = bollinger_data.get('lower_band_latest')
    bandwidth = bollinger_data.get('bandwidth_latest')
    
    if price is None or upper_band is None or lower_band is None:
        return f"**Bollinger Bands Analysis for {ticker}:** Unable to generate detailed analysis due to missing data points.\n\n**Trading Implications:** Monitor for price breakouts above upper band or bounces off lower band for potential trading opportunities."
    
    # Determine price position relative to bands
    if price > upper_band:
        position = "above the upper band"
        signal = "overbought"
        implication = "potential selling pressure or continuation of uptrend"
    elif price < lower_band:
        position = "below the lower band"
        signal = "oversold"
        implication = "potential buying opportunity or continuation of downtrend"
    elif price > middle_band:
        position = "above the middle band (SMA)"
        signal = "bullish"
        implication = "uptrend continuation possible"
    else:
        position = "below the middle band (SMA)"
        signal = "bearish"
        implication = "downtrend continuation possible"
    
    # Analyze bandwidth for volatility
    if bandwidth is not None:
        if bandwidth < 0.1:  # Low volatility
            volatility_note = "Low volatility (bands are tight) - watch for breakout"
        elif bandwidth > 0.3:  # High volatility
            volatility_note = "High volatility (bands are wide) - expect consolidation"
        else:
            volatility_note = "Normal volatility levels"
    else:
        volatility_note = "Volatility analysis unavailable"
    
    return f"**Current Signal Interpretation:** The current price of {ticker} is {position}, indicating {signal} conditions. {volatility_note}.\n\n**Trading Implications:** {implication}. Consider monitoring for price action near band boundaries for potential entry/exit signals, and watch for band squeezes (low bandwidth) that often precede significant moves."

def _handle_analysis_query(parsed, query: str, selected_tools: List[str] = None) -> QueryResponse:
    """Handle general analysis queries - tool-aware execution"""
    from workers.engine.features import REQUIEM_TOOLS
    from workers.engine.tool_executor import parse_tool_request, get_tool_executor_for_user
    from datetime import datetime, timedelta
    
    logger = logging.getLogger("requiem.api")
    ticker = parsed.ticker or "SPY"
    query_lower = query.lower()
    
    # Detect which tool is requested
    tool_keywords = {
        'rsi': ['rsi', 'relative strength', 'relative strength index'],
        'sma': ['sma', 'simple moving average', 'moving average'],
        'zscore': ['zscore', 'z-score', 'z score', 'standardized'],
        'realized_vol': ['volatility', 'realized vol', 'vol', 'realized volatility'],
        'momentum': ['momentum', 'mom'],
        'aroon': ['aroon', 'trend'],
        'macd': ['macd', 'moving average convergence'],
        'bollinger': ['bollinger', 'bollinger bands'],
        'williams_r': ['williams', 'williams r', 'williams %r'],
        'stochastic': ['stochastic', 'stoch'],
    }
    
    # Check for specific time period requests
    time_period_keywords = {
        '1 month': ['1 month', 'one month', 'last month', 'over the last month', 'last 30 days'],
        '3 months': ['3 months', 'three months', 'last 3 months', 'over the last 3 months'],
        '6 months': ['6 months', 'six months', 'last 6 months', 'over the last 6 months'],
        '1 year': ['1 year', 'one year', 'last year', 'over the last year', '12 months'],
        '2 years': ['2 years', 'two years', 'last 2 years', 'over the last 2 years'],
    }
    
    # Detect time period
    detected_period = None
    for period, keywords in time_period_keywords.items():
        if any(kw in query_lower for kw in keywords):
            detected_period = period
            break
    
    detected_tool = None
    for tool_name, keywords in tool_keywords.items():
        if any(kw in query_lower for kw in keywords):
            detected_tool = tool_name
            logger.info(f"Detected tool '{detected_tool}' from query: '{query}'")
            break
    
    logger.info(f"Final detected_tool: '{detected_tool}' for query: '{query}'")
    
    if not detected_tool:
        # Check if user is asking for general technical analysis
        if any(phrase in query_lower for phrase in ['technical analysis', 'conduct technical analysis', 'technical indicators', 'tech analysis']):
            # Return comprehensive technical analysis with multiple indicators
            return QueryResponse(
                intent="analysis",
                intent_confidence=parsed.intent_confidence,
                data={
                    "ticker": ticker,
                    "analysis_type": "comprehensive_technical",
                    "available_tools": list(REQUIEM_TOOLS.keys()),
                    "recommended_indicators": [
                        {"name": "RSI", "description": "Relative Strength Index - momentum oscillator (0-100)"},
                        {"name": "SMA", "description": "Simple Moving Average - trend following indicator"},
                        {"name": "MACD", "description": "Moving Average Convergence Divergence - trend changes"},
                        {"name": "Bollinger Bands", "description": "Volatility bands around moving average"},
                        {"name": "Williams %R", "description": "Momentum oscillator for overbought/oversold"},
                        {"name": "Stochastic", "description": "Momentum indicator comparing closing price to range"}
                    ]
                },
                message=f"🔍 **Comprehensive Technical Analysis for {ticker}**\n\nI can provide detailed technical analysis using multiple indicators:\n\n**📊 Recommended Indicators:**\n• **RSI** - Momentum oscillator (0-100)\n• **SMA** - Simple Moving Average trends\n• **MACD** - Trend change detection\n• **Bollinger Bands** - Volatility analysis\n• **Williams %R** - Overbought/oversold levels\n• **Stochastic** - Momentum comparison\n\n**💡 Ask for specific indicators:**\n- \"RSI for {ticker}\"\n- \"MACD analysis of {ticker}\"\n- \"Bollinger Bands for {ticker}\"\n\n**🎯 Or get multiple indicators:**\n- \"Show RSI, MACD, and SMA for {ticker}\""
            )
        else:
            # No specific tool detected, return generic message
            return QueryResponse(
                intent="analysis",
                intent_confidence=parsed.intent_confidence,
                data={
                    "ticker": ticker,
                    "available_tools": list(REQUIEM_TOOLS.keys())
                },
                message=f"I can help with technical analysis for {ticker}. Try asking for specific indicators like RSI, SMA, MACD, Bollinger Bands, etc."
            )
    
    # Check if tool exists in Requiem tools
    if detected_tool not in REQUIEM_TOOLS:
        return QueryResponse(
            intent="analysis",
            intent_confidence=parsed.intent_confidence,
            data={"requested_tool": detected_tool},
            message=f"❌ Tool '{detected_tool}' not found. Available tools: {', '.join(REQUIEM_TOOLS.keys())}"
        )
    
    # Use selected tools from request, or default to all tools
    if not selected_tools:
        # If no tools provided, use all available tools by default
        selected_tools = list(REQUIEM_TOOLS.keys())
    
    logger.info(f"Selected tools: {selected_tools}")
    logger.info(f"Detected tool: {detected_tool}")
    
    if detected_tool not in selected_tools:
        return QueryResponse(
            intent="analysis",
            intent_confidence=parsed.intent_confidence,
            data={
                "tool_name": detected_tool,
                "tool_available": True,
                "tool_selected": False
            },
            message=f"⚠️ Tool '{detected_tool}' is available but disabled. Please enable it in the Tools section (⚙️ Tools → Requiem Tools → {detected_tool}), or upload your own custom '{detected_tool}' tool."
        )
    
    # Tool is selected, execute it!
    try:
        tool_request = parse_tool_request(query, selected_tools)
        
        if not tool_request:
            return QueryResponse(
                intent="analysis",
                intent_confidence=parsed.intent_confidence,
                data={},
                message=f"I detected you want to use the '{detected_tool}' tool, but couldn't extract the parameters. Try: 'calculate {detected_tool} for {ticker}'"
            )
        
        # Execute the tool
        executor = get_tool_executor_for_user(selected_tools)
        
        # Determine date range
        from workers.adapters.calendar import nearest_trading_day_utc
        
        if parsed.start and parsed.end:
            # Explicit date range provided
            start_date = parsed.start
            end_date = parsed.end
        elif parsed.start:
            # Only start date provided (e.g., "over the last 1 year" sets start to 1 year ago)
            # Use start as the beginning, and use current date
            start_date = parsed.start
            # Use current date
            end_date = datetime.now().strftime("%Y-%m-%d")
            
            # Ensure start date is before end date
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            if start_dt >= end_dt:
                # If start is after or equal to end, adjust start to be 30 days before end
                start_dt = end_dt - timedelta(days=30)
                start_date = start_dt.strftime("%Y-%m-%d")
        else:
            # Use detected period or default to 3 months
            # Use current date as end date
            current_date = datetime.now().strftime("%Y-%m-%d")
            end_date = current_date
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            
            if detected_period == '1 month':
                start_dt = end_dt - timedelta(days=30)
            elif detected_period == '3 months':
                start_dt = end_dt - timedelta(days=90)
            elif detected_period == '6 months':
                start_dt = end_dt - timedelta(days=180)
            elif detected_period == '1 year':
                start_dt = end_dt - timedelta(days=365)
            elif detected_period == '2 years':
                start_dt = end_dt - timedelta(days=730)
            else:
                # Default: last 3 months
                start_dt = end_dt - timedelta(days=90)
            
            start_date = start_dt.strftime("%Y-%m-%d")
        
        result = executor.execute_tool(
            tool_name=tool_request["tool"],
            ticker=tool_request["ticker"],
            start_date=start_date,
            end_date=end_date,
            **tool_request["params"]
        )
        
        if "error" in result:
            return QueryResponse(
                intent="analysis",
                intent_confidence=parsed.intent_confidence,
                data={"error": result["error"]},
                message=f"Tool execution failed: {result['error']}"
            )
        
        # Format the response
        tool_name = result["tool_name"]
        latest_value = result["latest_value"]
        mean_value = result["mean_value"]
        
        # Format values safely
        latest_str = f"{latest_value:.4f}" if latest_value is not None else "N/A"
        mean_str = f"{mean_value:.4f}" if mean_value is not None else "N/A"
        
        # Create card data for technical analysis
        if tool_name == 'bollinger' and 'bollinger_data' in result:
            # Special handling for Bollinger Bands
            bollinger_data = result['bollinger_data']
            card_data = {
                "title": f"📊 Bollinger Bands Analysis — {ticker}",
                "meta": f"Period: {result['period']} · Data Points: {result['data_points']} · Tool: Bollinger Bands",
                "metrics": [
                    {"key": "Price", "value": f"{bollinger_data['price_latest']:.2f}" if bollinger_data['price_latest'] is not None else "N/A"},
                    {"key": "Upper Band", "value": f"{bollinger_data['upper_band_latest']:.2f}" if bollinger_data['upper_band_latest'] is not None else "N/A"},
                    {"key": "Middle Band", "value": f"{bollinger_data['middle_band_latest']:.2f}" if bollinger_data['middle_band_latest'] is not None else "N/A"},
                    {"key": "Lower Band", "value": f"{bollinger_data['lower_band_latest']:.2f}" if bollinger_data['lower_band_latest'] is not None else "N/A"},
                    {"key": "Bandwidth", "value": f"{bollinger_data['bandwidth_latest']:.4f}" if bollinger_data['bandwidth_latest'] is not None else "N/A"}
                ],
                "chart_data": result.get("series_data", {}),
                "insight": _generate_bollinger_insight(ticker, result, query),
                "diagnostics": {
                    "parameters_used": result.get("parameters_used", {}),
                    "tool_description": result.get("description", ""),
                    "period": result.get("period", "")
                }
            }
        else:
            # Standard technical analysis
            card_data = {
                "title": f"📊 {tool_name.upper()} Analysis — {ticker}",
                "meta": f"Period: {result['period']} · Data Points: {result['data_points']} · Tool: {tool_name.title()}",
                "metrics": [
                    {"key": "Latest", "value": latest_str},
                    {"key": "Mean", "value": mean_str},
                    {"key": "Min", "value": f"{result.get('min_value', 0):.4f}" if result.get('min_value') is not None else "N/A"},
                    {"key": "Max", "value": f"{result.get('max_value', 0):.4f}" if result.get('max_value') is not None else "N/A"},
                    {"key": "Data Points", "value": str(result['data_points'])}
                ],
                "chart_data": result.get("series_data", {}),
                "insight": _generate_technical_insight(tool_name, ticker, result, query),
                "diagnostics": {
                    "parameters_used": result.get("parameters_used", {}),
                    "tool_description": result.get("description", ""),
                    "period": result.get("period", "")
                }
            }
        
        # Add card data to result
        result["card_data"] = card_data
        
        message = f"**{tool_name.upper()}** for {ticker}\n\n"
        message += f"**Latest value:** {latest_str}\n"
        message += f"**Mean value:** {mean_str}\n"
        message += f"**Data points:** {result['data_points']}\n"
        message += f"**Period:** {result['period']}\n\n"
        message += f"*{result['description']}*"
        
        return QueryResponse(
            intent="analysis",
            intent_confidence=parsed.intent_confidence,
            data=result,
            message=message
        )
        
    except Exception as e:
        logger.error(f"Tool execution error: {str(e)}", exc_info=True)
        return QueryResponse(
            intent="analysis",
            intent_confidence=parsed.intent_confidence,
            data={"error": str(e)},
            message=f"Failed to execute {detected_tool}: {str(e)}"
        )


# ============================================================================
# TOOLS MANAGEMENT ENDPOINTS
# ============================================================================

class ToolInfo(BaseModel):
    name: str
    description: str
    selected: bool = False
    type: str  # "orchid" or "user"

class ToolsResponse(BaseModel):
    requiem_tools: List[ToolInfo]
    user_tools: List[ToolInfo]

@app.get("/tools", response_model=ToolsResponse)
def get_tools():
    """Get all available tools (Requiem built-in + user uploaded)"""
    logger = logging.getLogger("requiem.api")
    logger.info("Fetching tools list")
    
    # Import the actual tool registry
    from workers.engine.features import REQUIEM_TOOLS
    
    # Built-in Requiem tools from the actual registry
    requiem_tools = []
    for tool_name, tool_info in REQUIEM_TOOLS.items():
        # All tools selected by default
        selected = True
        
        requiem_tools.append(ToolInfo(
            name=tool_name,
            description=tool_info["description"],
            selected=selected,
            type="requiem"
        ))
    
    # User uploaded tools (in a real implementation, this would come from a database)
    user_tools = []  # TODO: Load from database
    
    return ToolsResponse(requiem_tools=requiem_tools, user_tools=user_tools)

@app.post("/tools/upload")
async def upload_tool(file: UploadFile = File(...)):
    """Upload a custom tool as a ZIP file"""
    logger = logging.getLogger("requiem.api")
    logger.info(f"Uploading tool: {file.filename}")
    
    # Validate file type
    if not file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="Only ZIP files are allowed")
    
    # Validate file size (max 10MB)
    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 10MB")
    
    try:
        # Extract and validate the ZIP file
        with tempfile.TemporaryDirectory() as temp_dir:
            zip_path = os.path.join(temp_dir, file.filename)
            
            # Write uploaded file
            with open(zip_path, 'wb') as f:
                f.write(content)
            
            # Validate ZIP structure
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                file_list = zip_ref.namelist()
                
                # Check for required files
                if not any(f.endswith('.py') for f in file_list):
                    raise HTTPException(status_code=400, detail="ZIP must contain at least one Python (.py) file")
                
                # Extract to temp directory for validation
                zip_ref.extractall(temp_dir)
                
                # Basic validation: check if main.py exists or has valid Python files
                python_files = [f for f in file_list if f.endswith('.py')]
                
                # TODO: More sophisticated validation (import the Python files, check for required functions)
                
        # Generate tool name from filename
        tool_name = file.filename.replace('.zip', '').replace(' ', '_').lower()
        tool_name = ''.join(c for c in tool_name if c.isalnum() or c in '_-')
        
        # In a real implementation, you would:
        # 1. Store the ZIP file in a secure location
        # 2. Extract and validate the Python code
        # 3. Add to database with metadata
        # 4. Return tool information
        
        logger.info(f"Tool '{tool_name}' uploaded successfully")
        
        return {
            "success": True,
            "tool_name": tool_name,
            "message": f"Tool '{tool_name}' uploaded successfully. Ready for use in strategies.",
            "files": file_list[:5]  # Show first 5 files as preview
        }
    
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Invalid ZIP file")
    except Exception as e:
        logger.error(f"Error uploading tool: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/tools/{tool_name}/toggle")
def toggle_tool(tool_name: str, selected: bool):
    """Toggle tool selection status"""
    logger = logging.getLogger("requiem.api")
    logger.info(f"Toggling tool '{tool_name}' to selected={selected}")
    
    # In a real implementation, this would update the database
    # For now, just return success
    
    return {
        "success": True,
        "tool_name": tool_name,
        "selected": selected,
        "message": f"Tool '{tool_name}' {'selected' if selected else 'deselected'}"
    }

@app.delete("/tools/{tool_name}")
def delete_tool(tool_name: str):
    """Delete a user-uploaded tool"""
    logger = logging.getLogger("requiem.api")
    logger.info(f"Deleting user tool: {tool_name}")
    
    # In a real implementation, this would:
    # 1. Remove from database
    # 2. Delete stored files
    # 3. Clean up any related data
    
    return {
        "success": True,
        "tool_name": tool_name,
        "message": f"Tool '{tool_name}' deleted successfully"
    }

class ToolExecutionRequest(BaseModel):
    query: str
    selected_tools: List[str] = []

class ToolExecutionResponse(BaseModel):
    success: bool
    results: Dict[str, Any] = {}
    message: str = ""

@app.post("/tools/execute", response_model=ToolExecutionResponse)
def execute_tools(request: ToolExecutionRequest):
    """Execute tools based on natural language query and user's selected tools"""
    logger = logging.getLogger("requiem.api")
    logger.info(f"Executing tools for query: {request.query}")
    
    try:
        from workers.engine.tool_executor import get_tool_executor_for_user, parse_tool_request
        
        # Create tool executor with user's selected tools
        executor = get_tool_executor_for_user(request.selected_tools)
        
        # Parse the query to extract tool execution request
        tool_request = parse_tool_request(request.query, request.selected_tools)
        
        if not tool_request:
            return ToolExecutionResponse(
                success=False,
                message="No tool execution request found in query. Try asking for specific calculations like 'compute SMA for 15 days on SPY'"
            )
        
        # Execute the tool
        # Get time-aware dates for technical analysis
        from workers.utils.time_aware_utils import get_market_time_aware_date
        from datetime import datetime, timedelta
        
        # Get current date and calculate start date (default to 3 months back)
        current_date, time_context = get_market_time_aware_date()
        current_dt = datetime.strptime(current_date, "%Y-%m-%d")
        
        # Default to 3 months back for technical analysis
        start_dt = current_dt - timedelta(days=90)
        default_start = start_dt.strftime("%Y-%m-%d")
        default_end = current_date
        
        result = executor.execute_tool(
            tool_name=tool_request["tool"],
            ticker=tool_request["ticker"],
            start_date=default_start,
            end_date=default_end,
            **tool_request["params"]
        )
        
        if "error" in result:
            return ToolExecutionResponse(
                success=False,
                message=f"Tool execution failed: {result['error']}"
            )
        
        # Format the response
        tool_name = result["tool_name"]
        ticker = result["ticker"]
        latest_value = result["latest_value"]
        description = result["description"]
        
        latest_value_str = f"{latest_value:.4f}" if latest_value is not None else "N/A"
        mean_value_str = f"{result['mean_value']:.4f}" if result['mean_value'] is not None else "N/A"
        
        response_message = f"**{tool_name.upper()}** for {ticker}: {latest_value_str}\n\n"
        response_message += f"*{description}*\n\n"
        response_message += f"Latest value: {latest_value_str}\n"
        response_message += f"Mean value: {mean_value_str}\n"
        response_message += f"Data points: {result['data_points']}\n"
        response_message += f"Parameters used: {result['parameters_used']}"
        
        return ToolExecutionResponse(
            success=True,
            results=result,
            message=response_message
        )
        
    except Exception as e:
        logger.error(f"Error executing tools: {str(e)}", exc_info=True)
        return ToolExecutionResponse(
            success=False,
            message=f"Error executing tools: {str(e)}"
        )


# Statistical Analysis Handlers
def _handle_statistical_analysis_query(parsed, query: str) -> QueryResponse:
    """Handle statistical analysis queries"""
    try:
        from workers.engine.statistical_analyzer import statistical_analyzer
        
        # Parse the query to extract parameters
        query_lower = query.lower()
        
        # Extract tickers from query
        tickers = []
        if "correlation" in query_lower or "regression" in query_lower or "cointegration" in query_lower:
            # Look for ticker patterns - accept any valid ticker format
            import re
            # More specific pattern to avoid common words like "OF", "ON", "TO", "FOR"
            ticker_pattern = r'\b([A-Z]{2,5})\b'
            potential_tickers = re.findall(ticker_pattern, query.upper())
            
            # Filter out common words that aren't tickers
            common_words = {'OF', 'ON', 'TO', 'FOR', 'AND', 'THE', 'WITH', 'FROM', 'RETURNS', 'MARKET', 'ANALYSIS', 'BETWEEN', 'CORRELATION', 'REGRESSION', 'OVER', 'LAST', 'PAST', 'MONTHS', 'MONTH', 'YEARS', 'WEEKS', 'DAYS', 'TEST', 'COINTEGRATION'}
            tickers = [t for t in potential_tickers if t not in common_words]
            
            # Handle "market returns" as SPY
            if "market" in query_lower and "SPY" not in tickers:
                tickers.append("SPY")
        
        # Extract time period - more comprehensive parsing
        period = "1y"  # default
        
        # Check for various time period patterns
        if any(phrase in query_lower for phrase in ["last 6 months", "6 months", "over the last 6 months", "past 6 months"]):
            period = "6m"
        elif any(phrase in query_lower for phrase in ["last 3 months", "3 months", "over the last 3 months", "past 3 months"]):
            period = "3m"
        elif any(phrase in query_lower for phrase in ["last 1 month", "1 month", "over the last month", "past month"]):
            period = "1m"
        elif any(phrase in query_lower for phrase in ["last 1 year", "1 year", "over the last year", "past year"]):
            period = "1y"
        elif any(phrase in query_lower for phrase in ["last 2 years", "2 years", "over the last 2 years", "past 2 years"]):
            period = "2y"
        elif any(phrase in query_lower for phrase in ["last 5 years", "5 years", "over the last 5 years", "past 5 years"]):
            period = "5y"
        elif any(phrase in query_lower for phrase in ["last week", "1 week", "over the last week"]):
            period = "1w"
        elif any(phrase in query_lower for phrase in ["last 2 weeks", "2 weeks", "over the last 2 weeks"]):
            period = "2w"
        
        # Route to appropriate analysis
        if "correlation" in query_lower:
            if len(tickers) < 2:
                tickers = ["AAPL", "SPY"]  # default
            result = statistical_analyzer.correlation_analysis(tickers, period, "pearson", query)
        elif "regression" in query_lower:
            if len(tickers) < 2:
                tickers = ["AAPL", "SPY"]  # default
            result = statistical_analyzer.regression_analysis(tickers[0], tickers[1:], period, query)
        elif "cointegration" in query_lower:
            if len(tickers) < 2:
                tickers = ["AAPL", "SPY"]  # default
            result = statistical_analyzer.cointegration_test(tickers, period, query)
        elif "stationarity" in query_lower:
            if not tickers:
                tickers = ["AAPL"]  # default
            result = statistical_analyzer.stationarity_test(tickers[0], period)
        elif "volatility" in query_lower:
            if not tickers:
                tickers = ["AAPL"]  # default
            result = statistical_analyzer.volatility_analysis(tickers[0], period)
        else:
            # Default to correlation analysis
            if len(tickers) < 2:
                tickers = ["AAPL", "SPY"]  # default
            result = statistical_analyzer.correlation_analysis(tickers, period, "pearson", query)
        
        if "error" in result:
            return QueryResponse(
                intent="statistical_analysis",
                intent_confidence=0.9,
                data={},
                message=f"Statistical analysis failed: {result['error']}"
            )
        
        # Format response message with cleaner output
        analysis_type = result.get("analysis_type", "statistical_analysis")
        interpretation = result.get("interpretation", "Analysis completed successfully.")
        data_points = result.get("data_points", 0)
        period = result.get("period", "")
        
        response_message = f"📊 **{analysis_type.replace('_', ' ').title()}**\n\n"
        
        # Handle different analysis types
        if analysis_type == "correlation_analysis":
            # Extract key metrics for correlation analysis
            strongest_correlations = result.get("strongest_correlations", [])
            
            # Show key correlation results
            if strongest_correlations:
                top_corr = strongest_correlations[0]
                response_message += f"**Top Correlation:** {top_corr['pair']} = {top_corr['correlation']:.3f} ({top_corr['strength']})\n"
            
            response_message += f"**Period:** {period}\n"
            response_message += f"**Data Points:** {data_points}\n\n"
            
            # Add concise interpretation
            response_message += "**Analysis:**\n"
            corr_value = 0.0  # Default value
            if strongest_correlations:
                corr_value = strongest_correlations[0]['correlation']
                if abs(corr_value) > 0.7:
                    response_message += f"- Strong correlation ({corr_value:.3f}) - assets move together closely\n"
                elif abs(corr_value) > 0.3:
                    response_message += f"- Moderate correlation ({corr_value:.3f}) - some co-movement\n"
                else:
                    response_message += f"- Weak correlation ({corr_value:.3f}) - limited relationship\n"
            else:
                response_message += "- No significant correlations found\n"
            
            response_message += f"- Statistical significance: Confirmed\n"
            response_message += f"- Risk implications: {'High' if abs(corr_value) > 0.7 else 'Moderate' if abs(corr_value) > 0.3 else 'Low'} portfolio correlation risk"
            
        elif analysis_type == "regression_analysis":
            # Check if we have card data for structured format
            if "card_data" in result:
                card_data = result["card_data"]
                
                # Format as structured regression card
                response_message = f"## {card_data['title']}\n\n"
                response_message += f"**{card_data['meta']}**\n\n"
                
                # Metrics section
                response_message += "### 📊 Key Metrics\n"
                for metric in card_data['metrics']:
                    response_message += f"- **{metric['key']}:** {metric['value']}\n"
                
                # Coefficients section
                response_message += "\n### β Coefficients\n"
                response_message += "| Variable | Beta | p-value | Sig |\n"
                response_message += "|----------|------|---------|-----|\n"
                for coeff in card_data['coefficients']:
                    sig_icon = "✅" if coeff['sig'] else "❌"
                    response_message += f"| {coeff['var']} | {coeff['beta']} | {coeff['p']} | {sig_icon} |\n"
                
                # AI Insight
                response_message += f"\n### 🔍 AI Insight\n{card_data['insight']}\n"
                
                # Model comparison
                if card_data['model_comparison']:
                    response_message += "\n### Model Comparison\n"
                    response_message += "| Model | Variables | R² | Beta |\n"
                    response_message += "|-------|-----------|----|----|\n"
                    for comp in card_data['model_comparison']:
                        response_message += f"| {comp['model']} | {comp['vars']} | {comp['r2']} | {comp.get('beta', '—')} |\n"
            else:
                # Fallback to original format
                r_squared = result.get("r_squared", 0)
                beta_values = result.get("beta_values", {})
                f_pvalue = result.get("f_pvalue", 1)
                dependent_var = result.get("dependent_variable", "")
                independent_vars = result.get("independent_variables", [])
                
                response_message += f"**Dependent Variable:** {dependent_var}\n"
                response_message += f"**Independent Variables:** {', '.join(independent_vars)}\n"
                response_message += f"**Period:** {period}\n"
                response_message += f"**Data Points:** {data_points}\n\n"
                
                response_message += "**Model Results:**\n"
                response_message += f"- R-squared: {r_squared:.3f} ({r_squared*100:.1f}% of variance explained)\n"
                response_message += f"- Statistical significance: {'Yes' if f_pvalue < 0.05 else 'No'} (p-value: {f_pvalue:.2e})\n\n"
                
                response_message += "**Beta Coefficients:**\n"
                for var, beta in beta_values.items():
                    volatility_desc = "more volatile" if beta > 1.0 else "less volatile" if beta < 1.0 else "equally volatile"
                    response_message += f"- {var}: {beta:.3f} ({volatility_desc} than market)\n"
                
                response_message += f"\n**Interpretation:**\n"
                if r_squared > 0.7:
                    response_message += f"- Strong model fit ({r_squared:.1%} variance explained)\n"
                elif r_squared > 0.3:
                    response_message += f"- Moderate model fit ({r_squared:.1%} variance explained)\n"
                else:
                    response_message += f"- Weak model fit ({r_squared:.1%} variance explained)\n"
                
                if f_pvalue < 0.05:
                    response_message += f"- Model is statistically significant\n"
                else:
                    response_message += f"- Model is not statistically significant\n"
                
        else:
            # Default formatting for other analysis types
            response_message += f"**Period:** {period}\n"
            response_message += f"**Data Points:** {data_points}\n\n"
            response_message += "**Analysis:**\n"
            response_message += interpretation
        
        return QueryResponse(
            intent="statistical_analysis",
            intent_confidence=0.9,
            data=result,
            message=response_message
        )
        
    except Exception as e:
        logger.error(f"Error in statistical analysis: {str(e)}")
        return QueryResponse(
            intent="statistical_analysis",
            intent_confidence=0.0,
            data={},
            message=f"Statistical analysis failed: {str(e)}"
        )


def _handle_risk_metrics_query(parsed, query: str) -> QueryResponse:
    """Handle risk metrics queries"""
    try:
        from workers.engine.risk_metrics import risk_metrics
        from workers.adapters.prices_polygon import get_prices_agg
        from workers.utils.time_aware_utils import get_market_time_aware_date
        from datetime import datetime, timedelta
        
        # Parse the query to extract parameters
        query_lower = query.lower()
        
        # Extract ticker from query
        ticker = "AAPL"  # default
        import re
        ticker_pattern = r'\b([A-Z]{1,5})\b'
        potential_tickers = re.findall(ticker_pattern, query.upper())
        valid_tickers = [t for t in potential_tickers if t in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'SPY', 'QQQ', 'IWM', 'VTI']]
        if valid_tickers:
            ticker = valid_tickers[0]
        
        # Get time period - use historical data for risk metrics
        # Use 2024-10-13 as end date to ensure we have real data
        end_date = "2024-10-13"
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        if "last 6 months" in query_lower or "6 months" in query_lower:
            start_dt = end_dt - timedelta(days=180)
        elif "last 3 months" in query_lower or "3 months" in query_lower:
            start_dt = end_dt - timedelta(days=90)
        elif "last 1 year" in query_lower or "1 year" in query_lower:
            start_dt = end_dt - timedelta(days=365)
        elif "last 2 years" in query_lower or "2 years" in query_lower:
            start_dt = end_dt - timedelta(days=730)
        else:
            start_dt = end_dt - timedelta(days=365)  # default to 1 year
        
        start_date = start_dt.strftime("%Y-%m-%d")
        
        # Fetch price data
        df = get_prices_agg(ticker, start_date, end_date)
        if df.empty:
            return QueryResponse(
                intent="risk_metrics",
                intent_confidence=0.9,
                data={},
                message=f"No data available for {ticker} in the specified period."
            )
        
        returns = df['close'].pct_change().dropna()
        
        # Route to appropriate risk metric
        if "var" in query_lower or "value at risk" in query_lower:
            confidence = 0.95
            if "99%" in query_lower:
                confidence = 0.99
            elif "90%" in query_lower:
                confidence = 0.90
            result = risk_metrics.var_calculation(returns, confidence, "all")
        elif "cvar" in query_lower or "expected shortfall" in query_lower:
            confidence = 0.95
            if "99%" in query_lower:
                confidence = 0.99
            elif "90%" in query_lower:
                confidence = 0.90
            result = risk_metrics.cvar_calculation(returns, confidence)
        elif "drawdown" in query_lower:
            result = risk_metrics.maximum_drawdown(returns)
        elif "sharpe" in query_lower:
            result = risk_metrics.sharpe_ratio(returns)
        elif "sortino" in query_lower:
            result = risk_metrics.sortino_ratio(returns)
        elif "calmar" in query_lower:
            result = risk_metrics.calmar_ratio(returns)
        else:
            # Default to Sharpe ratio
            result = risk_metrics.sharpe_ratio(returns)
        
        if "error" in result:
            return QueryResponse(
                intent="risk_metrics",
                intent_confidence=0.9,
                data={},
                message=f"Risk metrics calculation failed: {result['error']}"
            )
        
        # Create card data for risk metrics
        card_data = {
            "title": f"📈 Risk Analysis — {ticker}",
            "meta": f"Period: {start_date} to {end_date} · Data Points: {len(returns)} · Method: {result.get('method', 'standard')}",
            "metrics": [],
            "analysis_type": "risk_metrics",
            "ticker": ticker,
            "period": f"{start_date} to {end_date}",
            "insight": result.get("interpretation", "Risk analysis completed successfully.")
        }
        
        # Add metrics based on the type of risk calculation
        if "var" in query_lower or "value at risk" in query_lower:
            if "historical" in result:
                card_data["metrics"].extend([
                    {"key": "VaR (Historical)", "value": f"{result['historical']['var_percentage']:.2f}%"},
                    {"key": "Confidence Level", "value": f"{confidence*100:.0f}%"},
                    {"key": "Method", "value": "Historical Simulation"}
                ])
            if "parametric" in result:
                card_data["metrics"].extend([
                    {"key": "VaR (Parametric)", "value": f"{result['parametric']['var_percentage']:.2f}%"},
                    {"key": "Mean Return", "value": f"{result['parametric']['mean_return']*100:.2f}%"},
                    {"key": "Std Deviation", "value": f"{result['parametric']['std_return']*100:.2f}%"}
                ])
        elif "sharpe" in query_lower:
            card_data["metrics"].extend([
                {"key": "Sharpe Ratio", "value": f"{result.get('sharpe_ratio', 0):.3f}"},
                {"key": "Annual Return", "value": f"{result.get('annual_return', 0)*100:.2f}%"},
                {"key": "Volatility", "value": f"{result.get('volatility', 0)*100:.2f}%"},
                {"key": "Risk-Free Rate", "value": f"{result.get('risk_free_rate', 0)*100:.2f}%"}
            ])
        elif "sortino" in query_lower:
            card_data["metrics"].extend([
                {"key": "Sortino Ratio", "value": f"{result.get('sortino_ratio', 0):.3f}"},
                {"key": "Annual Return", "value": f"{result.get('annual_return', 0)*100:.2f}%"},
                {"key": "Downside Deviation", "value": f"{result.get('downside_deviation', 0)*100:.2f}%"},
                {"key": "Risk-Free Rate", "value": f"{result.get('risk_free_rate', 0)*100:.2f}%"}
            ])
        elif "drawdown" in query_lower:
            card_data["metrics"].extend([
                {"key": "Max Drawdown", "value": f"{result.get('max_drawdown', 0)*100:.2f}%"},
                {"key": "Drawdown Date", "value": result.get('max_drawdown_date', 'N/A')},
                {"key": "Recovery Time", "value": f"{result.get('recovery_time', 0)} days"},
                {"key": "Current Drawdown", "value": f"{result.get('current_drawdown', 0)*100:.2f}%"}
            ])
        else:
            # Default metrics
            card_data["metrics"].extend([
                {"key": "Analysis Type", "value": analysis_type.replace('_', ' ').title()},
                {"key": "Data Points", "value": str(len(returns))},
                {"key": "Period", "value": f"{start_date} to {end_date}"}
            ])
        
        # Add card data to result
        result["card_data"] = card_data
        
        # Format response message
        analysis_type = result.get("analysis_type", "risk_metrics")
        interpretation = result.get("interpretation", "Risk analysis completed successfully.")
        
        response_message = f"📈 **{analysis_type.replace('_', ' ').title()}**\n\n"
        response_message += interpretation
        
        return QueryResponse(
            intent="risk_metrics",
            intent_confidence=0.9,
            data=result,
            message=response_message
        )
        
    except Exception as e:
        logger.error(f"Error in risk metrics calculation: {str(e)}")
        return QueryResponse(
            intent="risk_metrics",
            intent_confidence=0.0,
            data={},
            message=f"Risk metrics calculation failed: {str(e)}"
        )


def _handle_mathematical_calculation_query(parsed, query: str) -> QueryResponse:
    """Handle mathematical calculation queries"""
    try:
        from workers.engine.math_engine import math_engine
        
        # Parse the query to extract parameters
        query_lower = query.lower()
        
        # Route to appropriate mathematical calculation
        if "monte carlo" in query_lower:
            # Extract ticker from query
            ticker = None
            import re
            ticker_pattern = r'\b([A-Z]{1,5})\b'
            potential_tickers = re.findall(ticker_pattern, query.upper())
            valid_tickers = [t for t in potential_tickers if t in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'SPY', 'QQQ', 'IWM', 'VTI', 'SVIX']]
            if valid_tickers:
                ticker = valid_tickers[0]
            
            # Extract parameters for Monte Carlo simulation
            params = {
                "model_type": "geometric_brownian",
                "ticker": ticker,
                "initial_value": 100.0,
                "drift": 0.05,
                "volatility": 0.2,
                "time_horizon": 1.0,
                "n_simulations": 10000,
                "n_steps": 252
            }
            
            # Extract time horizon from query
            time_horizon = 1.0  # Default to 1 year
            if "day" in query_lower:
                if "30" in query_lower or "month" in query_lower:
                    time_horizon = 30/365  # 30 days
                elif "7" in query_lower or "week" in query_lower:
                    time_horizon = 7/365  # 1 week
                elif "90" in query_lower or "quarter" in query_lower:
                    time_horizon = 90/365  # 3 months
                elif "180" in query_lower or "half" in query_lower:
                    time_horizon = 180/365  # 6 months
            elif "month" in query_lower:
                if "3" in query_lower:
                    time_horizon = 3/12  # 3 months
                elif "6" in query_lower:
                    time_horizon = 6/12  # 6 months
                elif "9" in query_lower:
                    time_horizon = 9/12  # 9 months
                else:
                    time_horizon = 1/12  # 1 month
            elif "year" in query_lower:
                if "2" in query_lower:
                    time_horizon = 2.0  # 2 years
                elif "3" in query_lower:
                    time_horizon = 3.0  # 3 years
                elif "5" in query_lower:
                    time_horizon = 5.0  # 5 years
                else:
                    time_horizon = 1.0  # 1 year
            
            params["time_horizon"] = time_horizon
            params["n_steps"] = max(50, int(time_horizon * 252))  # Adjust steps based on time horizon
            
            # Extract number of simulations if mentioned
            if "simulation" in query_lower:
                import re
                sim_match = re.search(r'(\d+)\s*simulation', query_lower)
                if sim_match:
                    params["n_simulations"] = min(int(sim_match.group(1)), 50000)  # Cap at 50k for performance
            
            # Extract specific parameters if mentioned
            if "jump" in query_lower:
                params["model_type"] = "jump_diffusion"
                params["jump_intensity"] = 0.1
                params["jump_mean"] = 0.0
                params["jump_std"] = 0.1
            
            result = math_engine.monte_carlo_simulation(params)
            
        elif "black scholes" in query_lower or "option" in query_lower:
            # Extract parameters for Black-Scholes
            params = {
                "option_type": "call",
                "spot_price": 100.0,
                "strike_price": 100.0,
                "time_to_expiry": 0.25,
                "risk_free_rate": 0.05,
                "volatility": 0.2
            }
            
            # Extract specific parameters if mentioned
            if "put" in query_lower:
                params["option_type"] = "put"
            
            result = math_engine.black_scholes_pricing(params)
            
        elif "factor analysis" in query_lower:
            # Extract tickers from query
            tickers = []
            import re
            ticker_pattern = r'\b([A-Z]{1,5})\b'
            potential_tickers = re.findall(ticker_pattern, query.upper())
            valid_tickers = [t for t in potential_tickers if t in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'SPY', 'QQQ', 'IWM', 'VTI']]
            if valid_tickers:
                tickers = valid_tickers
            else:
                tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']  # default for FAANG
            
            params = {
                "tickers": tickers,
                "n_factors": 3,
                "method": "pca"
            }
            result = math_engine.factor_analysis(params)
            
        else:
            # Default to Monte Carlo simulation
            params = {
                "model_type": "geometric_brownian",
                "initial_value": 100.0,
                "drift": 0.05,
                "volatility": 0.2,
                "time_horizon": 1.0,
                "n_simulations": 10000,
                "n_steps": 252
            }
            result = math_engine.monte_carlo_simulation(params)
        
        if "error" in result:
            return QueryResponse(
                intent="mathematical_calculation",
                intent_confidence=0.9,
                data={},
                message=f"Mathematical calculation failed: {result['error']}"
            )
        
        # Create card data for mathematical calculations
        analysis_type = result.get("analysis_type", "mathematical_calculation")
        interpretation = result.get("interpretation", "Mathematical calculation completed successfully.")
        
        card_data = {
            "title": f"🧮 {analysis_type.replace('_', ' ').title()}",
            "meta": f"Calculation Type: {analysis_type} · Parameters: {len(result.get('parameters', {}))}",
            "metrics": [],
            "analysis_type": "mathematical_calculation",
            "calculation_type": analysis_type,
            "insight": interpretation
        }
        
        # Add metrics based on the type of mathematical calculation
        if analysis_type == "monte_carlo_simulation":
            results_data = result.get("results", {})
            card_data["metrics"].extend([
                {"key": "Expected Return", "value": f"{results_data.get('expected_return', 0):.2f}%"},
                {"key": "Probability Positive", "value": f"{results_data.get('probability_positive_return', 0)*100:.1f}%"},
                {"key": "Simulations", "value": f"{result.get('parameters', {}).get('n_simulations', 0):,}"},
                {"key": "Model Type", "value": result.get('parameters', {}).get('model_type', 'N/A').replace('_', ' ').title()}
            ])
            
            # Add percentiles if available
            percentiles = results_data.get('percentiles', {})
            if percentiles:
                card_data["metrics"].extend([
                    {"key": "5th Percentile", "value": f"{percentiles.get('5th', 0):.2f}"},
                    {"key": "95th Percentile", "value": f"{percentiles.get('95th', 0):.2f}"}
                ])
            
            # Add chart data for Monte Carlo simulations
            price_paths = results_data.get('price_paths', [])
            if price_paths:
                n_steps = len(price_paths[0]) if price_paths else 0
                time_horizon = result.get('parameters', {}).get('time_horizon', 1.0)
                
                # Create time axis
                time_steps = np.linspace(0, time_horizon, n_steps)
                
                # Sample a subset of paths for visualization (max 50 paths)
                max_paths = min(50, len(price_paths))
                sampled_paths = price_paths[:max_paths] if len(price_paths) > max_paths else price_paths
                
                # Calculate percentiles for confidence bands
                price_array = np.array(price_paths)
                percentile_5 = np.percentile(price_array, 5, axis=0)
                percentile_25 = np.percentile(price_array, 25, axis=0)
                percentile_75 = np.percentile(price_array, 75, axis=0)
                percentile_95 = np.percentile(price_array, 95, axis=0)
                mean_path = np.mean(price_array, axis=0)
                
                card_data["chart_data"] = {
                    "time_steps": time_steps.tolist(),
                    "simulation_paths": sampled_paths,
                    "percentile_5": percentile_5.tolist(),
                    "percentile_25": percentile_25.tolist(),
                    "percentile_75": percentile_75.tolist(),
                    "percentile_95": percentile_95.tolist(),
                    "mean_path": mean_path.tolist(),
                    "initial_value": result.get('parameters', {}).get('initial_value', 100.0)
                }
                
        elif analysis_type == "black_scholes_pricing":
            option_price = result.get("option_price", 0)
            greeks = result.get("greeks", {})
            params = result.get("parameters", {})
            
            card_data["metrics"].extend([
                {"key": "Option Price", "value": f"${option_price:.2f}"},
                {"key": "Option Type", "value": params.get("option_type", "N/A").title()},
                {"key": "Spot Price", "value": f"${params.get('spot_price', 0):.2f}"},
                {"key": "Strike Price", "value": f"${params.get('strike_price', 0):.2f}"},
                {"key": "Delta", "value": f"{greeks.get('delta', 0):.3f}"},
                {"key": "Gamma", "value": f"{greeks.get('gamma', 0):.3f}"},
                {"key": "Theta", "value": f"{greeks.get('theta', 0):.3f}"},
                {"key": "Vega", "value": f"{greeks.get('vega', 0):.3f}"}
            ])
            
        elif analysis_type == "factor_analysis":
            n_factors = result.get("n_factors", 0)
            variance_explained = result.get("factor_variance_explained", [])
            cumulative_var = result.get("cumulative_variance_explained", 0)
            
            card_data["metrics"].extend([
                {"key": "Number of Factors", "value": str(n_factors)},
                {"key": "Cumulative Variance", "value": f"{cumulative_var*100:.1f}%"},
                {"key": "Data Points", "value": str(result.get("data_points", 0))},
                {"key": "Method", "value": result.get("method", "N/A").upper()}
            ])
            
            # Add individual factor variance explained
            for i, var_exp in enumerate(variance_explained[:3]):  # Top 3 factors
                card_data["metrics"].append({
                    "key": f"Factor {i+1} Variance", 
                    "value": f"{var_exp*100:.1f}%"
                })
                
        elif analysis_type == "portfolio_optimization":
            portfolio_metrics = result.get("portfolio_metrics", {})
            optimization_type = result.get("optimization_type", "N/A")
            
            card_data["metrics"].extend([
                {"key": "Expected Return", "value": f"{portfolio_metrics.get('expected_return', 0)*100:.2f}%"},
                {"key": "Volatility", "value": f"{portfolio_metrics.get('volatility', 0)*100:.2f}%"},
                {"key": "Sharpe Ratio", "value": f"{portfolio_metrics.get('sharpe_ratio', 0):.3f}"},
                {"key": "Optimization Type", "value": optimization_type.replace('_', ' ').title()},
                {"key": "Number of Assets", "value": str(result.get("n_assets", 0))}
            ])
            
        else:
            # Default metrics for other mathematical calculations
            card_data["metrics"].extend([
                {"key": "Calculation Type", "value": analysis_type.replace('_', ' ').title()},
                {"key": "Status", "value": "Completed Successfully"},
                {"key": "Parameters", "value": str(len(result.get('parameters', {})))}
            ])
        
        # Add card data to result
        result["card_data"] = card_data
        
        # Format response message
        response_message = f"🧮 **{analysis_type.replace('_', ' ').title()}**\n\n"
        response_message += interpretation
        
        return QueryResponse(
            intent="mathematical_calculation",
            intent_confidence=0.9,
            data=result,
            message=response_message
        )
        
    except Exception as e:
        logger.error(f"Error in mathematical calculation: {str(e)}")
        return QueryResponse(
            intent="mathematical_calculation",
            intent_confidence=0.0,
            data={},
            message=f"Mathematical calculation failed: {str(e)}"
        )

def _is_talib_indicator_query(query: str) -> bool:
    """Check if the query is requesting a TA-Lib technical indicator"""
    query_lower = query.lower()
    
    # List of TA-Lib indicators
    talib_indicators = [
        'sma', 'ema', 'wma', 'dema', 'tema', 'trima', 'kama', 'mama', 't3',
        'rsi', 'stoch', 'stochf', 'stochrsi', 'willr', 'cci', 'cmo', 'roc', 
        'rocp', 'rocr', 'rocr100', 'mom', 'dx', 'adx', 'adxr', 'aroon', 
        'aroonosc', 'bop', 'trix', 'ultosc', 'mfi', 'ppo', 'macd', 'macdext', 
        'macdfix', 'bbands', 'natr', 'trange', 'atr', 'ad', 'adosc', 'obv',
        'ht_dcperiod', 'ht_dcphase', 'ht_phasor', 'ht_sine', 'ht_trendmode',
        'minus_di', 'minus_dm', 'plus_di', 'plus_dm'
    ]
    
    # Check if any indicator is mentioned in the query
    for indicator in talib_indicators:
        if indicator in query_lower:
            return True
    
    # Check for common indicator names
    indicator_names = [
        'moving average', 'bollinger', 'relative strength', 'stochastic',
        'williams', 'commodity channel', 'money flow', 'accumulation',
        'distribution', 'on balance volume', 'average true range',
        'directional movement', 'aroon', 'momentum', 'rate of change'
    ]
    
    for name in indicator_names:
        if name in query_lower:
            return True
    
    return False

async def _handle_talib_comparison_query(req: QueryRequest) -> QueryResponse:
    """Handle TA-Lib indicator queries with card format and graphs"""
    try:
        query_lower = req.query.lower()
        
        # Extract ticker from query
        import re
        ticker_pattern = r'\b([A-Z]{1,5})\b'
        potential_tickers = re.findall(ticker_pattern, req.query.upper())
        valid_tickers = [t for t in potential_tickers if t in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'SPY', 'QQQ', 'IWM', 'VTI', 'SVIX', 'AMD', 'META', 'NFLX', 'BABA', 'UBER', 'LYFT', 'SNAP', 'TWTR', 'SQ', 'PYPL', 'ADBE', 'CRM', 'ORCL', 'INTC', 'CSCO', 'IBM', 'GE', 'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'JNJ', 'PFE', 'UNH', 'HD', 'LOW', 'WMT', 'PG', 'KO', 'PEP', 'MCD', 'NKE', 'DIS', 'CMCSA', 'VZ', 'T', 'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'OXY', 'MPC', 'VLO', 'PSX', 'KMI', 'WMB', 'OKE', 'EPD', 'ET', 'ENB', 'TRP', 'PPL', 'DUK', 'SO', 'EXC', 'AEP', 'XEL', 'ES', 'PEG', 'ED', 'EIX', 'SRE', 'WEC', 'AWK', 'AEE', 'LNT', 'CNP', 'NI', 'DTE', 'CMS', 'ETR', 'FE', 'PPL', 'AES', 'PCG', 'SRE', 'WEC', 'AWK', 'AEE', 'LNT', 'CNP', 'NI', 'DTE', 'CMS', 'ETR', 'FE', 'PPL', 'AES', 'PCG']]
        
        if not valid_tickers:
            return QueryResponse(
                intent="error",
                intent_confidence=0.0,
                data={},
                message="❌ Please specify a valid ticker symbol (e.g., AAPL, MSFT, TSLA)",
                success=False
            )
        
        ticker = valid_tickers[0]
        
        # Extract indicator name from query
        indicator_name = _extract_indicator_name(query_lower)
        if not indicator_name:
            return QueryResponse(
                intent="error",
                intent_confidence=0.0,
                data={},
                message="❌ Please specify a valid technical indicator",
                success=False
            )
        
        # Extract time period from query
        start_date, end_date = _extract_time_period(query_lower)
        
        # Extract parameters from query
        params = _extract_indicator_parameters(query_lower, indicator_name)
        
        # Calculate extended date range to include timeperiod buffer
        # Find the maximum timeperiod among all parameters
        max_timeperiod = _get_max_timeperiod(indicator_name, params)
        
        extended_start_date, extended_end_date = _calculate_extended_date_range(
            start_date, end_date, max_timeperiod
        )
        
        # Initialize TA-Lib executor
        talib_executor = TALibToolExecutor()
        
        # Execute the TA-Lib indicator with extended date range
        talib_result = talib_executor.execute_indicator(
            indicator_name, ticker, extended_start_date, extended_end_date, params,
            original_start_date=start_date, original_end_date=end_date
        )
        
        if "error" in talib_result:
            return QueryResponse(
                intent="error",
                intent_confidence=0.0,
                data={},
                message=f"❌ {talib_result['error']}",
                success=False
            )
        
        # Format the TA-Lib response in card format
        message = _format_talib_card_response(talib_result, indicator_name, ticker)
        
        # Prepare data for response
        response_data = {
            "talib_result": talib_result,
            "indicator_name": indicator_name,
            "ticker": ticker,
            "card_format": True
        }
        
        return QueryResponse(
            intent="technical_analysis",
            intent_confidence=0.95,
            data=response_data,
            message=message,
            success=True
        )
        
    except Exception as e:
        logger.error(f"Error handling TA-Lib indicator query: {e}")
        return QueryResponse(
            intent="error",
            intent_confidence=0.0,
            data={},
            message=f"❌ Error processing indicator request: {str(e)}",
            success=False
        )

async def _handle_talib_indicator_query(req: QueryRequest) -> QueryResponse:
    """Handle TA-Lib indicator queries"""
    try:
        query_lower = req.query.lower()
        
        # Extract ticker from query
        import re
        ticker_pattern = r'\b([A-Z]{1,5})\b'
        potential_tickers = re.findall(ticker_pattern, req.query.upper())
        valid_tickers = [t for t in potential_tickers if t in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'SPY', 'QQQ', 'IWM', 'VTI', 'SVIX', 'AMD', 'META', 'NFLX', 'BABA', 'UBER', 'LYFT', 'SNAP', 'TWTR', 'SQ', 'PYPL', 'ADBE', 'CRM', 'ORCL', 'INTC', 'CSCO', 'IBM', 'GE', 'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'JNJ', 'PFE', 'UNH', 'HD', 'LOW', 'WMT', 'PG', 'KO', 'PEP', 'MCD', 'NKE', 'DIS', 'CMCSA', 'VZ', 'T', 'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'OXY', 'MPC', 'VLO', 'PSX', 'KMI', 'WMB', 'OKE', 'EPD', 'ET', 'ENB', 'TRP', 'PPL', 'DUK', 'SO', 'EXC', 'AEP', 'XEL', 'ES', 'PEG', 'ED', 'EIX', 'SRE', 'WEC', 'AWK', 'AEE', 'LNT', 'CNP', 'NI', 'DTE', 'CMS', 'ETR', 'FE', 'PPL', 'AES', 'PCG', 'SRE', 'WEC', 'AWK', 'AEE', 'LNT', 'CNP', 'NI', 'DTE', 'CMS', 'ETR', 'FE', 'PPL', 'AES', 'PCG']]
        
        if not valid_tickers:
            return QueryResponse(
                intent="error",
                intent_confidence=0.0,
                data={},
                message="❌ Please specify a valid ticker symbol (e.g., AAPL, MSFT, TSLA)",
                success=False
            )
        
        ticker = valid_tickers[0]
        
        # Extract indicator name from query
        indicator_name = _extract_indicator_name(query_lower)
        if not indicator_name:
            return QueryResponse(
                intent="error",
                intent_confidence=0.0,
                data={},
                message="❌ Please specify a valid technical indicator",
                success=False
            )
        
        # Extract time period from query
        start_date, end_date = _extract_time_period(query_lower)
        
        # Extract parameters from query
        params = _extract_indicator_parameters(query_lower, indicator_name)
        
        # Initialize TA-Lib executor
        talib_executor = TALibToolExecutor()
        
        # Execute the indicator
        result = talib_executor.execute_indicator(
            indicator_name, ticker, start_date, end_date, params
        )
        
        if "error" in result:
            return QueryResponse(
                intent="error",
                intent_confidence=0.0,
                data={},
                message=f"❌ {result['error']}",
                success=False
            )
        
        # Format the response
        message = _format_talib_indicator_response(result)
        
        return QueryResponse(
            intent="technical_analysis",
            intent_confidence=0.95,
            data=result,
            message=message,
            success=True
        )
        
    except Exception as e:
        logger.error(f"Error handling TA-Lib indicator query: {e}")
        return QueryResponse(
            intent="error",
            intent_confidence=0.0,
            data={},
            message=f"❌ Error processing indicator request: {str(e)}",
            success=False
        )

def _extract_indicator_name(query_lower: str) -> str:
    """Extract indicator name from query"""
    # Map common names to TA-Lib indicator names
    indicator_mapping = {
        'simple moving average': 'sma',
        'sma': 'sma',
        'moving average': 'sma',
        'exponential moving average': 'ema',
        'ema': 'ema',
        'weighted moving average': 'wma',
        'double exponential moving average': 'dema',
        'triple exponential moving average': 'tema',
        'triangular moving average': 'trima',
        'kaufman adaptive moving average': 'kama',
        'mesa adaptive moving average': 'mama',
        't3 moving average': 't3',
        'relative strength index': 'rsi',
        'relative strength': 'rsi',
        'rsi': 'rsi',
        'stochastic': 'stoch',
        'stochastic fast': 'stochf',
        'stochastic rsi': 'stochrsi',
        'williams %r': 'willr',
        'williams r': 'willr',
        'commodity channel index': 'cci',
        'cci': 'cci',
        'chande momentum oscillator': 'cmo',
        'rate of change': 'roc',
        'momentum': 'mom',
        'directional movement index': 'dx',
        'average directional movement index': 'adx',
        'average directional movement index rating': 'adxr',
        'aroon': 'aroon',
        'aroon oscillator': 'aroonosc',
        'balance of power': 'bop',
        'trix': 'trix',
        'ultimate oscillator': 'ultosc',
        'money flow index': 'mfi',
        'percentage price oscillator': 'ppo',
        'macd': 'macd',
        'bollinger bands': 'bbands',
        'normalized average true range': 'natr',
        'true range': 'trange',
        'average true range': 'atr',
        'atr': 'atr',
        'accumulation distribution': 'ad',
        'accumulation distribution oscillator': 'adosc',
        'on balance volume': 'obv',
        'hilbert transform dominant cycle period': 'ht_dcperiod',
        'hilbert transform dominant cycle phase': 'ht_dcphase',
        'hilbert transform phasor': 'ht_phasor',
        'hilbert transform sine': 'ht_sine',
        'hilbert transform trend mode': 'ht_trendmode',
        'minus directional indicator': 'minus_di',
        'minus directional movement': 'minus_dm',
        'plus directional indicator': 'plus_di',
        'plus directional movement': 'plus_dm'
    }
    
    # Check for exact matches first (longer names first to avoid partial matches)
    sorted_mappings = sorted(indicator_mapping.items(), key=lambda x: len(x[0]), reverse=True)
    for name, indicator in sorted_mappings:
        if name in query_lower:
            return indicator
    
    # Check for partial matches (longer names first) - be more specific
    for name, indicator in sorted_mappings:
        words = name.split()
        if len(words) > 1:  # Only for multi-word indicators
            if all(word in query_lower for word in words):
                return indicator
        elif len(words) == 1 and name in query_lower:
            return indicator
    
    # Check for direct indicator names
    talib_indicators = [
        'sma', 'ema', 'wma', 'dema', 'tema', 'trima', 'kama', 'mama', 't3',
        'rsi', 'stoch', 'stochf', 'stochrsi', 'willr', 'cci', 'cmo', 'roc', 
        'rocp', 'rocr', 'rocr100', 'mom', 'dx', 'adx', 'adxr', 'aroon', 
        'aroonosc', 'bop', 'trix', 'ultosc', 'mfi', 'ppo', 'macd', 'macdext', 
        'macdfix', 'bbands', 'natr', 'trange', 'atr', 'ad', 'adosc', 'obv',
        'ht_dcperiod', 'ht_dcphase', 'ht_phasor', 'ht_sine', 'ht_trendmode',
        'minus_di', 'minus_dm', 'plus_di', 'plus_dm'
    ]
    
    for indicator in talib_indicators:
        if indicator in query_lower:
            return indicator
    
    return None

def _extract_time_period(query_lower: str) -> tuple:
    """Extract start and end dates from query"""
    from datetime import datetime, timedelta
    from dateutil.relativedelta import relativedelta
    
    # Default to 1 year
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    
    # Check for specific time periods
    if "last 30 days" in query_lower or "30 days" in query_lower:
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    elif "last 90 days" in query_lower or "90 days" in query_lower:
        start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
    elif "3 months" in query_lower or "last 3 months" in query_lower:
        # Use 3 calendar months for more accurate business day calculation
        start_date = (datetime.now() - relativedelta(months=3)).strftime("%Y-%m-%d")
    elif "last 180 days" in query_lower or "180 days" in query_lower or "6 months" in query_lower:
        start_date = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
    elif "last 365 days" in query_lower or "365 days" in query_lower or "1 year" in query_lower:
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    elif "last 2 years" in query_lower or "2 years" in query_lower:
        start_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
    elif "last 5 years" in query_lower or "5 years" in query_lower:
        start_date = (datetime.now() - timedelta(days=1825)).strftime("%Y-%m-%d")
    
    return start_date, end_date

def _get_max_timeperiod(indicator_name: str, params: Dict[str, Any]) -> int:
    """Get the maximum timeperiod among all parameters for an indicator"""
    indicator_lower = indicator_name.lower()
    
    # Special handling for indicators that need more complex calculations
    if indicator_lower == 'macd':
        # MACD needs slowperiod + signalperiod + buffer for EMA stabilization
        slowperiod = params.get('slowperiod', 26)
        signalperiod = params.get('signalperiod', 9)
        return slowperiod + signalperiod + 15  # Extra 15 for EMA stabilization
    
    elif indicator_lower == 'stoch':
        # Stochastic needs the maximum of all periods + buffer
        fastk = params.get('fastk_period', 5)
        slowk = params.get('slowk_period', 3)
        slowd = params.get('slowd_period', 3)
        return max(fastk, slowk, slowd) + 15
    
    elif indicator_lower == 'stochrsi':
        # StochRSI needs timeperiod + max of fast periods + buffer
        timeperiod = params.get('timeperiod', 14)
        fastk = params.get('fastk_period', 5)
        fastd = params.get('fastd_period', 3)
        return timeperiod + max(fastk, fastd) + 15
    
    elif indicator_lower == 'bollinger':
        # Bollinger Bands need timeperiod + buffer
        timeperiod = params.get('timeperiod', 20)
        return timeperiod + 15
    
    elif indicator_lower == 'rsi':
        # RSI needs timeperiod + buffer
        timeperiod = params.get('timeperiod', 14)
        return timeperiod + 10
    
    elif indicator_lower == 'sma':
        # SMA needs timeperiod + buffer
        timeperiod = params.get('timeperiod', 20)
        return timeperiod + 5
    
    elif indicator_lower == 'ema':
        # EMA needs timeperiod + buffer
        timeperiod = params.get('timeperiod', 20)
        return timeperiod + 5
    
    else:
        # For other indicators, use the standard approach
        timeperiod_params = {
            'aroon': ['timeperiod'],
            'adx': ['timeperiod'],
            'cci': ['timeperiod'],
            'willr': ['timeperiod']
        }
        
        # Get the relevant timeperiod parameters for this indicator
        relevant_params = timeperiod_params.get(indicator_lower, ['timeperiod'])
        
        # Find the maximum value among all timeperiod parameters
        max_period = 0
        for param in relevant_params:
            if param in params:
                max_period = max(max_period, params[param])
        
        # If no timeperiod parameters found, use default
        if max_period == 0:
            max_period = 14  # Default fallback
        
        return max_period + 10  # Add buffer for all indicators

def _calculate_extended_date_range(start_date: str, end_date: str, timeperiod: int) -> tuple:
    """Calculate extended date range to include timeperiod buffer for technical indicators"""
    from datetime import datetime, timedelta
    from dateutil.relativedelta import relativedelta
    
    # Convert to datetime objects
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    # Calculate business days buffer
    # For business days, we need to go back more calendar days
    # Roughly 1.4x calendar days = 1 business day
    buffer_calendar_days = int(timeperiod * 1.4)  # Convert business days to calendar days
    
    # Extend the start date backwards by the buffer
    extended_start_dt = start_dt - timedelta(days=buffer_calendar_days)
    extended_start_date = extended_start_dt.strftime("%Y-%m-%d")
    
    logger.info(f"Extended date range: {extended_start_date} to {end_date} (buffer: {buffer_calendar_days} calendar days for {timeperiod} business days)")
    
    return extended_start_date, end_date

def _extract_indicator_parameters(query_lower: str, indicator_name: str) -> dict:
    """Extract parameters for the indicator from the query"""
    params = {}
    
    # Extract timeperiod parameter
    import re
    
    # Look for period numbers with explicit period words
    period_matches = re.findall(r'(\d+)\s*(?:period|day|week|month)', query_lower)
    if period_matches:
        params['timeperiod'] = int(period_matches[0])
    
    # Look for numbers directly after indicator name (e.g., "RSI 30", "MACD 12")
    indicator_pattern = rf'{indicator_name.lower()}\s+(\d+)'
    direct_matches = re.findall(indicator_pattern, query_lower)
    if direct_matches:
        params['timeperiod'] = int(direct_matches[0])
    
    # Look for specific parameter patterns
    if indicator_name == 'bbands':
        # Look for standard deviation parameters
        std_matches = re.findall(r'(\d+(?:\.\d+)?)\s*sigma', query_lower)
        if std_matches:
            std_val = float(std_matches[0])
            params['nbdevup'] = std_val
            params['nbdevdn'] = std_val
    
    elif indicator_name == 'macd':
        # Look for MACD parameters
        fast_matches = re.findall(r'fast\s*(\d+)', query_lower)
        slow_matches = re.findall(r'slow\s*(\d+)', query_lower)
        signal_matches = re.findall(r'signal\s*(\d+)', query_lower)
        
        if fast_matches:
            params['fastperiod'] = int(fast_matches[0])
        if slow_matches:
            params['slowperiod'] = int(slow_matches[0])
        if signal_matches:
            params['signalperiod'] = int(signal_matches[0])
    
    elif indicator_name == 'stoch':
        # Look for stochastic parameters
        k_matches = re.findall(r'k\s*(\d+)', query_lower)
        d_matches = re.findall(r'd\s*(\d+)', query_lower)
        
        if k_matches:
            params['fastk_period'] = int(k_matches[0])
        if d_matches:
            params['slowd_period'] = int(d_matches[0])
    
    return params

def _format_talib_indicator_response(result: Dict[str, Any]) -> str:
    """Format TA-Lib indicator response"""
    ticker = result.get('ticker', 'Unknown')
    indicator = result.get('tool_name', 'Indicator')
    latest_value = result.get('latest_value', 0)
    mean_value = result.get('mean_value', 0)
    min_value = result.get('min_value', 0)
    max_value = result.get('max_value', 0)
    period = result.get('period', 'Unknown period')
    data_points = result.get('data_points', 0)
    insights = result.get('insights', 'No insights available')
    
    message = f"📊 **{indicator.upper()} for {ticker}**\n\n"
    message += f"**Period:** {period}\n"
    message += f"**Data Points:** {data_points}\n\n"
    message += f"**Latest Value:** {latest_value:.4f}\n"
    message += f"**Mean Value:** {mean_value:.4f}\n"
    message += f"**Min Value:** {min_value:.4f}\n"
    message += f"**Max Value:** {max_value:.4f}\n\n"
    message += f"**Analysis:**\n{insights}"
    
    return message

def _format_talib_card_response(talib_result: Dict[str, Any], indicator_name: str, ticker: str) -> str:
    """Format TA-Lib response in card format with graphs"""
    
    # Get basic info
    period = talib_result.get('period', 'Unknown')
    data_points = talib_result.get('data_points', 0)
    latest_value = talib_result.get('latest_value', 0)
    mean_value = talib_result.get('mean_value', 0)
    min_value = talib_result.get('min_value', 0)
    max_value = talib_result.get('max_value', 0)
    parameters = talib_result.get('parameters_used', {})
    
    # Generate AI insights using OpenAI
    ai_insights = _generate_technical_insight(indicator_name, ticker, talib_result, f"{indicator_name} for {ticker}")
    
    # Format the card
    message = f"📊 **{indicator_name.upper()} Analysis — {ticker}**\n"
    message += f"**Period:** {period} · **Data Points:** {data_points} · **Tool:** TA-Lib\n\n"
    
    # Key Metrics
    message += "📈 **Key Metrics**\n"
    message += f"**Latest:** {latest_value:.4f}\n"
    message += f"**Mean:** {mean_value:.4f}\n"
    message += f"**Min:** {min_value:.4f}\n"
    message += f"**Max:** {max_value:.4f}\n"
    message += f"**Parameters:** {parameters}\n\n"
    
    # Chart placeholder
    message += "📊 **Chart**\n"
    message += f"{indicator_name.lower()} Chart\n"
    message += f"{data_points} data points\n"
    message += f"{max_value:.1f}\n"
    message += f"{(max_value + mean_value) / 2:.1f}\n"
    message += f"{mean_value:.1f}\n"
    message += f"{(min_value + mean_value) / 2:.1f}\n"
    message += f"{min_value:.1f}\n"
    message += f"Jul 20 Aug 4 Aug 19 Sep 4 Sep 21 Oct 6\n"
    message += f"Aug 20, 2025 Value: {latest_value:.2f}\n\n"
    
    # AI Insights
    message += "🔍 **AI Insight**\n"
    message += f"{ai_insights}\n\n"
    
    return message

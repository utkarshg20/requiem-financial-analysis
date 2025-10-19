# workers/engine/planning.py
from typing import Dict, List, Any, Optional, Tuple
import logging
import uuid
from datetime import datetime
import inspect
import importlib
import os

logger = logging.getLogger("requiem.planning")

class PlanStep:
    """Represents a single step in an execution plan"""
    
    def __init__(self, action: str, tool_function: str, inputs: Dict[str, Any], 
                 expected_output: str, reasoning: str, step_id: str = None):
        self.step_id = step_id or str(uuid.uuid4())[:8]
        self.action = action
        self.tool_function = tool_function
        self.inputs = inputs
        self.expected_output = expected_output
        self.reasoning = reasoning
        self.status = "pending"  # pending, approved, executing, completed, failed
        self.result = None
        self.error = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "action": self.action,
            "tool_function": self.tool_function,
            "inputs": self.inputs,
            "expected_output": self.expected_output,
            "reasoning": self.reasoning,
            "status": self.status,
            "result": self.result,
            "error": self.error
        }

class ExecutionPlan:
    """Represents a complete execution plan with multiple steps"""
    
    def __init__(self, plan_id: str, intent: str, query: str, steps: List[PlanStep]):
        self.plan_id = plan_id
        self.intent = intent
        self.query = query
        self.steps = steps
        self.status = "pending"  # pending, approved, executing, completed, failed
        self.created_at = datetime.now()
        self.approved_at = None
        self.completed_at = None
    
    def to_markdown(self) -> str:
        """Convert plan to markdown format for display"""
        md = f"## Intent\n{self.intent}\n\n"
        md += f"**Resolved query:** {self.query}\n\n"
        md += "## Proposed Plan (review only)\n"
        
        for i, step in enumerate(self.steps, 1):
            md += f"{i}) Action: {step.action}\n"
            md += f"   Tool/Function: {step.tool_function}\n"
            md += f"   Inputs: {self._format_inputs(step.inputs)}\n"
            md += f"   Output: {step.expected_output}\n"
            md += f"   Why: {step.reasoning}\n\n"
        
        md += "**Proceed with this plan?**"
        return md
    
    def _format_inputs(self, inputs: Dict[str, Any]) -> str:
        """Format inputs for display"""
        if not inputs:
            return "None"
        
        formatted = []
        for key, value in inputs.items():
            if isinstance(value, str):
                formatted.append(f"{key}='{value}'")
            else:
                formatted.append(f"{key}={value}")
        return ", ".join(formatted)

class ToolCatalog:
    """Discovers and catalogs available tools in the system"""
    
    def __init__(self):
        self.tools = {}
        self._discover_tools()
    
    def _discover_tools(self):
        """Discover available tools from the codebase"""
        # Discover adapters
        self._discover_module_tools("workers.adapters", "adapter")
        
        # Discover engine functions
        self._discover_module_tools("workers.engine", "engine")
        
        # Discover API endpoints
        self._discover_api_endpoints()
    
    def _discover_module_tools(self, module_path: str, tool_type: str):
        """Discover tools from a specific module"""
        try:
            module = importlib.import_module(module_path)
            for name in dir(module):
                obj = getattr(module, name)
                if callable(obj) and not name.startswith('_'):
                    self.tools[f"{module_path}.{name}"] = {
                        "type": tool_type,
                        "function": obj,
                        "signature": str(inspect.signature(obj)),
                        "docstring": inspect.getdoc(obj) or "No documentation"
                    }
        except Exception as e:
            logger.warning(f"Could not discover tools from {module_path}: {e}")
    
    def _discover_api_endpoints(self):
        """Discover API endpoints"""
        # Add known API endpoints
        api_endpoints = [
            "api.main.get_prices_agg",
            "api.main.get_ticker_suggestions", 
            "api.main.execute_backtest",
            "api.main.get_tearsheet"
        ]
        
        for endpoint in api_endpoints:
            self.tools[endpoint] = {
                "type": "api",
                "function": endpoint,
                "signature": "HTTP endpoint",
                "docstring": "API endpoint"
            }
    
    def get_tool(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get tool information by name"""
        return self.tools.get(tool_name)
    
    def list_tools(self) -> Dict[str, Dict[str, Any]]:
        """List all available tools"""
        return self.tools.copy()

class PlanGenerator:
    """Generates execution plans based on user queries"""
    
    def __init__(self):
        self.tool_catalog = ToolCatalog()
    
    def _detect_multiple_intents(self, query: str) -> List[str]:
        """Detect multiple intents in a compound query"""
        intents = []
        query_lower = query.lower()
        
        # Skip compound detection for tool-based backtests
        if any(pattern in query_lower for pattern in ["buy when", "sell when", "long when", "short when"]):
            return []
        
        # Keywords for each intent type
        intent_keywords = {
            "valuation": ["overvalued", "undervalued", "fair value", "valuation", "expensive", "cheap", "worth"],
            "analysis": ["technical analysis", "analyze", "calculate rsi", "calculate macd", "calculate sma", "show indicators", "show trend"],
            "price_query": ["price", "what is", "current price", "trading at"],
            "backtest": ["backtest", "test strategy", "run strategy", "simulate"],
            "comparison": ["compare", "versus", "vs", "better than"]
        }
        
        # Check for each intent
        for intent, keywords in intent_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    if intent not in intents:
                        intents.append(intent)
                    break
        
        return intents if len(intents) > 1 else [intents[0]] if intents else []
    
    def generate_plan(self, intent: str, query: str, parsed_query: Dict[str, Any]) -> ExecutionPlan:
        """Generate an execution plan based on query intent"""
        plan_id = str(uuid.uuid4())[:8]
        steps = []
        
        # Check for compound queries (multiple intents)
        detected_intents = self._detect_multiple_intents(query)
        
        if len(detected_intents) > 1:
            # Compound query - combine multiple analyses
            logger.info(f"Detected compound query with intents: {detected_intents}")
            for detected_intent in detected_intents:
                if detected_intent == "backtest":
                    steps.extend(self._generate_backtest_plan(parsed_query))
                elif detected_intent == "price_query":
                    steps.extend(self._generate_price_query_plan(parsed_query))
                elif detected_intent == "valuation":
                    steps.extend(self._generate_valuation_plan(parsed_query))
                elif detected_intent == "analysis":
                    steps.extend(self._generate_analysis_plan(parsed_query))
                elif detected_intent == "comparison":
                    steps.extend(self._generate_comparison_plan(parsed_query))
            
            # Use "compound" as the intent for multi-intent queries
            intent = "compound"
        else:
            # Single intent query
            if intent == "backtest":
                steps = self._generate_backtest_plan(parsed_query)
            elif intent == "price_query":
                steps = self._generate_price_query_plan(parsed_query)
            elif intent == "valuation":
                steps = self._generate_valuation_plan(parsed_query)
            elif intent == "analysis":
                steps = self._generate_analysis_plan(parsed_query)
            elif intent == "comparison":
                steps = self._generate_comparison_plan(parsed_query)
            elif intent == "statistical_analysis":
                steps = self._generate_statistical_analysis_plan(parsed_query)
            elif intent == "risk_metrics":
                steps = self._generate_risk_metrics_plan(parsed_query)
            elif intent == "mathematical_calculation":
                steps = self._generate_mathematical_calculation_plan(parsed_query)
            else:
                steps = self._generate_generic_plan(parsed_query)
        
        return ExecutionPlan(plan_id, intent, query, steps)
    
    def _generate_backtest_plan(self, parsed: Dict[str, Any]) -> List[PlanStep]:
        """Generate plan for backtest queries"""
        from ..utils.date_utils import get_default_backtest_dates
        default_start, default_end = get_default_backtest_dates()
        
        steps = []
        
        # Step 1: Get historical data
        steps.append(PlanStep(
            action=f"Fetch historical price data for {parsed.get('ticker', 'SPY')}",
            tool_function="workers.adapters.prices_polygon.get_prices_agg",
            inputs={
                "symbol": parsed.get('ticker', 'SPY'),
                "start": parsed.get('start', default_start),
                "end": parsed.get('end', default_end),
                "timespan": "day"
            },
            expected_output="DataFrame with OHLCV data",
            reasoning="Baseline price history needed for strategy backtesting"
        ))
        
        # Step 2: Execute backtest
        steps.append(PlanStep(
            action="Run strategy backtest simulation",
            tool_function="api.main.execute_backtest",
            inputs={
                "strategy_spec": {
                    "ticker": parsed.get('ticker', 'SPY'),
                    "start_date": parsed.get('start', default_start),
                    "end_date": parsed.get('end', default_end),
                    "features": parsed.get('features', ['momentum_12m_skip_1m']),
                    "signals": parsed.get('signals', ['rank_top_frac']),
                    "rebalance": parsed.get('rebalance', 'monthly')
                }
            },
            expected_output="Backtest results with performance metrics",
            reasoning="Execute the strategy to get performance statistics"
        ))
        
        # Step 3: Generate tearsheet
        steps.append(PlanStep(
            action="Generate detailed performance report",
            tool_function="api.main.get_tearsheet",
            inputs={"run_id": "from_previous_step"},
            expected_output="Tearsheet with charts and metrics",
            reasoning="Create comprehensive performance visualization"
        ))
        
        return steps
    
    def _generate_price_query_plan(self, parsed: Dict[str, Any]) -> List[PlanStep]:
        """Generate plan for price queries"""
        steps = []
        
        if parsed.get('time'):
            # Intraday query
            steps.append(PlanStep(
                action=f"Fetch intraday price data for {parsed.get('ticker', 'SPY')}",
                tool_function="workers.adapters.prices_polygon.get_intraday_price",
                inputs={
                    "symbol": parsed.get('ticker', 'SPY'),
                    "date": parsed.get('start', '2024-01-01'),
                    "time": parsed.get('time', '12:00')
                },
                expected_output="Intraday OHLCV data",
                reasoning="Get minute-level price data for specific time"
            ))
        else:
            # Daily query
            steps.append(PlanStep(
                action=f"Fetch daily price data for {parsed.get('ticker', 'SPY')}",
                tool_function="workers.adapters.prices_polygon.get_prices_agg",
                inputs={
                    "symbol": parsed.get('ticker', 'SPY'),
                    "start": parsed.get('start', '2024-01-01'),
                    "end": parsed.get('end', '2024-01-01'),
                    "timespan": "day"
                },
                expected_output="Daily OHLCV data",
                reasoning="Get daily price information"
            ))
        
        return steps
    
    def _generate_valuation_plan(self, parsed: Dict[str, Any]) -> List[PlanStep]:
        """Generate plan for valuation queries"""
        steps = []
        
        # Step 1: Get fundamental data
        steps.append(PlanStep(
            action=f"Fetch fundamental data for {parsed.get('ticker', 'SPY')}",
            tool_function="workers.adapters.alpha_vantage.alpha_vantage.get_overview",
            inputs={"symbol": parsed.get('ticker', 'SPY')},
            expected_output="Fundamental metrics (P/E, P/B, P/S, etc.)",
            reasoning="Get company fundamentals for valuation analysis"
        ))
        
        # Step 2: Get current price
        steps.append(PlanStep(
            action="Get current stock price",
            tool_function="workers.adapters.prices_polygon.get_prices_agg",
            inputs={
                "symbol": parsed.get('ticker', 'SPY'),
                "start": "today",
                "end": "today",
                "timespan": "day"
            },
            expected_output="Current price data",
            reasoning="Get latest price for valuation calculations"
        ))
        
        # Step 3: Calculate valuation metrics
        steps.append(PlanStep(
            action="Calculate comprehensive valuation metrics",
            tool_function="workers.engine.valuation.valuation_analyzer.analyze_valuation",
            inputs={"symbol": parsed.get('ticker', 'SPY')},
            expected_output="Valuation analysis with ratings",
            reasoning="Combine fundamental data and price for valuation assessment"
        ))
        
        return steps
    
    def _generate_analysis_plan(self, parsed: Dict[str, Any]) -> List[PlanStep]:
        """Generate plan for analysis queries"""
        steps = []
        ticker = parsed.get('ticker', 'SPY')
        
        # Step 1: Get price data
        steps.append(PlanStep(
            action=f"Fetch price data for {ticker}",
            tool_function="workers.adapters.prices_polygon.get_prices_agg",
            inputs={
                "symbol": ticker,
                "start": parsed.get('start', '6_months_ago'),
                "end": "today",
                "timespan": "day"
            },
            expected_output="Price history DataFrame",
            reasoning="Get historical data for technical analysis"
        ))
        
        # Step 2: Calculate RSI (momentum indicator)
        steps.append(PlanStep(
            action=f"Calculate RSI for {ticker}",
            tool_function="workers.engine.tool_executor.execute_tool",
            inputs={
                "tool_name": "rsi",
                "ticker": ticker,
                "window": 14,
                "start_date": parsed.get('start', '6_months_ago'),
                "end_date": "today"
            },
            expected_output="RSI values (overbought/oversold indicator)",
            reasoning="Identify overbought (>70) or oversold (<30) conditions"
        ))
        
        # Step 3: Calculate MACD (trend indicator)
        steps.append(PlanStep(
            action=f"Calculate MACD for {ticker}",
            tool_function="workers.engine.tool_executor.execute_tool",
            inputs={
                "tool_name": "macd",
                "ticker": ticker,
                "start_date": parsed.get('start', '6_months_ago'),
                "end_date": "today"
            },
            expected_output="MACD line, signal line, and histogram",
            reasoning="Identify trend direction and momentum shifts"
        ))
        
        # Step 4: Calculate SMA (trend confirmation)
        steps.append(PlanStep(
            action=f"Calculate moving averages for {ticker}",
            tool_function="workers.engine.tool_executor.execute_tool",
            inputs={
                "tool_name": "sma",
                "ticker": ticker,
                "window": 50,
                "start_date": parsed.get('start', '6_months_ago'),
                "end_date": "today"
            },
            expected_output="50-day simple moving average",
            reasoning="Determine overall trend direction (price above/below SMA)"
        ))
        
        return steps
    
    def _generate_comparison_plan(self, parsed: Dict[str, Any]) -> List[PlanStep]:
        """Generate plan for comparison queries"""
        steps = []
        
        # This would be implemented when comparison feature is built
        steps.append(PlanStep(
            action="Compare multiple assets",
            tool_function="workers.engine.comparison.compare_assets",
            inputs={"tickers": parsed.get('tickers', [])},
            expected_output="Comparative analysis results",
            reasoning="Compare performance and metrics across assets"
        ))
        
        return steps
    
    def _generate_generic_plan(self, parsed: Dict[str, Any]) -> List[PlanStep]:
        """Generate generic plan for unknown intents"""
        steps = []
        
        steps.append(PlanStep(
            action="Process query with available tools",
            tool_function="api.main.handle_query",
            inputs={"query": parsed.get('query', '')},
            expected_output="Query response",
            reasoning="Use general query handler for unknown intents"
        ))
        
        return steps
    
    def _generate_statistical_analysis_plan(self, parsed: Dict[str, Any]) -> List[PlanStep]:
        """Generate plan for statistical analysis queries"""
        steps = []
        
        # Step 1: Fetch price data for all assets
        steps.append(PlanStep(
            action="Fetch historical price data",
            tool_function="workers.adapters.prices_polygon.get_prices_agg",
            inputs={
                "symbols": "extracted from query",
                "start_date": "calculated from period",
                "end_date": "2024-10-13",
                "timespan": "day"
            },
            expected_output="Price history DataFrame for all assets",
            reasoning="Get historical data needed for statistical analysis"
        ))
        
        # Step 2: Calculate returns
        steps.append(PlanStep(
            action="Calculate daily returns",
            tool_function="pandas.Series.pct_change",
            inputs={
                "price_series": "from previous step",
                "method": "percentage change"
            },
            expected_output="Returns series for correlation analysis",
            reasoning="Convert prices to returns for statistical analysis"
        ))
        
        # Step 3: Perform correlation analysis
        steps.append(PlanStep(
            action="Calculate correlation matrix",
            tool_function="workers.engine.statistical_analyzer.correlation_analysis",
            inputs={
                "tickers": "extracted from query",
                "period": "extracted from query",
                "method": "pearson"
            },
            expected_output="Correlation matrix with significance testing",
            reasoning="Analyze statistical relationships between assets"
        ))
        
        # Step 4: Generate interpretation
        steps.append(PlanStep(
            action="Generate intelligent interpretation",
            tool_function="LLM analysis",
            inputs={
                "correlation_results": "from previous step",
                "context": "financial market analysis"
            },
            expected_output="Professional interpretation of correlation results",
            reasoning="Provide actionable insights for investment decisions"
        ))
        
        return steps
    
    def _generate_risk_metrics_plan(self, parsed: Dict[str, Any]) -> List[PlanStep]:
        """Generate plan for risk metrics queries"""
        steps = []
        
        # Step 1: Fetch price data
        steps.append(PlanStep(
            action="Fetch historical price data",
            tool_function="workers.adapters.prices_polygon.get_prices_agg",
            inputs={
                "symbol": "extracted from query",
                "start_date": "calculated from period",
                "end_date": "2024-10-13",
                "timespan": "day"
            },
            expected_output="Price history DataFrame",
            reasoning="Get historical data for risk calculation"
        ))
        
        # Step 2: Calculate returns
        steps.append(PlanStep(
            action="Calculate daily returns",
            tool_function="pandas.Series.pct_change",
            inputs={
                "price_series": "from previous step"
            },
            expected_output="Returns series",
            reasoning="Convert prices to returns for risk metrics"
        ))
        
        # Step 3: Calculate risk metrics
        steps.append(PlanStep(
            action="Calculate risk metrics",
            tool_function="workers.engine.risk_metrics",
            inputs={
                "metric_type": "extracted from query (Sharpe, VaR, etc.)",
                "returns": "from previous step",
                "confidence_level": "95% default"
            },
            expected_output="Risk metrics with interpretation",
            reasoning="Calculate specific risk measures requested"
        ))
        
        return steps
    
    def _generate_mathematical_calculation_plan(self, parsed: Dict[str, Any]) -> List[PlanStep]:
        """Generate plan for mathematical calculation queries"""
        steps = []
        
        # Step 1: Determine calculation type
        steps.append(PlanStep(
            action="Identify mathematical calculation type",
            tool_function="query parsing",
            inputs={
                "query": "user input",
                "calculation_types": ["Monte Carlo", "Black-Scholes", "Portfolio Optimization"]
            },
            expected_output="Specific calculation type and parameters",
            reasoning="Determine which mathematical model to apply"
        ))
        
        # Step 2: Execute mathematical calculation
        steps.append(PlanStep(
            action="Execute mathematical calculation",
            tool_function="workers.engine.math_engine",
            inputs={
                "calculation_type": "from previous step",
                "parameters": "extracted from query"
            },
            expected_output="Mathematical results with interpretation",
            reasoning="Perform the requested mathematical analysis"
        ))
        
        return steps

class PlanExecutor:
    """Executes approved plans step by step"""
    
    def __init__(self):
        self.active_plans = {}
    
    def execute_plan(self, plan: ExecutionPlan) -> Dict[str, Any]:
        """Execute an approved plan"""
        plan.status = "executing"
        self.active_plans[plan.plan_id] = plan
        
        results = {
            "plan_id": plan.plan_id,
            "status": "executing",
            "steps_completed": 0,
            "total_steps": len(plan.steps),
            "step_results": [],
            "final_result": None,
            "errors": []
        }
        
        try:
            for i, step in enumerate(plan.steps):
                step.status = "executing"
                logger.info(f"Executing step {i+1}: {step.action}")
                
                try:
                    # Execute the step
                    step_result = self._execute_step(step)
                    step.result = step_result
                    step.status = "completed"
                    
                    results["step_results"].append({
                        "step_id": step.step_id,
                        "action": step.action,
                        "status": "completed",
                        "result": step_result
                    })
                    
                    results["steps_completed"] += 1
                    
                except Exception as e:
                    step.error = str(e)
                    step.status = "failed"
                    results["errors"].append({
                        "step_id": step.step_id,
                        "action": step.action,
                        "error": str(e)
                    })
                    logger.error(f"Step {i+1} failed: {e}")
            
            plan.status = "completed" if not results["errors"] else "failed"
            plan.completed_at = datetime.now()
            
            results["status"] = plan.status
            results["final_result"] = self._summarize_results(plan)
            
        except Exception as e:
            plan.status = "failed"
            results["status"] = "failed"
            results["errors"].append({"general_error": str(e)})
            logger.error(f"Plan execution failed: {e}")
        
        return results
    
    def _execute_step(self, step: PlanStep) -> Any:
        """Execute a single plan step"""
        # This would contain the actual execution logic
        # For now, return a placeholder
        return {
            "step_executed": True,
            "action": step.action,
            "tool_function": step.tool_function,
            "inputs": step.inputs
        }
    
    def _summarize_results(self, plan: ExecutionPlan) -> Dict[str, Any]:
        """Summarize the results of plan execution"""
        completed_steps = [s for s in plan.steps if s.status == "completed"]
        failed_steps = [s for s in plan.steps if s.status == "failed"]
        
        return {
            "total_steps": len(plan.steps),
            "completed": len(completed_steps),
            "failed": len(failed_steps),
            "success_rate": len(completed_steps) / len(plan.steps) if plan.steps else 0,
            "execution_time": (plan.completed_at - plan.created_at).total_seconds() if plan.completed_at else None
        }

# Global instances
plan_generator = PlanGenerator()
plan_executor = PlanExecutor()

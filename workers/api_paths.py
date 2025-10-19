from datetime import datetime, timedelta
from typing import Literal, Optional, List, Dict, Any
from pydantic import BaseModel, ValidationError as PydanticValidationError
import pandas as pd
import hashlib
import uuid
import logging
from workers.adapters.prices_polygon import get_prices_agg
from workers.exceptions import ValidationError, DataError, InternalError, ExternalAPIError
from workers.adapters.options_polygon import build_chain_with_prev_close
from workers.adapters.calendar import nearest_trading_day_utc
from workers.engine.timeseries import momentum_12m_skip_1m, simple_vector_backtest, perf_metrics, yearly_returns, rolling_sharpe, yearly_turnover, underwater_heatmap_data
from workers.engine.options import select_strikes_by_delta, strangle_credit
from workers.engine.tearsheet import save_equity_and_drawdown, save_tearsheet_json, capture_env_snapshot
from workers.engine.features import FEATURES
from workers.engine.signals import signal_rank_top_frac, signal_threshold, signal_crossover, signal_band, signal_tool_based
from workers.engine.rebalance import monthly_rebalance_signal_from_rank, gate_signal_to_schedule


# Nested models for structured config
class TradingRules(BaseModel):
    rebalance: Literal["daily", "weekly", "monthly"] = "daily"
    transaction_costs_bps: float = 5.0


class SignalRule(BaseModel):
    type: Literal["rank_top_frac", "threshold", "crossover", "band", "tool_based"] = "rank_top_frac"
    # For rank_top_frac
    top_frac: Optional[float] = 0.1
    # For threshold and band
    lower: Optional[float] = None
    upper: Optional[float] = None
    # For crossover
    fast_ma: Optional[int] = 20
    slow_ma: Optional[int] = 50
    # For tool_based
    tool_rules: Optional[List[Dict[str, Any]]] = None


class CustomConfig(BaseModel):
    feature: str = "momentum_12m_skip_1m"
    signal_rule: Optional[SignalRule] = None


class StrategySpec(BaseModel):
    # Required fields
    spec_id: str
    domain: str
    
    # Equities factor fields
    ticker: Optional[str] = None
    start: Optional[str] = None
    end: Optional[str] = None
    
    # Optional nested configs
    trading_rules: Optional[TradingRules] = None
    custom: Optional[CustomConfig] = None
    
    # Options feasibility fields
    underlying: Optional[str] = None
    expiry: Optional[str] = None
    target_call_delta: Optional[float] = 0.16
    target_put_delta: Optional[float] = -0.16


def validate_strategy_spec(spec: StrategySpec, trace_id: str) -> None:
    """Validate strategy specification with detailed error messages"""
    
    if spec.domain == "equities_factor":
        if not spec.ticker or not spec.ticker.strip():
            raise ValidationError(
                "Ticker is required for equities_factor domain",
                details={"field": "ticker", "domain": spec.domain},
                trace_id=trace_id
            )
        
        if not spec.start or not spec.end:
            raise ValidationError(
                "Start and end dates are required for equities_factor domain",
                details={"field": "start/end", "domain": spec.domain},
                trace_id=trace_id
            )
        
        try:
            start_date = pd.to_datetime(spec.start)
            end_date = pd.to_datetime(spec.end)
        except Exception as e:
            raise ValidationError(
                "Invalid date format",
                details={"field": "start/end", "error": str(e)},
                trace_id=trace_id
            )
        
        if start_date >= end_date:
            raise ValidationError(
                "Start date must be before end date",
                details={"start": spec.start, "end": spec.end},
                trace_id=trace_id
            )
        
        # Validate date range is reasonable
        days_diff = (end_date - start_date).days
        if days_diff < 1:
            raise ValidationError(
                "Date range too short (minimum 1 day)",
                details={"days": days_diff},
                trace_id=trace_id
            )
        if days_diff > 365 * 10:  # 10 years max
            raise ValidationError(
                "Date range too long (maximum 10 years)",
                details={"days": days_diff},
                trace_id=trace_id
            )
    
    elif spec.domain == "options_feasibility":
        if not spec.underlying or not spec.underlying.strip():
            raise ValidationError(
                "Underlying is required for options_feasibility domain",
                details={"field": "underlying", "domain": spec.domain},
                trace_id=trace_id
            )
        
        if not spec.expiry:
            raise ValidationError(
                "Expiry is required for options_feasibility domain",
                details={"field": "expiry", "domain": spec.domain},
                trace_id=trace_id
            )
    
    else:
        raise ValidationError(
            f"Unknown domain: {spec.domain}",
            details={"domain": spec.domain, "supported": ["equities_factor", "options_feasibility"]},
            trace_id=trace_id
        )

def run_spec_to_tearsheet(spec: StrategySpec, trace_id: str = None) -> dict:
    """Main entry point with comprehensive error handling and logging"""
    
    # Generate trace ID if not provided
    if trace_id is None:
        trace_id = str(uuid.uuid4())
    
    # Set up logging
    logger = logging.getLogger(f"requiem.{spec.spec_id}")
    logger.setLevel(logging.INFO)
    
    # Create run-specific log file
    import os
    os.makedirs(f"runs/{spec.spec_id}", exist_ok=True)
    log_file = f"runs/{spec.spec_id}/run.log"
    
    # Clear any existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Add file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    try:
        logger.info(f"Starting backtest run {spec.spec_id} with trace_id {trace_id}")
        logger.info(f"Input spec: {spec.model_dump()}")
        
        # Validate inputs
        validate_strategy_spec(spec, trace_id)
        logger.info("Input validation passed")
        
        if spec.domain == "equities_factor":
            # Extract config with defaults
            trading_rules = spec.trading_rules or TradingRules()
            custom = spec.custom or CustomConfig()
            signal_rule = custom.signal_rule or SignalRule()
            logger.info(f"Using config: feature={custom.feature}, signal={signal_rule.type}, rebalance={trading_rules.rebalance}")
        
        # 1. Get price data with warmup period
        # Features like momentum need historical data before the backtest period
        # Fetch extra ~1.5 years of warmup data to ensure features are ready
        warmup_days = 400  # ~1.5 years of trading days for safety
        
        # Use today's date for warmup calculation
        from datetime import datetime
        reasonable_end_str = datetime.now().strftime("%Y-%m-%d")
        
        from workers.adapters.calendar import trading_days
        all_days = trading_days("2020-01-01", reasonable_end_str)  # Get all trading days
        if len(all_days) < warmup_days:
            # If not enough history, use what we have
            data_start = all_days[0]
        else:
            # Find warmup start date
            target_idx = next((i for i, d in enumerate(all_days) if d >= spec.start), 0)
            warmup_start_idx = max(0, target_idx - warmup_days)
            data_start = all_days[warmup_start_idx]
        
        # Fetch data from warmup start to reasonable end (not future dates)
        logger.info(f"Fetching data for {spec.ticker} from {data_start} to {reasonable_end_str}")
        try:
            df = get_prices_agg(spec.ticker, data_start, reasonable_end_str)
        except Exception as e:
            logger.error(f"Failed to fetch price data: {e}")
            raise ExternalAPIError(
                f"Failed to fetch price data for {spec.ticker}",
                details={"ticker": spec.ticker, "start": str(data_start), "end": reasonable_end_str, "error": str(e)},
                trace_id=trace_id
            )
        
        # Validate data
        if df is None or df.empty:
            logger.error(f"No data returned for {spec.ticker}")
            raise DataError(
                f"No price data available for {spec.ticker}",
                details={"ticker": spec.ticker, "start": str(data_start), "end": spec.end},
                trace_id=trace_id
            )
        
        logger.info(f"Fetched {len(df)} rows of price data")
        # Ensure proper DatetimeIndex for safe time-series operations
        if 'date' in df.columns:
            df = df.set_index('date')
        df.index = pd.to_datetime(df.index)
        
        # Store the actual backtest period for later filtering
        backtest_start = pd.to_datetime(spec.start)
        backtest_end = pd.to_datetime(spec.end)
        
        # Generate data snapshot ID for reproducibility
        close_hash = hashlib.sha256(str(df["close"].values).encode()).hexdigest()[:16]
        data_snapshot_id = f"polygon_{spec.end}_{len(df)}_{close_hash}"
        
        # 2. Compute feature (use FEATURES dict for flexibility)
        feature_name = custom.feature
        if feature_name in FEATURES:
            feat = FEATURES[feature_name](df["close"])
        else:
            # fallback to momentum
            feat = momentum_12m_skip_1m(df["close"])
        
        # 3. Generate signal based on signal_rule type
        if signal_rule.type == "rank_top_frac":
            sig = signal_rank_top_frac(feat, top_frac=signal_rule.top_frac or 0.1)
            
        elif signal_rule.type == "threshold":
            sig = signal_threshold(feat, lower=signal_rule.lower, upper=signal_rule.upper)
            
        elif signal_rule.type == "crossover":
            # Crossover requires computing two MAs from price
            fast_window = signal_rule.fast_ma or 20
            slow_window = signal_rule.slow_ma or 50
            fast_ma = df["close"].rolling(fast_window).mean()
            slow_ma = df["close"].rolling(slow_window).mean()
            sig = signal_crossover(fast_ma, slow_ma)
            logger.info(f"Crossover signal: fast={fast_window}, slow={slow_window}")
            
        elif signal_rule.type == "band":
            # Mean-reversion: long when feature < lower
            lower_band = signal_rule.lower
            if lower_band is None:
                # Default to median for mean-reversion
                lower_band = feat.median()
                logger.info(f"Band signal: using median={lower_band:.4f} as lower band")
            sig = signal_band(feat, lower=lower_band)
            
        elif signal_rule.type == "tool_based":
            # Tool-based signals: execute tools and generate signals based on rules
            if hasattr(spec, '_tool_signals') and spec._tool_signals:
                sig = _generate_tool_based_signals(df, spec._tool_signals, logger)
                logger.info(f"Tool-based signal: generated signals from {len(spec._tool_signals)} tool rules")
            else:
                # Fallback to default
                sig = signal_rank_top_frac(feat, top_frac=0.1)
                logger.warning("Tool-based signal requested but no tool signals found, defaulting to rank_top_frac")
            
        else:
            # default: rank_top_frac
            sig = signal_rank_top_frac(feat, top_frac=0.1)
            logger.warning(f"Unknown signal type '{signal_rule.type}', defaulting to rank_top_frac")
        
        # 4. Apply rebalancing constraint if requested
        if trading_rules.rebalance == "monthly":
            # Monthly: use rank-based helper with rank series, not raw feature
            from workers.engine.rebalance import monthly_rebalance_signal_from_rank
            # Convert feature to rank series for proper monthly rebalancing
            feat_rank = feat.rank(pct=True)
            sig = monthly_rebalance_signal_from_rank(feat_rank, top_frac=signal_rule.top_frac or 0.1)
            logger.info(f"Monthly rebalance: using rank-based helper with top_frac={signal_rule.top_frac or 0.1}")
            
        elif trading_rules.rebalance == "weekly":
            # Weekly: gate to actual last trading day of each week using calendar
            from workers.engine.rebalance import last_trading_day_of_week
            ltdws = last_trading_day_of_week(spec.start, spec.end)
            allowed_dates = set(pd.to_datetime(ltdws))
            sig = gate_signal_to_schedule(sig, allowed_dates)
            logger.info(f"Weekly rebalance: {len(ltdws)} last trading days of week")
            
        # Daily rebalance: raw signal (already generated above, no additional gating)
        
        # 5. Filter to backtest period only (exclude warmup data)
        # But also ensure we have valid signals (no NaN values)
        mask = (df.index >= backtest_start) & (df.index <= backtest_end)
        df_backtest = df[mask]
        sig_backtest = sig[mask]
        
        # Remove NaN signals (features that need warmup)
        valid_signal_mask = ~sig_backtest.isna()
        if not valid_signal_mask.any():
            logger.error("No valid signals generated - feature may need more warmup data")
            raise DataError(
                f"No valid signals generated for feature {feature_name}",
                details={"feature": feature_name, "signal_type": signal_rule.type},
                trace_id=trace_id
            )
        
        # Adjust backtest period to actual data availability
        first_valid_idx = sig_backtest[valid_signal_mask].index[0]
        last_valid_idx = sig_backtest[valid_signal_mask].index[-1]
        
        if first_valid_idx > backtest_start:
            logger.info(f"Feature {feature_name} requires warmup - first signal at {first_valid_idx.date()}")
            actual_start = first_valid_idx
        else:
            actual_start = backtest_start
            
        # Final mask for valid signals in requested period
        final_mask = (df.index >= actual_start) & (df.index <= backtest_end) & ~sig.isna()
        df_final = df[final_mask]
        sig_final = sig[final_mask]
        
        # 6. Backtest (only on the requested period)
        res = simple_vector_backtest(df_final["close"], sig_final, tc_bps=trading_rules.transaction_costs_bps)
        
        # Calculate performance metrics with enhanced data
        pnl = res["equity_curve"].pct_change(fill_method=None).fillna(0.0)
        metrics = perf_metrics(pnl, res["equity_curve"], sig_final)
        
        # Calculate rolling Sharpe (use 63-day window = ~3 months for better visualization)
        # 252 days would require a full year of data before showing any points
        rolling_window = min(63, len(pnl) // 2)  # Use 63 days or half the data, whichever is smaller
        rolling_sharpe_series = rolling_sharpe(pnl, window=rolling_window) if rolling_window >= 20 else None
        
        # Generate enhanced charts
        figs = save_equity_and_drawdown(spec.spec_id, res["equity_curve"], rolling_sharpe_series)
        
        # Create yearly returns table
        yt = yearly_returns(pnl)
        year_table = {
        "id": "perf_by_year",
        "title": "Performance by Year",
            "columns": ["Year", "Return"],
        "rows": [[int(idx), float(val)] for idx, val in yt["return"].items()]
        }
        
        # Create yearly turnover table
        turnover_df = yearly_turnover(sig_final)
        turnover_table = {
            "id": "turnover_by_year",
            "title": "Turnover by Year",
            "columns": ["Year", "Turnover"],
            "rows": [[int(idx), float(val)] for idx, val in turnover_df["turnover"].items()]
        }
        
        # Create underwater heatmap data
        underwater_data = underwater_heatmap_data(pnl)

        # Build strategy description for title
        strat_desc = feature_name.replace("_", " ").title()
        rebal_note = f" ({trading_rules.rebalance.title()} Rebal)" if trading_rules.rebalance != "daily" else ""

        ts = {
        "run_id": f"run_{spec.spec_id}",
        "generated_at": datetime.utcnow().isoformat()+"Z",
        "trace_id": trace_id,
        "summary": {
            "title": f"{strat_desc}{rebal_note} — {spec.ticker}", 
            "domain": "equities_factor",
            "universe": spec.ticker, 
            "period": {"start": spec.start, "end": spec.end},
            "actual_period": {"start": actual_start.date().isoformat(), "end": spec.end},
            "config": {
                "feature": feature_name,
                "signal_type": signal_rule.type,
                "signal_params": signal_rule.model_dump(exclude_none=True),
                "rebalance": trading_rules.rebalance,
                "tc_bps": trading_rules.transaction_costs_bps
            }
        },
        "metrics": {"performance": metrics},
        "figures": [
            {
                "id":"equity_curve",
                "title":"Equity Curve",
                "type":"equity_curve",
                "path": figs["equity_curve"],
                "data": {
                    "labels": [str(d.date()) for d in res["equity_curve"].index],
                    "values": [float(v) if not pd.isna(v) else 0.0 for v in res["equity_curve"].values]
                }
            },
            {
                "id":"drawdown",
                "title":"Drawdown",
                "type":"drawdown", 
                "path": figs["drawdown"],
                "data": {
                    "labels": [str(d.date()) for d in res["equity_curve"].index],
                    "values": [float(v) if not pd.isna(v) else 0.0 for v in ((res["equity_curve"] / res["equity_curve"].cummax() - 1.0) * 100).values]
                }
            },
            {
                "id":"rolling_sharpe",
                "title":"Rolling Sharpe (252d)",
                "type":"rolling_sharpe",
                "path": figs.get("rolling_sharpe", ""),
                "data": {
                    "labels": [str(d.date()) for d in rolling_sharpe_series.index[~rolling_sharpe_series.isna()]] if rolling_sharpe_series is not None else [],
                    "values": [float(v) for v in rolling_sharpe_series[~rolling_sharpe_series.isna()].values] if rolling_sharpe_series is not None else []
                }
            }
        ],
        "tables": [year_table, turnover_table],
        "narrative": {
            "assumptions": [
                "PIT not enforced in v0",
                f"{trading_rules.transaction_costs_bps} bps TC",
                f"Feature: {feature_name}",
                f"Signal: {signal_rule.type}",
                f"Rebalance: {trading_rules.rebalance}"
            ],
            "findings": [], 
            "limitations": [], 
            "recommendation": ""
        },
        "env": capture_env_snapshot(),
        "data_snapshot_id": data_snapshot_id
        }
        save_tearsheet_json(spec.spec_id, ts)
        
        # Create run manifest for reproducibility
        run_manifest = {
            "run_id": ts["run_id"],
            "trace_id": trace_id,
            "generated_at": ts["generated_at"],
            "inputs": {
                "spec_id": spec.spec_id,
                "domain": spec.domain,
                "ticker": spec.ticker,
                "start": spec.start,
                "end": spec.end,
                "trading_rules": trading_rules.model_dump() if trading_rules else None,
                "custom": custom.model_dump() if custom else None
            },
            "data_snapshot_id": data_snapshot_id,
            "artifacts": {
                "tearsheet": f"runs/{spec.spec_id}/tearsheet.json",
                "equity_curve": figs["equity_curve"],
                "drawdown": figs["drawdown"],
                "run_log": f"runs/{spec.spec_id}/run.log"
            },
            "env": capture_env_snapshot()
        }
        
        # Save run manifest
        import json
        import os
        os.makedirs(f"runs/{spec.spec_id}", exist_ok=True)
        with open(f"runs/{spec.spec_id}/run_manifest.json", "w") as f:
            json.dump(run_manifest, f, indent=2)
        
            logger.info(f"Backtest completed successfully. CAGR: {metrics.get('cagr', 'N/A')}")
        return ts

    except ValidationError as e:
        # Re-raise validation errors as-is
        logger.error(f"Validation error: {str(e)}")
        raise
    except DataError as e:
        # Re-raise data errors as-is  
        logger.error(f"Data error: {str(e)}")
        raise
    except ExternalAPIError as e:
        # Re-raise API errors as-is
        logger.error(f"External API error: {str(e)}")
        raise
    except Exception as e:
        # Catch any other errors and wrap as InternalError
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise InternalError(
            f"Internal error during backtest execution",
            details={"error": str(e), "spec_id": spec.spec_id},
            trace_id=trace_id
        )

    if spec.domain == "options_feasibility":
        expiry = nearest_trading_day_utc(spec.expiry)
        chain = build_chain_with_prev_close(spec.underlying, expiry)
        # spot proxy from recent close of underlying
        spot_df = get_prices_agg(spec.underlying, expiry, expiry)
        S = float(spot_df["close"].iloc[-1]) if not spot_df.empty else 200.0
        # candidate strikes from chain (fallback to ±50% grid if chain empty)
        strikes = sorted(chain["strike"].unique().tolist()) if not chain.empty else None
        T = 30/365  # crude proxy for v0
        r = 0.02
        from workers.engine.timeseries import realized_vol_30d
        sigma = 0.2
        try:
            # Use 1 year of historical data before expiry
            hist_start = (datetime.fromisoformat(expiry) - timedelta(days=365)).date().isoformat()
            hist = get_prices_agg(spec.underlying, hist_start, expiry)
            sigma = max(0.01, min(1.0, 1.2 * realized_vol_30d(hist["close"])))
        except Exception:
            pass
        Kp, Kc = select_strikes_by_delta(S, T, r, sigma, spec.target_call_delta, spec.target_put_delta, strikes)
        credit = strangle_credit(S, Kp, Kc, T, r, sigma)
        return {
            "run_id": f"run_{spec.spec_id}",
            "generated_at": datetime.utcnow().isoformat()+"Z",
            "summary": {"title": f"{spec.underlying} Short Strangle Feasibility", "domain": "options_feasibility",
                        "universe": f"{spec.underlying} options", "period": {"start": "hist", "end": expiry}},
            "metrics": {"performance": {"pnl": None, "sharpe": None, "max_drawdown_pct": None},
                        "risk": {"exposures": {"theta": "short", "vega": "short"}}},
            "tables": [
                {"id":"strikes","title":"Suggested Strikes","columns":["PutK","CallK","Credit","Sigma"],
                 "rows":[[float(Kp), float(Kc), float(credit), float(sigma)]]}
            ],
            "figures": [],
            "narrative": {"assumptions": [f"σ fallback={sigma:.3f} from 30d RV", "prev_close quotes used"],
                          "findings": ["Credit & strikes suggested by delta targeting"],
                          "limitations": ["No margin or liquidity model in v0"], "recommendation": ""}
        }

    raise ValueError(f"Unknown domain: {spec.domain}")


def _generate_tool_based_signals(df: pd.DataFrame, tool_signals: List[Dict[str, Any]], logger) -> pd.Series:
    """
    Generate trading signals based on tool execution and signal rules.
    
    Args:
        df: DataFrame with OHLCV data
        tool_signals: List of signal rules (e.g., [{"tool_name": "rsi", "action": "buy", "comparison": "less_than", "threshold": 30.0}])
        logger: Logger instance
    
    Returns:
        pd.Series: Trading signals (0 or 1)
    """
    import pandas as pd
    import numpy as np
    
    # Initialize signal series with zeros
    final_signal = pd.Series(0, index=df.index)
    
    # Group signals by tool to avoid duplicate tool execution
    tool_groups = {}
    for rule in tool_signals:
        tool_name = rule["tool_name"]
        if tool_name not in tool_groups:
            tool_groups[tool_name] = []
        tool_groups[tool_name].append(rule)
    
    # Execute each tool and generate signals
    for tool_name, rules in tool_groups.items():
        logger.info(f"Executing tool: {tool_name} with {len(rules)} rules")
        
        try:
            # Execute the tool to get indicator values
            tool_values = _execute_tool(df, tool_name, logger)
            
            if tool_values is None or tool_values.empty:
                logger.warning(f"Tool {tool_name} returned no data, skipping")
                continue
            
            # Generate signals for each rule
            tool_signal = pd.Series(0, index=df.index)
            
            for rule in rules:
                action = rule["action"]  # "buy" or "sell"
                comparison = rule["comparison"]  # "less_than" or "greater_than"
                threshold = rule["threshold"]
                
                # Generate signal based on rule
                if comparison == "less_than":
                    condition = tool_values < threshold
                elif comparison == "greater_than":
                    condition = tool_values > threshold
                elif comparison == "less_than_or_equal":
                    condition = tool_values <= threshold
                elif comparison == "greater_than_or_equal":
                    condition = tool_values >= threshold
                else:
                    logger.warning(f"Unknown comparison operator: {comparison}")
                    continue
                
                # Apply signal based on action
                if action in ["buy", "long"]:
                    tool_signal = tool_signal | condition.astype(int)
                elif action in ["sell", "short"]:
                    tool_signal = tool_signal | condition.astype(int)
                else:
                    logger.warning(f"Unknown action: {action}")
                    continue
                
                logger.info(f"Rule: {action} when {tool_name} {comparison} {threshold}")
            
            # Combine with final signal (OR logic for multiple tools)
            final_signal = final_signal | tool_signal
            
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {str(e)}")
            continue
    
    logger.info(f"Generated tool-based signals: {final_signal.sum()} buy signals out of {len(final_signal)} total")
    return final_signal


def _execute_tool(df: pd.DataFrame, tool_name: str, logger) -> pd.Series:
    """
    Execute a specific tool to get indicator values.
    
    Args:
        df: DataFrame with OHLCV data
        tool_name: Name of the tool to execute
        logger: Logger instance
    
    Returns:
        pd.Series: Tool indicator values
    """
    import pandas as pd
    import numpy as np
    
    try:
        # Map tool names to their implementations
        if tool_name.lower() == "rsi":
            return _calculate_rsi(df["close"])
        elif tool_name.lower() == "sma":
            return _calculate_sma(df["close"])
        elif tool_name.lower() == "macd":
            return _calculate_macd(df["close"])
        elif tool_name.lower() == "bollinger":
            return _calculate_bollinger_bands(df["close"])
        elif tool_name.lower() == "stochastic":
            return _calculate_stochastic(df["high"], df["low"], df["close"])
        elif tool_name.lower() == "williams_r":
            return _calculate_williams_r(df["high"], df["low"], df["close"])
        elif tool_name.lower() == "aroon":
            return _calculate_aroon(df["high"], df["low"])
        elif tool_name.lower() == "momentum":
            return _calculate_momentum(df["close"])
        elif tool_name.lower() == "zscore":
            return _calculate_zscore(df["close"])
        elif tool_name.lower() == "realized_vol":
            return _calculate_realized_vol(df["close"])
        else:
            logger.warning(f"Unknown tool: {tool_name}")
            return pd.Series(index=df.index)
            
    except Exception as e:
        logger.error(f"Error executing tool {tool_name}: {str(e)}")
        return pd.Series(index=df.index)


def _calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _calculate_sma(prices: pd.Series, window: int = 20) -> pd.Series:
    """Calculate Simple Moving Average"""
    return prices.rolling(window=window).mean()


def _calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """Calculate MACD indicator"""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    return macd


def _calculate_bollinger_bands(prices: pd.Series, window: int = 20, std_dev: float = 2) -> pd.Series:
    """Calculate Bollinger Bands (returns middle band)"""
    sma = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    # Return the middle band (SMA) for signal generation
    return sma


def _calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_window: int = 14, d_window: int = 3) -> pd.Series:
    """Calculate Stochastic Oscillator"""
    lowest_low = low.rolling(window=k_window).min()
    highest_high = high.rolling(window=k_window).max()
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    return k_percent


def _calculate_williams_r(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Calculate Williams %R"""
    highest_high = high.rolling(window=window).max()
    lowest_low = low.rolling(window=window).min()
    williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
    return williams_r


def _calculate_aroon(high: pd.Series, low: pd.Series, window: int = 14) -> pd.Series:
    """Calculate Aroon indicator"""
    aroon_up = high.rolling(window=window).apply(lambda x: (window - x.argmax()) / window * 100)
    aroon_down = low.rolling(window=window).apply(lambda x: (window - x.argmin()) / window * 100)
    aroon = aroon_up - aroon_down
    return aroon


def _calculate_momentum(prices: pd.Series, window: int = 12) -> pd.Series:
    """Calculate momentum indicator"""
    return prices.pct_change(window)


def _calculate_zscore(prices: pd.Series, window: int = 20) -> pd.Series:
    """Calculate Z-score"""
    rolling_mean = prices.rolling(window=window).mean()
    rolling_std = prices.rolling(window=window).std()
    zscore = (prices - rolling_mean) / rolling_std
    return zscore


def _calculate_realized_vol(prices: pd.Series, window: int = 20) -> pd.Series:
    """Calculate realized volatility"""
    returns = prices.pct_change()
    realized_vol = returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
    return realized_vol

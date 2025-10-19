#!/usr/bin/env python3
"""Streamlit app to showcase Requiem strategy backtesting features"""

import streamlit as st
import json
from datetime import datetime
import os
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, os.path.dirname(__file__))

# Load .env file
env_file = Path(__file__).parent / ".env"
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ[key.strip()] = value.strip()

from workers.api_paths import StrategySpec, run_spec_to_tearsheet
from workers.engine.prompt_parser import parse_prompt, prompt_to_spec_skeleton, validate_spec_skeleton

st.set_page_config(page_title="Requiem Strategy Backtester", page_icon="📊", layout="wide")

# Header
st.title("📊 Requiem Strategy Backtester")
st.markdown("Interactive demo of the new feature-agnostic pipeline")

# Prompt-Audit Feature
st.header("🎯 Quick Start: Natural Language Prompt")

prompt_examples = [
    "Test 12-month momentum on SPY since 2015 monthly, 5 bps TC",
    "Run zscore below -1 on QQQ from 2020-2023 weekly, 10 bps",
    "SMA crossover 20-day and 50-day on SPY, 2022 to 2023 weekly",
    "Momentum top 20% on SPY from 2023-01-01 to 2023-12-31",
]

prompt_input = st.text_input(
    "Describe your strategy in plain English:",
    placeholder="e.g., Test 12-month momentum on SPY since 2015 monthly, 5 bps TC",
    help="Type a natural language description and we'll parse it into a strategy"
)

col1, col2 = st.columns([1, 4])
with col1:
    parse_button = st.button("🔍 Parse Prompt", type="primary")
with col2:
    example_prompt = st.selectbox("Or try an example:", [""] + prompt_examples, label_visibility="collapsed")
    if example_prompt:
        prompt_input = example_prompt
        st.rerun()

# Initialize session state for parsed values
if 'parsed_values' not in st.session_state:
    st.session_state.parsed_values = None

# Parse prompt when button clicked
if parse_button and prompt_input:
    with st.spinner("Parsing your prompt..."):
        try:
            parsed = parse_prompt(prompt_input)
            spec_skeleton = prompt_to_spec_skeleton(parsed)
            is_valid, errors = validate_spec_skeleton(spec_skeleton)
            
            # Store parsed values
            st.session_state.parsed_values = {
                'parsed': parsed,
                'spec_skeleton': spec_skeleton,
                'is_valid': is_valid,
                'errors': errors,
            }
            
            st.success(f"✅ Parsed with {parsed.confidence:.0%} confidence!")
            
            # Show what was parsed
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Ticker", parsed.ticker or "❓")
                st.metric("Feature", parsed.feature or "❓")
            with col2:
                st.metric("Start Date", parsed.start or "❓")
                st.metric("Signal Type", parsed.signal_type or "❓")
            with col3:
                st.metric("Rebalance", parsed.rebalance or "❓")
                st.metric("TC (bps)", parsed.transaction_costs_bps or "❓")
            
            # Show questions if any
            if parsed.questions:
                st.warning(f"❓ **Need clarification ({len(parsed.questions)} questions):**")
                for i, q in enumerate(parsed.questions, 1):
                    st.write(f"{i}. {q}")
            else:
                st.success("✨ All fields extracted! Ready to run.")
            
            # Show validation results
            if errors:
                st.error("⚠️ **Validation Issues:**")
                for error in errors:
                    st.write(f"- {error}")
            
        except Exception as e:
            st.error(f"Failed to parse prompt: {str(e)}")

st.markdown("---")

# Sidebar - Strategy Configuration
st.sidebar.header("⚙️ Strategy Configuration")

with st.sidebar:
    st.subheader("Basic Info")
    spec_id = st.text_input("Strategy ID", value=f"demo_{datetime.now().strftime('%H%M%S')}")
    domain = st.selectbox("Domain", ["equities_factor", "options_feasibility"], index=0)
    
    if domain == "equities_factor":
        # Use parsed values if available
        parsed_data = st.session_state.parsed_values
        if parsed_data:
            parsed = parsed_data['parsed']
            default_ticker = parsed.ticker or "SPY"
            default_start = datetime.fromisoformat(parsed.start) if parsed.start else datetime(2023, 1, 1)
            default_end = datetime.fromisoformat(parsed.end) if parsed.end else datetime(2023, 12, 31)
            default_feature = parsed.feature or "momentum_12m_skip_1m"
            default_signal_type = parsed.signal_type or "rank_top_frac"
            default_rebalance = parsed.rebalance or "monthly"
            default_tc = parsed.transaction_costs_bps or 5.0
            default_signal_params = parsed.signal_params
        else:
            default_ticker = "SPY"
            default_start = datetime(2023, 1, 1)
            default_end = datetime(2023, 12, 31)
            default_feature = "momentum_12m_skip_1m"
            default_signal_type = "rank_top_frac"
            default_rebalance = "monthly"
            default_tc = 5.0
            default_signal_params = {}
        
        st.subheader("Market Data")
        ticker = st.text_input("Ticker", value=default_ticker)
        col1, col2 = st.columns(2)
        with col1:
            start = st.date_input("Start Date", value=default_start).isoformat()
        with col2:
            end = st.date_input("End Date", value=default_end).isoformat()
        st.info("💡 Warmup data is automatically fetched for feature calculation")
        
        st.subheader("Feature Selection")
        feature_options = ["momentum_12m_skip_1m", "zscore_20d", "sma_20", "sma_50", "rsi_14"]
        feature_index = feature_options.index(default_feature) if default_feature in feature_options else 0
        feature = st.selectbox(
            "Feature",
            feature_options,
            index=feature_index,
            help="Choose which feature to compute from price data"
        )
        
        st.subheader("Signal Rules")
        signal_options = ["rank_top_frac", "threshold", "crossover", "band"]
        signal_index = signal_options.index(default_signal_type) if default_signal_type in signal_options else 0
        signal_type = st.selectbox(
            "Signal Type",
            signal_options,
            index=signal_index,
            help="How to convert features into trading signals"
        )
        
        if signal_type == "rank_top_frac":
            default_top_frac = default_signal_params.get('top_frac', 0.1)
            top_frac = st.slider("Top Fraction", 0.05, 1.0, float(default_top_frac), 0.05,
                                help="Trade the top X% ranked securities")
            signal_params = {"type": signal_type, "top_frac": top_frac}
        elif signal_type == "threshold":
            default_lower = default_signal_params.get('lower', -1.0)
            default_upper = default_signal_params.get('upper', 1.0)
            col1, col2 = st.columns(2)
            with col1:
                lower = st.number_input("Lower Threshold", value=float(default_lower))
            with col2:
                upper = st.number_input("Upper Threshold", value=float(default_upper))
            signal_params = {"type": signal_type, "lower": lower, "upper": upper}
        elif signal_type == "crossover":
            default_fast = default_signal_params.get('fast_ma', 20)
            default_slow = default_signal_params.get('slow_ma', 50)
            col1, col2 = st.columns(2)
            with col1:
                fast_ma = st.number_input("Fast MA Window", value=int(default_fast), min_value=1, max_value=100)
            with col2:
                slow_ma = st.number_input("Slow MA Window", value=int(default_slow), min_value=1, max_value=200)
            signal_params = {"type": signal_type, "fast_ma": fast_ma, "slow_ma": slow_ma}
            st.info("Golden cross: long when fast MA > slow MA")
        elif signal_type == "band":
            default_band_lower = default_signal_params.get('lower', 0.0)
            lower = st.number_input("Lower Band", value=float(default_band_lower), 
                                   help="Long when feature < lower (mean-reversion)")
            signal_params = {"type": signal_type, "lower": lower}
            st.info("Mean-reversion: buy when feature drops below lower band")
        else:
            signal_params = {"type": signal_type}
        
        st.subheader("Trading Rules")
        rebalance_options = ["daily", "weekly", "monthly"]
        rebalance_index = rebalance_options.index(default_rebalance) if default_rebalance in rebalance_options else 2
        rebalance = st.selectbox(
            "Rebalance Frequency",
            rebalance_options,
            index=rebalance_index,
            help="How often positions can change"
        )
        
        # Show rebalance mechanics explanation
        if rebalance == "daily":
            st.info("📅 **Daily**: Raw signal with 1-bar lookahead shift (highest turnover)")
        elif rebalance == "weekly":
            st.info("📅 **Weekly**: Gates to actual last trading day of each week (medium turnover)")
        elif rebalance == "monthly":
            st.info("📅 **Monthly**: Uses rank-based helper with rank series (lowest turnover)")
            
        tc_bps = st.slider("Transaction Costs (bps)", 0, 50, int(default_tc), 5)

# Main content
if domain == "equities_factor":
    # Create spec
    spec = StrategySpec(
        spec_id=spec_id,
        domain=domain,
        ticker=ticker,
        start=start,
        end=end,
        trading_rules={"rebalance": rebalance, "transaction_costs_bps": tc_bps},
        custom={"feature": feature, "signal_rule": signal_params}
    )
    
    # Display JSON spec
    with st.expander("📄 View Strategy Spec JSON"):
        st.json(spec.model_dump(exclude_none=True))
    
    # Run button
    if st.button("🚀 Run Backtest", type="primary", use_container_width=True):
        with st.spinner("Running backtest..."):
            try:
                tearsheet = run_spec_to_tearsheet(spec)
                
                # Check if we got valid results
                if tearsheet['metrics']['performance']['sharpe'] == 0.0 and tearsheet['metrics']['performance']['cagr'] == 0.0:
                    st.warning("⚠️ Backtest returned zero performance. Possible issues:")
                    st.write("- Feature may need more historical warmup data")
                    st.write("- No valid signals were generated") 
                    st.write("- Check data availability for this ticker and date range")
                
                # Success message
                st.success(f"✅ Backtest complete! Run ID: {tearsheet['run_id']}")
                
                # Metrics in columns
                st.header("📈 Performance Metrics")
                metrics = tearsheet['metrics']['performance']
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("CAGR", f"{metrics['cagr']:.2%}")
                    st.metric("Sharpe", f"{metrics['sharpe']:.2f}")
                with col2:
                    st.metric("Sortino", f"{metrics['sortino']:.2f}")
                    st.metric("Hit Rate", f"{metrics['hit_rate']:.2%}")
                with col3:
                    st.metric("Annual Vol", f"{metrics['vol_annual']:.2%}")
                    st.metric("Max Drawdown", f"{metrics['max_drawdown_pct']:.2%}")
                with col4:
                    st.metric("Avg Turnover", f"{metrics.get('avg_turnover', 0):.1%}")
                    st.metric("Exposure", f"{metrics.get('exposure_share', 0):.1%}")
                
                # Max drawdown date
                st.info(f"**Max Drawdown Date:** {metrics.get('max_drawdown_date', 'N/A')}")
                
                # Charts
                st.header("📊 Performance Charts")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Equity Curve")
                    equity_path = tearsheet['figures'][0]['path']
                    if os.path.exists(equity_path):
                        st.image(equity_path, use_container_width=True)
                    else:
                        st.warning(f"Chart not found: {equity_path}")
                
                with col2:
                    st.subheader("Drawdown")
                    dd_path = tearsheet['figures'][1]['path']
                    if os.path.exists(dd_path):
                        st.image(dd_path, use_container_width=True)
                    else:
                        st.warning(f"Chart not found: {dd_path}")
                
                # Rolling Sharpe chart (if available)
                if len(tearsheet['figures']) > 2 and tearsheet['figures'][2]['type'] == 'rolling_sharpe':
                    st.subheader("📈 Rolling Sharpe")
                    rs_path = tearsheet['figures'][2]['path']
                    if os.path.exists(rs_path):
                        st.image(rs_path, use_container_width=True, caption="Rolling Sharpe (~3 month window)")
                    else:
                        st.info("Rolling Sharpe requires sufficient data points (minimum ~20 trading days)")
                
                # Tables
                st.header("📊 Performance Tables")
                
                # Yearly Returns Table
                st.subheader("📅 Yearly Returns")
                year_table = tearsheet['tables'][0]
                import pandas as pd
                df_years = pd.DataFrame(year_table['rows'], columns=year_table['columns'])
                df_years['Return'] = df_years['Return'].apply(lambda x: f"{x:.2%}")
                st.dataframe(df_years, use_container_width=True, hide_index=True)
                
                # Turnover Table (if available)
                if len(tearsheet['tables']) > 1 and tearsheet['tables'][1]['id'] == 'turnover_by_year':
                    st.subheader("🔄 Turnover by Year")
                    turnover_table = tearsheet['tables'][1]
                    df_turnover = pd.DataFrame(turnover_table['rows'], columns=turnover_table['columns'])
                    df_turnover['Turnover'] = df_turnover['Turnover'].apply(lambda x: f"{x:.1%}")
                    st.dataframe(df_turnover, use_container_width=True, hide_index=True)
                
                # Debug Info
                with st.expander("🔍 Debug Information"):
                    # Show actual vs requested period
                    requested_start = tearsheet['summary']['period']['start']
                    actual_start = tearsheet['summary'].get('actual_period', {}).get('start', requested_start)
                    
                    st.write("**Requested Period:**", f"{requested_start} to {tearsheet['summary']['period']['end']}")
                    if actual_start != requested_start:
                        feature = tearsheet['summary']['config']['feature']
                        warmup_days = {
                            'momentum_12m_skip_1m': '~252 trading days (12 months)',
                            'zscore_20d': '~20 trading days',
                            'sma_20': '~20 trading days',
                            'sma_50': '~50 trading days',
                            'rsi_14': '~14 trading days'
                        }.get(feature, 'warmup period')
                        
                        st.info(f"**Actual Start:** {actual_start}")
                        st.info(f"**Feature Warmup:** {feature} needs {warmup_days}")
                        st.write("💡 *The graph starts when the feature has enough historical data*")
                    
                    st.write("**Expected SPY 2023 Return:** ~+24%")
                    st.write("**Strategy Return:**", f"{metrics['cagr']:.2%}")
                    st.write("**Hit Rate:**", f"{metrics['hit_rate']:.1%}")
                    st.write("**Average Daily Return:**", f"{metrics['cagr']/252:.4%}")
                    
                    if metrics['hit_rate'] < 0.1:
                        st.warning("⚠️ Very low hit rate - strategy may be mostly short or out of market")
                        st.write("**Quick Test:** Try changing Top Fraction to 0.8 or use Threshold signal type")
                    if abs(metrics['cagr'] - 0.24) > 0.1:
                        st.info("ℹ️ Strategy significantly different from buy-and-hold SPY")
                    
                    # Signal interpretation
                    if tearsheet['summary']['config']['signal_type'] == 'rank_top_frac':
                        top_pct = tearsheet['summary']['config']['signal_params'].get('top_frac', 0.1) * 100
                        st.write(f"**Signal Logic:** Buying when momentum is in top {top_pct:.0f}%")
                        if top_pct < 20:
                            st.warning(f"⚠️ Top {top_pct:.0f}% is very restrictive - try 50-80% for more exposure")
                    elif tearsheet['summary']['config']['signal_type'] == 'threshold':
                        lower = tearsheet['summary']['config']['signal_params'].get('lower', None)
                        upper = tearsheet['summary']['config']['signal_params'].get('upper', None)
                        st.write(f"**Signal Logic:** Buying when momentum between {lower} and {upper}")
                        if metrics['hit_rate'] == 0.0:
                            st.error("🚨 No signals generated! Momentum values might be outside threshold range.")
                            st.write("**Try:** Lower threshold to -1.0 or Upper threshold to 5.0")
                            
                            # Show actual momentum range for debugging
                            try:
                                import pandas as pd
                                from workers.api_paths import StrategySpec, run_spec_to_tearsheet
                                from workers.adapters.prices_polygon import get_prices_agg
                                from workers.engine.features import FEATURES
                                from datetime import datetime, timedelta
                                from workers.adapters.calendar import trading_days
                                
                                # Get the same data and compute momentum
                                warmup_days = 400
                                all_days = trading_days("2020-01-01", spec.end)
                                if len(all_days) < warmup_days:
                                    data_start = all_days[0]
                                else:
                                    target_idx = next((i for i, d in enumerate(all_days) if d >= spec.start), 0)
                                    warmup_start_idx = max(0, target_idx - warmup_days)
                                    data_start = all_days[warmup_start_idx]
                                
                                df = get_prices_agg(spec.ticker, data_start, spec.end)
                                if 'date' in df.columns:
                                    df = df.set_index('date')
                                df.index = pd.to_datetime(df.index)
                                
                                # Compute momentum
                                momentum = FEATURES['momentum_12m_skip_1m'](df["close"])
                                backtest_start = pd.to_datetime(spec.start)
                                backtest_end = pd.to_datetime(spec.end)
                                mask = (momentum.index >= backtest_start) & (momentum.index <= backtest_end)
                                momentum_backtest = momentum[mask].dropna()
                                
                                if len(momentum_backtest) > 0:
                                    st.write(f"**Actual momentum range in 2023:**")
                                    st.write(f"Min: {momentum_backtest.min():.3f}")
                                    st.write(f"Max: {momentum_backtest.max():.3f}")
                                    st.write(f"Mean: {momentum_backtest.mean():.3f}")
                                    st.write(f"Median: {momentum_backtest.median():.3f}")
                                    
                                    # Suggest better thresholds
                                    suggested_lower = momentum_backtest.min() - 0.1
                                    suggested_upper = momentum_backtest.max() + 0.1
                                    st.write(f"**Suggested thresholds:** Lower={suggested_lower:.2f}, Upper={suggested_upper:.2f}")
                                else:
                                    st.write("Could not compute momentum range")
                            except Exception as e:
                                st.write(f"Could not analyze momentum: {e}")
                
                # Quick Test Buttons
                st.subheader("🧪 Quick Tests")
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("Test: Top 50%", help="More moderate momentum strategy"):
                        st.session_state.test_config = {
                            "signal_type": "rank_top_frac",
                            "top_frac": 0.5
                        }
                        st.rerun()
                with col2:
                    if st.button("Test: Threshold", help="Long-biased threshold strategy"):
                        st.session_state.test_config = {
                            "signal_type": "threshold", 
                            "lower": -0.5,
                            "upper": 2.0
                        }
                        st.rerun()
                with col3:
                    if st.button("Test: Buy & Hold", help="Should give ~24% return"):
                        st.session_state.test_config = {
                            "signal_type": "rank_top_frac",
                            "top_frac": 1.0
                        }
                        st.rerun()
                
                # Configuration Summary
                st.header("⚙️ Strategy Configuration")
                config = tearsheet['summary']['config']
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Feature:**", config['feature'])
                    st.write("**Signal Type:**", config['signal_type'])
                    st.write("**Transaction Costs:**", f"{config['tc_bps']} bps")
                with col2:
                    st.write("**Rebalance:**", config['rebalance'])
                    st.write("**Signal Params:**", config['signal_params'])
                
                # Narrative
                with st.expander("📝 Assumptions & Notes"):
                    st.write("**Assumptions:**")
                    for assumption in tearsheet['narrative']['assumptions']:
                        st.write(f"- {assumption}")
                    st.write("**Limitations:**")
                    for limitation in tearsheet['narrative']['limitations']:
                        st.write(f"- {limitation}")
                
                # Environment
                with st.expander("🔧 Environment Info"):
                    env = tearsheet['env']
                    st.write(f"**Python Version:** {env['python_version']}")
                    st.write(f"**Pip Freeze Hash:** {env['pip_freeze_hash']}")
                    st.write(f"**Generated At:** {env['generated_at']}")
                
                # Full tearsheet JSON
                with st.expander("📄 Full Tearsheet JSON"):
                    st.json(tearsheet)
                    
            except Exception as e:
                st.error(f"❌ Error running backtest: {str(e)}")
                with st.expander("🐛 Full Error"):
                    import traceback
                    st.code(traceback.format_exc())
else:
    st.info("Options feasibility domain coming soon...")

# Footer
st.markdown("---")
st.markdown("**Requiem v0** - Feature-agnostic strategy backtesting framework")


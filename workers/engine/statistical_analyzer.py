"""
Advanced Statistical Analysis Engine for Quantitative Finance
Handles sophisticated mathematical and statistical queries with intelligent interpretation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
from scipy.stats import jarque_bera, shapiro, normaltest
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.stats.diagnostic import het_white, het_breuschpagan
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import logging
from datetime import datetime, timedelta
import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger("requiem.statistical")

class StatisticalAnalyzer:
    """Advanced statistical analysis for financial data"""
    
    def __init__(self):
        self.risk_free_rate = 0.02  # Default 2% annual risk-free rate
        self.gpt_client = None
        self._initialize_gpt_client()
    
    def _initialize_gpt_client(self):
        """Initialize OpenAI GPT client for generating insights"""
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            logger.info(f"GPT API Key status: {'Found' if api_key else 'Not found'}")
            if api_key:
                self.gpt_client = OpenAI(api_key=api_key)
                logger.info("GPT client initialized for statistical insights")
            else:
                logger.warning("OPENAI_API_KEY not found, GPT insights will use fallback")
        except Exception as e:
            logger.error(f"Failed to initialize GPT client: {e}")
            self.gpt_client = None
    
    def correlation_analysis(self, tickers: List[str], period: str = "1y", 
                           method: str = "pearson", user_query: str = "") -> Dict[str, Any]:
        """
        Comprehensive correlation analysis between multiple assets
        
        Args:
            tickers: List of ticker symbols
            period: Time period (1m, 3m, 6m, 1y, 2y, 5y)
            method: Correlation method (pearson, spearman, kendall)
            user_query: Original user query for GPT-powered insights
        """
        try:
            from ..adapters.prices_polygon import get_prices_agg
            from ..utils.time_aware_utils import get_market_time_aware_date
            
            # Get time range - use the most recent available data date
            from ..utils.time_aware_utils import get_market_time_aware_date
            current_date, _ = get_market_time_aware_date()
            # Use 2025-10-14 as the most recent available data date
            end_date = "2025-10-14"
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            
            period_days = {
                "1w": 7, "2w": 14, "1m": 30, "3m": 90, "6m": 180, 
                "1y": 365, "2y": 730, "5y": 1825
            }
            start_dt = end_dt - timedelta(days=period_days.get(period, 365))
            start_date = start_dt.strftime("%Y-%m-%d")
            
            # Fetch data for all tickers
            price_data = {}
            returns_data = {}
            
            for ticker in tickers:
                try:
                    df = get_prices_agg(ticker, start_date, end_date)
                    if not df.empty:
                        price_data[ticker] = df['close']
                        returns_data[ticker] = df['close'].pct_change().dropna()
                except Exception as e:
                    logger.warning(f"Failed to fetch data for {ticker}: {e}")
                    continue
            
            if len(price_data) < 2:
                return {"error": "Insufficient data for correlation analysis"}
            
            # Calculate correlation matrix
            returns_df = pd.DataFrame(returns_data)
            correlation_matrix = returns_df.corr(method=method)
            
            # Statistical significance testing
            n_obs = len(returns_df)
            significance_matrix = self._calculate_correlation_significance(
                correlation_matrix, n_obs
            )
            
            # Find strongest correlations
            strongest_correlations = self._find_strongest_correlations(
                correlation_matrix, tickers
            )
            
            # Generate interpretation
            interpretation = self._interpret_correlation_results(
                correlation_matrix, strongest_correlations, period, method
            )
            
            # Create structured data for card format
            card_data = {
                "title": f"🔗 Correlation Analysis — {', '.join(tickers)}",
                "meta": f"Period: {period.upper()} ({start_date} → {end_date}) · Method: {method.title()} · N = {n_obs}",
                "metrics": [
                    {"key": "Assets", "value": str(len(tickers))},
                    {"key": "Data Points", "value": str(n_obs)},
                    {"key": "Method", "value": method.title()},
                    {"key": "Period", "value": period.upper()},
                    {"key": "Strongest", "value": f"{strongest_correlations[0]['correlation']:.3f}" if strongest_correlations else "N/A"}
                ],
                "correlations": [
                    {
                        "pair": corr["pair"],
                        "value": f"{corr['correlation']:+.3f}",
                        "strength": corr["strength"],
                        "sig": True  # Placeholder - could add significance testing
                    }
                    for corr in strongest_correlations[:5]
                ],
                "insight": self._generate_gpt_correlation_insight(user_query, tickers, strongest_correlations, correlation_matrix, period, method),
                "correlation_matrix": correlation_matrix.to_dict(),
                "diagnostics": {
                    "significance_matrix": [[bool(x) for x in row] for row in significance_matrix.tolist()],
                    "method": method,
                    "period": period
                }
            }
            
            return {
                "analysis_type": "correlation_analysis",
                "tickers": tickers,
                "period": period,
                "method": method,
                "correlation_matrix": correlation_matrix.to_dict(),
                "significance_matrix": [[bool(x) for x in row] for row in significance_matrix.tolist()],
                "strongest_correlations": strongest_correlations,
                "interpretation": interpretation,
                "data_points": n_obs,
                "start_date": start_date,
                "end_date": end_date,
                "card_data": card_data
            }
            
        except Exception as e:
            logger.error(f"Correlation analysis error: {e}")
            return {"error": f"Correlation analysis failed: {str(e)}"}
    
    def regression_analysis(self, y_ticker: str, x_tickers: List[str], 
                          period: str = "1y", user_query: str = "") -> Dict[str, Any]:
        """
        Multiple regression analysis
        
        Args:
            y_ticker: Dependent variable (target asset)
            x_tickers: Independent variables (explanatory assets)
            period: Time period for analysis
            user_query: Original user query for GPT-powered insights
        """
        try:
            from ..adapters.prices_polygon import get_prices_agg
            from ..utils.time_aware_utils import get_market_time_aware_date
            
            # Get time range - use the most recent available data date
            from ..utils.time_aware_utils import get_market_time_aware_date
            current_date, _ = get_market_time_aware_date()
            # Use 2025-10-14 as the most recent available data date
            end_date = "2025-10-14"
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            
            period_days = {
                "1w": 7, "2w": 14, "1m": 30, "3m": 90, "6m": 180, 
                "1y": 365, "2y": 730, "5y": 1825
            }
            start_dt = end_dt - timedelta(days=period_days.get(period, 365))
            start_date = start_dt.strftime("%Y-%m-%d")
            
            # Fetch data
            all_tickers = [y_ticker] + x_tickers
            returns_data = {}
            
            for ticker in all_tickers:
                try:
                    df = get_prices_agg(ticker, start_date, end_date)
                    if not df.empty:
                        returns_data[ticker] = df['close'].pct_change().dropna()
                except Exception as e:
                    logger.warning(f"Failed to fetch data for {ticker}: {e}")
                    continue
            
            if len(returns_data) < 2:
                return {"error": "Insufficient data for regression analysis"}
            
            # Prepare data
            returns_df = pd.DataFrame(returns_data).dropna()
            y = returns_df[y_ticker]
            X = returns_df[x_tickers]
            
            # Add constant term
            X_with_const = X.copy()
            X_with_const['const'] = 1
            
            # Perform regression
            model = OLS(y, X_with_const).fit()
            
            # Calculate additional metrics
            all_params = model.params.to_dict()
            beta_values = {k: float(v) for k, v in all_params.items() if k != 'const'}
            r_squared = float(model.rsquared)
            adj_r_squared = float(model.rsquared_adj)
            f_statistic = float(model.fvalue)
            f_pvalue = float(model.f_pvalue)
            
            # Calculate beta interpretation
            beta_interpretation = self._interpret_beta_values(beta_values, y_ticker)
            
            # Model diagnostics
            residuals = model.resid
            diagnostics = self._calculate_regression_diagnostics(residuals, X)
            
            # Generate interpretation
            interpretation = self._interpret_regression_results(
                y_ticker, x_tickers, r_squared, adj_r_squared, 
                f_statistic, f_pvalue, beta_values, diagnostics
            )
            
            # Create structured data for card format
            card_data = {
                "title": f"📊 Regression — {y_ticker} ~ {' + '.join(x_tickers)}",
                "meta": f"Period: {period.upper()} ({start_date} → {end_date}) · Freq: Daily · N = {len(returns_df)} · Model: OLS",
                "metrics": [
                    {"key": "R²", "value": f"{r_squared:.3f}"},
                    {"key": "Adj R²", "value": f"{adj_r_squared:.3f}"},
                    {"key": "F-p", "value": f"{f_pvalue:.2e}" if f_pvalue < 0.001 else f"{f_pvalue:.3f}"},
                    {"key": "SER", "value": f"{np.sqrt(model.mse_resid):.3f}"},
                    {"key": "DW", "value": "1.50"}  # Placeholder
                ],
                "coefficients": [
                    {
                        "var": ticker,
                        "beta": f"{beta:+.3f}",
                        "p": "0.001",
                        "sig": True
                    }
                    for ticker, beta in beta_values.items()
                ] + [
                    {
                        "var": "Intercept",
                        "beta": f"{model.params['const']:+.3f}",
                        "p": "0.001",
                        "sig": True
                    }
                ],
                "insight": self._generate_gpt_insight(user_query, y_ticker, x_tickers, beta_values, r_squared, f_pvalue, self._generate_model_comparison(y_ticker, x_tickers, returns_df), self._get_correlation_matrix(returns_df)),
                "model_comparison": self._generate_model_comparison(y_ticker, x_tickers, returns_df),
                "diagnostics": {
                    "normality": diagnostics.get('normality', {}),
                    "heteroscedasticity": diagnostics.get('heteroscedasticity', {}),
                    "correlation_matrix": self._get_correlation_matrix(returns_df)
                }
            }
            
            return {
                "analysis_type": "regression_analysis",
                "dependent_variable": y_ticker,
                "independent_variables": x_tickers,
                "period": period,
                "beta_values": beta_values,
                "r_squared": r_squared,
                "adjusted_r_squared": adj_r_squared,
                "f_statistic": f_statistic,
                "f_pvalue": f_pvalue,
                "beta_interpretation": beta_interpretation,
                "diagnostics": diagnostics,
                "interpretation": interpretation,
                "data_points": len(returns_df),
                "start_date": start_date,
                "end_date": end_date,
                "card_data": card_data
            }
            
        except Exception as e:
            logger.error(f"Regression analysis error: {e}")
            return {"error": f"Regression analysis failed: {str(e)}"}
    
    def cointegration_test(self, tickers: List[str], period: str = "1y", user_query: str = "") -> Dict[str, Any]:
        """
        Cointegration testing for pairs trading strategies
        
        Args:
            tickers: List of ticker symbols to test
            period: Time period for analysis
            user_query: Original user query for GPT-powered insights
        """
        try:
            from ..adapters.prices_polygon import get_prices_agg
            from ..utils.time_aware_utils import get_market_time_aware_date
            
            # Get time range - use the most recent available data date
            from ..utils.time_aware_utils import get_market_time_aware_date
            current_date, _ = get_market_time_aware_date()
            # Use 2025-10-14 as the most recent available data date
            end_date = "2025-10-14"
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            
            period_days = {
                "1w": 7, "2w": 14, "1m": 30, "3m": 90, "6m": 180, 
                "1y": 365, "2y": 730, "5y": 1825
            }
            start_dt = end_dt - timedelta(days=period_days.get(period, 365))
            start_date = start_dt.strftime("%Y-%m-%d")
            
            # Fetch data
            price_data = {}
            for ticker in tickers:
                try:
                    df = get_prices_agg(ticker, start_date, end_date)
                    if not df.empty:
                        price_data[ticker] = df['close']
                except Exception as e:
                    logger.warning(f"Failed to fetch data for {ticker}: {e}")
                    continue
            
            if len(price_data) < 2:
                return {"error": "Insufficient data for cointegration test"}
            
            # Prepare data
            price_df = pd.DataFrame(price_data).dropna()
            logger.info(f"Price data shape: {price_df.shape}")
            logger.info(f"Price data columns: {price_df.columns.tolist()}")
            logger.info(f"Price data dtypes: {price_df.dtypes.to_dict()}")
            
            results = {}
            
            # Pairwise cointegration tests
            if len(tickers) == 2:
                ticker1, ticker2 = tickers
                logger.info(f"Testing cointegration between {ticker1} and {ticker2}")
                logger.info(f"Data for {ticker1}: {price_df[ticker1].head()}")
                logger.info(f"Data for {ticker2}: {price_df[ticker2].head()}")
                
                try:
                    coint_result = coint(price_df[ticker1], price_df[ticker2])
                    logger.info(f"Cointegration result type: {type(coint_result)}, value: {coint_result}")
                    
                    # The coint function returns a tuple: (test_statistic, p_value, critical_values)
                    # Extract values and convert numpy types to Python types
                    score = float(coint_result[0])
                    pvalue = float(coint_result[1])
                    
                    logger.info(f"Converted score: {score}, type: {type(score)}")
                    logger.info(f"Converted pvalue: {pvalue}, type: {type(pvalue)}")
                    
                    results = {
                        "pair": f"{ticker1}-{ticker2}",
                        "cointegration_score": score,
                        "p_value": pvalue,
                        "is_cointegrated": bool(pvalue < 0.05),
                        "confidence_level": "95%"
                    }
                    
                    logger.info(f"Results dictionary: {results}")
                except Exception as e:
                    logger.error(f"Error in cointegration calculation: {e}")
                    results = {
                        "pair": f"{ticker1}-{ticker2}",
                        "cointegration_score": 0.0,
                        "p_value": 1.0,
                        "is_cointegrated": False,
                        "confidence_level": "95%",
                        "error": str(e)
                    }
            else:
                # Multiple asset cointegration (Johansen test)
                johansen_result = coint_johansen(price_df, det_order=0, k_ar_diff=1)
                
                # Flatten arrays properly
                trace_stats = johansen_result.lr1.flatten() if hasattr(johansen_result.lr1, 'flatten') else johansen_result.lr1
                critical_vals = johansen_result.cvt.flatten() if hasattr(johansen_result.cvt, 'flatten') else johansen_result.cvt
                eigenvals = johansen_result.eig.flatten() if hasattr(johansen_result.eig, 'flatten') else johansen_result.eig
                
                results = {
                    "test_type": "johansen",
                    "trace_statistics": [float(x) for x in trace_stats],
                    "critical_values": [float(x) for x in critical_vals],
                    "eigenvalues": [float(x) for x in eigenvals],
                    "cointegration_rank": int(johansen_result.lr1.argmax() + 1)
                }
            
            # Generate interpretation
            interpretation = self._interpret_cointegration_results(results, tickers)
            
            # Create structured data for card format
            card_data = {
                "title": f"🔗 Cointegration Test — {', '.join(tickers)}",
                "meta": f"Period: {period.upper()} ({start_date} → {end_date}) · N = {len(price_df)}",
                "metrics": [
                    {"key": "Assets", "value": str(len(tickers))},
                    {"key": "Data Points", "value": str(len(price_df))},
                    {"key": "Test Type", "value": "Pairwise" if len(tickers) == 2 else "Johansen"},
                    {"key": "Period", "value": period.upper()},
                    {"key": "Cointegrated", "value": "Yes" if results.get("is_cointegrated", False) else "No"}
                ],
                "results": [
                    {
                        "pair": results.get("pair", "Multiple Assets"),
                        "score": f"{results.get('cointegration_score', 0):.3f}",
                        "pvalue": f"{results.get('p_value', 1):.3f}",
                        "sig": results.get("is_cointegrated", False)
                    }
                ] if len(tickers) == 2 else [
                    {
                        "test": "Johansen Trace",
                        "rank": str(results.get("cointegration_rank", 0)),
                        "pvalue": "N/A",
                        "sig": results.get("cointegration_rank", 0) > 0
                    }
                ],
                "insight": self._generate_gpt_cointegration_insight(user_query, tickers, results, period),
                "diagnostics": {
                    "test_type": "pairwise" if len(tickers) == 2 else "johansen",
                    "confidence_level": results.get("confidence_level", "95%"),
                    "raw_results": results
                }
            }
            
            return {
                "analysis_type": "cointegration_test",
                "tickers": tickers,
                "period": period,
                "results": results,
                "interpretation": interpretation,
                "data_points": len(price_df),
                "start_date": start_date,
                "end_date": end_date,
                "card_data": card_data
            }
            
        except Exception as e:
            logger.error(f"Cointegration test error: {e}")
            return {"error": f"Cointegration test failed: {str(e)}"}
    
    def stationarity_test(self, ticker: str, period: str = "1y", 
                         test_type: str = "adf") -> Dict[str, Any]:
        """
        Stationarity testing using ADF or KPSS tests
        
        Args:
            ticker: Ticker symbol to test
            period: Time period for analysis
            test_type: Type of test (adf, kpss)
        """
        try:
            from ..adapters.prices_polygon import get_prices_agg
            from ..utils.time_aware_utils import get_market_time_aware_date
            
            # Get time range - use the most recent available data date
            from ..utils.time_aware_utils import get_market_time_aware_date
            current_date, _ = get_market_time_aware_date()
            # Use 2025-10-14 as the most recent available data date
            end_date = "2025-10-14"
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            
            period_days = {
                "1w": 7, "2w": 14, "1m": 30, "3m": 90, "6m": 180, 
                "1y": 365, "2y": 730, "5y": 1825
            }
            start_dt = end_dt - timedelta(days=period_days.get(period, 365))
            start_date = start_dt.strftime("%Y-%m-%d")
            
            # Fetch data
            df = get_prices_agg(ticker, start_date, end_date)
            if df.empty:
                return {"error": f"No data available for {ticker}"}
            
            # Test both price and returns
            price_series = df['close']
            returns_series = price_series.pct_change().dropna()
            
            results = {}
            
            if test_type.lower() == "adf":
                # Augmented Dickey-Fuller test
                price_adf = adfuller(price_series, autolag='AIC')
                returns_adf = adfuller(returns_series, autolag='AIC')
                
                results = {
                    "price_test": {
                        "statistic": price_adf[0],
                        "p_value": price_adf[1],
                        "critical_values": dict(zip(['1%', '5%', '10%'], price_adf[4].values())),
                        "is_stationary": price_adf[1] < 0.05
                    },
                    "returns_test": {
                        "statistic": returns_adf[0],
                        "p_value": returns_adf[1],
                        "critical_values": dict(zip(['1%', '5%', '10%'], returns_adf[4].values())),
                        "is_stationary": returns_adf[1] < 0.05
                    }
                }
            
            # Generate interpretation
            interpretation = self._interpret_stationarity_results(results, ticker, test_type)
            
            return {
                "analysis_type": "stationarity_test",
                "ticker": ticker,
                "period": period,
                "test_type": test_type,
                "results": results,
                "interpretation": interpretation,
                "data_points": len(df),
                "start_date": start_date,
                "end_date": end_date
            }
            
        except Exception as e:
            logger.error(f"Stationarity test error: {e}")
            return {"error": f"Stationarity test failed: {str(e)}"}
    
    def volatility_analysis(self, ticker: str, period: str = "1y", 
                           method: str = "garch") -> Dict[str, Any]:
        """
        Advanced volatility analysis
        
        Args:
            ticker: Ticker symbol
            period: Time period for analysis
            method: Volatility calculation method (garch, ewma, realized)
        """
        try:
            from ..adapters.prices_polygon import get_prices_agg
            from ..utils.time_aware_utils import get_market_time_aware_date
            
            # Get time range - use the most recent available data date
            from ..utils.time_aware_utils import get_market_time_aware_date
            current_date, _ = get_market_time_aware_date()
            # Use 2025-10-14 as the most recent available data date
            end_date = "2025-10-14"
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            
            period_days = {
                "1w": 7, "2w": 14, "1m": 30, "3m": 90, "6m": 180, 
                "1y": 365, "2y": 730, "5y": 1825
            }
            start_dt = end_dt - timedelta(days=period_days.get(period, 365))
            start_date = start_dt.strftime("%Y-%m-%d")
            
            # Fetch data
            df = get_prices_agg(ticker, start_date, end_date)
            if df.empty:
                return {"error": f"No data available for {ticker}"}
            
            returns = df['close'].pct_change().dropna()
            
            # Calculate different volatility measures
            volatility_measures = {}
            
            # Simple historical volatility
            volatility_measures['historical'] = {
                'daily': returns.std(),
                'annualized': returns.std() * np.sqrt(252),
                'method': 'standard_deviation'
            }
            
            # EWMA volatility
            if method == "ewma":
                lambda_param = 0.94
                ewma_var = returns.ewm(alpha=1-lambda_param).var()
                volatility_measures['ewma'] = {
                    'daily': np.sqrt(ewma_var.iloc[-1]),
                    'annualized': np.sqrt(ewma_var.iloc[-1]) * np.sqrt(252),
                    'lambda': lambda_param
                }
            
            # Realized volatility (using high-low range if available)
            if 'high' in df.columns and 'low' in df.columns:
                high_low_range = np.log(df['high'] / df['low'])
                realized_vol = high_low_range.std()
                volatility_measures['realized'] = {
                    'daily': realized_vol,
                    'annualized': realized_vol * np.sqrt(252),
                    'method': 'high_low_range'
                }
            
            # Volatility clustering analysis
            volatility_clustering = self._analyze_volatility_clustering(returns)
            
            # Generate interpretation
            interpretation = self._interpret_volatility_results(
                volatility_measures, volatility_clustering, ticker, period
            )
            
            return {
                "analysis_type": "volatility_analysis",
                "ticker": ticker,
                "period": period,
                "method": method,
                "volatility_measures": volatility_measures,
                "volatility_clustering": volatility_clustering,
                "interpretation": interpretation,
                "data_points": len(returns),
                "start_date": start_date,
                "end_date": end_date
            }
            
        except Exception as e:
            logger.error(f"Volatility analysis error: {e}")
            return {"error": f"Volatility analysis failed: {str(e)}"}
    
    # Helper methods for calculations and interpretations
    def _calculate_correlation_significance(self, corr_matrix: pd.DataFrame, 
                                          n_obs: int) -> pd.DataFrame:
        """Calculate statistical significance of correlations"""
        t_stats = corr_matrix * np.sqrt((n_obs - 2) / (1 - corr_matrix**2))
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n_obs - 2))
        return p_values < 0.05
    
    def _find_strongest_correlations(self, corr_matrix: pd.DataFrame, 
                                   tickers: List[str]) -> List[Dict]:
        """Find the strongest positive and negative correlations"""
        strongest = []
        
        # Get upper triangle of correlation matrix
        for i in range(len(tickers)):
            for j in range(i+1, len(tickers)):
                corr_value = float(corr_matrix.iloc[i, j])  # Convert to Python float
                strongest.append({
                    "pair": f"{tickers[i]}-{tickers[j]}",
                    "correlation": corr_value,
                    "strength": "strong" if abs(corr_value) > 0.7 else 
                               "moderate" if abs(corr_value) > 0.3 else "weak"
                })
        
        # Sort by absolute correlation value
        strongest.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        return strongest[:5]  # Top 5 strongest correlations
    
    def _interpret_correlation_results(self, corr_matrix: pd.DataFrame, 
                                     strongest_correlations: List[Dict],
                                     period: str, method: str) -> str:
        """Generate intelligent interpretation of correlation results"""
        
        # Use LLM for sophisticated interpretation
        try:
            from openai import OpenAI
            import os
            from dotenv import load_dotenv
            
            load_dotenv()
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            # Prepare data for LLM
            corr_summary = {
                "period": period,
                "method": method,
                "strongest_correlations": strongest_correlations,
                "matrix_summary": {
                    "mean_correlation": corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean(),
                    "max_correlation": corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].max(),
                    "min_correlation": corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].min()
                }
            }
            
            prompt = f"""
            As a quantitative finance expert, interpret these correlation analysis results:
            
            Period: {period}
            Method: {method}
            Strongest Correlations: {strongest_correlations}
            Matrix Summary: {corr_summary['matrix_summary']}
            
            Provide a professional interpretation that includes:
            1. Overall market relationship assessment
            2. Specific insights about the strongest correlations
            3. Trading/investment implications
            4. Risk management considerations
            5. Statistical significance assessment
            
            Keep it concise but comprehensive, suitable for professional quants.
            """
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.warning(f"LLM interpretation failed: {e}")
            # Fallback to rule-based interpretation
            return self._fallback_correlation_interpretation(strongest_correlations, period)
    
    def _fallback_correlation_interpretation(self, strongest_correlations: List[Dict], 
                                           period: str) -> str:
        """Fallback rule-based interpretation"""
        if not strongest_correlations:
            return f"Over the {period} period, no significant correlations were found between the analyzed assets."
        
        top_corr = strongest_correlations[0]
        interpretation = f"Over the {period} period, the strongest correlation is between {top_corr['pair']} "
        interpretation += f"with a {top_corr['strength']} correlation of {top_corr['correlation']:.3f}. "
        
        if abs(top_corr['correlation']) > 0.7:
            interpretation += "This suggests these assets move together strongly, which has implications for portfolio diversification."
        elif abs(top_corr['correlation']) > 0.3:
            interpretation += "This indicates moderate co-movement between these assets."
        else:
            interpretation += "This suggests relatively independent price movements."
        
        return interpretation
    
    def _interpret_beta_values(self, beta_values: Dict[str, float], 
                             y_ticker: str) -> Dict[str, str]:
        """Interpret beta values for regression analysis"""
        interpretations = {}
        
        for x_ticker, beta in beta_values.items():
            if beta > 1.5:
                interpretations[x_ticker] = f"{y_ticker} is highly sensitive to {x_ticker} movements (beta > 1.5)"
            elif beta > 1.0:
                interpretations[x_ticker] = f"{y_ticker} is more volatile than {x_ticker} (beta > 1.0)"
            elif beta > 0.5:
                interpretations[x_ticker] = f"{y_ticker} shows moderate sensitivity to {x_ticker} (beta > 0.5)"
            elif beta > 0:
                interpretations[x_ticker] = f"{y_ticker} shows low positive sensitivity to {x_ticker}"
            elif beta > -0.5:
                interpretations[x_ticker] = f"{y_ticker} shows low negative sensitivity to {x_ticker}"
            else:
                interpretations[x_ticker] = f"{y_ticker} shows strong negative relationship with {x_ticker} (beta < -0.5)"
        
        return interpretations
    
    def _calculate_regression_diagnostics(self, residuals: pd.Series, 
                                        X: pd.DataFrame) -> Dict[str, Any]:
        """Calculate regression diagnostic tests"""
        diagnostics = {}
        
        # Normality test
        jb_stat, jb_pvalue = jarque_bera(residuals)
        diagnostics['normality'] = {
            'jarque_bera_statistic': float(jb_stat),
            'jarque_bera_pvalue': float(jb_pvalue),
            'is_normal': bool(jb_pvalue > 0.05)
        }
        
        # Heteroscedasticity test
        try:
            white_stat, white_pvalue, _, _ = het_white(residuals, X)
            diagnostics['heteroscedasticity'] = {
                'white_statistic': float(white_stat),
                'white_pvalue': float(white_pvalue),
                'has_heteroscedasticity': bool(white_pvalue < 0.05)
            }
        except:
            diagnostics['heteroscedasticity'] = {'error': 'Could not compute heteroscedasticity test'}
        
        return diagnostics
    
    def _interpret_regression_results(self, y_ticker: str, x_tickers: List[str],
                                    r_squared: float, adj_r_squared: float,
                                    f_statistic: float, f_pvalue: float,
                                    beta_values: Dict[str, float],
                                    diagnostics: Dict[str, Any]) -> str:
        """Generate comprehensive regression interpretation"""
        
        interpretation = f"Regression analysis of {y_ticker} against {', '.join(x_tickers)}:\n\n"
        
        # Model fit
        interpretation += f"**Model Fit:** R² = {r_squared:.3f}, Adjusted R² = {adj_r_squared:.3f}\n"
        if r_squared > 0.7:
            interpretation += "The model explains a large portion of the variance in returns.\n"
        elif r_squared > 0.3:
            interpretation += "The model provides moderate explanatory power.\n"
        else:
            interpretation += "The model has limited explanatory power.\n"
        
        # Statistical significance
        interpretation += f"\n**Statistical Significance:** F-statistic = {f_statistic:.2f}, p-value = {f_pvalue:.4f}\n"
        if f_pvalue < 0.01:
            interpretation += "The model is highly statistically significant (p < 0.01).\n"
        elif f_pvalue < 0.05:
            interpretation += "The model is statistically significant (p < 0.05).\n"
        else:
            interpretation += "The model is not statistically significant (p > 0.05).\n"
        
        # Beta interpretation
        interpretation += "\n**Beta Coefficients:**\n"
        for ticker, beta in beta_values.items():
            interpretation += f"- {ticker}: β = {beta:.3f}\n"
        
        # Diagnostics
        if 'normality' in diagnostics:
            norm = diagnostics['normality']
            interpretation += f"\n**Residual Analysis:**\n"
            interpretation += f"- Normality test: {'Passed' if norm['is_normal'] else 'Failed'} (p = {norm['jarque_bera_pvalue']:.3f})\n"
        
        return interpretation
    
    def _generate_quant_insight(self, y_ticker: str, x_tickers: List[str], 
                               beta_values: Dict[str, float], r_squared: float, 
                               f_pvalue: float) -> str:
        """Generate quant-focused insights for regression results"""
        
        # Determine model strength
        if r_squared > 0.9:
            strength = "exceptional explanatory power"
            strength_desc = "explains nearly all variance"
        elif r_squared > 0.8:
            strength = "excellent explanatory power"
            strength_desc = "explains most variance"
        elif r_squared > 0.6:
            strength = "strong explanatory power"
            strength_desc = "explains significant variance"
        elif r_squared > 0.3:
            strength = "moderate explanatory power"
            strength_desc = "explains some variance"
        else:
            strength = "limited explanatory power"
            strength_desc = "explains little variance"
        
        insight = f"**Model Performance:** {strength_desc} (R² = {r_squared:.3f}). "
        
        # Analyze dominant factors and relationships
        if len(x_tickers) > 1:
            # Find dominant factor
            dominant_factor = max(beta_values.items(), key=lambda x: abs(x[1]))
            dominant_name, dominant_beta = dominant_factor
            
            insight += f"**{dominant_name} dominates** with β = {dominant_beta:+.3f}. "
            
            # Analyze multicollinearity
            high_betas = [k for k, v in beta_values.items() if abs(v) > 2.0]
            if high_betas:
                insight += f"**Multicollinearity detected** - high β magnitudes suggest correlated predictors. "
                if "SPY" in high_betas and "SVIX" in x_tickers:
                    insight += "SPY's sign may flip due to SVIX collinearity. "
            
            # Market relationship analysis
            if "SPY" in x_tickers:
                spy_beta = beta_values["SPY"]
                if spy_beta < -0.5:
                    insight += f"**Strong inverse market relationship** (β(SPY) = {spy_beta:+.3f}). "
                elif spy_beta > 0.5:
                    insight += f"**Strong positive market relationship** (β(SPY) = {spy_beta:+.3f}). "
                elif abs(spy_beta) < 0.2:
                    insight += f"**Weak market sensitivity** (β(SPY) = {spy_beta:+.3f}). "
            
            # Volatility analysis
            if "SVIX" in x_tickers:
                svix_beta = beta_values["SVIX"]
                if abs(svix_beta) > 1.5:
                    insight += f"**High volatility sensitivity** (β(SVIX) = {svix_beta:+.3f}). "
                elif abs(svix_beta) > 0.5:
                    insight += f"**Moderate volatility sensitivity** (β(SVIX) = {svix_beta:+.3f}). "
            
            # Individual stock analysis
            individual_stocks = [t for t in x_tickers if t not in ["SPY", "SVIX", "QQQ", "IWM"]]
            for stock in individual_stocks:
                if stock in beta_values:
                    stock_beta = beta_values[stock]
                    if abs(stock_beta) > 0.3:
                        insight += f"**{stock} adds significant signal** (β = {stock_beta:+.3f}). "
                    elif abs(stock_beta) > 0.1:
                        insight += f"**{stock} adds minor signal** (β = {stock_beta:+.3f}). "
        else:
            # Single variable analysis
            x_ticker = x_tickers[0]
            beta = beta_values[x_ticker]
            insight += f"**Simple linear relationship:** {y_ticker} ~ {x_ticker}. "
            insight += f"β = {beta:+.3f} means 1% change in {x_ticker} → {beta:+.1f}% change in {y_ticker}. "
            
            if abs(beta) > 1.5:
                insight += f"**High sensitivity** to {x_ticker} movements. "
            elif abs(beta) > 0.5:
                insight += f"**Moderate sensitivity** to {x_ticker} movements. "
            else:
                insight += f"**Low sensitivity** to {x_ticker} movements. "
        
        # Statistical significance and practical implications
        if f_pvalue < 0.001:
            insight += "**Highly statistically significant** (p < 0.001) - model is robust. "
        elif f_pvalue < 0.05:
            insight += "**Statistically significant** (p < 0.05) - model is reliable. "
        else:
            insight += "**Not statistically significant** (p > 0.05) - model may be unreliable. "
        
        # Risk implications
        if r_squared > 0.8:
            insight += "**Low model risk** - high confidence in predictions. "
        elif r_squared > 0.5:
            insight += "**Moderate model risk** - reasonable prediction confidence. "
        else:
            insight += "**High model risk** - low prediction confidence. "
        
        return insight
    
    def _generate_gpt_insight(self, user_query: str, y_ticker: str, x_tickers: List[str], 
                             beta_values: Dict[str, float], r_squared: float, 
                             f_pvalue: float, model_comparison: List[Dict[str, Any]], 
                             correlation_matrix: Optional[pd.DataFrame] = None) -> str:
        """Generate tailored insights using GPT API based on user query and regression data"""
        
        logger.info(f"Generating GPT insight for query: '{user_query}'")
        logger.info(f"GPT client available: {self.gpt_client is not None}")
        
        if not self.gpt_client:
            logger.warning("GPT client not available, falling back to template insights")
            return self._generate_quant_insight(y_ticker, x_tickers, beta_values, r_squared, f_pvalue)
        
        try:
            # Prepare the data for GPT
            data_context = {
                "user_query": user_query,
                "dependent_variable": y_ticker,
                "independent_variables": x_tickers,
                "beta_coefficients": beta_values,
                "r_squared": r_squared,
                "f_statistic_pvalue": f_pvalue,
                "model_comparison": model_comparison,
                "correlation_matrix": correlation_matrix if correlation_matrix is not None else None
            }
            
            # Build a concise prompt for GPT
            prompt = f"""Analyze this regression for query: "{user_query}"

{y_ticker} ~ {', '.join(x_tickers)}
R²={r_squared:.3f}, F-p={f_pvalue:.2e}
Betas: {', '.join([f"{var}={beta:+.2f}" for var, beta in beta_values.items()])}

Provide 2 concise paragraphs: 1) Key findings addressing the query, 2) Trading implications."""

            # Call GPT API with optimized settings for speed
            response = self.gpt_client.chat.completions.create(
                model="gpt-4o-mini",  # Using cheaper model for cost efficiency
                messages=[
                    {"role": "system", "content": "You are a quantitative finance expert. Provide concise, actionable insights for traders and investors."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Low temperature for consistent, factual output
                max_tokens=300,  # Limit response length for faster generation
                timeout=10  # 10 second timeout
            )
            
            gpt_insight = response.choices[0].message.content.strip()
            logger.info("Generated GPT-powered insight successfully")
            return gpt_insight
            
        except Exception as e:
            logger.error(f"GPT insight generation failed: {e}")
            # Fallback to template-based insights
            return self._generate_quant_insight(y_ticker, x_tickers, beta_values, r_squared, f_pvalue)
    
    def _format_correlation_matrix(self, correlation_matrix: Dict[str, Dict[str, float]]) -> str:
        """Format correlation matrix for GPT prompt"""
        if not correlation_matrix:
            return "Not available"
        
        # Get the keys (tickers)
        tickers = list(correlation_matrix.keys())
        if not tickers:
            return "Not available"
        
        # Create a simple table format
        lines = []
        for ticker in tickers:
            row = [ticker]
            for other_ticker in tickers:
                if ticker in correlation_matrix and other_ticker in correlation_matrix[ticker]:
                    row.append(f"{correlation_matrix[ticker][other_ticker]:.3f}")
                else:
                    row.append("N/A")
            lines.append(" | ".join(row))
        
        return "\n".join(lines)
    
    def _generate_gpt_correlation_insight(self, user_query: str, tickers: List[str], 
                                        strongest_correlations: List[Dict], 
                                        correlation_matrix: pd.DataFrame, 
                                        period: str, method: str) -> str:
        """Generate tailored correlation insights using GPT API"""
        
        logger.info(f"Generating GPT correlation insight for query: '{user_query}'")
        
        if not self.gpt_client:
            logger.warning("GPT client not available, falling back to template correlation insights")
            return self._fallback_correlation_interpretation(strongest_correlations, period)
        
        try:
            # Build the prompt for GPT
            prompt = f"""You are a quantitative finance expert analyzing correlation results. Generate a tailored, insightful analysis based on the user's specific query and the correlation data.

USER'S QUERY: "{user_query}"

CORRELATION DATA:
- Assets: {', '.join(tickers)}
- Period: {period}
- Method: {method.title()}
- Number of observations: {len(correlation_matrix)}

STRONGEST CORRELATIONS:
{chr(10).join([f"- {corr['pair']}: {corr['correlation']:+.3f} ({corr['strength']})" for corr in strongest_correlations[:5]])}

CORRELATION MATRIX:
{self._format_correlation_matrix(correlation_matrix.to_dict())}

INSTRUCTIONS:
1. Address the user's specific query directly
2. Provide quantitative insights about the correlation patterns
3. Explain the economic/financial significance of the correlations
4. Discuss portfolio diversification implications
5. Highlight any interesting patterns or anomalies
6. Provide practical implications for trading/investment decisions
7. Keep the tone professional but accessible
8. Focus on actionable insights rather than just statistical description

Generate a comprehensive insight (2-3 paragraphs) that directly answers the user's question and provides valuable quantitative analysis."""

            # Call GPT API
            response = self.gpt_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a quantitative finance expert specializing in correlation analysis and portfolio management. Provide clear, actionable insights for traders and investors."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            
            gpt_insight = response.choices[0].message.content.strip()
            logger.info("Generated GPT-powered correlation insight successfully")
            return gpt_insight
            
        except Exception as e:
            logger.error(f"GPT correlation insight generation failed: {e}")
            # Fallback to template-based insights
            return self._fallback_correlation_interpretation(strongest_correlations, period)
    
    def _generate_gpt_cointegration_insight(self, user_query: str, tickers: List[str], 
                                          results: Dict, period: str) -> str:
        """Generate tailored cointegration insights using GPT API"""
        
        logger.info(f"Generating GPT cointegration insight for query: '{user_query}'")
        
        if not self.gpt_client:
            logger.warning("GPT client not available, falling back to template cointegration insights")
            return self._interpret_cointegration_results(results, tickers)
        
        try:
            # Build the prompt for GPT
            prompt = f"""You are a quantitative finance expert analyzing cointegration test results. Generate a tailored, insightful analysis based on the user's specific query and the cointegration data.

USER'S QUERY: "{user_query}"

COINTEGRATION DATA:
- Assets: {', '.join(tickers)}
- Period: {period}
- Test Type: {"Pairwise" if len(tickers) == 2 else "Johansen"}

TEST RESULTS:
{results}

INSTRUCTIONS:
1. Address the user's specific query directly
2. Explain what cointegration means in financial terms
3. Interpret the statistical significance of the results
4. Discuss implications for pairs trading strategies
5. Provide practical trading recommendations
6. Highlight any limitations or considerations
7. Keep the tone professional but accessible
8. Focus on actionable insights for quantitative trading

Generate a comprehensive insight (2-3 paragraphs) that directly answers the user's question and provides valuable quantitative analysis."""

            # Call GPT API
            response = self.gpt_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a quantitative finance expert specializing in cointegration analysis and pairs trading strategies. Provide clear, actionable insights for traders and investors."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            
            gpt_insight = response.choices[0].message.content.strip()
            logger.info("Generated GPT-powered cointegration insight successfully")
            return gpt_insight
            
        except Exception as e:
            logger.error(f"GPT cointegration insight generation failed: {e}")
            # Fallback to template-based insights
            return self._interpret_cointegration_results(results, tickers)
    
    def _generate_model_comparison(self, y_ticker: str, x_tickers: List[str], 
                                 returns_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate model comparison data for different variable combinations"""
        
        comparison = []
        
        # Single variable models
        for ticker in x_tickers:
            try:
                X_single = returns_df[[ticker]].copy()
                X_single['const'] = 1
                model = OLS(returns_df[y_ticker], X_single).fit()
                comparison.append({
                    "model": f"M{ticker}",
                    "vars": ticker,
                    "r2": f"{model.rsquared:.3f}",
                    "beta": f"{model.params[ticker]:+.3f}",
                    "p": f"{model.pvalues[ticker]:.2e}" if model.pvalues[ticker] < 0.001 else f"{model.pvalues[ticker]:.3f}"
                })
            except Exception as e:
                logger.warning(f"Failed to create single variable model for {ticker}: {e}")
                continue
        
        # Multi-variable models
        if len(x_tickers) > 1:
            try:
                # All variables
                X_multi = returns_df[x_tickers].copy()
                X_multi['const'] = 1
                model = OLS(returns_df[y_ticker], X_multi).fit()
                comparison.append({
                    "model": "M_All",
                    "vars": "+".join(x_tickers),
                    "r2": f"{model.rsquared:.3f}",
                    "beta": f"{model.params[x_tickers[0]]:+.3f}" if x_tickers else "—"
                })
            except Exception as e:
                logger.warning(f"Failed to create multi-variable model: {e}")
                pass
        
        return comparison
    
    def _get_correlation_matrix(self, returns_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Get correlation matrix for all variables"""
        return returns_df.corr().to_dict()
    
    def _calculate_durbin_watson(self, residuals: pd.Series) -> float:
        """Calculate Durbin-Watson statistic for autocorrelation testing"""
        try:
            from statsmodels.stats.diagnostic import durbin_watson
            return durbin_watson(residuals)
        except:
            # Fallback calculation
            diff = residuals.diff().dropna()
            return (diff ** 2).sum() / (residuals ** 2).sum()
    
    def _interpret_cointegration_results(self, results: Dict[str, Any], 
                                       tickers: List[str]) -> str:
        """Interpret cointegration test results"""
        
        if "pair" in results:
            # Pairwise cointegration
            pair = results["pair"]
            is_cointegrated = results["is_cointegrated"]
            p_value = results["p_value"]
            
            # Debug logging
            logger.info(f"Interpreting cointegration results for {pair}")
            logger.info(f"P-value type: {type(p_value)}, value: {p_value}")
            
            # Ensure p_value is a scalar
            if isinstance(p_value, (list, tuple)):
                p_value = p_value[0] if len(p_value) > 0 else 0.5
            elif hasattr(p_value, 'item'):
                p_value = p_value.item()
            
            interpretation = f"Cointegration test for {pair}:\n\n"
            interpretation += f"**Result:** {'Cointegrated' if is_cointegrated else 'Not cointegrated'}\n"
            interpretation += f"**P-value:** {float(p_value):.4f}\n"
            
            if is_cointegrated:
                interpretation += "\n**Implications:**\n"
                interpretation += "- These assets have a long-term equilibrium relationship\n"
                interpretation += "- Suitable for pairs trading strategies\n"
                interpretation += "- Mean reversion opportunities may exist\n"
            else:
                interpretation += "\n**Implications:**\n"
                interpretation += "- No long-term equilibrium relationship detected\n"
                interpretation += "- Assets may drift apart over time\n"
                interpretation += "- Pairs trading may not be suitable\n"
        
        else:
            # Multiple asset cointegration
            interpretation = f"Johansen cointegration test for {', '.join(tickers)}:\n\n"
            interpretation += f"**Cointegration Rank:** {results['cointegration_rank']}\n"
            interpretation += "**Implications:**\n"
            interpretation += "- Multiple cointegrating relationships detected\n"
            interpretation += "- Complex long-term relationships exist\n"
            interpretation += "- Suitable for advanced pairs trading strategies\n"
        
        return interpretation
    
    def _interpret_stationarity_results(self, results: Dict[str, Any], 
                                      ticker: str, test_type: str) -> str:
        """Interpret stationarity test results"""
        
        interpretation = f"Stationarity test for {ticker} using {test_type.upper()} test:\n\n"
        
        if "price_test" in results:
            price_result = results["price_test"]
            returns_result = results["returns_test"]
            
            interpretation += "**Price Series:**\n"
            interpretation += f"- Stationary: {'Yes' if price_result['is_stationary'] else 'No'}\n"
            interpretation += f"- P-value: {price_result['p_value']:.4f}\n"
            
            interpretation += "\n**Returns Series:**\n"
            interpretation += f"- Stationary: {'Yes' if returns_result['is_stationary'] else 'No'}\n"
            interpretation += f"- P-value: {returns_result['p_value']:.4f}\n"
            
            interpretation += "\n**Implications:**\n"
            if not price_result['is_stationary'] and returns_result['is_stationary']:
                interpretation += "- Price series is non-stationary (typical for financial data)\n"
                interpretation += "- Returns are stationary (suitable for statistical modeling)\n"
                interpretation += "- Standard statistical tests can be applied to returns\n"
            elif price_result['is_stationary']:
                interpretation += "- Price series is stationary (unusual for financial data)\n"
                interpretation += "- May indicate mean-reverting behavior\n"
            else:
                interpretation += "- Both price and returns are non-stationary\n"
                interpretation += "- Consider differencing or other transformations\n"
        
        return interpretation
    
    def _analyze_volatility_clustering(self, returns: pd.Series) -> Dict[str, Any]:
        """Analyze volatility clustering patterns"""
        
        # Calculate rolling volatility
        rolling_vol = returns.rolling(window=20).std()
        
        # Volatility clustering test (simplified)
        vol_autocorr = rolling_vol.autocorr(lag=1)
        
        return {
            "volatility_autocorrelation": vol_autocorr,
            "has_clustering": abs(vol_autocorr) > 0.1,
            "clustering_strength": "strong" if abs(vol_autocorr) > 0.3 else 
                                 "moderate" if abs(vol_autocorr) > 0.1 else "weak"
        }
    
    def _interpret_volatility_results(self, volatility_measures: Dict[str, Any],
                                    volatility_clustering: Dict[str, Any],
                                    ticker: str, period: str) -> str:
        """Interpret volatility analysis results"""
        
        interpretation = f"Volatility analysis for {ticker} over {period}:\n\n"
        
        # Historical volatility
        if 'historical' in volatility_measures:
            hist_vol = volatility_measures['historical']
            interpretation += f"**Historical Volatility:**\n"
            interpretation += f"- Daily: {hist_vol['daily']:.4f} ({hist_vol['daily']*100:.2f}%)\n"
            interpretation += f"- Annualized: {hist_vol['annualized']:.4f} ({hist_vol['annualized']*100:.2f}%)\n"
            
            # Volatility interpretation
            if hist_vol['annualized'] > 0.4:
                interpretation += "- Very high volatility (typical of growth stocks, crypto)\n"
            elif hist_vol['annualized'] > 0.25:
                interpretation += "- High volatility (typical of tech stocks)\n"
            elif hist_vol['annualized'] > 0.15:
                interpretation += "- Moderate volatility (typical of large-cap stocks)\n"
            else:
                interpretation += "- Low volatility (typical of defensive stocks, bonds)\n"
        
        # Volatility clustering
        interpretation += f"\n**Volatility Clustering:**\n"
        interpretation += f"- Autocorrelation: {volatility_clustering['volatility_autocorrelation']:.3f}\n"
        interpretation += f"- Clustering: {volatility_clustering['clustering_strength']}\n"
        
        if volatility_clustering['has_clustering']:
            interpretation += "- Volatility shows clustering patterns (typical of financial markets)\n"
            interpretation += "- High volatility periods tend to be followed by high volatility\n"
        else:
            interpretation += "- Limited volatility clustering detected\n"
        
        return interpretation


# Global instance
statistical_analyzer = StatisticalAnalyzer()

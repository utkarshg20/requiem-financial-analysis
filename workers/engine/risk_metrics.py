"""
Advanced Risk Metrics Engine for Quantitative Finance
Calculates sophisticated risk measures with intelligent interpretation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
from scipy.optimize import minimize
import logging
from datetime import datetime, timedelta

logger = logging.getLogger("requiem.risk_metrics")

class RiskMetrics:
    """Advanced risk metrics calculations for portfolio analysis"""
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate  # Annual risk-free rate
    
    def var_calculation(self, returns: pd.Series, confidence: float = 0.95, 
                       method: str = "historical") -> Dict[str, Any]:
        """
        Value at Risk (VaR) calculation using multiple methods
        
        Args:
            returns: Series of returns
            confidence: Confidence level (0.95 for 95% VaR)
            method: Calculation method (historical, parametric, monte_carlo)
        """
        try:
            returns_clean = returns.dropna()
            
            if len(returns_clean) < 30:
                return {"error": "Insufficient data for VaR calculation (minimum 30 observations)"}
            
            results = {}
            
            if method == "historical":
                # Historical simulation
                var_historical = np.percentile(returns_clean, (1 - confidence) * 100)
                results["historical"] = {
                    "var": var_historical,
                    "var_percentage": var_historical * 100,
                    "method": "historical_simulation"
                }
            
            elif method == "parametric":
                # Parametric (normal distribution assumption)
                mean_return = returns_clean.mean()
                std_return = returns_clean.std()
                z_score = stats.norm.ppf(1 - confidence)
                var_parametric = mean_return + z_score * std_return
                
                results["parametric"] = {
                    "var": var_parametric,
                    "var_percentage": var_parametric * 100,
                    "mean_return": mean_return,
                    "std_return": std_return,
                    "z_score": z_score,
                    "method": "parametric_normal"
                }
            
            elif method == "monte_carlo":
                # Monte Carlo simulation
                mean_return = returns_clean.mean()
                std_return = returns_clean.std()
                n_simulations = 10000
                
                simulated_returns = np.random.normal(mean_return, std_return, n_simulations)
                var_monte_carlo = np.percentile(simulated_returns, (1 - confidence) * 100)
                
                results["monte_carlo"] = {
                    "var": var_monte_carlo,
                    "var_percentage": var_monte_carlo * 100,
                    "simulations": n_simulations,
                    "method": "monte_carlo_simulation"
                }
            
            # Calculate all methods for comparison
            if method == "all":
                # Historical
                var_historical = np.percentile(returns_clean, (1 - confidence) * 100)
                results["historical"] = {
                    "var": var_historical,
                    "var_percentage": var_historical * 100,
                    "method": "historical_simulation"
                }
                
                # Parametric
                mean_return = returns_clean.mean()
                std_return = returns_clean.std()
                z_score = stats.norm.ppf(1 - confidence)
                var_parametric = mean_return + z_score * std_return
                
                results["parametric"] = {
                    "var": var_parametric,
                    "var_percentage": var_parametric * 100,
                    "mean_return": mean_return,
                    "std_return": std_return,
                    "z_score": z_score,
                    "method": "parametric_normal"
                }
                
                # Monte Carlo
                n_simulations = 10000
                simulated_returns = np.random.normal(mean_return, std_return, n_simulations)
                var_monte_carlo = np.percentile(simulated_returns, (1 - confidence) * 100)
                
                results["monte_carlo"] = {
                    "var": var_monte_carlo,
                    "var_percentage": var_monte_carlo * 100,
                    "simulations": n_simulations,
                    "method": "monte_carlo_simulation"
                }
            
            # Generate interpretation
            interpretation = self._interpret_var_results(results, confidence, len(returns_clean))
            
            return {
                "analysis_type": "var_calculation",
                "confidence_level": confidence,
                "method": method,
                "results": results,
                "interpretation": interpretation,
                "data_points": len(returns_clean)
            }
            
        except Exception as e:
            logger.error(f"VaR calculation error: {e}")
            return {"error": f"VaR calculation failed: {str(e)}"}
    
    def cvar_calculation(self, returns: pd.Series, confidence: float = 0.95) -> Dict[str, Any]:
        """
        Conditional Value at Risk (CVaR) / Expected Shortfall calculation
        
        Args:
            returns: Series of returns
            confidence: Confidence level (0.95 for 95% CVaR)
        """
        try:
            returns_clean = returns.dropna()
            
            if len(returns_clean) < 30:
                return {"error": "Insufficient data for CVaR calculation (minimum 30 observations)"}
            
            # Calculate VaR first
            var_threshold = np.percentile(returns_clean, (1 - confidence) * 100)
            
            # Calculate CVaR (expected value of returns below VaR threshold)
            tail_returns = returns_clean[returns_clean <= var_threshold]
            cvar = tail_returns.mean()
            
            # Additional tail statistics
            tail_probability = len(tail_returns) / len(returns_clean)
            tail_volatility = tail_returns.std()
            
            # Generate interpretation
            interpretation = self._interpret_cvar_results(
                var_threshold, cvar, tail_probability, tail_volatility, confidence
            )
            
            return {
                "analysis_type": "cvar_calculation",
                "confidence_level": confidence,
                "var_threshold": var_threshold,
                "cvar": cvar,
                "cvar_percentage": cvar * 100,
                "tail_probability": tail_probability,
                "tail_volatility": tail_volatility,
                "tail_observations": len(tail_returns),
                "interpretation": interpretation,
                "data_points": len(returns_clean)
            }
            
        except Exception as e:
            logger.error(f"CVaR calculation error: {e}")
            return {"error": f"CVaR calculation failed: {str(e)}"}
    
    def maximum_drawdown(self, returns: pd.Series) -> Dict[str, Any]:
        """
        Maximum drawdown calculation with detailed analysis
        
        Args:
            returns: Series of returns
        """
        try:
            returns_clean = returns.dropna()
            
            if len(returns_clean) < 2:
                return {"error": "Insufficient data for drawdown calculation"}
            
            # Calculate cumulative returns
            cumulative_returns = (1 + returns_clean).cumprod()
            
            # Calculate running maximum
            running_max = cumulative_returns.expanding().max()
            
            # Calculate drawdown
            drawdown = (cumulative_returns - running_max) / running_max
            
            # Maximum drawdown
            max_drawdown = drawdown.min()
            max_drawdown_date = drawdown.index[drawdown.argmin()]
            # Convert to datetime if it's not already
            if hasattr(max_drawdown_date, 'strftime'):
                pass  # Already a datetime
            else:
                max_drawdown_date = pd.to_datetime(max_drawdown_date)
            
            # Find drawdown periods
            drawdown_periods = self._find_drawdown_periods(drawdown)
            
            # Recovery analysis
            recovery_analysis = self._analyze_recovery_times(cumulative_returns, drawdown_periods)
            
            # Generate interpretation
            interpretation = self._interpret_drawdown_results(
                max_drawdown, max_drawdown_date, drawdown_periods, recovery_analysis
            )
            
            return {
                "analysis_type": "maximum_drawdown",
                "max_drawdown": max_drawdown,
                "max_drawdown_percentage": max_drawdown * 100,
                "max_drawdown_date": max_drawdown_date,
                "drawdown_periods": drawdown_periods,
                "recovery_analysis": recovery_analysis,
                "interpretation": interpretation,
                "data_points": len(returns_clean)
            }
            
        except Exception as e:
            logger.error(f"Maximum drawdown calculation error: {e}")
            return {"error": f"Maximum drawdown calculation failed: {str(e)}"}
    
    def sharpe_ratio(self, returns: pd.Series, risk_free_rate: Optional[float] = None) -> Dict[str, Any]:
        """
        Sharpe ratio calculation with interpretation
        
        Args:
            returns: Series of returns
            risk_free_rate: Risk-free rate (uses instance default if None)
        """
        try:
            returns_clean = returns.dropna()
            
            if len(returns_clean) < 2:
                return {"error": "Insufficient data for Sharpe ratio calculation"}
            
            rf_rate = risk_free_rate if risk_free_rate is not None else self.risk_free_rate
            
            # Calculate excess returns
            excess_returns = returns_clean - (rf_rate / 252)  # Daily risk-free rate
            
            # Sharpe ratio
            mean_excess_return = excess_returns.mean()
            std_excess_return = excess_returns.std()
            sharpe_ratio = mean_excess_return / std_excess_return if std_excess_return != 0 else 0
            
            # Annualized Sharpe ratio
            annualized_sharpe = sharpe_ratio * np.sqrt(252)
            
            # Additional metrics
            mean_return = returns_clean.mean()
            volatility = returns_clean.std()
            annualized_return = mean_return * 252
            annualized_volatility = volatility * np.sqrt(252)
            
            # Generate interpretation
            interpretation = self._interpret_sharpe_results(
                annualized_sharpe, annualized_return, annualized_volatility, rf_rate
            )
            
            return {
                "analysis_type": "sharpe_ratio",
                "sharpe_ratio": sharpe_ratio,
                "annualized_sharpe_ratio": annualized_sharpe,
                "mean_return": mean_return,
                "annualized_return": annualized_return,
                "volatility": volatility,
                "annualized_volatility": annualized_volatility,
                "risk_free_rate": rf_rate,
                "excess_return": mean_excess_return,
                "interpretation": interpretation,
                "data_points": len(returns_clean)
            }
            
        except Exception as e:
            logger.error(f"Sharpe ratio calculation error: {e}")
            return {"error": f"Sharpe ratio calculation failed: {str(e)}"}
    
    def sortino_ratio(self, returns: pd.Series, risk_free_rate: Optional[float] = None,
                     target_return: float = 0.0) -> Dict[str, Any]:
        """
        Sortino ratio calculation (downside deviation focus)
        
        Args:
            returns: Series of returns
            risk_free_rate: Risk-free rate
            target_return: Target return (default 0)
        """
        try:
            returns_clean = returns.dropna()
            
            if len(returns_clean) < 2:
                return {"error": "Insufficient data for Sortino ratio calculation"}
            
            rf_rate = risk_free_rate if risk_free_rate is not None else self.risk_free_rate
            
            # Calculate excess returns
            excess_returns = returns_clean - (rf_rate / 252)
            
            # Calculate downside deviation
            downside_returns = excess_returns[excess_returns < target_return]
            downside_deviation = np.sqrt(np.mean(downside_returns**2)) if len(downside_returns) > 0 else 0
            
            # Sortino ratio
            mean_excess_return = excess_returns.mean()
            sortino_ratio = mean_excess_return / downside_deviation if downside_deviation != 0 else 0
            
            # Annualized Sortino ratio
            annualized_sortino = sortino_ratio * np.sqrt(252)
            
            # Additional metrics
            downside_frequency = len(downside_returns) / len(returns_clean)
            mean_downside = downside_returns.mean() if len(downside_returns) > 0 else 0
            
            # Generate interpretation
            interpretation = self._interpret_sortino_results(
                annualized_sortino, mean_excess_return, downside_deviation, 
                downside_frequency, target_return
            )
            
            return {
                "analysis_type": "sortino_ratio",
                "sortino_ratio": sortino_ratio,
                "annualized_sortino_ratio": annualized_sortino,
                "downside_deviation": downside_deviation,
                "downside_frequency": downside_frequency,
                "mean_downside": mean_downside,
                "target_return": target_return,
                "risk_free_rate": rf_rate,
                "interpretation": interpretation,
                "data_points": len(returns_clean)
            }
            
        except Exception as e:
            logger.error(f"Sortino ratio calculation error: {e}")
            return {"error": f"Sortino ratio calculation failed: {str(e)}"}
    
    def calmar_ratio(self, returns: pd.Series) -> Dict[str, Any]:
        """
        Calmar ratio calculation (annual return / maximum drawdown)
        
        Args:
            returns: Series of returns
        """
        try:
            returns_clean = returns.dropna()
            
            if len(returns_clean) < 2:
                return {"error": "Insufficient data for Calmar ratio calculation"}
            
            # Calculate annualized return
            mean_return = returns_clean.mean()
            annualized_return = mean_return * 252
            
            # Calculate maximum drawdown
            cumulative_returns = (1 + returns_clean).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = abs(drawdown.min())
            
            # Calmar ratio
            calmar_ratio = annualized_return / max_drawdown if max_drawdown != 0 else 0
            
            # Generate interpretation
            interpretation = self._interpret_calmar_results(
                calmar_ratio, annualized_return, max_drawdown
            )
            
            return {
                "analysis_type": "calmar_ratio",
                "calmar_ratio": calmar_ratio,
                "annualized_return": annualized_return,
                "max_drawdown": max_drawdown,
                "max_drawdown_percentage": max_drawdown * 100,
                "interpretation": interpretation,
                "data_points": len(returns_clean)
            }
            
        except Exception as e:
            logger.error(f"Calmar ratio calculation error: {e}")
            return {"error": f"Calmar ratio calculation failed: {str(e)}"}
    
    def portfolio_risk_metrics(self, returns_matrix: pd.DataFrame, 
                             weights: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Portfolio-level risk metrics calculation
        
        Args:
            returns_matrix: DataFrame of returns for multiple assets
            weights: Portfolio weights (equal weights if None)
        """
        try:
            returns_clean = returns_matrix.dropna()
            
            if len(returns_clean) < 2:
                return {"error": "Insufficient data for portfolio risk calculation"}
            
            n_assets = len(returns_clean.columns)
            
            # Default to equal weights if not provided
            if weights is None:
                weights = [1.0 / n_assets] * n_assets
            
            if len(weights) != n_assets:
                return {"error": f"Weights length ({len(weights)}) must match number of assets ({n_assets})"}
            
            # Calculate portfolio returns
            portfolio_returns = (returns_clean * weights).sum(axis=1)
            
            # Calculate portfolio metrics
            portfolio_mean = portfolio_returns.mean()
            portfolio_volatility = portfolio_returns.std()
            
            # Calculate correlation matrix
            correlation_matrix = returns_clean.corr()
            
            # Calculate portfolio VaR
            portfolio_var = np.percentile(portfolio_returns, 5)  # 95% VaR
            
            # Calculate portfolio Sharpe ratio
            excess_returns = portfolio_returns - (self.risk_free_rate / 252)
            portfolio_sharpe = excess_returns.mean() / portfolio_volatility if portfolio_volatility != 0 else 0
            
            # Calculate diversification ratio
            weighted_vol = np.sqrt(np.sum([w**2 * returns_clean[col].var() for w, col in zip(weights, returns_clean.columns)]))
            diversification_ratio = weighted_vol / portfolio_volatility if portfolio_volatility != 0 else 0
            
            # Generate interpretation
            interpretation = self._interpret_portfolio_risk_results(
                portfolio_mean, portfolio_volatility, portfolio_var, 
                portfolio_sharpe, diversification_ratio, correlation_matrix
            )
            
            return {
                "analysis_type": "portfolio_risk_metrics",
                "portfolio_returns": portfolio_returns.to_dict(),
                "portfolio_mean": portfolio_mean,
                "portfolio_volatility": portfolio_volatility,
                "portfolio_var": portfolio_var,
                "portfolio_sharpe": portfolio_sharpe,
                "diversification_ratio": diversification_ratio,
                "weights": dict(zip(returns_clean.columns, weights)),
                "correlation_matrix": correlation_matrix.to_dict(),
                "interpretation": interpretation,
                "data_points": len(returns_clean)
            }
            
        except Exception as e:
            logger.error(f"Portfolio risk calculation error: {e}")
            return {"error": f"Portfolio risk calculation failed: {str(e)}"}
    
    # Helper methods for analysis and interpretation
    def _find_drawdown_periods(self, drawdown: pd.Series) -> List[Dict[str, Any]]:
        """Find all drawdown periods"""
        periods = []
        in_drawdown = False
        start_date = None
        
        for date, dd in drawdown.items():
            if dd < -0.001 and not in_drawdown:  # Start of drawdown
                in_drawdown = True
                start_date = date
            elif dd >= -0.001 and in_drawdown:  # End of drawdown
                in_drawdown = False
                periods.append({
                    "start_date": start_date,
                    "end_date": date,
                    "duration": self._safe_days_calculation(date - start_date),
                    "max_drawdown": drawdown.loc[start_date:date].min()
                })
        
        # Handle case where drawdown period extends to end of data
        if in_drawdown:
            periods.append({
                "start_date": start_date,
                "end_date": drawdown.index[-1],
                "duration": self._safe_days_calculation(drawdown.index[-1] - start_date),
                "max_drawdown": drawdown.loc[start_date:].min()
            })
        
        return periods
    
    def _safe_days_calculation(self, time_delta) -> int:
        """Safely calculate days from a time delta object"""
        try:
            return time_delta.days
        except AttributeError:
            try:
                return int(time_delta.total_seconds() / 86400)
            except AttributeError:
                return 0
    
    def _analyze_recovery_times(self, cumulative_returns: pd.Series, 
                              drawdown_periods: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze recovery times from drawdowns"""
        if not drawdown_periods:
            return {"average_recovery_time": 0, "max_recovery_time": 0}
        
        recovery_times = []
        
        for period in drawdown_periods:
            start_date = period["start_date"]
            end_date = period["end_date"]
            
            # Find when cumulative returns recover to pre-drawdown level
            pre_drawdown_level = cumulative_returns.loc[start_date]
            recovery_mask = cumulative_returns.loc[end_date:] >= pre_drawdown_level
            
            if recovery_mask.any():
                # Find the first True value in the mask
                recovery_indices = recovery_mask[recovery_mask].index
                if len(recovery_indices) > 0:
                    recovery_date = recovery_indices[0]
                    recovery_time = self._safe_days_calculation(recovery_date - end_date)
                    recovery_times.append(recovery_time)
        
        return {
            "average_recovery_time": np.mean(recovery_times) if recovery_times else 0,
            "max_recovery_time": max(recovery_times) if recovery_times else 0,
            "recovery_times": recovery_times
        }
    
    def _interpret_var_results(self, results: Dict[str, Any], confidence: float, 
                             data_points: int) -> str:
        """Generate intelligent VaR interpretation"""
        
        confidence_pct = confidence * 100
        
        interpretation = f"Value at Risk (VaR) Analysis at {confidence_pct}% confidence level:\n\n"
        
        for method, result in results.items():
            var_pct = result["var_percentage"]
            interpretation += f"**{method.title()} Method:**\n"
            interpretation += f"- VaR: {var_pct:.2f}% daily loss\n"
            
            if var_pct < -5:
                interpretation += "- Very high risk (typical of volatile assets)\n"
            elif var_pct < -2:
                interpretation += "- High risk (typical of growth stocks)\n"
            elif var_pct < -1:
                interpretation += "- Moderate risk (typical of large-cap stocks)\n"
            else:
                interpretation += "- Low risk (typical of defensive assets)\n"
            
            interpretation += "\n"
        
        interpretation += f"**Interpretation:**\n"
        interpretation += f"- Based on {data_points} observations\n"
        interpretation += f"- {confidence_pct}% confidence means this loss level should not be exceeded more than {(1-confidence)*100:.1f}% of the time\n"
        interpretation += f"- VaR is a measure of potential loss, not guaranteed loss\n"
        
        return interpretation
    
    def _interpret_cvar_results(self, var_threshold: float, cvar: float, 
                              tail_probability: float, tail_volatility: float,
                              confidence: float) -> str:
        """Generate intelligent CVaR interpretation"""
        
        confidence_pct = confidence * 100
        
        interpretation = f"Conditional Value at Risk (CVaR) Analysis at {confidence_pct}% confidence:\n\n"
        interpretation += f"**VaR Threshold:** {var_threshold*100:.2f}%\n"
        interpretation += f"**CVaR (Expected Shortfall):** {cvar*100:.2f}%\n"
        interpretation += f"**Tail Probability:** {tail_probability*100:.1f}%\n"
        interpretation += f"**Tail Volatility:** {tail_volatility*100:.2f}%\n\n"
        
        interpretation += f"**Interpretation:**\n"
        interpretation += f"- CVaR represents the expected loss when losses exceed the VaR threshold\n"
        interpretation += f"- {confidence_pct}% of the time, losses should not exceed {var_threshold*100:.2f}%\n"
        interpretation += f"- When losses do exceed this threshold, the expected loss is {cvar*100:.2f}%\n"
        
        if cvar < var_threshold * 1.5:
            interpretation += "- Tail risk is relatively contained\n"
        else:
            interpretation += "- Significant tail risk exists (CVaR much worse than VaR)\n"
        
        return interpretation
    
    def _interpret_drawdown_results(self, max_drawdown: float, max_drawdown_date,
                                  drawdown_periods: List[Dict[str, Any]], 
                                  recovery_analysis: Dict[str, Any]) -> str:
        """Generate intelligent drawdown interpretation"""
        
        interpretation = f"Maximum Drawdown Analysis:\n\n"
        interpretation += f"**Maximum Drawdown:** {max_drawdown*100:.2f}%\n"
        interpretation += f"**Date of Maximum Drawdown:** {max_drawdown_date.strftime('%Y-%m-%d')}\n"
        interpretation += f"**Number of Drawdown Periods:** {len(drawdown_periods)}\n"
        interpretation += f"**Average Recovery Time:** {recovery_analysis['average_recovery_time']:.0f} days\n"
        interpretation += f"**Maximum Recovery Time:** {recovery_analysis['max_recovery_time']:.0f} days\n\n"
        
        interpretation += f"**Interpretation:**\n"
        
        if max_drawdown > -0.2:
            interpretation += "- Relatively low maximum drawdown (good risk control)\n"
        elif max_drawdown > -0.4:
            interpretation += "- Moderate maximum drawdown (typical for equity strategies)\n"
        else:
            interpretation += "- High maximum drawdown (high-risk strategy)\n"
        
        if len(drawdown_periods) > 10:
            interpretation += "- Frequent drawdown periods (volatile performance)\n"
        elif len(drawdown_periods) > 5:
            interpretation += "- Moderate number of drawdown periods\n"
        else:
            interpretation += "- Few drawdown periods (stable performance)\n"
        
        if recovery_analysis['average_recovery_time'] > 100:
            interpretation += "- Long recovery times (persistent losses)\n"
        elif recovery_analysis['average_recovery_time'] > 30:
            interpretation += "- Moderate recovery times\n"
        else:
            interpretation += "- Quick recovery times (resilient strategy)\n"
        
        return interpretation
    
    def _interpret_sharpe_results(self, annualized_sharpe: float, annualized_return: float,
                                annualized_volatility: float, risk_free_rate: float) -> str:
        """Generate intelligent Sharpe ratio interpretation"""
        
        interpretation = f"Sharpe Ratio Analysis:\n\n"
        interpretation += f"**Annualized Sharpe Ratio:** {annualized_sharpe:.2f}\n"
        interpretation += f"**Annualized Return:** {annualized_return*100:.2f}%\n"
        interpretation += f"**Annualized Volatility:** {annualized_volatility*100:.2f}%\n"
        interpretation += f"**Risk-Free Rate:** {risk_free_rate*100:.2f}%\n\n"
        
        interpretation += f"**Interpretation:**\n"
        interpretation += f"- Sharpe ratio measures excess return per unit of risk\n"
        interpretation += f"- Note: Sharpe ratio is more commonly used for strategies/portfolios than individual stocks\n"
        
        if annualized_sharpe > 2.0:
            interpretation += f"- Excellent risk-adjusted performance (Sharpe > 2.0)\n"
        elif annualized_sharpe > 1.0:
            interpretation += f"- Good risk-adjusted performance (Sharpe > 1.0)\n"
        elif annualized_sharpe > 0.5:
            interpretation += f"- Acceptable risk-adjusted performance (Sharpe > 0.5)\n"
        elif annualized_sharpe > 0:
            interpretation += f"- Positive but low risk-adjusted performance\n"
        else:
            interpretation += "- Poor risk-adjusted returns (negative Sharpe)\n"
        
        interpretation += f"- Higher values indicate better risk-adjusted performance\n"
        
        return interpretation
    
    def _interpret_sortino_results(self, annualized_sortino: float, mean_excess_return: float,
                                 downside_deviation: float, downside_frequency: float,
                                 target_return: float) -> str:
        """Generate intelligent Sortino ratio interpretation"""
        
        interpretation = f"Sortino Ratio Analysis:\n\n"
        interpretation += f"**Annualized Sortino Ratio:** {annualized_sortino:.2f}\n"
        interpretation += f"**Mean Excess Return:** {mean_excess_return*100:.2f}%\n"
        interpretation += f"**Downside Deviation:** {downside_deviation*100:.2f}%\n"
        interpretation += f"**Downside Frequency:** {downside_frequency*100:.1f}%\n"
        interpretation += f"**Target Return:** {target_return*100:.2f}%\n\n"
        
        interpretation += f"**Interpretation:**\n"
        
        if annualized_sortino > 2.0:
            interpretation += "- Excellent downside risk-adjusted returns\n"
        elif annualized_sortino > 1.0:
            interpretation += "- Good downside risk-adjusted returns\n"
        elif annualized_sortino > 0.5:
            interpretation += "- Acceptable downside risk-adjusted returns\n"
        else:
            interpretation += "- Poor downside risk-adjusted returns\n"
        
        interpretation += f"- Sortino ratio focuses on downside risk (volatility of negative returns)\n"
        interpretation += f"- More relevant than Sharpe ratio for asymmetric return distributions\n"
        interpretation += f"- {downside_frequency*100:.1f}% of returns were below target\n"
        
        return interpretation
    
    def _interpret_calmar_results(self, calmar_ratio: float, annualized_return: float,
                                max_drawdown: float) -> str:
        """Generate intelligent Calmar ratio interpretation"""
        
        interpretation = f"Calmar Ratio Analysis:\n\n"
        interpretation += f"**Calmar Ratio:** {calmar_ratio:.2f}\n"
        interpretation += f"**Annualized Return:** {annualized_return*100:.2f}%\n"
        interpretation += f"**Maximum Drawdown:** {max_drawdown*100:.2f}%\n\n"
        
        interpretation += f"**Interpretation:**\n"
        
        if calmar_ratio > 1.0:
            interpretation += "- Excellent risk-adjusted returns relative to maximum drawdown\n"
        elif calmar_ratio > 0.5:
            interpretation += "- Good risk-adjusted returns relative to maximum drawdown\n"
        elif calmar_ratio > 0.2:
            interpretation += "- Acceptable risk-adjusted returns relative to maximum drawdown\n"
        else:
            interpretation += "- Poor risk-adjusted returns relative to maximum drawdown\n"
        
        interpretation += f"- Calmar ratio measures annual return per unit of maximum drawdown\n"
        interpretation += f"- Higher values indicate better performance relative to worst-case losses\n"
        interpretation += f"- Particularly relevant for strategies with significant drawdown periods\n"
        
        return interpretation
    
    def _interpret_portfolio_risk_results(self, portfolio_mean: float, portfolio_volatility: float,
                                        portfolio_var: float, portfolio_sharpe: float,
                                        diversification_ratio: float, 
                                        correlation_matrix: pd.DataFrame) -> str:
        """Generate intelligent portfolio risk interpretation"""
        
        interpretation = f"Portfolio Risk Analysis:\n\n"
        interpretation += f"**Portfolio Return:** {portfolio_mean*252*100:.2f}% (annualized)\n"
        interpretation += f"**Portfolio Volatility:** {portfolio_volatility*np.sqrt(252)*100:.2f}% (annualized)\n"
        interpretation += f"**Portfolio VaR (95%):** {portfolio_var*100:.2f}%\n"
        interpretation += f"**Portfolio Sharpe Ratio:** {portfolio_sharpe*np.sqrt(252):.2f}\n"
        interpretation += f"**Diversification Ratio:** {diversification_ratio:.2f}\n\n"
        
        interpretation += f"**Interpretation:**\n"
        
        # Diversification analysis
        if diversification_ratio > 1.5:
            interpretation += "- Excellent diversification benefits (ratio > 1.5)\n"
        elif diversification_ratio > 1.2:
            interpretation += "- Good diversification benefits (ratio > 1.2)\n"
        elif diversification_ratio > 1.0:
            interpretation += "- Some diversification benefits\n"
        else:
            interpretation += "- Limited diversification benefits\n"
        
        # Correlation analysis
        avg_correlation = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean()
        interpretation += f"- Average correlation: {avg_correlation:.3f}\n"
        
        if avg_correlation < 0.3:
            interpretation += "- Low correlations provide good diversification\n"
        elif avg_correlation < 0.7:
            interpretation += "- Moderate correlations provide some diversification\n"
        else:
            interpretation += "- High correlations limit diversification benefits\n"
        
        return interpretation


# Global instance
risk_metrics = RiskMetrics()

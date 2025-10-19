"""
Advanced Mathematical Engine for Quantitative Finance
Handles sophisticated mathematical calculations and modeling
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from scipy import stats, optimize
from scipy.stats import norm, t
from scipy.optimize import minimize, differential_evolution
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger("requiem.math_engine")

class MathEngine:
    """Advanced mathematical calculations for quantitative finance"""
    
    def __init__(self):
        self.default_iterations = 10000
        self.default_tolerance = 1e-6
    
    def monte_carlo_simulation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Monte Carlo simulation for various financial models
        
        Args:
            params: Dictionary containing simulation parameters
                - model_type: 'geometric_brownian', 'jump_diffusion', 'heston', 'portfolio'
                - ticker: Stock ticker symbol (optional, for real data)
                - initial_value: Starting value
                - drift: Drift parameter
                - volatility: Volatility parameter
                - time_horizon: Time horizon in years
                - n_simulations: Number of simulations
                - n_steps: Number of time steps
        """
        try:
            model_type = params.get('model_type', 'geometric_brownian')
            ticker = params.get('ticker', None)
            initial_value = params.get('initial_value', 100.0)
            drift = params.get('drift', 0.05)
            volatility = params.get('volatility', 0.2)
            time_horizon = params.get('time_horizon', 1.0)
            n_simulations = params.get('n_simulations', self.default_iterations)
            n_steps = params.get('n_steps', 252)
            
            # If ticker is provided, use real stock data
            if ticker:
                try:
                    from ..adapters.prices_polygon import get_prices_agg
                    # Get 1 year of historical data
                    end_date = "2024-10-13"
                    start_date = "2023-10-13"
                    df = get_prices_agg(ticker, start_date, end_date)
                    
                    if not df.empty:
                        returns = df['close'].pct_change().dropna()
                        initial_value = df['close'].iloc[-1]  # Use latest price
                        drift = returns.mean() * 252  # Annualized drift
                        volatility = returns.std() * np.sqrt(252)  # Annualized volatility
                        logger.info(f"Using real data for {ticker}: drift={drift:.4f}, volatility={volatility:.4f}")
                except Exception as e:
                    logger.warning(f"Failed to fetch real data for {ticker}: {e}")
                    # Fall back to default parameters
            
            dt = time_horizon / n_steps
            
            if model_type == 'geometric_brownian':
                results = self._geometric_brownian_motion(
                    initial_value, drift, volatility, dt, n_steps, n_simulations
                )
            elif model_type == 'jump_diffusion':
                jump_intensity = params.get('jump_intensity', 0.1)
                jump_mean = params.get('jump_mean', 0.0)
                jump_std = params.get('jump_std', 0.1)
                results = self._jump_diffusion_model(
                    initial_value, drift, volatility, jump_intensity, 
                    jump_mean, jump_std, dt, n_steps, n_simulations
                )
            elif model_type == 'heston':
                kappa = params.get('kappa', 2.0)
                theta = params.get('theta', 0.04)
                sigma_v = params.get('sigma_v', 0.3)
                rho = params.get('rho', -0.7)
                results = self._heston_model(
                    initial_value, drift, kappa, theta, volatility, 
                    sigma_v, rho, dt, n_steps, n_simulations
                )
            elif model_type == 'portfolio':
                weights = params.get('weights', [0.5, 0.5])
                correlations = params.get('correlations', [[1.0, 0.3], [0.3, 1.0]])
                results = self._portfolio_simulation(
                    initial_value, weights, correlations, drift, volatility, 
                    dt, n_steps, n_simulations
                )
            else:
                return {"error": f"Unknown model type: {model_type}"}
            
            # Generate interpretation
            interpretation = self._interpret_monte_carlo_results(results, model_type, params)
            
            return {
                "analysis_type": "monte_carlo_simulation",
                "model_type": model_type,
                "parameters": params,
                "results": results,
                "interpretation": interpretation
            }
            
        except Exception as e:
            logger.error(f"Monte Carlo simulation error: {e}")
            return {"error": f"Monte Carlo simulation failed: {str(e)}"}
    
    def black_scholes_pricing(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Black-Scholes option pricing with Greeks
        
        Args:
            params: Dictionary containing option parameters
                - option_type: 'call' or 'put'
                - spot_price: Current stock price
                - strike_price: Strike price
                - time_to_expiry: Time to expiry in years
                - risk_free_rate: Risk-free rate
                - volatility: Volatility
        """
        try:
            option_type = params.get('option_type', 'call').lower()
            S = params.get('spot_price', 100.0)
            K = params.get('strike_price', 100.0)
            T = params.get('time_to_expiry', 0.25)
            r = params.get('risk_free_rate', 0.05)
            sigma = params.get('volatility', 0.2)
            
            # Black-Scholes formula
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            if option_type == 'call':
                price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            else:  # put
                price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            
            # Calculate Greeks
            greeks = self._calculate_greeks(S, K, T, r, sigma, d1, d2, option_type)
            
            # Generate interpretation
            interpretation = self._interpret_black_scholes_results(
                price, greeks, params
            )
            
            return {
                "analysis_type": "black_scholes_pricing",
                "option_type": option_type,
                "parameters": params,
                "option_price": price,
                "greeks": greeks,
                "interpretation": interpretation
            }
            
        except Exception as e:
            logger.error(f"Black-Scholes pricing error: {e}")
            return {"error": f"Black-Scholes pricing failed: {str(e)}"}
    
    def portfolio_optimization(self, returns: pd.DataFrame, 
                             method: str = "markowitz") -> Dict[str, Any]:
        """
        Portfolio optimization using various methods
        
        Args:
            returns: DataFrame of asset returns
            method: Optimization method ('markowitz', 'risk_parity', 'max_sharpe', 'min_variance')
        """
        try:
            returns_clean = returns.dropna()
            
            if len(returns_clean) < 2:
                return {"error": "Insufficient data for portfolio optimization"}
            
            n_assets = len(returns_clean.columns)
            
            # Calculate expected returns and covariance matrix
            expected_returns = returns_clean.mean() * 252  # Annualized
            cov_matrix = returns_clean.cov() * 252  # Annualized
            
            if method == "markowitz":
                results = self._markowitz_optimization(expected_returns, cov_matrix)
            elif method == "risk_parity":
                results = self._risk_parity_optimization(cov_matrix)
            elif method == "max_sharpe":
                risk_free_rate = 0.02
                results = self._max_sharpe_optimization(expected_returns, cov_matrix, risk_free_rate)
            elif method == "min_variance":
                results = self._min_variance_optimization(cov_matrix)
            else:
                return {"error": f"Unknown optimization method: {method}"}
            
            # Calculate portfolio metrics
            portfolio_metrics = self._calculate_portfolio_metrics(
                results['weights'], expected_returns, cov_matrix
            )
            
            # Generate interpretation
            interpretation = self._interpret_portfolio_optimization_results(
                results, portfolio_metrics, method
            )
            
            return {
                "analysis_type": "portfolio_optimization",
                "method": method,
                "n_assets": n_assets,
                "optimization_results": results,
                "portfolio_metrics": portfolio_metrics,
                "interpretation": interpretation
            }
            
        except Exception as e:
            logger.error(f"Portfolio optimization error: {e}")
            return {"error": f"Portfolio optimization failed: {str(e)}"}
    
    def factor_analysis(self, returns: pd.DataFrame, n_factors: Optional[int] = None) -> Dict[str, Any]:
        """
        Factor analysis for return decomposition
        
        Args:
            returns: DataFrame of asset returns
            n_factors: Number of factors (auto-determined if None)
        """
        try:
            returns_clean = returns.dropna()
            
            if len(returns_clean) < 2:
                return {"error": "Insufficient data for factor analysis"}
            
            # Standardize returns
            returns_standardized = (returns_clean - returns_clean.mean()) / returns_clean.std()
            
            # Calculate correlation matrix
            corr_matrix = returns_standardized.corr()
            
            # Eigenvalue decomposition
            eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)
            
            # Sort by eigenvalues (descending)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # Determine number of factors
            if n_factors is None:
                # Kaiser criterion: factors with eigenvalues > 1
                n_factors = np.sum(eigenvalues > 1.0)
                if n_factors == 0:
                    n_factors = 1  # At least one factor
            
            # Extract factors
            factor_loadings = eigenvectors[:, :n_factors]
            factor_variance_explained = eigenvalues[:n_factors] / np.sum(eigenvalues)
            
            # Calculate factor scores
            factor_scores = returns_standardized @ factor_loadings
            
            # Calculate communalities
            communalities = np.sum(factor_loadings**2, axis=1)
            
            # Generate interpretation
            interpretation = self._interpret_factor_analysis_results(
                n_factors, factor_variance_explained, factor_loadings, 
                communalities, returns_clean.columns.tolist()
            )
            
            return {
                "analysis_type": "factor_analysis",
                "n_factors": n_factors,
                "eigenvalues": eigenvalues.tolist(),
                "factor_loadings": factor_loadings.tolist(),
                "factor_variance_explained": factor_variance_explained.tolist(),
                "cumulative_variance_explained": np.cumsum(factor_variance_explained).tolist(),
                "communalities": dict(zip(returns_clean.columns, communalities)),
                "factor_scores": factor_scores.to_dict(),
                "interpretation": interpretation
            }
            
        except Exception as e:
            logger.error(f"Factor analysis error: {e}")
            return {"error": f"Factor analysis failed: {str(e)}"}
    
    def garch_modeling(self, returns: pd.Series, model_type: str = "GARCH(1,1)") -> Dict[str, Any]:
        """
        GARCH modeling for volatility forecasting
        
        Args:
            returns: Series of returns
            model_type: GARCH model type ('GARCH(1,1)', 'EGARCH', 'GJR-GARCH')
        """
        try:
            returns_clean = returns.dropna()
            
            if len(returns_clean) < 100:
                return {"error": "Insufficient data for GARCH modeling (minimum 100 observations)"}
            
            # Simple GARCH(1,1) implementation
            if model_type == "GARCH(1,1)":
                results = self._garch_11_estimation(returns_clean)
            else:
                return {"error": f"GARCH model type {model_type} not implemented yet"}
            
            # Generate interpretation
            interpretation = self._interpret_garch_results(results, model_type)
            
            return {
                "analysis_type": "garch_modeling",
                "model_type": model_type,
                "results": results,
                "interpretation": interpretation
            }
            
        except Exception as e:
            logger.error(f"GARCH modeling error: {e}")
            return {"error": f"GARCH modeling failed: {str(e)}"}
    
    # Helper methods for Monte Carlo simulations
    def _geometric_brownian_motion(self, S0: float, mu: float, sigma: float, 
                                 dt: float, n_steps: int, n_simulations: int) -> Dict[str, Any]:
        """Geometric Brownian Motion simulation"""
        
        # Generate random shocks
        random_shocks = np.random.normal(0, 1, (n_simulations, n_steps))
        
        # Calculate price paths
        price_paths = np.zeros((n_simulations, n_steps + 1))
        price_paths[:, 0] = S0
        
        for t in range(n_steps):
            price_paths[:, t + 1] = price_paths[:, t] * np.exp(
                (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * random_shocks[:, t]
            )
        
        # Calculate statistics
        final_prices = price_paths[:, -1]
        mean_final_price = np.mean(final_prices)
        std_final_price = np.std(final_prices)
        percentiles = np.percentile(final_prices, [5, 25, 50, 75, 95])
        
        return {
            "price_paths": price_paths.tolist(),
            "final_prices": final_prices.tolist(),
            "mean_final_price": mean_final_price,
            "std_final_price": std_final_price,
            "percentiles": {
                "5th": percentiles[0],
                "25th": percentiles[1],
                "50th": percentiles[2],
                "75th": percentiles[3],
                "95th": percentiles[4]
            },
            "probability_positive_return": np.mean(final_prices > S0),
            "expected_return": (mean_final_price / S0 - 1) * 100
        }
    
    def _jump_diffusion_model(self, S0: float, mu: float, sigma: float, 
                            jump_intensity: float, jump_mean: float, jump_std: float,
                            dt: float, n_steps: int, n_simulations: int) -> Dict[str, Any]:
        """Jump diffusion model simulation"""
        
        # Generate random shocks
        random_shocks = np.random.normal(0, 1, (n_simulations, n_steps))
        jump_events = np.random.poisson(jump_intensity * dt, (n_simulations, n_steps))
        jump_sizes = np.random.normal(jump_mean, jump_std, (n_simulations, n_steps))
        
        # Calculate price paths
        price_paths = np.zeros((n_simulations, n_steps + 1))
        price_paths[:, 0] = S0
        
        for t in range(n_steps):
            # Continuous component
            continuous_component = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * random_shocks[:, t]
            
            # Jump component
            jump_component = jump_events[:, t] * jump_sizes[:, t]
            
            price_paths[:, t + 1] = price_paths[:, t] * np.exp(continuous_component + jump_component)
        
        # Calculate statistics
        final_prices = price_paths[:, -1]
        mean_final_price = np.mean(final_prices)
        std_final_price = np.std(final_prices)
        percentiles = np.percentile(final_prices, [5, 25, 50, 75, 95])
        
        return {
            "price_paths": price_paths.tolist(),
            "final_prices": final_prices.tolist(),
            "mean_final_price": mean_final_price,
            "std_final_price": std_final_price,
            "percentiles": {
                "5th": percentiles[0],
                "25th": percentiles[1],
                "50th": percentiles[2],
                "75th": percentiles[3],
                "95th": percentiles[4]
            },
            "probability_positive_return": np.mean(final_prices > S0),
            "expected_return": (mean_final_price / S0 - 1) * 100,
            "jump_statistics": {
                "total_jumps": np.sum(jump_events),
                "jumps_per_simulation": np.mean(np.sum(jump_events, axis=1))
            }
        }
    
    def _heston_model(self, S0: float, mu: float, kappa: float, theta: float,
                     v0: float, sigma_v: float, rho: float, dt: float, 
                     n_steps: int, n_simulations: int) -> Dict[str, Any]:
        """Heston stochastic volatility model simulation"""
        
        # Generate correlated random shocks
        random_shocks = np.random.normal(0, 1, (n_simulations, n_steps, 2))
        
        # Calculate price and variance paths
        price_paths = np.zeros((n_simulations, n_steps + 1))
        variance_paths = np.zeros((n_simulations, n_steps + 1))
        
        price_paths[:, 0] = S0
        variance_paths[:, 0] = v0
        
        for t in range(n_steps):
            # Variance process
            variance_paths[:, t + 1] = np.maximum(
                variance_paths[:, t] + kappa * (theta - variance_paths[:, t]) * dt +
                sigma_v * np.sqrt(variance_paths[:, t] * dt) * random_shocks[:, t, 1],
                0.001  # Ensure variance stays positive
            )
            
            # Price process
            price_paths[:, t + 1] = price_paths[:, t] * np.exp(
                (mu - 0.5 * variance_paths[:, t]) * dt +
                np.sqrt(variance_paths[:, t] * dt) * (
                    rho * random_shocks[:, t, 1] + 
                    np.sqrt(1 - rho**2) * random_shocks[:, t, 0]
                )
            )
        
        # Calculate statistics
        final_prices = price_paths[:, -1]
        mean_final_price = np.mean(final_prices)
        std_final_price = np.std(final_prices)
        percentiles = np.percentile(final_prices, [5, 25, 50, 75, 95])
        
        return {
            "price_paths": price_paths.tolist(),
            "variance_paths": variance_paths.tolist(),
            "final_prices": final_prices.tolist(),
            "mean_final_price": mean_final_price,
            "std_final_price": std_final_price,
            "percentiles": {
                "5th": percentiles[0],
                "25th": percentiles[1],
                "50th": percentiles[2],
                "75th": percentiles[3],
                "95th": percentiles[4]
            },
            "probability_positive_return": np.mean(final_prices > S0),
            "expected_return": (mean_final_price / S0 - 1) * 100,
            "volatility_statistics": {
                "mean_volatility": np.mean(np.sqrt(variance_paths)),
                "volatility_of_volatility": np.std(np.sqrt(variance_paths))
            }
        }
    
    def _portfolio_simulation(self, initial_value: float, weights: List[float],
                            correlations: List[List[float]], drift: List[float],
                            volatility: List[float], dt: float, n_steps: int,
                            n_simulations: int) -> Dict[str, Any]:
        """Multi-asset portfolio simulation"""
        
        n_assets = len(weights)
        
        # Generate correlated random shocks
        corr_matrix = np.array(correlations)
        L = np.linalg.cholesky(corr_matrix)
        
        random_shocks = np.random.normal(0, 1, (n_simulations, n_steps, n_assets))
        correlated_shocks = np.dot(random_shocks, L.T)
        
        # Calculate individual asset paths
        asset_paths = np.zeros((n_simulations, n_steps + 1, n_assets))
        asset_paths[:, 0, :] = initial_value / n_assets  # Equal initial allocation
        
        for t in range(n_steps):
            for i in range(n_assets):
                asset_paths[:, t + 1, i] = asset_paths[:, t, i] * np.exp(
                    (drift[i] - 0.5 * volatility[i]**2) * dt +
                    volatility[i] * np.sqrt(dt) * correlated_shocks[:, t, i]
                )
        
        # Calculate portfolio paths
        portfolio_paths = np.sum(asset_paths * weights, axis=2)
        
        # Calculate statistics
        final_portfolio_values = portfolio_paths[:, -1]
        mean_final_value = np.mean(final_portfolio_values)
        std_final_value = np.std(final_portfolio_values)
        percentiles = np.percentile(final_portfolio_values, [5, 25, 50, 75, 95])
        
        return {
            "portfolio_paths": portfolio_paths.tolist(),
            "asset_paths": asset_paths.tolist(),
            "final_portfolio_values": final_portfolio_values.tolist(),
            "mean_final_value": mean_final_value,
            "std_final_value": std_final_value,
            "percentiles": {
                "5th": percentiles[0],
                "25th": percentiles[1],
                "50th": percentiles[2],
                "75th": percentiles[3],
                "95th": percentiles[4]
            },
            "probability_positive_return": np.mean(final_portfolio_values > initial_value),
            "expected_return": (mean_final_value / initial_value - 1) * 100
        }
    
    # Helper methods for Black-Scholes
    def _calculate_greeks(self, S: float, K: float, T: float, r: float, 
                         sigma: float, d1: float, d2: float, option_type: str) -> Dict[str, float]:
        """Calculate option Greeks"""
        
        # Delta
        if option_type == 'call':
            delta = norm.cdf(d1)
        else:  # put
            delta = norm.cdf(d1) - 1
        
        # Gamma (same for call and put)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        
        # Theta
        if option_type == 'call':
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - 
                    r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        else:  # put
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + 
                    r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
        
        # Vega (same for call and put)
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100
        
        # Rho
        if option_type == 'call':
            rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        else:  # put
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
        
        return {
            "delta": delta,
            "gamma": gamma,
            "theta": theta,
            "vega": vega,
            "rho": rho
        }
    
    # Helper methods for portfolio optimization
    def _markowitz_optimization(self, expected_returns: pd.Series, 
                              cov_matrix: pd.DataFrame) -> Dict[str, Any]:
        """Markowitz mean-variance optimization"""
        
        n_assets = len(expected_returns)
        
        # Objective function: minimize portfolio variance
        def objective(weights):
            return np.dot(weights, np.dot(cov_matrix, weights))
        
        # Constraints
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        # Initial guess
        x0 = np.array([1.0 / n_assets] * n_assets)
        
        # Optimize
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        return {
            "weights": result.x.tolist(),
            "expected_return": np.dot(result.x, expected_returns),
            "variance": result.fun,
            "volatility": np.sqrt(result.fun),
            "sharpe_ratio": np.dot(result.x, expected_returns) / np.sqrt(result.fun)
        }
    
    def _risk_parity_optimization(self, cov_matrix: pd.DataFrame) -> Dict[str, Any]:
        """Risk parity optimization"""
        
        n_assets = len(cov_matrix)
        
        # Objective function: minimize sum of squared risk contributions
        def objective(weights):
            portfolio_var = np.dot(weights, np.dot(cov_matrix, weights))
            risk_contributions = weights * np.dot(cov_matrix, weights) / portfolio_var
            return np.sum((risk_contributions - 1.0/n_assets)**2)
        
        # Constraints
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        # Initial guess
        x0 = np.array([1.0 / n_assets] * n_assets)
        
        # Optimize
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        return {
            "weights": result.x.tolist(),
            "risk_contributions": (result.x * np.dot(cov_matrix, result.x) / 
                                 np.dot(result.x, np.dot(cov_matrix, result.x))).tolist()
        }
    
    def _max_sharpe_optimization(self, expected_returns: pd.Series, 
                               cov_matrix: pd.DataFrame, risk_free_rate: float) -> Dict[str, Any]:
        """Maximum Sharpe ratio optimization"""
        
        n_assets = len(expected_returns)
        
        # Objective function: minimize negative Sharpe ratio
        def objective(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_var = np.dot(weights, np.dot(cov_matrix, weights))
            return -(portfolio_return - risk_free_rate) / np.sqrt(portfolio_var)
        
        # Constraints
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        # Initial guess
        x0 = np.array([1.0 / n_assets] * n_assets)
        
        # Optimize
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        return {
            "weights": result.x.tolist(),
            "expected_return": np.dot(result.x, expected_returns),
            "variance": np.dot(result.x, np.dot(cov_matrix, result.x)),
            "sharpe_ratio": -result.fun
        }
    
    def _min_variance_optimization(self, cov_matrix: pd.DataFrame) -> Dict[str, Any]:
        """Minimum variance optimization"""
        
        n_assets = len(cov_matrix)
        
        # Objective function: minimize portfolio variance
        def objective(weights):
            return np.dot(weights, np.dot(cov_matrix, weights))
        
        # Constraints
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        # Initial guess
        x0 = np.array([1.0 / n_assets] * n_assets)
        
        # Optimize
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        return {
            "weights": result.x.tolist(),
            "variance": result.fun,
            "volatility": np.sqrt(result.fun)
        }
    
    def _calculate_portfolio_metrics(self, weights: List[float], 
                                   expected_returns: pd.Series, 
                                   cov_matrix: pd.DataFrame) -> Dict[str, float]:
        """Calculate portfolio performance metrics"""
        
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        return {
            "expected_return": portfolio_return,
            "variance": portfolio_variance,
            "volatility": portfolio_volatility,
            "sharpe_ratio": portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
        }
    
    def _garch_11_estimation(self, returns: pd.Series) -> Dict[str, Any]:
        """Simple GARCH(1,1) estimation"""
        
        # Initial parameters
        omega = np.var(returns) * 0.1
        alpha = 0.1
        beta = 0.8
        
        # Simple estimation (in practice, use proper MLE)
        squared_returns = returns**2
        variance = np.zeros_like(returns)
        variance[0] = np.var(returns)
        
        for t in range(1, len(returns)):
            variance[t] = omega + alpha * squared_returns[t-1] + beta * variance[t-1]
        
        # Calculate volatility
        volatility = np.sqrt(variance)
        
        return {
            "omega": omega,
            "alpha": alpha,
            "beta": beta,
            "variance": variance.tolist(),
            "volatility": volatility.tolist(),
            "persistence": alpha + beta,
            "unconditional_variance": omega / (1 - alpha - beta)
        }
    
    # Interpretation methods
    def _interpret_monte_carlo_results(self, results: Dict[str, Any], 
                                     model_type: str, params: Dict[str, Any]) -> str:
        """Generate intelligent Monte Carlo interpretation with clear time context and actionable insights"""
        
        # Extract key parameters
        time_horizon = params.get('time_horizon', 1.0)
        n_simulations = params.get('n_simulations', 10000)
        initial_value = params.get('initial_value', 100.0)
        ticker = params.get('ticker', 'Asset')
        
        # Convert time horizon to more readable format
        if time_horizon < 1/12:  # Less than 1 month
            time_str = f"{int(time_horizon * 365)} days"
        elif time_horizon < 1:  # Less than 1 year
            time_str = f"{int(time_horizon * 12)} months"
        else:  # 1 year or more
            time_str = f"{time_horizon:.1f} years"
        
        interpretation = f"Monte Carlo Simulation Results ({model_type}):\n\n"
        
        # Time context and parameters
        interpretation += f"**Simulation Parameters:**\n"
        interpretation += f"- Asset: {ticker}\n"
        interpretation += f"- Time Horizon: {time_str}\n"
        interpretation += f"- Simulations: {n_simulations:,}\n"
        interpretation += f"- Starting Price: ${initial_value:.2f}\n\n"
        
        # Results
        expected_return = results.get('expected_return', 0)
        prob_positive = results.get('probability_positive_return', 0)
        percentiles = results.get('percentiles', {})
        
        interpretation += f"**Expected Return:** {expected_return:.2f}% over {time_str}\n"
        interpretation += f"**Probability of Positive Return:** {prob_positive*100:.1f}%\n"
        interpretation += f"**Price Distribution at End of {time_str}:**\n"
        interpretation += f"- 5th percentile: ${percentiles.get('5th', 0):.2f} (worst 5% of outcomes)\n"
        interpretation += f"- 25th percentile: ${percentiles.get('25th', 0):.2f} (bottom quarter)\n"
        interpretation += f"- 50th percentile: ${percentiles.get('50th', 0):.2f} (median outcome)\n"
        interpretation += f"- 75th percentile: ${percentiles.get('75th', 0):.2f} (top quarter)\n"
        interpretation += f"- 95th percentile: ${percentiles.get('95th', 0):.2f} (best 5% of outcomes)\n\n"
        
        # Practical interpretation
        interpretation += f"**What This Means for You:**\n"
        
        # Risk assessment
        worst_case = percentiles.get('5th', 0)
        best_case = percentiles.get('95th', 0)
        potential_loss = ((worst_case - initial_value) / initial_value) * 100
        potential_gain = ((best_case - initial_value) / initial_value) * 100
        
        interpretation += f"- **Worst Case (5% chance):** Price drops to ${worst_case:.2f} ({potential_loss:.1f}% loss)\n"
        interpretation += f"- **Best Case (5% chance):** Price rises to ${best_case:.2f} ({potential_gain:.1f}% gain)\n"
        interpretation += f"- **Most Likely Range:** ${percentiles.get('25th', 0):.2f} - ${percentiles.get('75th', 0):.2f}\n\n"
        
        # Investment advice based on results
        if prob_positive > 0.7:
            interpretation += f"- **Investment Outlook:** Favorable - {prob_positive*100:.0f}% chance of positive returns\n"
        elif prob_positive > 0.5:
            interpretation += f"- **Investment Outlook:** Neutral - {prob_positive*100:.0f}% chance of positive returns\n"
        else:
            interpretation += f"- **Investment Outlook:** Risky - Only {prob_positive*100:.0f}% chance of positive returns\n"
        
        # Risk level assessment
        volatility = params.get('volatility', 0.2)
        if volatility > 0.3:
            interpretation += f"- **Risk Level:** High volatility ({volatility*100:.0f}% annual)\n"
        elif volatility > 0.15:
            interpretation += f"- **Risk Level:** Moderate volatility ({volatility*100:.0f}% annual)\n"
        else:
            interpretation += f"- **Risk Level:** Low volatility ({volatility*100:.0f}% annual)\n"
        
        # Time-specific insights
        if time_horizon <= 0.25:  # 3 months or less
            interpretation += f"- **Short-term Note:** High uncertainty in {time_str} - consider position sizing\n"
        elif time_horizon <= 1:  # 1 year or less
            interpretation += f"- **Medium-term Note:** {time_str} provides moderate predictability\n"
        else:  # More than 1 year
            interpretation += f"- **Long-term Note:** {time_str} allows for mean reversion and trend development\n"
        
        if model_type == "jump_diffusion":
            jump_stats = results.get('jump_statistics', {})
            interpretation += f"- **Jump Events:** Average {jump_stats.get('jumps_per_simulation', 0):.1f} jumps per simulation\n"
        
        return interpretation
    
    def _interpret_black_scholes_results(self, price: float, greeks: Dict[str, float], 
                                       params: Dict[str, Any]) -> str:
        """Generate intelligent Black-Scholes interpretation"""
        
        option_type = params.get('option_type', 'call')
        spot_price = params.get('spot_price', 100)
        strike_price = params.get('strike_price', 100)
        
        interpretation = f"Black-Scholes Option Pricing Results:\n\n"
        interpretation += f"**Option Price:** ${price:.2f}\n"
        interpretation += f"**Option Type:** {option_type.upper()}\n"
        interpretation += f"**Spot Price:** ${spot_price:.2f}\n"
        interpretation += f"**Strike Price:** ${strike_price:.2f}\n\n"
        
        interpretation += f"**Greeks:**\n"
        interpretation += f"- Delta: {greeks['delta']:.3f} (price sensitivity)\n"
        interpretation += f"- Gamma: {greeks['gamma']:.3f} (delta sensitivity)\n"
        interpretation += f"- Theta: {greeks['theta']:.3f} (time decay)\n"
        interpretation += f"- Vega: {greeks['vega']:.3f} (volatility sensitivity)\n"
        interpretation += f"- Rho: {greeks['rho']:.3f} (interest rate sensitivity)\n\n"
        
        interpretation += f"**Interpretation:**\n"
        
        # Delta interpretation
        if abs(greeks['delta']) > 0.7:
            interpretation += "- High price sensitivity (delta > 0.7)\n"
        elif abs(greeks['delta']) > 0.3:
            interpretation += "- Moderate price sensitivity (delta > 0.3)\n"
        else:
            interpretation += "- Low price sensitivity (delta < 0.3)\n"
        
        # Theta interpretation
        if greeks['theta'] < -0.1:
            interpretation += "- High time decay (theta < -0.1)\n"
        elif greeks['theta'] < -0.05:
            interpretation += "- Moderate time decay\n"
        else:
            interpretation += "- Low time decay\n"
        
        return interpretation
    
    def _interpret_portfolio_optimization_results(self, results: Dict[str, Any], 
                                                method: str,
                                                portfolio_metrics: Dict[str, float]) -> str:
        """Generate intelligent portfolio optimization interpretation"""
        
        interpretation = f"Portfolio Optimization Results ({method}):\n\n"
        
        weights = results.get('weights', [])
        expected_return = portfolio_metrics.get('expected_return', 0)
        volatility = portfolio_metrics.get('volatility', 0)
        sharpe_ratio = portfolio_metrics.get('sharpe_ratio', 0)
        
        interpretation += f"**Portfolio Metrics:**\n"
        interpretation += f"- Expected Return: {expected_return*100:.2f}%\n"
        interpretation += f"- Volatility: {volatility*100:.2f}%\n"
        interpretation += f"- Sharpe Ratio: {sharpe_ratio:.2f}\n\n"
        
        interpretation += f"**Optimal Weights:**\n"
        for i, weight in enumerate(weights):
            interpretation += f"- Asset {i+1}: {weight*100:.1f}%\n"
        
        interpretation += f"\n**Interpretation:**\n"
        
        if sharpe_ratio > 1.0:
            interpretation += "- Excellent risk-adjusted returns (Sharpe > 1.0)\n"
        elif sharpe_ratio > 0.5:
            interpretation += "- Good risk-adjusted returns (Sharpe > 0.5)\n"
        else:
            interpretation += "- Poor risk-adjusted returns (Sharpe < 0.5)\n"
        
        if method == "risk_parity":
            risk_contributions = results.get('risk_contributions', [])
            interpretation += f"- Risk contributions are approximately equal\n"
        
        return interpretation
    
    def _interpret_factor_analysis_results(self, n_factors: int, 
                                         factor_variance_explained: np.ndarray,
                                         factor_loadings: np.ndarray,
                                         communalities: np.ndarray,
                                         asset_names: List[str]) -> str:
        """Generate intelligent factor analysis interpretation"""
        
        interpretation = f"Factor Analysis Results:\n\n"
        interpretation += f"**Number of Factors:** {n_factors}\n"
        interpretation += f"**Variance Explained:**\n"
        
        for i, var_exp in enumerate(factor_variance_explained):
            interpretation += f"- Factor {i+1}: {var_exp*100:.1f}%\n"
        
        interpretation += f"**Cumulative Variance Explained:** {np.sum(factor_variance_explained)*100:.1f}%\n\n"
        
        interpretation += f"**Factor Loadings (Top 3 per factor):**\n"
        for i in range(n_factors):
            interpretation += f"\nFactor {i+1}:\n"
            # Find top 3 loadings for this factor
            top_indices = np.argsort(np.abs(factor_loadings[:, i]))[-3:][::-1]
            for idx in top_indices:
                interpretation += f"- {asset_names[idx]}: {factor_loadings[idx, i]:.3f}\n"
        
        interpretation += f"\n**Interpretation:**\n"
        
        if np.sum(factor_variance_explained) > 0.8:
            interpretation += "- Factors explain most of the variance (good fit)\n"
        elif np.sum(factor_variance_explained) > 0.6:
            interpretation += "- Factors explain moderate variance\n"
        else:
            interpretation += "- Factors explain limited variance (poor fit)\n"
        
        interpretation += f"- {n_factors} factors capture the main sources of return variation\n"
        
        return interpretation
    
    def _interpret_garch_results(self, results: Dict[str, Any], model_type: str) -> str:
        """Generate intelligent GARCH interpretation"""
        
        interpretation = f"GARCH Model Results ({model_type}):\n\n"
        
        omega = results.get('omega', 0)
        alpha = results.get('alpha', 0)
        beta = results.get('beta', 0)
        persistence = results.get('persistence', 0)
        
        interpretation += f"**Model Parameters:**\n"
        interpretation += f"- Omega (ω): {omega:.6f}\n"
        interpretation += f"- Alpha (α): {alpha:.3f}\n"
        interpretation += f"- Beta (β): {beta:.3f}\n"
        interpretation += f"- Persistence (α + β): {persistence:.3f}\n\n"
        
        interpretation += f"**Interpretation:**\n"
        
        if persistence > 0.95:
            interpretation += "- Very high persistence (volatility shocks persist for long periods)\n"
        elif persistence > 0.9:
            interpretation += "- High persistence (volatility clustering is strong)\n"
        elif persistence > 0.8:
            interpretation += "- Moderate persistence\n"
        else:
            interpretation += "- Low persistence (volatility mean-reverts quickly)\n"
        
        if alpha > 0.1:
            interpretation += "- High sensitivity to recent shocks (α > 0.1)\n"
        else:
            interpretation += "- Low sensitivity to recent shocks (α < 0.1)\n"
        
        return interpretation
    
    def portfolio_optimization(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Portfolio optimization using various methods
        
        Args:
            params: Dictionary containing optimization parameters
                - tickers: List of ticker symbols
                - optimization_type: 'max_sharpe', 'min_variance', 'risk_parity'
                - risk_free_rate: Risk-free rate
        """
        try:
            tickers = params.get('tickers', ['AAPL', 'MSFT', 'GOOGL'])
            optimization_type = params.get('optimization_type', 'max_sharpe')
            risk_free_rate = params.get('risk_free_rate', 0.02)
            
            # Fetch historical data
            from ..adapters.prices_polygon import get_prices_agg
            
            end_date = "2024-10-13"
            start_date = "2023-10-13"
            
            returns_data = {}
            for ticker in tickers:
                try:
                    df = get_prices_agg(ticker, start_date, end_date)
                    if not df.empty:
                        returns = df['close'].pct_change().dropna()
                        returns_data[ticker] = returns
                except Exception as e:
                    logger.warning(f"Failed to fetch data for {ticker}: {e}")
                    continue
            
            if len(returns_data) < 2:
                return {"error": "Insufficient data for portfolio optimization"}
            
            # Create returns DataFrame
            returns_df = pd.DataFrame(returns_data)
            returns_df = returns_df.dropna()
            
            if len(returns_df) < 30:
                return {"error": "Insufficient data points for optimization"}
            
            # Calculate expected returns and covariance matrix
            expected_returns = returns_df.mean() * 252  # Annualized
            cov_matrix = returns_df.cov() * 252  # Annualized
            
            # Perform optimization
            if optimization_type == 'max_sharpe':
                result = self._max_sharpe_optimization(expected_returns, cov_matrix, risk_free_rate)
            elif optimization_type == 'min_variance':
                result = self._min_variance_optimization(cov_matrix)
            elif optimization_type == 'risk_parity':
                result = self._risk_parity_optimization(cov_matrix)
            else:
                result = self._max_sharpe_optimization(expected_returns, cov_matrix, risk_free_rate)
            
            # Calculate portfolio metrics
            weights = np.array(result['weights'])
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
            portfolio_volatility = np.sqrt(portfolio_variance)
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
            
            # Generate interpretation
            interpretation = self._interpret_portfolio_optimization_results(
                result, optimization_type, {
                    'expected_return': portfolio_return,
                    'volatility': portfolio_volatility,
                    'sharpe_ratio': sharpe_ratio
                }
            )
            
            return {
                "analysis_type": "portfolio_optimization",
                "optimization_type": optimization_type,
                "tickers": tickers,
                "weights": result['weights'],
                "portfolio_metrics": {
                    "expected_return": portfolio_return,
                    "volatility": portfolio_volatility,
                    "sharpe_ratio": sharpe_ratio
                },
                "individual_metrics": {
                    "expected_returns": expected_returns.to_dict(),
                    "volatilities": (np.sqrt(np.diag(cov_matrix))).tolist()
                },
                "interpretation": interpretation,
                "data_points": len(returns_df)
            }
            
        except Exception as e:
            logger.error(f"Portfolio optimization error: {e}")
            return {"error": f"Portfolio optimization failed: {str(e)}"}
    
    def factor_analysis(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Factor analysis using PCA
        
        Args:
            params: Dictionary containing analysis parameters
                - tickers: List of ticker symbols
                - n_factors: Number of factors to extract
                - method: 'pca' or 'fa'
        """
        try:
            tickers = params.get('tickers', ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'])
            n_factors = params.get('n_factors', 3)
            method = params.get('method', 'pca')
            
            # Fetch historical data
            from ..adapters.prices_polygon import get_prices_agg
            
            end_date = "2024-10-13"
            start_date = "2023-10-13"
            
            returns_data = {}
            for ticker in tickers:
                try:
                    df = get_prices_agg(ticker, start_date, end_date)
                    if not df.empty:
                        returns = df['close'].pct_change().dropna()
                        returns_data[ticker] = returns
                except Exception as e:
                    logger.warning(f"Failed to fetch data for {ticker}: {e}")
                    continue
            
            if len(returns_data) < 3:
                return {"error": "Insufficient data for factor analysis"}
            
            # Create returns DataFrame
            returns_df = pd.DataFrame(returns_data)
            returns_df = returns_df.dropna()
            
            if len(returns_df) < 30:
                return {"error": "Insufficient data points for factor analysis"}
            
            # Standardize returns
            standardized_returns = (returns_df - returns_df.mean()) / returns_df.std()
            
            # Perform PCA
            from sklearn.decomposition import PCA
            
            pca = PCA(n_components=min(n_factors, len(tickers)))
            pca.fit(standardized_returns)
            
            # Extract results
            factor_variance_explained = pca.explained_variance_ratio_
            factor_loadings = pca.components_
            communalities = np.sum(factor_loadings**2, axis=0)
            
            # Generate interpretation
            interpretation = self._interpret_factor_analysis_results(
                n_factors, factor_variance_explained, factor_loadings, 
                communalities, tickers
            )
            
            return {
                "analysis_type": "factor_analysis",
                "method": method,
                "tickers": tickers,
                "n_factors": n_factors,
                "factor_variance_explained": factor_variance_explained.tolist(),
                "factor_loadings": factor_loadings.tolist(),
                "communalities": communalities.tolist(),
                "cumulative_variance_explained": np.sum(factor_variance_explained),
                "interpretation": interpretation,
                "data_points": len(returns_df)
            }
            
        except Exception as e:
            logger.error(f"Factor analysis error: {e}")
            return {"error": f"Factor analysis failed: {str(e)}"}


# Global instance
math_engine = MathEngine()

# workers/engine/timeseries.py
import pandas as pd
import numpy as np
def realized_vol_30d(close: pd.Series) -> float:
    r = close.pct_change(fill_method=None).dropna()
    return float(r.rolling(30).std().dropna().iloc[-1] * (252**0.5))

def momentum_12m_skip_1m(close: pd.Series) -> pd.Series:
    # 252d lookback, skip last 21d
    return (close.shift(21)/close.shift(252))-1.0

def monthly_rebalance_signals(rank_series: pd.Series, top_frac=0.1):
    by_month = rank_series.groupby([rank_series.index.year, rank_series.index.month])
    thresh = by_month.transform(lambda s: s.quantile(1-top_frac))
    return (rank_series >= thresh).astype(int)  # 1 = in portfolio

def simple_vector_backtest(close: pd.Series, signal: pd.Series, tc_bps=5):
    # equal-weight long-only on a single instrument is trivial; for demo, compute ret * signal
    ret = close.pct_change(fill_method=None).fillna(0.0)
    gross = (ret * signal.shift(1)).fillna(0.0)  # enter next day
    tc = (signal.diff().abs().clip(lower=0.0) * (tc_bps/1e4))  # turnover cost
    pnl = gross - tc
    eq = (1+pnl).cumprod()
    sharpe = pnl.mean()/ (pnl.std()+1e-12) * (252**0.5)
    mxdd = (eq/eq.cummax()-1).min()
    return {"sharpe": float(sharpe), "max_dd": float(mxdd), "equity_curve": eq}

def perf_metrics(pnl: pd.Series, equity: pd.Series, signal: pd.Series = None) -> dict:
    ann_factor = 252.0
    mu = pnl.mean() * ann_factor
    vol = pnl.std(ddof=0) * np.sqrt(ann_factor) + 1e-12
    sharpe = mu / vol
    downside = pnl.where(pnl < 0, 0.0).std(ddof=0) * np.sqrt(ann_factor) + 1e-12
    sortino = mu / downside
    hit_rate = (pnl > 0).mean()
    cagr = equity.iloc[-1] ** (ann_factor / max(1.0, len(pnl))) - 1.0
    dd = (equity / equity.cummax() - 1.0)
    max_dd = float(dd.min())
    max_dd_date = dd.idxmin().date().isoformat()
    
    # Calculate additional metrics
    avg_turnover = signal.diff().abs().mean() * ann_factor if signal is not None else 0.0
    exposure_share = signal.mean() if signal is not None else 0.0
    
    return {
        "cagr": float(cagr), 
        "vol_annual": float(vol), 
        "sharpe": float(sharpe),
        "sortino": float(sortino), 
        "hit_rate": float(hit_rate), 
        "max_drawdown_pct": max_dd,
        "max_drawdown_date": max_dd_date,
        "avg_turnover": float(avg_turnover),
        "exposure_share": float(exposure_share)
    }

def rolling_sharpe(pnl: pd.Series, window: int = 252) -> pd.Series:
    """Calculate rolling Sharpe ratio over specified window"""
    ann_factor = 252.0
    rolling_mean = pnl.rolling(window).mean() * ann_factor
    rolling_std = pnl.rolling(window).std() * np.sqrt(ann_factor) + 1e-12
    return rolling_mean / rolling_std

def yearly_returns(pnl: pd.Series) -> pd.DataFrame:
    # daily pnl → yearly return via cumprod
    eq = (1 + pnl).cumprod()
    eq_y = eq.groupby([eq.index.year]).agg(lambda s: (s.iloc[-1] / s.iloc[0]) - 1.0)
    return eq_y.rename("return").to_frame()

def yearly_turnover(signal: pd.Series) -> pd.DataFrame:
    """Calculate yearly turnover from signal changes"""
    turnover = signal.diff().abs()
    yearly_turnover = turnover.groupby([turnover.index.year]).mean() * 252  # Annualize
    return yearly_turnover.rename("turnover").to_frame()

def underwater_heatmap_data(pnl: pd.Series) -> pd.DataFrame:
    """Create data for underwater heatmap by year and month"""
    equity = (1 + pnl).cumprod()
    drawdown = (equity / equity.cummax() - 1.0) * 100  # Convert to percentage
    
    # Create year-month grid
    df = pd.DataFrame({
        'year': drawdown.index.year,
        'month': drawdown.index.month,
        'drawdown': drawdown.values
    })
    
    # Group by year-month and take mean drawdown
    heatmap_data = df.groupby(['year', 'month'])['drawdown'].mean().unstack(fill_value=0)
    
    return heatmap_data

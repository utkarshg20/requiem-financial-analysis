from __future__ import annotations
import os, json, hashlib, platform, subprocess
from datetime import datetime
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server-side plotting
import matplotlib.pyplot as plt

def _sha256_of_pip_freeze() -> str:
    try:
        out = subprocess.check_output(["pip", "freeze"], text=True, timeout=10)
        return hashlib.sha256(out.encode()).hexdigest()[:16]
    except Exception:
        return "unknown"

def save_equity_and_drawdown(run_id: str, eq: pd.Series, rolling_sharpe: pd.Series = None) -> dict:
    outdir = os.path.join("runs", run_id, "figs")
    os.makedirs(outdir, exist_ok=True)

    # Equity curve
    fig1 = plt.figure(figsize=(12, 6))
    eq.plot(linewidth=2, color='blue')
    plt.title("Equity Curve", fontsize=14, fontweight='bold')
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    p1 = os.path.join(outdir, "equity_curve.png")
    fig1.savefig(p1, dpi=150, bbox_inches='tight'); plt.close(fig1)

    # Drawdown curve
    dd = (eq / eq.cummax() - 1.0) * 100  # Convert to percentage
    fig2 = plt.figure(figsize=(12, 6))
    dd.plot(linewidth=2, color='red')
    plt.title("Drawdown", fontsize=14, fontweight='bold')
    plt.xlabel("Date")
    plt.ylabel("Drawdown (%)")
    plt.grid(True, alpha=0.3)
    plt.fill_between(dd.index, dd.values, 0, alpha=0.3, color='red')
    plt.tight_layout()
    p2 = os.path.join(outdir, "drawdown.png")
    fig2.savefig(p2, dpi=150, bbox_inches='tight'); plt.close(fig2)

    # Rolling Sharpe
    if rolling_sharpe is not None and len(rolling_sharpe.dropna()) > 0:
        fig3 = plt.figure(figsize=(12, 6))
        rolling_sharpe.dropna().plot(linewidth=2, color='green')
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.axhline(y=1, color='orange', linestyle='--', alpha=0.5, label='Sharpe = 1')
        plt.axhline(y=2, color='purple', linestyle='--', alpha=0.5, label='Sharpe = 2')
        window_days = len(rolling_sharpe.dropna()) + len(rolling_sharpe) - len(rolling_sharpe.dropna()) if len(rolling_sharpe) > 0 else 63
        plt.title(f"Rolling Sharpe (~3 month window)", fontsize=14, fontweight='bold')
        plt.xlabel("Date")
        plt.ylabel("Sharpe Ratio")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        p3 = os.path.join(outdir, "rolling_sharpe.png")
        fig3.savefig(p3, dpi=150, bbox_inches='tight'); plt.close(fig3)
        
        return {"equity_curve": p1, "drawdown": p2, "rolling_sharpe": p3}
    else:
        return {"equity_curve": p1, "drawdown": p2}

def save_tearsheet_json(run_id: str, tearsheet: dict) -> str:
    outdir = os.path.join("runs", run_id)
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "tearsheet.json")
    with open(path, "w") as f: json.dump(tearsheet, f, indent=2)
    return path

def capture_env_snapshot() -> dict:
    return {
        "python_version": platform.python_version(),
        "pip_freeze_hash": _sha256_of_pip_freeze(),
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }

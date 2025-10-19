"""
10-line script: Complete strategy study using Requiem SDK
"""

from sdk import RequiemClient

# 1. Initialize client
client = RequiemClient("http://localhost:8000")

# 2. Run backtest from natural language
tearsheet = client.quick_backtest(
    "Test 12-month momentum on SPY since 2020 monthly, 5 bps TC",
    output_html="spy_momentum.html"
)

# 3. Print results
print(f"✅ Backtest complete!")
print(f"   CAGR: {tearsheet['metrics']['performance']['cagr']:.2%}")
print(f"   Sharpe: {tearsheet['metrics']['performance']['sharpe']:.2f}")
print(f"   Max DD: {tearsheet['metrics']['performance']['max_drawdown_pct']:.2%}")


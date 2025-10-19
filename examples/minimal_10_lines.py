from sdk import RequiemClient
client = RequiemClient()
tearsheet = client.quick_backtest("Test momentum on SPY since 2020", "results.html")
print(f"CAGR: {tearsheet['metrics']['performance']['cagr']:.2%}")
print(f"Sharpe: {tearsheet['metrics']['performance']['sharpe']:.2f}")


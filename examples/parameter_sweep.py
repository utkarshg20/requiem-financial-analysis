"""
Parameter sweep: Test different rebalance frequencies
"""

from sdk import RequiemClient

client = RequiemClient()

# Test different rebalance frequencies
frequencies = ["daily", "weekly", "monthly"]

print("=" * 80)
print("REBALANCE FREQUENCY SWEEP")
print("=" * 80)

for freq in frequencies:
    prompt = f"Test momentum on SPY since 2023 {freq}, 5 bps TC"
    
    spec = client.prompt_to_spec(prompt)
    tearsheet = client.execute_and_fetch(spec)
    
    metrics = tearsheet['metrics']['performance']
    
    print(f"\n{freq.upper()}:")
    print(f"  CAGR: {metrics['cagr']:>8.2%}")
    print(f"  Sharpe: {metrics['sharpe']:>6.2f}")
    print(f"  Turnover: {metrics['avg_turnover']:>6.1%}")


"""
Multi-strategy comparison using Requiem SDK
"""

from sdk import RequiemClient
import pandas as pd

# Initialize
client = RequiemClient()

# Test multiple strategies
strategies = [
    "Test momentum on SPY since 2020 monthly",
    "Run zscore on SPY since 2020 weekly",
    "SMA crossover 20/50 on SPY since 2020",
]

results = []

for prompt in strategies:
    print(f"Testing: {prompt}")
    
    # Convert to spec and run
    spec = client.prompt_to_spec(prompt)
    tearsheet = client.execute_and_fetch(spec)
    
    # Extract metrics
    metrics = tearsheet['metrics']['performance']
    results.append({
        'strategy': prompt,
        'cagr': metrics['cagr'],
        'sharpe': metrics['sharpe'],
        'max_dd': metrics['max_drawdown_pct'],
    })

# Compare results
df = pd.DataFrame(results)
print("\n" + "=" * 80)
print("STRATEGY COMPARISON")
print("=" * 80)
print(df.to_string(index=False))

# Find best Sharpe
best = df.loc[df['sharpe'].idxmax()]
print(f"\n🏆 Best Sharpe: {best['strategy']}")
print(f"   Sharpe: {best['sharpe']:.2f}")


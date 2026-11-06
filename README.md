# Quant Algorithms Automated Developer (QAAD)

Builds, backtests, scores and ranks quantitative trading strategies.  
Can also take an existing strategy or bot and give it a 0-100 Algo Level score.

### What’s working

- Metrics engine — Sharpe, Sortino, CAGR, MaxDD, Calmar, Profit Factor, Ulcer Index
- Vectorized backtester — signals → equity curve + trade log (fees & slippage)
- Algo Level Scorer — 0–100 across Performance / Robustness / Risk / Sophistication / Implementation
- Strategy base + examples — SMA, RSI, Bollinger, breakout
- Advanced strategies — regime filters, vol targeting, multi-factor momentum
- Walk-forward analysis
- Comparison demos

Evolutionary / LLM agents still planned.

### Algo Level Score (0–100)

| Dimension       | Weight | Looks at                                      |
|-----------------|--------|-----------------------------------------------|
| Performance     | 25%    | Sharpe, Sortino, CAGR, Profit Factor          |
| Robustness      | 25%    | Walk-forward stability, parameter sensitivity |
| Risk Control    | 20%    | Max Drawdown, Ulcer Index                     |
| Sophistication  | 15%    | Regime filters, adaptive sizing, feature depth|
| Implementation  | 15%    | Transaction costs, no look-ahead, clean code  |

**Labels:** 90+ Elite · 75–89 Advanced · 60–74 Intermediate · 40–59 Basic · <40 Weak / Overfit

### Quick start

```bash
git clone https://github.com/7NSE/Quant-Algorithms-Automated-Developer.git
cd Quant-Algorithms-Automated-Developer

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .

# Rank built-in strategies on synthetic data
python examples/run_comparison.py

# Advanced ranking with walk-forward
python examples/run_advanced_comparison.py

# Scoring demo only
python -m qaad.score --example
```

### Project layout

```
qaad/
├── core/
│   ├── metrics.py
│   ├── backtester.py
│   ├── scoring.py
│   ├── walkforward.py
│   ├── regime.py
│   └── position_sizing.py
├── strategies/
│   ├── base.py
│   ├── examples.py
│   └── advanced.py
├── utils/
│   └── data.py
├── score.py
└── evolve.py          # scaffold
examples/
├── run_comparison.py
└── run_advanced_comparison.py
```

### Using it in your own code

```python
from qaad.utils.data import make_synthetic_ohlcv
from qaad.strategies import RSIMeanReversion

data = make_synthetic_ohlcv(n=1200)
strat = RSIMeanReversion(period=14, lower=30, upper=70)

bt = strat.run(data, fee_bps=5, slippage_bps=2)
print(bt.summary())

score = strat.score(data)
print(f"Level score: {score.total:.1f} → {score.level}")
```

### Roadmap

- [x] Metrics + backtester + scoring
- [x] Strategy base + examples
- [x] Walk-forward module
- [x] Comparison scripts
- [ ] Monte-Carlo robustness
- [ ] Natural-language → strategy code agent
- [ ] Multi-agent evolutionary loop
- [ ] Live / paper-trading adapter

### Disclaimer

Research / educational only.  
Past performance does not guarantee future results.  
Validate thoroughly before risking real capital.

**Author:** [7NSE](https://github.com/7NSE)  
**License:** MIT

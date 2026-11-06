# Quant Algorithms Automated Developer (QAAD)

This is a system that builds, backtests, scores and ranks quantitative trading strategies.  
It can also take an existing strategy (or bot) and give it a 0-100 “Algo Level” score so you can see how strong it actually is.

### What’s working right now

- **Metrics engine** — Sharpe, Sortino, CAGR, Max Drawdown, Calmar, Profit Factor, Ulcer Index  
- **Vectorized backtester** — takes signals → equity curve + trade log (with fees & slippage)  
- **Algo Level Scorer** — 0–100 score across Performance / Robustness / Risk / Sophistication / Implementation  
- **Strategy base + a few examples** — SMA crossover, RSI mean-reversion, Bollinger, Volatility breakout  
- **Comparison demo** — run several strategies side-by-side and rank them  

Evolutionary / LLM agents are still planned.

### Algo Level Score (0–100)

| Dimension       | Weight | What it looks at                              |
|-----------------|--------|-----------------------------------------------|
| Performance     | 25%    | Sharpe, Sortino, CAGR, Profit Factor          |
| Robustness      | 25%    | Walk-forward stability, parameter sensitivity |
| Risk Control    | 20%    | Max Drawdown, Ulcer Index                     |
| Sophistication  | 15%    | Regime filters, adaptive sizing, feature depth|
| Implementation  | 15%    | Transaction costs, no look-ahead, clean code  |

**Labels:**  
90+ Elite · 75–89 Advanced · 60–74 Intermediate · 40–59 Basic · <40 Weak / Overfit

### Quick start

```bash
git clone https://github.com/7NSE/Quant-Algorithms-Automated-Developer.git
cd Quant-Algorithms-Automated-Developer

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install numpy pandas rich

# Rank a few built-in strategies on synthetic data
python examples/run_comparison.py

# Or just the scoring demo
python -m qaad.score --example
```

### Project layout

```
qaad/
├── core/
│   ├── metrics.py       # performance & risk numbers
│   ├── backtester.py    # vectorized backtest engine
│   └── scoring.py       # Algo Level Score (0-100)
├── strategies/
│   ├── base.py          # Strategy abstract class
│   └── examples.py      # SMA, RSI, Bollinger, Breakout
├── utils/
│   └── data.py          # CSV loader + synthetic OHLCV
├── score.py             # CLI for scoring
└── evolve.py            # evolutionary loop (scaffold)
examples/
└── run_comparison.py    # end-to-end ranking demo
```

### Using it in your own code

```python
from qaad.utils.data import make_synthetic_ohlcv
from qaad.strategies import RSIMeanReversion
from qaad.core.scoring import AlgoLevelScorer

data = make_synthetic_ohlcv(n=1200)
strat = RSIMeanReversion(period=14, lower=30, upper=70)

bt = strat.run(data, fee_bps=5, slippage_bps=2)
print(bt.summary())

score = strat.score(data)
print(f"Level score: {score.total:.1f} → {score.level}")
```

### Roadmap

- [x] Metrics + backtester + scoring engine  
- [x] Strategy base class + example strategies  
- [x] End-to-end comparison script  
- [ ] Walk-forward & Monte-Carlo robustness module  
- [ ] Natural-language → strategy code agent  
- [ ] Multi-agent evolutionary loop  
- [ ] Optional live / paper-trading adapter  

### Disclaimer

Research / educational software only.  
Past performance does not guarantee future results.  
Always validate thoroughly before risking real capital.

**Author:** [7NSE](https://github.com/7NSE)  
**License:** MIT

# Quant Algorithms Automated Developer (QAAD)

**Agentic AI system that automatically develops, evolves, backtests, and ranks quantitative trading algorithms.**

It can also **score the sophistication / strength level** of any existing quant algo or trading bot and help you decide which ones are stronger.

> Built for researchers and traders who want an automated quant research pipeline instead of writing every strategy by hand.

---

## What it does

| Capability | Description |
|------------|-------------|
| **Auto Strategy Generation** | Describe an idea in plain English → system generates executable Python strategy code |
| **Evolutionary Search** | Multi-agent evolutionary framework that mutates & improves strategies |
| **Backtesting Engine** | Vectorized + event-driven backtests with leakage controls |
| **Algo Level / Strength Scoring** | Scores any algo/bot on a 0–100 scale across multiple dimensions |
| **Ranking & Comparison** | Compare multiple strategies side-by-side and pick the strongest |
| **Hypothesis → Code Pipeline** | Literature-inspired hypothesis generation → code → evaluation loop |

---

## Algo Level Score (0–100)

The system evaluates any strategy / bot on these dimensions:

| Dimension | Weight | What it measures |
|-----------|--------|------------------|
| **Performance** | 25% | Sharpe, Sortino, Calmar, CAGR, Profit Factor |
| **Robustness** | 25% | Walk-forward stability, out-of-sample decay, parameter sensitivity |
| **Risk Control** | 20% | Max Drawdown, Ulcer Index, tail risk, position sizing quality |
| **Sophistication** | 15% | Feature engineering depth, regime awareness, adaptive logic |
| **Implementation Quality** | 15% | Code cleanliness, transaction cost realism, no look-ahead bias |

**Final Level Labels:**
- 90–100 → Elite / Institutional-grade
- 75–89 → Advanced
- 60–74 → Intermediate
- 40–59 → Basic / Retail
- <40 → Weak / Overfit or broken

---

## Architecture (High Level)

```
User Idea / Existing Algo
        ↓
┌──────────────────────────────────────────┐
│  Orchestrator Agent (LLM)                  │
│  - Plans research DAG                       │
│  - Decides mutate / combine / pivot        │
├──────────────────────────────────────────┤
│  Research Agents                            │
│  • Hypothesis Generator                     │
│  • Literature / Idea Miner                  │
│  • Strategy Coder                           │
│  • Backtest Runner                          │
│  • Evaluator & Ranker                       │
│  • Memory / Experience Store                │
└───────────────────────────────────────────┘
        ↓
Ranked Strategies + Level Scores + Reports
```

---

## Quick Start

```bash
git clone https://github.com/7NSE/Quant-Algorithms-Automated-Developer.git
cd Quant-Algorithms-Automated-Developer

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Score an existing strategy
python -m qaad.score --strategy path/to/your_strategy.py --data path/to/ohlcv.csv

# Generate + evolve new strategies from a natural language idea
python -m qaad.evolve --idea "Mean reversion on RSI + volume confirmation for crypto" --generations 20
```

---

## Project Structure

```
qaad/
├── agents/           # LLM agents (orchestrator, coder, evaluator...)
├── core/             # Backtester, metrics, scoring engine
├── strategies/       # Generated & example strategies
├── memory/           # Experience store & trajectory memory
├── utils/            # Data loaders, helpers
├── evolve.py         # Evolutionary loop entrypoint
├── score.py          # Algo level scoring entrypoint
└── __main__.py
configs/              # Example research configs
examples/             # Sample strategies + notebooks
tests/
```

---

## Roadmap

- [x] Repo + core architecture scaffold
- [ ] Full scoring engine (metrics + robustness tests)
- [ ] Multi-agent evolutionary loop
- [ ] Natural language → strategy code pipeline
- [ ] Walk-forward & Monte-Carlo robustness module
- [ ] Live paper-trading adapter (optional)
- [ ] Web / CLI dashboard for ranking

---

## Disclaimer

This is research / educational software.  
Past performance does not guarantee future results.  
Use at your own risk. Always paper-trade and validate thoroughly before any real capital.

---

**Author:** [7NSE](https://github.com/7NSE)  
**License:** MIT

"""
Lightweight vectorized backtester.

Takes a signal series (-1 / 0 / 1) and OHLCV data, returns equity curve +
metrics + a simple trade log. Easy to read and extend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any
import numpy as np
import pandas as pd

from .metrics import full_metrics


@dataclass
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
    positions: pd.Series
    metrics: Dict[str, float]
    trades: pd.DataFrame = field(default_factory=pd.DataFrame)
    meta: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        m = self.metrics
        lines = [
            "=== Backtest Summary ===",
            f"Sharpe         {m.get('sharpe', 0):>8.2f}",
            f"Sortino        {m.get('sortino', 0):>8.2f}",
            f"CAGR           {m.get('cagr', 0):>8.1%}",
            f"Max Drawdown   {m.get('max_drawdown', 0):>8.1%}",
            f"Calmar         {m.get('calmar', 0):>8.2f}",
            f"Profit Factor  {m.get('profit_factor', 0):>8.2f}",
            f"Win Rate       {m.get('win_rate', 0):>8.1%}",
            f"Total Return   {m.get('total_return', 0):>8.1%}",
        ]
        if not self.trades.empty:
            lines.append(f"Trades         {len(self.trades):>8d}")
        return "\n".join(lines)


def run_backtest(
    data: pd.DataFrame,
    signal: pd.Series,
    initial_capital: float = 100_000.0,
    fee_bps: float = 5.0,
    slippage_bps: float = 2.0,
    periods_per_year: int = 252,
    price_col: str = "close",
) -> BacktestResult:
    """
    Position-based backtest.

    signal should align with data.index and contain values in {-1, 0, 1}
    (or continuous weights for partial exposure).

    Costs are charged on every change in position size.
    """
    if price_col not in data.columns:
        raise ValueError(f"Column '{price_col}' not found in data")

    price = data[price_col].astype(float)
    sig = signal.reindex(price.index).fillna(0.0).astype(float)

    # trade on next bar — avoids look-ahead
    position = sig.shift(1).fillna(0.0)

    raw_ret = price.pct_change().fillna(0.0)
    strat_ret = position * raw_ret

    pos_change = position.diff().abs().fillna(0.0)
    cost = pos_change * (fee_bps + slippage_bps) / 10_000.0
    strat_ret = strat_ret - cost

    equity = (1.0 + strat_ret).cumprod() * initial_capital
    equity.iloc[0] = initial_capital

    metrics = full_metrics(equity, periods=periods_per_year)
    trades = _extract_trades(position, price)

    return BacktestResult(
        equity=equity,
        returns=strat_ret,
        positions=position,
        metrics=metrics,
        trades=trades,
        meta={
            "initial_capital": initial_capital,
            "fee_bps": fee_bps,
            "slippage_bps": slippage_bps,
            "periods_per_year": periods_per_year,
        },
    )


def _extract_trades(position: pd.Series, price: pd.Series) -> pd.DataFrame:
    records = []
    in_trade = False
    entry_time = None
    entry_price = None
    direction = 0

    for t, pos in position.items():
        if not in_trade and pos != 0:
            in_trade = True
            entry_time = t
            entry_price = float(price.loc[t])
            direction = 1 if pos > 0 else -1
        elif in_trade and (pos == 0 or np.sign(pos) != direction):
            exit_price = float(price.loc[t])
            pnl = (exit_price / entry_price - 1.0) * direction
            records.append(
                {
                    "entry_time": entry_time,
                    "exit_time": t,
                    "direction": "long" if direction > 0 else "short",
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl_pct": pnl,
                }
            )
            if pos != 0:
                entry_time = t
                entry_price = exit_price
                direction = 1 if pos > 0 else -1
                in_trade = True
            else:
                in_trade = False

    return pd.DataFrame(records)

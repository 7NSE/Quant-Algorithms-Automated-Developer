"""
Lightweight vectorized backtester.

Signal in, equity curve + metrics + trade log out.
Works with discrete (-1/0/1) or continuous position weights.
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
    Basic position-based backtest.

    signal can be -1/0/1 or continuous weights.
    We shift by 1 bar so there's no look-ahead.
    Costs hit every time position size changes.
    """
    if price_col not in data.columns:
        raise ValueError(f"Column '{price_col}' not found in data")

    price = data[price_col].astype(float)
    sig = signal.reindex(price.index).fillna(0.0).astype(float)

    # next-bar execution
    position = sig.shift(1).fillna(0.0)

    raw_ret = price.pct_change().fillna(0.0)
    strat_ret = position * raw_ret

    # cost on turnover
    turnover = position.diff().abs().fillna(0.0)
    cost = turnover * (fee_bps + slippage_bps) / 10_000.0
    strat_ret = strat_ret - cost

    equity = initial_capital * (1.0 + strat_ret).cumprod()
    # first bar has no return yet
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
    """
    Build a simple trade log.

    For continuous weights we treat any meaningful size change
    (crossing zero or flipping sign) as an exit/entry.
    Small noise in vol-targeted weights is ignored.
    """
    records = []
    in_trade = False
    entry_time = None
    entry_price = None
    direction = 0
    min_size = 0.05  # ignore tiny residual weights

    for t, pos in position.items():
        size = abs(pos)

        if not in_trade:
            if size >= min_size:
                in_trade = True
                entry_time = t
                entry_price = float(price.loc[t])
                direction = 1 if pos > 0 else -1
        else:
            flipped = np.sign(pos) != direction and size >= min_size
            closed = size < min_size

            if flipped or closed:
                exit_price = float(price.loc[t])
                pnl = (exit_price / entry_price - 1.0) * direction
                records.append({
                    "entry_time": entry_time,
                    "exit_time": t,
                    "direction": "long" if direction > 0 else "short",
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl_pct": pnl,
                })

                if flipped:
                    # immediately open the other side
                    entry_time = t
                    entry_price = exit_price
                    direction = 1 if pos > 0 else -1
                    in_trade = True
                else:
                    in_trade = False

    return pd.DataFrame(records)

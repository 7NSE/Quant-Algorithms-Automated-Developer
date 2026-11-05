"""
Walk-forward analysis.

This is the minimum bar for claiming a strategy is robust.
Train on a window, test on the next window, roll forward, repeat.
If the edge survives out-of-sample, it might be real.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Any, Optional
import numpy as np
import pandas as pd

from .backtester import run_backtest, BacktestResult
from .metrics import full_metrics


@dataclass
class WalkForwardFold:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_metrics: Dict[str, float] = field(default_factory=dict)
    test_metrics: Dict[str, float] = field(default_factory=dict)
    test_equity: Optional[pd.Series] = None


@dataclass
class WalkForwardResult:
    folds: List[WalkForwardFold]
    combined_test_equity: pd.Series
    combined_test_metrics: Dict[str, float]
    avg_is_sharpe: float
    avg_oos_sharpe: float
    oos_is_ratio: float  # oos / is sharpe — main robustness signal

    def summary(self) -> str:
        lines = [
            "=== Walk-Forward Summary ===",
            f"Folds              {len(self.folds):>6d}",
            f"Avg IS Sharpe      {self.avg_is_sharpe:>8.2f}",
            f"Avg OOS Sharpe     {self.avg_oos_sharpe:>8.2f}",
            f"OOS / IS ratio     {self.oos_is_ratio:>8.2f}",
            f"Combined OOS Sharpe{self.combined_test_metrics.get('sharpe', 0):>8.2f}",
            f"Combined OOS MaxDD {self.combined_test_metrics.get('max_drawdown', 0):>8.1%}",
            f"Combined OOS CAGR  {self.combined_test_metrics.get('cagr', 0):>8.1%}",
        ]
        return "\n".join(lines)


def walk_forward(
    data: pd.DataFrame,
    signal_fn: Callable[[pd.DataFrame], pd.Series],
    train_bars: int = 252,
    test_bars: int = 63,
    step_bars: Optional[int] = None,
    fee_bps: float = 5.0,
    slippage_bps: float = 2.0,
    periods_per_year: int = 252,
) -> WalkForwardResult:
    """
    Rolling walk-forward.

    signal_fn(data_slice) -> signal Series
    We only evaluate the signal on the test window (no peeking).
    """
    if step_bars is None:
        step_bars = test_bars

    n = len(data)
    folds: List[WalkForwardFold] = []
    oos_equities: List[pd.Series] = []

    start = 0
    while start + train_bars + test_bars <= n:
        train_slice = data.iloc[start : start + train_bars]
        test_slice = data.iloc[start + train_bars : start + train_bars + test_bars]

        # fit / generate on train (for strategies that need fitting this is where it happens)
        # for pure rule-based strategies signal_fn just uses the data it receives
        train_signal = signal_fn(train_slice)
        test_signal = signal_fn(test_slice)

        train_bt = run_backtest(
            train_slice, train_signal,
            fee_bps=fee_bps, slippage_bps=slippage_bps,
            periods_per_year=periods_per_year,
        )
        test_bt = run_backtest(
            test_slice, test_signal,
            fee_bps=fee_bps, slippage_bps=slippage_bps,
            periods_per_year=periods_per_year,
        )

        fold = WalkForwardFold(
            train_start=train_slice.index[0],
            train_end=train_slice.index[-1],
            test_start=test_slice.index[0],
            test_end=test_slice.index[-1],
            train_metrics=train_bt.metrics,
            test_metrics=test_bt.metrics,
            test_equity=test_bt.equity,
        )
        folds.append(fold)
        oos_equities.append(test_bt.equity)

        start += step_bars

    if not folds:
        raise ValueError("Not enough data for even one walk-forward fold")

    # stitch OOS equity curves (re-base each fold to continue from previous)
    combined = _stitch_equity(oos_equities)
    combined_metrics = full_metrics(combined, periods=periods_per_year)

    is_sharpes = [f.train_metrics.get("sharpe", 0.0) for f in folds]
    oos_sharpes = [f.test_metrics.get("sharpe", 0.0) for f in folds]
    avg_is = float(np.mean(is_sharpes))
    avg_oos = float(np.mean(oos_sharpes))
    ratio = avg_oos / avg_is if avg_is != 0 else 0.0

    return WalkForwardResult(
        folds=folds,
        combined_test_equity=combined,
        combined_test_metrics=combined_metrics,
        avg_is_sharpe=avg_is,
        avg_oos_sharpe=avg_oos,
        oos_is_ratio=ratio,
    )


def _stitch_equity(equities: List[pd.Series]) -> pd.Series:
    if not equities:
        return pd.Series(dtype=float)

    pieces = []
    level = 1.0
    for eq in equities:
        # normalise fold to start at 1, then scale to current level
        norm = eq / eq.iloc[0]
        scaled = norm * level
        pieces.append(scaled)
        level = float(scaled.iloc[-1])

    return pd.concat(pieces)

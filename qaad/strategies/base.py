"""
Base class for strategies.

Just implement generate_signals and the rest of the pipeline works.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd

from qaad.core.backtester import run_backtest, BacktestResult
from qaad.core.scoring import AlgoLevelScorer, ScoreBreakdown


class Strategy(ABC):
    name: str = "unnamed"

    def __init__(self, **params):
        if "name" in params:
            self.name = params.pop("name")
        self.params = params

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Return signal series aligned with data.index (-1/0/1 or continuous)."""
        ...

    def run(
        self,
        data: pd.DataFrame,
        initial_capital: float = 100_000.0,
        fee_bps: float = 5.0,
        slippage_bps: float = 2.0,
        periods_per_year: int = 252,
    ) -> BacktestResult:
        signal = self.generate_signals(data)
        return run_backtest(
            data=data,
            signal=signal,
            initial_capital=initial_capital,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            periods_per_year=periods_per_year,
        )

    def score(
        self,
        data: pd.DataFrame,
        robustness_metrics: Optional[Dict[str, float]] = None,
        sophistication_signals: Optional[Dict[str, float]] = None,
        implementation_signals: Optional[Dict[str, float]] = None,
        **backtest_kwargs,
    ) -> ScoreBreakdown:
        result = self.run(data, **backtest_kwargs)
        scorer = AlgoLevelScorer()

        # defaults if nothing passed in
        soph = sophistication_signals or {
            "num_features": 2,
            "has_regime_filter": 0.0,
            "adaptive_sizing": 0.0,
        }
        impl = implementation_signals or {
            "has_transaction_costs": 1.0,
            "no_lookahead": 1.0,
            "realistic_slippage": 1.0,
            "position_sizing_present": 0.5,
            "clean_code": 1.0,
        }

        return scorer.score(
            metrics=result.metrics,
            robustness_metrics=robustness_metrics or {},
            sophistication_signals=soph,
            implementation_signals=impl,
        )

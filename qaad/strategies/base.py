"""
Base class for strategies.

Every strategy just needs to implement `generate_signals`.
The rest of the pipeline (backtest, score, rank) stays the same.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd

from qaad.core.backtester import run_backtest, BacktestResult
from qaad.core.scoring import AlgoLevelScorer, ScoreBreakdown


class Strategy(ABC):
    """
    Minimal interface.

    Subclass this, implement generate_signals, and you can plug the strategy
    straight into the backtester and scorer.
    """

    name: str = "unnamed"
    params: Dict[str, Any] = {}

    def __init__(self, **params):
        self.params = params
        # allow subclasses to set a nicer display name
        if "name" in params:
            self.name = params.pop("name")

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Return a Series aligned with data.index.
        Values should be roughly in {-1, 0, 1} or continuous weights.
        """
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

        # sensible defaults if caller doesn't supply extra signals
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

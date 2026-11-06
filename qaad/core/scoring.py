"""
Algo Level Scoring Engine

Takes backtest numbers + a few qualitative signals and turns them
into a 0-100 score across five dimensions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd


@dataclass
class ScoreBreakdown:
    performance: float = 0.0
    robustness: float = 0.0
    risk_control: float = 0.0
    sophistication: float = 0.0
    implementation: float = 0.0
    total: float = 0.0
    level: str = "Unknown"
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "performance": round(self.performance, 2),
            "robustness": round(self.robustness, 2),
            "risk_control": round(self.risk_control, 2),
            "sophistication": round(self.sophistication, 2),
            "implementation": round(self.implementation, 2),
            "total": round(self.total, 2),
            "level": self.level,
            "details": self.details,
        }


class AlgoLevelScorer:
    """
    Weights:
        Performance    25%
        Robustness     25%
        Risk Control   20%
        Sophistication 15%
        Implementation 15%
    """

    WEIGHTS = {
        "performance": 0.25,
        "robustness": 0.25,
        "risk_control": 0.20,
        "sophistication": 0.15,
        "implementation": 0.15,
    }

    LEVEL_LABELS = [
        (90, "Elite / Institutional-grade"),
        (75, "Advanced"),
        (60, "Intermediate"),
        (40, "Basic / Retail"),
        (0, "Weak / Overfit or broken"),
    ]

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        if weights:
            self.WEIGHTS = weights

    def score(
        self,
        metrics: Dict[str, float],
        robustness_metrics: Optional[Dict[str, float]] = None,
        sophistication_signals: Optional[Dict[str, float]] = None,
        implementation_signals: Optional[Dict[str, float]] = None,
    ) -> ScoreBreakdown:
        robustness_metrics = robustness_metrics or {}
        sophistication_signals = sophistication_signals or {}
        implementation_signals = implementation_signals or {}

        perf = self._score_performance(metrics)
        rob = self._score_robustness(metrics, robustness_metrics)
        risk = self._score_risk(metrics)
        soph = self._score_sophistication(sophistication_signals)
        impl = self._score_implementation(implementation_signals)

        total = (
            perf * self.WEIGHTS["performance"]
            + rob * self.WEIGHTS["robustness"]
            + risk * self.WEIGHTS["risk_control"]
            + soph * self.WEIGHTS["sophistication"]
            + impl * self.WEIGHTS["implementation"]
        )

        return ScoreBreakdown(
            performance=perf,
            robustness=rob,
            risk_control=risk,
            sophistication=soph,
            implementation=impl,
            total=total,
            level=self._label(total),
            details={
                "raw_metrics": metrics,
                "robustness_metrics": robustness_metrics,
                "sophistication_signals": sophistication_signals,
                "implementation_signals": implementation_signals,
            },
        )

    def _get(self, m: Dict[str, float], *keys, default=0.0):
        for k in keys:
            if k in m and m[k] is not None:
                return m[k]
        return default

    def _score_performance(self, m: Dict[str, float]) -> float:
        score = 50.0

        sharpe = self._get(m, "sharpe", "Sharpe")
        if sharpe >= 2.0:
            score += 25
        elif sharpe >= 1.5:
            score += 18
        elif sharpe >= 1.0:
            score += 10
        elif sharpe >= 0.5:
            score += 3
        else:
            score -= 15

        sortino = self._get(m, "sortino", "Sortino")
        if sortino >= 2.5:
            score += 10
        elif sortino >= 1.5:
            score += 5

        pf = self._get(m, "profit_factor", "Profit Factor", default=1.0)
        if pf >= 2.0:
            score += 10
        elif pf >= 1.5:
            score += 5
        elif pf < 1.0:
            score -= 15

        cagr = self._get(m, "cagr", "CAGR")
        if cagr >= 0.30:
            score += 5
        elif cagr >= 0.15:
            score += 2

        return float(np.clip(score, 0, 100))

    def _score_robustness(self, m: Dict[str, float], rob: Dict[str, float]) -> float:
        score = 50.0

        wf_ratio = self._get(rob, "walk_forward_ratio", "oos_sharpe_ratio", default=0.7)
        if wf_ratio >= 0.85:
            score += 25
        elif wf_ratio >= 0.70:
            score += 15
        elif wf_ratio >= 0.50:
            score += 5
        else:
            score -= 20

        sens = self._get(rob, "param_sensitivity", default=0.5)
        if sens < 0.2:
            score += 15
        elif sens < 0.4:
            score += 8
        elif sens > 0.7:
            score -= 15

        if self._get(m, "win_rate", default=0.5) > 0.55 and self._get(m, "profit_factor", default=1.0) > 1.3:
            score += 5

        return float(np.clip(score, 0, 100))

    def _score_risk(self, m: Dict[str, float]) -> float:
        score = 50.0

        max_dd = abs(self._get(m, "max_drawdown", "Max Drawdown", default=0.25))
        if max_dd <= 0.08:
            score += 25
        elif max_dd <= 0.15:
            score += 15
        elif max_dd <= 0.25:
            score += 5
        else:
            score -= 20

        ulcer = self._get(m, "ulcer_index")
        if 0 < ulcer < 5:
            score += 10
        elif ulcer > 15:
            score -= 10

        if max_dd > 0.40:
            score -= 15

        return float(np.clip(score, 0, 100))

    def _score_sophistication(self, signals: Dict[str, float]) -> float:
        if not signals:
            return 40.0

        score = 30.0
        score += signals.get("num_features", 0) * 3
        score += signals.get("has_regime_filter", 0) * 15
        score += signals.get("adaptive_sizing", 0) * 12
        score += signals.get("multi_timeframe", 0) * 10
        score += signals.get("ml_component", 0) * 15
        score += signals.get("risk_parity_or_vol_target", 0) * 10

        return float(np.clip(score, 0, 100))

    def _score_implementation(self, signals: Dict[str, float]) -> float:
        if not signals:
            return 50.0

        score = 40.0
        score += signals.get("has_transaction_costs", 0) * 15
        score += signals.get("no_lookahead", 0) * 20
        score += signals.get("realistic_slippage", 0) * 10
        score += signals.get("clean_code", 0) * 10
        score += signals.get("position_sizing_present", 0) * 10

        return float(np.clip(score, 0, 100))

    def _label(self, total: float) -> str:
        for threshold, label in self.LEVEL_LABELS:
            if total >= threshold:
                return label
        return "Weak / Overfit or broken"


def compare_algos(scores: Dict[str, ScoreBreakdown]) -> pd.DataFrame:
    rows = []
    for name, sb in scores.items():
        rows.append({
            "algo": name,
            "total": sb.total,
            "level": sb.level,
            "performance": sb.performance,
            "robustness": sb.robustness,
            "risk_control": sb.risk_control,
            "sophistication": sb.sophistication,
            "implementation": sb.implementation,
        })
    df = pd.DataFrame(rows).sort_values("total", ascending=False).reset_index(drop=True)
    df.index = df.index + 1
    df.index.name = "rank"
    return df

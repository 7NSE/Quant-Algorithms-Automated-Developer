"""
Performance & risk metrics.

Simple, readable functions that most quant researchers actually use
when deciding whether a strategy is worth keeping.
"""

from __future__ import annotations

from typing import Dict
import numpy as np
import pandas as pd


def _clean(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").dropna()


def compute_returns(equity: pd.Series) -> pd.Series:
    eq = _clean(equity)
    return eq.pct_change().dropna()


def sharpe_ratio(returns: pd.Series, periods: int = 252, risk_free: float = 0.0) -> float:
    r = _clean(returns)
    if len(r) < 2:
        return 0.0
    excess = r - risk_free / periods
    vol = excess.std()
    if vol == 0 or np.isnan(vol):
        return 0.0
    return float(np.sqrt(periods) * excess.mean() / vol)


def sortino_ratio(returns: pd.Series, periods: int = 252, risk_free: float = 0.0) -> float:
    r = _clean(returns)
    if len(r) < 2:
        return 0.0
    excess = r - risk_free / periods
    downside = excess[excess < 0]
    if len(downside) == 0:
        return float("inf") if excess.mean() > 0 else 0.0
    down_std = downside.std()
    if down_std == 0 or np.isnan(down_std):
        return 0.0
    return float(np.sqrt(periods) * excess.mean() / down_std)


def max_drawdown(equity: pd.Series) -> float:
    eq = _clean(equity)
    if len(eq) < 2:
        return 0.0
    peak = eq.cummax()
    dd = (eq - peak) / peak
    return float(abs(dd.min()))


def cagr(equity: pd.Series, periods: int = 252) -> float:
    eq = _clean(equity)
    if len(eq) < 2 or eq.iloc[0] <= 0:
        return 0.0
    total = eq.iloc[-1] / eq.iloc[0]
    years = len(eq) / periods
    if years <= 0:
        return 0.0
    return float(total ** (1 / years) - 1)


def calmar_ratio(equity: pd.Series, periods: int = 252) -> float:
    growth = cagr(equity, periods)
    mdd = max_drawdown(equity)
    if mdd == 0:
        return float("inf") if growth > 0 else 0.0
    return float(growth / mdd)


def profit_factor(returns: pd.Series) -> float:
    r = _clean(returns)
    gains = r[r > 0].sum()
    losses = abs(r[r < 0].sum())
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def win_rate(returns: pd.Series) -> float:
    r = _clean(returns)
    if len(r) == 0:
        return 0.0
    return float((r > 0).mean())


def ulcer_index(equity: pd.Series) -> float:
    """How painful the drawdowns feel (depth + duration). Lower is better."""
    eq = _clean(equity)
    if len(eq) < 2:
        return 0.0
    peak = eq.cummax()
    dd_pct = 100.0 * (eq - peak) / peak
    return float(np.sqrt((dd_pct ** 2).mean()))


def full_metrics(equity: pd.Series, periods: int = 252, risk_free: float = 0.0) -> Dict[str, float]:
    """Everything in one dict so the scorer can consume it easily."""
    returns = compute_returns(equity)
    return {
        "sharpe": sharpe_ratio(returns, periods, risk_free),
        "sortino": sortino_ratio(returns, periods, risk_free),
        "max_drawdown": max_drawdown(equity),
        "calmar": calmar_ratio(equity, periods),
        "cagr": cagr(equity, periods),
        "profit_factor": profit_factor(returns),
        "win_rate": win_rate(returns),
        "ulcer_index": ulcer_index(equity),
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1) if len(equity) > 1 else 0.0,
        "n_periods": float(len(returns)),
    }


def print_metrics(m: Dict[str, float]) -> None:
    print(f"Sharpe         {m.get('sharpe', 0):>8.2f}")
    print(f"Sortino        {m.get('sortino', 0):>8.2f}")
    print(f"CAGR           {m.get('cagr', 0):>8.1%}")
    print(f"Max Drawdown   {m.get('max_drawdown', 0):>8.1%}")
    print(f"Calmar         {m.get('calmar', 0):>8.2f}")
    print(f"Profit Factor  {m.get('profit_factor', 0):>8.2f}")
    print(f"Win Rate       {m.get('win_rate', 0):>8.1%}")
    print(f"Ulcer Index    {m.get('ulcer_index', 0):>8.2f}")
    print(f"Total Return   {m.get('total_return', 0):>8.1%}")

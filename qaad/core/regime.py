"""
Market regime detection.

Simple but useful: trend regime + volatility regime.
Strategies can use these filters to stay out of hostile environments
instead of blindly firing signals all the time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def trend_regime(close: pd.Series, slow: int = 100, fast: int = 20) -> pd.Series:
    """
    +1 = uptrend, -1 = downtrend, 0 = chop.

    Uses a slow MA for direction and distance from MA as a strength filter.
    """
    ma_slow = close.rolling(slow).mean()
    ma_fast = close.rolling(fast).mean()
    dist = (close - ma_slow) / ma_slow

    regime = pd.Series(0, index=close.index, dtype=float)
    regime[(ma_fast > ma_slow) & (dist > 0.01)] = 1.0
    regime[(ma_fast < ma_slow) & (dist < -0.01)] = -1.0
    return regime


def volatility_regime(
    close: pd.Series,
    short_window: int = 20,
    long_window: int = 100,
    high_mult: float = 1.25,
    low_mult: float = 0.75,
) -> pd.Series:
    """
    1 = high vol, 0 = normal, -1 = low vol.

    Compares short-term realised vol to a longer baseline.
    """
    ret = close.pct_change()
    short_vol = ret.rolling(short_window).std()
    long_vol = ret.rolling(long_window).std()

    ratio = short_vol / long_vol.replace(0, np.nan)
    regime = pd.Series(0.0, index=close.index)
    regime[ratio > high_mult] = 1.0
    regime[ratio < low_mult] = -1.0
    return regime


def risk_on_off(close: pd.Series, lookback: int = 60) -> pd.Series:
    """
    Crude risk-on / risk-off from momentum + drawdown.

    +1 risk-on, -1 risk-off, 0 neutral.
    """
    mom = close.pct_change(lookback)
    roll_max = close.rolling(lookback).max()
    dd = close / roll_max - 1.0

    state = pd.Series(0.0, index=close.index)
    state[(mom > 0) & (dd > -0.08)] = 1.0
    state[(mom < 0) | (dd < -0.15)] = -1.0
    return state


def combine_regimes(
    trend: pd.Series,
    vol: pd.Series,
    risk: pd.Series | None = None,
) -> pd.Series:
    """
    Final exposure multiplier in [0, 1].

    Idea: full size only in favourable trend + normal/low vol + risk-on.
    """
    exposure = pd.Series(1.0, index=trend.index)

    # cut size in downtrends
    exposure[trend < 0] *= 0.25
    exposure[trend == 0] *= 0.5

    # cut size in high vol
    exposure[vol > 0] *= 0.4

    if risk is not None:
        exposure[risk < 0] *= 0.3

    return exposure.clip(0.0, 1.0)

"""
Position sizing.

Volatility targeting is the single most useful upgrade most retail
strategies never bother with. Same signal, much more stable equity curve.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def realised_vol(returns: pd.Series, window: int = 20, periods: int = 252) -> pd.Series:
    return returns.rolling(window).std() * np.sqrt(periods)


def vol_target_weights(
    signal: pd.Series,
    close: pd.Series,
    target_vol: float = 0.12,
    vol_window: int = 20,
    periods: int = 252,
    max_leverage: float = 2.0,
) -> pd.Series:
    """
    Scale a raw signal so the portfolio aims for a constant volatility.

    weight = signal * (target_vol / realised_vol)
    clipped to max_leverage.
    """
    ret = close.pct_change()
    vol = realised_vol(ret, window=vol_window, periods=periods)
    scale = target_vol / vol.replace(0, np.nan)
    scale = scale.clip(upper=max_leverage).fillna(0.0)

    raw = signal.reindex(close.index).fillna(0.0)
    return (raw * scale).clip(-max_leverage, max_leverage)


def risk_parity_weights(vols: pd.DataFrame, max_weight: float = 0.4) -> pd.DataFrame:
    """
    Inverse-vol weights across multiple assets.
    vols: DataFrame of realised volatilities (columns = assets).
    """
    inv = 1.0 / vols.replace(0, np.nan)
    w = inv.div(inv.sum(axis=1), axis=0)
    return w.clip(upper=max_weight).fillna(0.0)


def kelly_fraction(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """
    Classic Kelly. Usually too aggressive — most people use half-Kelly.
    avg_loss should be positive number (magnitude).
    """
    if avg_loss <= 0:
        return 0.0
    b = avg_win / avg_loss
    q = 1.0 - win_rate
    f = win_rate - (q / b)
    return float(max(0.0, f))


def apply_exposure_cap(weights: pd.Series, max_gross: float = 1.0) -> pd.Series:
    gross = weights.abs()
    scale = (max_gross / gross).clip(upper=1.0)
    scale = scale.where(gross > 0, 1.0)
    return weights * scale

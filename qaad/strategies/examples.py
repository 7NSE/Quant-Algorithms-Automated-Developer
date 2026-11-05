"""
A few simple strategies so the rest of the pipeline has something real to run.

Nothing fancy — just clean, readable signal logic that a human would write.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from .base import Strategy


class SMACrossover(Strategy):
    """Classic fast/slow moving average crossover."""

    name = "sma_crossover"

    def __init__(self, fast: int = 10, slow: int = 30, **kwargs):
        super().__init__(fast=fast, slow=slow, **kwargs)
        self.fast = fast
        self.slow = slow

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"]
        fast_ma = close.rolling(self.fast).mean()
        slow_ma = close.rolling(self.slow).mean()

        signal = pd.Series(0.0, index=data.index)
        signal[fast_ma > slow_ma] = 1.0
        signal[fast_ma < slow_ma] = -1.0
        # stay flat until both MAs exist
        signal[fast_ma.isna() | slow_ma.isna()] = 0.0
        return signal


class RSIMeanReversion(Strategy):
    """Buy when RSI is washed out, sell when it's overbought."""

    name = "rsi_mean_reversion"

    def __init__(self, period: int = 14, lower: float = 30.0, upper: float = 70.0, **kwargs):
        super().__init__(period=period, lower=lower, upper=upper, **kwargs)
        self.period = period
        self.lower = lower
        self.upper = upper

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"]
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)

        avg_gain = gain.rolling(self.period).mean()
        avg_loss = loss.rolling(self.period).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))

        signal = pd.Series(0.0, index=data.index)
        signal[rsi < self.lower] = 1.0
        signal[rsi > self.upper] = -1.0
        signal[rsi.isna()] = 0.0
        return signal


class BollingerMeanReversion(Strategy):
    """Fade moves that touch the Bollinger bands."""

    name = "bollinger_mean_reversion"

    def __init__(self, period: int = 20, num_std: float = 2.0, **kwargs):
        super().__init__(period=period, num_std=num_std, **kwargs)
        self.period = period
        self.num_std = num_std

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"]
        mid = close.rolling(self.period).mean()
        std = close.rolling(self.period).std()
        upper = mid + self.num_std * std
        lower = mid - self.num_std * std

        signal = pd.Series(0.0, index=data.index)
        signal[close < lower] = 1.0
        signal[close > upper] = -1.0
        signal[mid.isna()] = 0.0
        return signal


class VolatilityBreakout(Strategy):
    """
    Simple breakout: go long when price closes above the recent high
    by more than a multiple of ATR, flat otherwise.
    """

    name = "vol_breakout"

    def __init__(self, lookback: int = 20, atr_mult: float = 1.5, **kwargs):
        super().__init__(lookback=lookback, atr_mult=atr_mult, **kwargs)
        self.lookback = lookback
        self.atr_mult = atr_mult

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        high = data["high"]
        low = data["low"]
        close = data["close"]

        tr = pd.concat(
            [
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)

        atr = tr.rolling(self.lookback).mean()
        recent_high = high.rolling(self.lookback).max().shift(1)

        signal = pd.Series(0.0, index=data.index)
        breakout = close > (recent_high + self.atr_mult * atr)
        signal[breakout.fillna(False)] = 1.0
        return signal

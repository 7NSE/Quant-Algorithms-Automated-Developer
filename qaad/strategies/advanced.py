"""
More serious strategies.

These combine multiple signals, respect regime, and use volatility targeting.
Still readable — no black-box ML yet, just better quant engineering.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy
from qaad.core.regime import trend_regime, volatility_regime, risk_on_off, combine_regimes
from qaad.core.position_sizing import vol_target_weights


class TrendFollowingVolTarget(Strategy):
    """
    Dual moving-average trend with volatility targeting and regime filter.

    This is closer to what a CTA sleeve actually looks like than a raw SMA cross.
    """

    name = "trend_vol_target"

    def __init__(
        self,
        fast: int = 20,
        slow: int = 100,
        target_vol: float = 0.12,
        vol_window: int = 20,
        max_leverage: float = 2.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.fast = fast
        self.slow = slow
        self.target_vol = target_vol
        self.vol_window = vol_window
        self.max_leverage = max_leverage

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"]
        fast_ma = close.rolling(self.fast).mean()
        slow_ma = close.rolling(self.slow).mean()

        raw = pd.Series(0.0, index=data.index)
        raw[fast_ma > slow_ma] = 1.0
        raw[fast_ma < slow_ma] = -1.0
        raw[fast_ma.isna() | slow_ma.isna()] = 0.0

        # regime filter — cut exposure in hostile environments
        trend = trend_regime(close, slow=self.slow, fast=self.fast)
        vol_reg = volatility_regime(close)
        risk = risk_on_off(close)
        exposure = combine_regimes(trend, vol_reg, risk)

        filtered = raw * exposure

        # volatility targeting
        weights = vol_target_weights(
            filtered,
            close,
            target_vol=self.target_vol,
            vol_window=self.vol_window,
            max_leverage=self.max_leverage,
        )
        return weights


class MultiFactorMomentum(Strategy):
    """
    Combines time-series momentum at multiple horizons + mean-reversion overlay.

    Score = weighted sum of z-scored features → position.
    """

    name = "multi_factor_momentum"

    def __init__(
        self,
        horizons: tuple = (20, 60, 120),
        weights: tuple = (0.5, 0.3, 0.2),
        mr_period: int = 5,
        mr_weight: float = 0.15,
        target_vol: float = 0.10,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.horizons = horizons
        self.weights = weights
        self.mr_period = mr_period
        self.mr_weight = mr_weight
        self.target_vol = target_vol

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"]
        score = pd.Series(0.0, index=data.index)

        for h, w in zip(self.horizons, self.weights):
            mom = close.pct_change(h)
            # z-score the momentum so horizons are comparable
            z = (mom - mom.rolling(252).mean()) / mom.rolling(252).std()
            score = score + w * z.fillna(0.0)

        # short-term mean-reversion overlay (fade recent moves)
        mr = -close.pct_change(self.mr_period)
        mr_z = (mr - mr.rolling(60).mean()) / mr.rolling(60).std()
        score = score + self.mr_weight * mr_z.fillna(0.0)

        # map score to position via tanh (smooth, bounded)
        raw = np.tanh(score)

        weights = vol_target_weights(
            pd.Series(raw, index=data.index),
            close,
            target_vol=self.target_vol,
            max_leverage=1.5,
        )
        return weights


class BreakoutWithFilter(Strategy):
    """
    Donchian-style breakout, but only takes trades when trend regime agrees
    and volatility is not extreme.
    """

    name = "breakout_filtered"

    def __init__(
        self,
        channel: int = 55,
        exit_channel: int = 20,
        target_vol: float = 0.15,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.channel = channel
        self.exit_channel = exit_channel
        self.target_vol = target_vol

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        high = data["high"]
        low = data["low"]
        close = data["close"]

        upper = high.rolling(self.channel).max().shift(1)
        lower = low.rolling(self.channel).min().shift(1)
        exit_upper = high.rolling(self.exit_channel).max().shift(1)
        exit_lower = low.rolling(self.exit_channel).min().shift(1)

        signal = pd.Series(0.0, index=data.index)

        # entry
        long_entry = close > upper
        short_entry = close < lower

        # simple state machine
        pos = 0.0
        positions = []
        for i in range(len(close)):
            c = close.iloc[i]
            if pos == 0:
                if long_entry.iloc[i]:
                    pos = 1.0
                elif short_entry.iloc[i]:
                    pos = -1.0
            elif pos > 0:
                if c < exit_lower.iloc[i]:
                    pos = 0.0
            elif pos < 0:
                if c > exit_upper.iloc[i]:
                    pos = 0.0
            positions.append(pos)

        signal = pd.Series(positions, index=data.index)

        # regime filter
        trend = trend_regime(close)
        vol_reg = volatility_regime(close)
        exposure = combine_regimes(trend, vol_reg)
        filtered = signal * exposure

        return vol_target_weights(filtered, close, target_vol=self.target_vol)


class ResidualMeanReversion(Strategy):
    """
    Mean-reversion on the residual after removing a slow trend.

    Idea: price = trend + residual. Trade the residual, not the raw price.
    """

    name = "residual_mean_reversion"

    def __init__(
        self,
        trend_window: int = 100,
        entry_z: float = 1.5,
        exit_z: float = 0.25,
        target_vol: float = 0.08,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.trend_window = trend_window
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.target_vol = target_vol

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"]
        trend = close.rolling(self.trend_window).mean()
        residual = close - trend
        z = (residual - residual.rolling(60).mean()) / residual.rolling(60).std()

        signal = pd.Series(0.0, index=data.index)
        pos = 0.0
        positions = []

        for i in range(len(z)):
            zi = z.iloc[i]
            if np.isnan(zi):
                positions.append(0.0)
                continue

            if pos == 0:
                if zi < -self.entry_z:
                    pos = 1.0
                elif zi > self.entry_z:
                    pos = -1.0
            elif pos > 0 and zi > -self.exit_z:
                pos = 0.0
            elif pos < 0 and zi < self.exit_z:
                pos = 0.0

            positions.append(pos)

        signal = pd.Series(positions, index=data.index)

        # only trade residual MR in low/normal vol
        vol_reg = volatility_regime(close)
        signal[vol_reg > 0] = 0.0

        return vol_target_weights(signal, close, target_vol=self.target_vol, max_leverage=1.0)

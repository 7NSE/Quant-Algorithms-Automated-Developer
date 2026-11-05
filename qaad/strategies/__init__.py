from .base import Strategy
from .examples import (
    SMACrossover,
    RSIMeanReversion,
    BollingerMeanReversion,
    VolatilityBreakout,
)
from .advanced import (
    TrendFollowingVolTarget,
    MultiFactorMomentum,
    BreakoutWithFilter,
    ResidualMeanReversion,
)

__all__ = [
    "Strategy",
    "SMACrossover",
    "RSIMeanReversion",
    "BollingerMeanReversion",
    "VolatilityBreakout",
    "TrendFollowingVolTarget",
    "MultiFactorMomentum",
    "BreakoutWithFilter",
    "ResidualMeanReversion",
]

"""
Small data helpers.

Keep it boring and reliable — load CSV, make synthetic data for tests,
maybe pull from yfinance later if the user has it installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd


def load_ohlcv(path: str | Path, parse_dates: bool = True) -> pd.DataFrame:
    """
    Load a CSV that has at least open/high/low/close.
    Tries to be flexible with column names.
    """
    path = Path(path)
    df = pd.read_csv(path)

    # normalise column names
    df.columns = [c.strip().lower() for c in df.columns]

    rename_map = {}
    for col in df.columns:
        if col in ("date", "datetime", "time", "timestamp"):
            rename_map[col] = "date"
        elif col.startswith("open"):
            rename_map[col] = "open"
        elif col.startswith("high"):
            rename_map[col] = "high"
        elif col.startswith("low"):
            rename_map[col] = "low"
        elif col.startswith("close") or col == "adj close":
            rename_map[col] = "close"
        elif col.startswith("vol"):
            rename_map[col] = "volume"

    df = df.rename(columns=rename_map)

    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if "date" in df.columns and parse_dates:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()

    return df


def make_synthetic_ohlcv(
    n: int = 1000,
    start_price: float = 100.0,
    volatility: float = 0.015,
    drift: float = 0.0002,
    seed: Optional[int] = 42,
) -> pd.DataFrame:
    """
    Generate random-walk OHLCV so we can test strategies without real data.
    """
    rng = np.random.default_rng(seed)
    returns = rng.normal(drift, volatility, size=n)
    close = start_price * np.cumprod(1.0 + returns)

    # crude OHLC from close
    noise = rng.uniform(0.002, 0.008, size=n)
    high = close * (1.0 + noise)
    low = close * (1.0 - noise)
    open_ = np.roll(close, 1)
    open_[0] = start_price
    volume = rng.integers(1000, 50_000, size=n).astype(float)

    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=idx,
    )


def try_yfinance(symbol: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """
    Optional helper. Only works if yfinance is installed.
    """
    try:
        import yfinance as yf
    except ImportError as e:
        raise ImportError("yfinance is not installed. pip install yfinance") from e

    df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data returned for {symbol}")

    df = df.rename(columns=str.lower)
    # yfinance sometimes returns multi-index columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    return df

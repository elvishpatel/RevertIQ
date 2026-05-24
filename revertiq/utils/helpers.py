"""
RevertIQ — Shared Helpers
=========================

Small, reusable utility functions used across multiple modules.
"""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────
# Path / Directory Helpers
# ─────────────────────────────────────────────────────────────────────

def ensure_dir(path: str) -> str:
    """Create directory (and parents) if it doesn't exist.  Returns *path*."""
    os.makedirs(path, exist_ok=True)
    return path


# ─────────────────────────────────────────────────────────────────────
# Date / Trading Calendar Helpers
# ─────────────────────────────────────────────────────────────────────

def trading_days_between(
    start: str,
    end: str,
    freq: str = "B",
) -> pd.DatetimeIndex:
    """
    Generate business-day index between two dates.

    This is an approximation — Indian market holidays are not removed.
    For exact calendars, filter against actual price data availability.
    """
    return pd.bdate_range(start=start, end=end, freq=freq)


def parse_date(date_str: str) -> pd.Timestamp:
    """Flexibly parse a date string."""
    return pd.Timestamp(date_str)


# ─────────────────────────────────────────────────────────────────────
# DataFrame Helpers
# ─────────────────────────────────────────────────────────────────────

def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Element-wise division that returns NaN instead of raising on zero."""
    return numerator / denominator.replace(0, np.nan)


def clip_outliers(
    series: pd.Series,
    lower_pct: float = 1.0,
    upper_pct: float = 99.0,
) -> pd.Series:
    """Winsorize a Series at the given percentiles."""
    lo = np.nanpercentile(series, lower_pct)
    hi = np.nanpercentile(series, upper_pct)
    return series.clip(lower=lo, upper=hi)


def normalize_min_max(series: pd.Series) -> pd.Series:
    """Min-max normalize a Series to [0, 1].  Returns NaN for constant input."""
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(np.nan, index=series.index)
    return (series - lo) / (hi - lo)


def multi_stock_pivot(
    long_df: pd.DataFrame,
    value_col: str,
    ticker_col: str = "ticker",
    date_col: str = "date",
) -> pd.DataFrame:
    """
    Pivot a long-format DataFrame to wide format.

    Parameters
    ----------
    long_df : DataFrame
        Must have columns for date, ticker, and the value.
    value_col : str
        Column name to pivot as values.

    Returns
    -------
    DataFrame
        Index = date, Columns = tickers, Values = value_col.
    """
    return long_df.pivot_table(
        index=date_col, columns=ticker_col, values=value_col
    )


def flatten_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten yfinance MultiIndex columns from ``(Price, Ticker)`` to
    simple lowercase names.  Works for both single- and multi-ticker
    downloads.
    """
    if isinstance(df.columns, pd.MultiIndex):
        # Multi-ticker: (Price, Ticker) — we keep just the price level
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    return df


def annualize_returns(daily_returns: pd.Series, periods: int = 252) -> float:
    """Compound daily returns to annualized return."""
    total = (1 + daily_returns).prod()
    n_years = len(daily_returns) / periods
    if n_years <= 0:
        return 0.0
    return float(total ** (1 / n_years) - 1)


def annualize_volatility(daily_returns: pd.Series, periods: int = 252) -> float:
    """Annualize standard deviation of daily returns."""
    return float(daily_returns.std() * np.sqrt(periods))

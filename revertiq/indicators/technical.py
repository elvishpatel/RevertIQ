"""
RevertIQ — Technical Indicators (Pure pandas/numpy)
====================================================

All indicators are implemented from scratch using pandas and numpy
for full transparency, educational value, and correctness.

NO external indicator libraries (``ta``, ``ta-lib``) are used.

Key Design Decisions
--------------------
- **Wilder's smoothing** is used for RSI and ATR — this is the industry
  standard and matches what charting platforms display.
- **Lookahead-bias prevention**: The ``zscore`` method shifts the rolling
  mean and std by 1 day so that today's signal uses only yesterday's
  statistics.
- **Vectorized**: Every method operates on entire Series/DataFrames via
  pandas built-ins — no Python ``for`` loops.

Usage
-----
>>> from revertiq.indicators.technical import TechnicalIndicators
>>> from revertiq.config.settings import IndicatorConfig
>>> ti = TechnicalIndicators(IndicatorConfig())
>>> enriched_df = ti.compute_all(single_stock_df)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from revertiq.config.settings import IndicatorConfig
from revertiq.utils.logger import get_logger

logger = get_logger(__name__)


class TechnicalIndicators:
    """Compute technical indicators for a single-stock OHLCV DataFrame.

    The input DataFrame must have lowercase columns:
    ``open``, ``high``, ``low``, ``close``, ``volume``.

    Each indicator is exposed as a ``@staticmethod`` so it can also be
    called independently without instantiation.

    Parameters
    ----------
    config : IndicatorConfig
        Dataclass containing all tuneable indicator parameters.

    Examples
    --------
    >>> cfg = IndicatorConfig(rsi_period=2, atr_period=14)
    >>> ti = TechnicalIndicators(cfg)
    >>> df = ti.compute_all(stock_df)
    >>> df.columns  # original cols + all indicator cols
    """

    # Columns required in the input DataFrame
    REQUIRED_COLUMNS = {"open", "high", "low", "close", "volume"}

    def __init__(self, config: Optional[IndicatorConfig] = None) -> None:
        self.config = config or IndicatorConfig()

    # ─────────────────────────────────────────────────────────────────
    # Public orchestrator
    # ─────────────────────────────────────────────────────────────────

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add every configured indicator column to *df* and return it.

        The DataFrame is modified **in-place** for memory efficiency and
        also returned for method-chaining convenience.

        Parameters
        ----------
        df : pd.DataFrame
            Single-stock OHLCV DataFrame (must contain ``open``, ``high``,
            ``low``, ``close``, ``volume``).

        Returns
        -------
        pd.DataFrame
            The same DataFrame with indicator columns appended.

        Raises
        ------
        ValueError
            If required columns are missing.
        """
        self._validate(df)
        cfg = self.config

        logger.info(
            "Computing all technical indicators (RSI=%d, ATR=%d, BB=%d) ...",
            cfg.rsi_period, cfg.atr_period, cfg.bb_period,
        )

        # ── RSI ──────────────────────────────────────────────────────
        df[f"rsi_{cfg.rsi_period}"] = self.rsi(df["close"], period=cfg.rsi_period)

        # ── SMA family ───────────────────────────────────────────────
        for p in cfg.sma_periods:
            df[f"sma_{p}"] = self.sma(df["close"], period=p)

        # ── EMA family ───────────────────────────────────────────────
        for p in cfg.ema_periods:
            df[f"ema_{p}"] = self.ema(df["close"], period=p)

        # ── ATR ──────────────────────────────────────────────────────
        df[f"atr_{cfg.atr_period}"] = self.atr(
            df["high"], df["low"], df["close"], period=cfg.atr_period,
        )

        # ── Bollinger Bands ──────────────────────────────────────────
        upper, middle, lower = self.bollinger_bands(
            df["close"], period=cfg.bb_period, std_dev=cfg.bb_std_dev,
        )
        df["bb_upper"] = upper
        df["bb_middle"] = middle
        df["bb_lower"] = lower
        # Percent-B: where price sits relative to the bands (0 = lower, 1 = upper)
        band_width = upper - lower
        df["bb_pct_b"] = np.where(
            band_width != 0,
            (df["close"] - lower) / band_width,
            np.nan,
        )

        # ── Rolling Returns ──────────────────────────────────────────
        returns_dict = self.rolling_returns(df["close"], periods=cfg.return_periods)
        for period, ret_series in returns_dict.items():
            df[f"return_{period}d"] = ret_series

        # ── Rolling Volatility ───────────────────────────────────────
        df["volatility_ann"] = self.rolling_volatility(
            df["close"], lookback=cfg.volatility_lookback,
        )

        # ── Z-Score (lookahead-safe) ─────────────────────────────────
        df["zscore"] = self.zscore(df["close"], lookback=cfg.zscore_lookback)

        # ── Volume Spike ─────────────────────────────────────────────
        df["volume_spike"] = self.volume_spike(
            df["volume"], ma_period=cfg.volume_ma_period,
        )

        # ── Distance from MA ────────────────────────────────────────
        df["dist_from_ma_20"] = self.distance_from_ma(
            df["close"], ma_period=cfg.zscore_lookback,
        )

        # ── Percentile Rank ──────────────────────────────────────────
        df["percentile_rank_252"] = self.percentile_rank(df["close"], lookback=252)

        logger.info(
            "Indicators complete — %d columns total.", len(df.columns),
        )
        return df

    # ─────────────────────────────────────────────────────────────────
    # Individual indicator methods
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def rsi(close: pd.Series, period: int = 2) -> pd.Series:
        """Relative Strength Index using **Wilder's smoothing** (EWM).

        Formula
        -------
        .. math::

            \\Delta_t = \\text{close}_t - \\text{close}_{t-1}

            \\text{gain}_t = \\max(\\Delta_t, 0)

            \\text{loss}_t = |\\min(\\Delta_t, 0)|

            \\text{avg\\_gain}_t = \\text{EWM}(\\text{gain}, \\alpha=1/\\text{period})

            \\text{avg\\_loss}_t = \\text{EWM}(\\text{loss}, \\alpha=1/\\text{period})

            RS = \\frac{\\text{avg\\_gain}}{\\text{avg\\_loss}}, \\quad
            RSI = 100 - \\frac{100}{1 + RS}

        Wilder's smoothing uses ``ewm(alpha=1/period, min_periods=period)``
        which is equivalent to ``ewm(com=period-1)``.

        Parameters
        ----------
        close : pd.Series
            Closing price series.
        period : int
            Lookback window (default 2 for Connors-style mean reversion).

        Returns
        -------
        pd.Series
            RSI values in [0, 100].  First ``period`` rows will be NaN.
        """
        # Daily price change
        delta = close.diff()

        # Separate gains (positive deltas) and losses (absolute negative deltas)
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)

        # Wilder's exponential moving average (alpha = 1/period)
        avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

        # Relative Strength → RSI (handle div-by-zero when avg_loss is 0)
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi_values = 100.0 - (100.0 / (1.0 + rs))

        # If avg_loss is exactly 0 (all gains), RSI should be 100
        rsi_values = rsi_values.fillna(100.0)

        # Keep first `period` values as NaN (not enough data)
        rsi_values.iloc[:period] = np.nan

        return rsi_values

    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average.

        Formula
        -------
        .. math::

            SMA_t = \\frac{1}{N} \\sum_{i=0}^{N-1} x_{t-i}

        Parameters
        ----------
        series : pd.Series
            Input price or indicator series.
        period : int
            Rolling window size *N*.

        Returns
        -------
        pd.Series
            SMA values.  First ``period - 1`` rows will be NaN.
        """
        return series.rolling(window=period, min_periods=period).mean()

    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average.

        Formula
        -------
        .. math::

            \\alpha = \\frac{2}{\\text{period} + 1}

            EMA_t = \\alpha \\cdot x_t + (1 - \\alpha) \\cdot EMA_{t-1}

        Parameters
        ----------
        series : pd.Series
            Input price or indicator series.
        period : int
            Span for the EWM computation.

        Returns
        -------
        pd.Series
            EMA values.
        """
        return series.ewm(span=period, min_periods=period, adjust=False).mean()

    @staticmethod
    def atr(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
    ) -> pd.Series:
        """Average True Range using **Wilder's smoothing**.

        Formula
        -------
        .. math::

            TR_t = \\max\\bigl(H_t - L_t,\\;|H_t - C_{t-1}|,\\;|L_t - C_{t-1}|\\bigr)

            ATR_t = \\text{EWM}(TR, \\alpha = 1/\\text{period})

        Parameters
        ----------
        high : pd.Series
            High prices.
        low : pd.Series
            Low prices.
        close : pd.Series
            Close prices.
        period : int
            Smoothing period (default 14).

        Returns
        -------
        pd.Series
            ATR values.  First ``period`` rows are NaN.
        """
        # Previous close (shifted by 1 to avoid using future data)
        prev_close = close.shift(1)

        # Three components of True Range
        tr1 = high - low                       # Intraday range
        tr2 = (high - prev_close).abs()        # Gap-up absorbed
        tr3 = (low - prev_close).abs()         # Gap-down absorbed

        # True Range = maximum of the three
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Wilder's smoothing via EWM (alpha = 1/period)
        atr_values = true_range.ewm(
            alpha=1.0 / period, min_periods=period, adjust=False,
        ).mean()

        return atr_values

    @staticmethod
    def bollinger_bands(
        close: pd.Series,
        period: int = 20,
        std_dev: float = 2.0,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands.

        Formula
        -------
        .. math::

            \\text{Middle} = SMA(\\text{close}, N)

            \\text{Upper}  = \\text{Middle} + k \\cdot \\sigma_N

            \\text{Lower}  = \\text{Middle} - k \\cdot \\sigma_N

        where :math:`\\sigma_N` is the rolling standard deviation over *N*
        periods and *k* is the number of standard deviations (default 2).

        Parameters
        ----------
        close : pd.Series
            Closing prices.
        period : int
            Rolling window for the SMA and std (default 20).
        std_dev : float
            Number of standard deviations (default 2.0).

        Returns
        -------
        tuple[pd.Series, pd.Series, pd.Series]
            ``(upper_band, middle_band, lower_band)``.
        """
        middle = close.rolling(window=period, min_periods=period).mean()
        rolling_std = close.rolling(window=period, min_periods=period).std(ddof=0)

        upper = middle + std_dev * rolling_std
        lower = middle - std_dev * rolling_std

        return upper, middle, lower

    @staticmethod
    def rolling_returns(
        close: pd.Series,
        periods: Optional[List[int]] = None,
    ) -> Dict[int, pd.Series]:
        """Percentage returns over multiple lookback windows.

        Formula
        -------
        .. math::

            R_{t,n} = \\frac{C_t - C_{t-n}}{C_{t-n}}
                    = \\text{pct\\_change}(n)

        Parameters
        ----------
        close : pd.Series
            Closing prices.
        periods : list[int], optional
            List of return horizons in trading days (default [1, 5, 10, 20]).

        Returns
        -------
        dict[int, pd.Series]
            Mapping of period → return Series.
        """
        if periods is None:
            periods = [1, 5, 10, 20]

        return {p: close.pct_change(periods=p) for p in periods}

    @staticmethod
    def rolling_volatility(close: pd.Series, lookback: int = 20) -> pd.Series:
        """Annualized rolling volatility.

        Formula
        -------
        .. math::

            \\sigma_{\\text{ann}} = \\text{std}(r_{t-N+1} \\dots r_t) \\times \\sqrt{252}

        where :math:`r_t = \\ln(C_t / C_{t-1})` are log returns and 252
        is the approximate number of trading days per year.

        Parameters
        ----------
        close : pd.Series
            Closing prices.
        lookback : int
            Rolling window in trading days (default 20).

        Returns
        -------
        pd.Series
            Annualized volatility (as a decimal, e.g. 0.25 = 25 %).
        """
        # Log returns for volatility (more statistically sound than simple returns)
        log_returns = np.log(close / close.shift(1))

        # Rolling standard deviation of daily log returns, annualized
        rolling_std = log_returns.rolling(window=lookback, min_periods=lookback).std(ddof=1)
        annualized = rolling_std * np.sqrt(252)

        return annualized

    @staticmethod
    def zscore(close: pd.Series, lookback: int = 20) -> pd.Series:
        """Rolling Z-score of price — **lookahead-bias safe**.

        The rolling mean and standard deviation are computed over the
        previous *lookback* days and then **shifted by 1** so that
        today's Z-score uses only information available *yesterday*.

        Formula
        -------
        .. math::

            \\mu_{t-1} = \\text{mean}(C_{t-N-1} \\dots C_{t-2})  \\quad \\text{(shifted)}

            \\sigma_{t-1} = \\text{std}(C_{t-N-1} \\dots C_{t-2})  \\quad \\text{(shifted)}

            Z_t = \\frac{C_t - \\mu_{t-1}}{\\sigma_{t-1}}

        CRITICAL: ``.shift(1)`` on rolling_mean and rolling_std prevents
        the current bar's price from influencing its own statistics.

        Parameters
        ----------
        close : pd.Series
            Closing prices.
        lookback : int
            Window for rolling statistics (default 20).

        Returns
        -------
        pd.Series
            Z-score values (unbounded, typically in [-3, +3]).
        """
        # Compute rolling stats using a lookback window
        rolling_mean = close.rolling(window=lookback, min_periods=lookback).mean()
        rolling_std = close.rolling(window=lookback, min_periods=lookback).std(ddof=1)

        # CRITICAL: shift by 1 to prevent lookahead bias
        # Today's z-score uses yesterday's completed rolling window stats
        shifted_mean = rolling_mean.shift(1)
        shifted_std = rolling_std.shift(1)

        # Z = (current_price - lagged_mean) / lagged_std
        # Replace zero std with NaN to avoid division by zero
        z = (close - shifted_mean) / shifted_std.replace(0, np.nan)

        return z

    @staticmethod
    def volume_spike(volume: pd.Series, ma_period: int = 20) -> pd.Series:
        """Volume spike detector — ratio of current volume to its moving average.

        Formula
        -------
        .. math::

            \\text{spike}_t = \\frac{V_t}{SMA(V, N)_t}

        A value > 1 means above-average volume; > 2 suggests a strong
        spike which often accompanies capitulation bottoms.

        Parameters
        ----------
        volume : pd.Series
            Trading volume series.
        ma_period : int
            Lookback for the volume moving average (default 20).

        Returns
        -------
        pd.Series
            Volume spike ratio (1.0 = average).
        """
        vol_ma = volume.rolling(window=ma_period, min_periods=ma_period).mean()

        # Divide current volume by its moving average (NaN-safe)
        spike = volume / vol_ma.replace(0, np.nan)

        return spike

    @staticmethod
    def distance_from_ma(close: pd.Series, ma_period: int = 20) -> pd.Series:
        """Percentage distance of price from its simple moving average.

        Formula
        -------
        .. math::

            d_t = \\frac{C_t - SMA_t}{SMA_t}

        Negative values mean price is *below* the SMA (potential mean-
        reversion buying opportunity).

        Parameters
        ----------
        close : pd.Series
            Closing prices.
        ma_period : int
            SMA lookback period (default 20).

        Returns
        -------
        pd.Series
            Fractional distance (e.g. -0.03 = 3 % below SMA).
        """
        sma_values = close.rolling(window=ma_period, min_periods=ma_period).mean()

        return (close - sma_values) / sma_values.replace(0, np.nan)

    @staticmethod
    def percentile_rank(close: pd.Series, lookback: int = 252) -> pd.Series:
        """Rolling percentile rank — where the current price sits in its
        trailing *N*-day range (0–100 scale).

        Formula
        -------
        .. math::

            \\text{rank}_t = \\frac{C_t - \\min(C_{t-N+1} \\dots C_t)}
                                   {\\max(C_{t-N+1} \\dots C_t)
                                    - \\min(C_{t-N+1} \\dots C_t)}
                            \\times 100

        A value near 0 means the stock is trading near its 252-day low
        (potential oversold); near 100 means near the high.

        Parameters
        ----------
        close : pd.Series
            Closing prices.
        lookback : int
            Window in trading days (default 252 ≈ 1 year).

        Returns
        -------
        pd.Series
            Percentile rank in [0, 100].
        """
        roll_min = close.rolling(window=lookback, min_periods=lookback).min()
        roll_max = close.rolling(window=lookback, min_periods=lookback).max()

        # Range (max - min); replace zero to avoid div-by-zero (flat stock)
        price_range = (roll_max - roll_min).replace(0, np.nan)

        pct_rank = ((close - roll_min) / price_range) * 100.0

        return pct_rank

    # ─────────────────────────────────────────────────────────────────
    # Validation
    # ─────────────────────────────────────────────────────────────────

    @classmethod
    def _validate(cls, df: pd.DataFrame) -> None:
        """Ensure required columns exist in the DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame to validate.

        Raises
        ------
        ValueError
            If any of the required OHLCV columns are missing.
        """
        missing = cls.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(
                f"Missing required columns: {sorted(missing)}. "
                f"DataFrame has: {list(df.columns)}"
            )

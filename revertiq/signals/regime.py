"""
RevertIQ — Market Regime Detection
====================================

Classifies the prevailing market regime into one of four states based on
NIFTY 50 price action and volatility conditions:

    - ``strong_bull``: Price > SMA200, SMA50 > SMA200, and low volatility
    - ``mild_bull``:   Price > SMA200 but missing one secondary condition
    - ``cautious``:    Price < SMA200 but SMA50 still > SMA200
    - ``bearish``:     Price < SMA200 and SMA50 < SMA200

Design decisions
----------------
1. A **1 % buffer zone** around SMA200 avoids rapid regime flips when
   price oscillates near the moving average ("whipsaw").
2. A **3-day confirmation** window prevents single-day noise from
   triggering a regime change.
3. Volatility is measured via India VIX when available; otherwise a
   20-day realized vol is compared against the 75th percentile of its
   own 252-day history.
4. Every rolling / moving-average result is **.shift(1)** so the regime
   label on day *T* only uses information available at close of *T−1*.

Usage
-----
>>> from revertiq.signals.regime import RegimeFilter
>>> rf = RegimeFilter()
>>> regime_series = rf.compute_regime(nifty_df, vix_df)
>>> rf.is_trading_allowed("strong_bull")  # True
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from revertiq.config.settings import RegimeConfig, get_default_config
from revertiq.utils.logger import get_logger

logger = get_logger(__name__)


class RegimeFilter:
    """Detect the prevailing market regime for NIFTY 50.

    Parameters
    ----------
    config : RegimeConfig, optional
        Regime-filter parameters.  Falls back to the global default when
        *None*.

    Attributes
    ----------
    config : RegimeConfig
        Active configuration.
    """

    # Human-readable regime ordering (worst → best)
    REGIME_ORDER = ["bearish", "cautious", "mild_bull", "strong_bull"]

    def __init__(self, config: Optional[RegimeConfig] = None) -> None:
        self.config: RegimeConfig = config or get_default_config().regime
        logger.info(
            "RegimeFilter initialised  |  MA=%d  buffer=%.1f%%  "
            "confirm=%d days  allowed=%s",
            self.config.regime_ma_period,
            self.config.regime_buffer_pct,
            self.config.confirmation_days,
            self.config.allowed_regimes,
        )

    # ────────────────────────────────────────────────────────────────
    # Public API
    # ────────────────────────────────────────────────────────────────

    def compute_regime(
        self,
        nifty_data: pd.DataFrame,
        vix_data: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """Compute the market regime for every date in *nifty_data*.

        Parameters
        ----------
        nifty_data : DataFrame
            Must contain at least a ``'close'`` column with a
            ``DatetimeIndex`` (or a ``'date'`` column that will be set
            as the index).
        vix_data : DataFrame, optional
            India VIX data with a ``'close'`` column.  When *None* the
            filter falls back to NIFTY realised volatility.

        Returns
        -------
        pd.Series
            Index aligned with *nifty_data*; values are one of
            ``'strong_bull'``, ``'mild_bull'``, ``'cautious'``,
            ``'bearish'``, or ``np.nan`` during the warm-up window.
        """
        nifty = self._prepare_dataframe(nifty_data)
        logger.info("Computing regime for %d trading days", len(nifty))

        # ── Step 1: Moving averages (shifted to avoid lookahead) ─────
        sma200 = nifty["close"].rolling(
            window=self.config.regime_ma_period, min_periods=self.config.regime_ma_period
        ).mean().shift(1)

        sma50 = nifty["close"].rolling(window=50, min_periods=50).mean().shift(1)

        # Use previous-day close for comparisons (signal at T uses T-1 data)
        close_prev = nifty["close"].shift(1)

        # ── Step 2: Buffer zone around SMA200 ───────────────────────
        buffer_frac = self.config.regime_buffer_pct / 100.0
        upper_band = sma200 * (1.0 + buffer_frac)
        lower_band = sma200 * (1.0 - buffer_frac)

        # ── Step 3: Volatility filter ────────────────────────────────
        low_vol = self._compute_low_volatility_flag(nifty, vix_data)

        # ── Step 4: Raw (unconfirmed) regime classification ──────────
        raw_regime = self._classify_raw(
            close_prev, sma200, sma50, upper_band, lower_band, low_vol
        )

        # ── Step 5: Confirmation — require N consecutive identical ───
        confirmed = self._apply_confirmation(raw_regime)

        logger.info(
            "Regime distribution:\n%s",
            confirmed.value_counts(dropna=False).to_string(),
        )
        return confirmed

    def is_trading_allowed(self, regime: str) -> bool:
        """Return *True* if the given regime permits new long entries.

        Parameters
        ----------
        regime : str
            One of ``'strong_bull'``, ``'mild_bull'``, ``'cautious'``,
            ``'bearish'``.

        Returns
        -------
        bool
        """
        return regime in self.config.allowed_regimes

    def get_regime_strength(self, regime: str) -> float:
        """Map a regime label to a numeric strength in [0, 1].

        Parameters
        ----------
        regime : str
            Regime label.

        Returns
        -------
        float
            0.0 for bearish → 1.0 for strong_bull.  NaN for unknown.
        """
        try:
            idx = self.REGIME_ORDER.index(regime)
            return idx / (len(self.REGIME_ORDER) - 1)
        except ValueError:
            return np.nan

    # ────────────────────────────────────────────────────────────────
    # Internal helpers
    # ────────────────────────────────────────────────────────────────

    @staticmethod
    def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure ``DatetimeIndex`` and lowercase column names.

        Parameters
        ----------
        df : DataFrame
            Raw input data.

        Returns
        -------
        DataFrame
            Copy with a proper DatetimeIndex.
        """
        df = df.copy()
        df.columns = [c.lower().strip() for c in df.columns]

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        df = df.sort_index()
        return df

    def _compute_low_volatility_flag(
        self,
        nifty: pd.DataFrame,
        vix_data: Optional[pd.DataFrame],
    ) -> pd.Series:
        """Return a boolean Series — *True* when volatility is "low".

        Uses India VIX if provided; otherwise falls back to realised
        volatility of NIFTY daily returns.

        Parameters
        ----------
        nifty : DataFrame
            NIFTY price data (``close`` column required).
        vix_data : DataFrame or None
            Optional VIX data (``close`` column required).

        Returns
        -------
        pd.Series[bool]
            Aligned to *nifty* index, shifted by 1 day.
        """
        if vix_data is not None and len(vix_data) > 0:
            return self._vix_low_vol(nifty, vix_data)
        return self._realized_low_vol(nifty)

    def _vix_low_vol(
        self, nifty: pd.DataFrame, vix_data: pd.DataFrame
    ) -> pd.Series:
        """VIX-based low-volatility flag.

        VIX is considered "low" when it sits below the rolling 75th
        percentile of its own trailing 252-day window.

        Parameters
        ----------
        nifty : DataFrame
            Used only for index alignment.
        vix_data : DataFrame
            Must contain ``'close'`` (VIX level).

        Returns
        -------
        pd.Series[bool]
        """
        logger.info("Using India VIX for volatility filter")
        vix = self._prepare_dataframe(vix_data)

        # Rolling 75th percentile of VIX over the benchmark window
        benchmark = self.config.realized_vol_benchmark
        pctile = self.config.vix_threshold_percentile  # 0.75

        vix_rolling_q = (
            vix["close"]
            .rolling(window=benchmark, min_periods=benchmark // 2)
            .quantile(pctile)
            .shift(1)  # no lookahead
        )
        vix_prev = vix["close"].shift(1)

        # Low vol when yesterday's VIX < rolling 75th percentile
        low_vol = vix_prev < vix_rolling_q

        # Reindex to NIFTY dates, forward-filling for minor date mismatches
        low_vol = low_vol.reindex(nifty.index, method="ffill")
        return low_vol.fillna(False)

    def _realized_low_vol(self, nifty: pd.DataFrame) -> pd.Series:
        """Realised-vol-based low-volatility flag (fallback).

        Computes 20-day realised volatility and compares it against the
        75th percentile of a trailing 252-day window.

        Parameters
        ----------
        nifty : DataFrame
            NIFTY price data.

        Returns
        -------
        pd.Series[bool]
        """
        logger.info("VIX not available — using realised vol fallback")
        lookback = self.config.realized_vol_lookback    # 20
        benchmark = self.config.realized_vol_benchmark  # 252
        pctile = self.config.vix_threshold_percentile   # 0.75

        # Daily log returns
        log_ret = np.log(nifty["close"] / nifty["close"].shift(1))

        # 20-day realised vol (annualised)
        realized_vol = log_ret.rolling(
            window=lookback, min_periods=lookback
        ).std() * np.sqrt(252)

        # 75th percentile of realised vol over the trailing year
        vol_threshold = (
            realized_vol
            .rolling(window=benchmark, min_periods=benchmark // 2)
            .quantile(pctile)
        )

        # Shift both by 1 day — avoid lookahead
        low_vol = realized_vol.shift(1) < vol_threshold.shift(1)
        return low_vol.fillna(False)

    # ── Raw classification ───────────────────────────────────────────

    @staticmethod
    def _classify_raw(
        close: pd.Series,
        sma200: pd.Series,
        sma50: pd.Series,
        upper_band: pd.Series,
        lower_band: pd.Series,
        low_vol: pd.Series,
    ) -> pd.Series:
        """Assign a raw (unconfirmed) regime label to each date.

        The buffer band replaces the simple ``close > sma200`` check:
        - close > upper_band  ⟹  above SMA200 zone
        - close < lower_band  ⟹  below SMA200 zone
        - otherwise           ⟹  inside the buffer (keep previous)

        Parameters
        ----------
        close, sma200, sma50 : Series
            All *already shifted* by 1 day.
        upper_band, lower_band : Series
            SMA200 ± buffer.
        low_vol : Series[bool]
            True when volatility is low.

        Returns
        -------
        pd.Series
            String regime labels (or NaN during warm-up).
        """
        n = len(close)

        # Boolean conditions (vectorised)
        above_zone = close > upper_band
        below_zone = close < lower_band
        sma50_above_200 = sma50 > sma200

        # Start with NaN, fill in outside-buffer cases first
        regime = pd.Series(np.nan, index=close.index, dtype=object)

        # ── Price clearly ABOVE the SMA200 buffer zone ──
        # strong_bull: above zone + SMA50 > SMA200 + low volatility
        strong = above_zone & sma50_above_200 & low_vol
        regime[strong] = "strong_bull"

        # mild_bull: above zone but missing at least one secondary condition
        mild = above_zone & ~strong
        regime[mild] = "mild_bull"

        # ── Price clearly BELOW the SMA200 buffer zone ──
        # cautious: below zone but SMA50 still > SMA200 (early decline)
        cautious = below_zone & sma50_above_200
        regime[cautious] = "cautious"

        # bearish: below zone AND SMA50 < SMA200
        bearish = below_zone & ~sma50_above_200
        regime[bearish] = "bearish"

        # ── Inside buffer zone — forward-fill from last decisive day ──
        regime = regime.ffill()

        return regime

    # ── Confirmation logic ───────────────────────────────────────────

    def _apply_confirmation(self, raw: pd.Series) -> pd.Series:
        """Require *N* consecutive identical regime labels to confirm.

        The regime changes only after ``confirmation_days`` consecutive
        days of the *same new label*.  Until confirmed, the previous
        regime is carried forward.

        Parameters
        ----------
        raw : Series
            Unconfirmed regime labels.

        Returns
        -------
        pd.Series
            Confirmed regime labels.
        """
        n_confirm = self.config.confirmation_days
        if n_confirm <= 1:
            # No confirmation needed — raw labels are final
            return raw

        # Build a "streak" counter: how many consecutive days has the
        # current raw label been the same?
        #   - When the label changes, streak resets to 1.
        #   - When it stays the same, streak increments.
        raw_values = raw.values.astype(str)
        confirmed = np.empty(len(raw_values), dtype=object)
        confirmed[:] = np.nan

        streak = 1
        prev_raw = raw_values[0]
        last_confirmed = raw_values[0]  # bootstrap with first label

        for i in range(len(raw_values)):
            current_raw = raw_values[i]

            if current_raw == "nan":
                # Warm-up period — nothing to confirm
                confirmed[i] = np.nan
                continue

            if current_raw == prev_raw:
                streak += 1
            else:
                streak = 1

            # Confirm if streak meets the threshold
            if streak >= n_confirm:
                last_confirmed = current_raw

            confirmed[i] = last_confirmed
            prev_raw = current_raw

        return pd.Series(confirmed, index=raw.index, dtype=object).replace(
            "nan", np.nan
        )

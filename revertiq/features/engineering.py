"""
RevertIQ — Cross-Sectional Feature Engineering
===============================================

Computes features ACROSS all stocks on each date, enabling
cross-sectional mean reversion ranking.

Key Features
------------
- Relative return vs market
- Relative weakness (rolling)
- Cross-sectional Z-score (rank among peers)
- Cross-sectional percentile rank
- Volume exhaustion scoring
- Composite mean reversion score

Anti-Lookahead Discipline
--------------------------
Every feature uses ``.shift(1)`` — we compute indicators from data
available *yesterday* so that signals generated today are tradeable
tomorrow at open.

Usage
-----
>>> from revertiq.features.engineering import FeatureEngineer
>>> fe = FeatureEngineer()
>>> enriched = fe.compute_all_features(stock_data, market_data)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from revertiq.config.settings import (
    IndicatorConfig,
    RankingConfig,
    get_default_config,
)
from revertiq.utils.helpers import normalize_min_max, safe_divide
from revertiq.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureEngineer:
    """Compute cross-sectional features for all stocks in the universe.

    Parameters
    ----------
    indicator_cfg : IndicatorConfig, optional
        Indicator parameters.  Falls back to defaults.
    ranking_cfg : RankingConfig, optional
        Ranking weights (used for composite score).  Falls back to defaults.
    """

    def __init__(
        self,
        indicator_cfg: Optional[IndicatorConfig] = None,
        ranking_cfg: Optional[RankingConfig] = None,
    ) -> None:
        cfg = get_default_config()
        self.ind_cfg = indicator_cfg or cfg.indicator
        self.rank_cfg = ranking_cfg or cfg.ranking

    # ─────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────

    def compute_all_features(
        self,
        stock_data: pd.DataFrame,
        market_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute all cross-sectional features.

        Parameters
        ----------
        stock_data : DataFrame
            Combined multi-stock DataFrame **with a ``ticker`` column**
            and columns: date (index or column), open, high, low, close,
            volume, plus any indicator columns already computed by
            ``TechnicalIndicators``.
        market_data : DataFrame
            NIFTY 50 index DataFrame with DatetimeIndex and column ``close``.

        Returns
        -------
        DataFrame
            Original *stock_data* enriched with new feature columns.
        """
        df = stock_data.copy()

        # Ensure 'date' is a column (not index) for groupby operations
        if df.index.name == "date" or "date" not in df.columns:
            if df.index.name == "date":
                df = df.reset_index()
            elif isinstance(df.index, pd.DatetimeIndex):
                df.index.name = "date"
                df = df.reset_index()

        logger.info("Computing cross-sectional features for %d rows …",
                     len(df))

        # ── Step 1: Market returns ──
        market_ret_5 = (
            market_data["close"].pct_change(5).shift(1)  # anti-lookahead
        )
        market_ret_20 = (
            market_data["close"].pct_change(20).shift(1)
        )
        # Map market returns to each stock row
        df["market_ret_5d"] = df["date"].map(market_ret_5)
        df["market_ret_20d"] = df["date"].map(market_ret_20)

        # ── Step 2: Relative return & weakness ──
        df = self._compute_relative_features(df)

        # ── Step 3: Cross-sectional Z-score ──
        df = self._compute_cross_sectional_zscore(df)

        # ── Step 4: Cross-sectional percentile rank ──
        df = self._compute_cross_sectional_percentile(df)

        # ── Step 5: Volume exhaustion ──
        df = self._compute_volume_exhaustion(df)

        # ── Step 6: Composite mean reversion score ──
        df = self._compute_mean_reversion_score(df)

        logger.info("Cross-sectional feature computation complete.")
        return df

    # ─────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────

    def _compute_relative_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Stock return minus market return (relative strength/weakness)."""
        # 5-day stock return — already shifted in indicators via return_5d
        stock_ret_5_col = "return_5d" if "return_5d" in df.columns else None
        stock_ret_20_col = "return_20d" if "return_20d" in df.columns else None

        if stock_ret_5_col:
            df["relative_return_5d"] = df[stock_ret_5_col] - df["market_ret_5d"]
        else:
            # Compute on the fly (shifted for anti-lookahead)
            df["relative_return_5d"] = (
                df.groupby("ticker")["close"]
                .pct_change(5)
                .shift(1)  # anti-lookahead
                - df["market_ret_5d"]
            )

        if stock_ret_20_col:
            df["relative_weakness"] = df[stock_ret_20_col] - df["market_ret_20d"]
        else:
            df["relative_weakness"] = (
                df.groupby("ticker")["close"]
                .pct_change(20)
                .shift(1)
                - df["market_ret_20d"]
            )

        return df

    def _compute_cross_sectional_zscore(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Standardise features *across stocks* on each date.

        Z_cs = (X_stock - mean_allstocks) / std_allstocks

        This tells us how extreme a stock is relative to peers *today*.
        """
        features_to_zscore = [
            col for col in [
                "zscore", "rsi", "distance_from_sma_20",
                "relative_weakness", "volume_spike",
            ]
            if col in df.columns
        ]

        for feat in features_to_zscore:
            col_name = f"cs_zscore_{feat}"
            group_mean = df.groupby("date")[feat].transform("mean")
            group_std = df.groupby("date")[feat].transform("std")
            # Avoid division by zero in cross-section
            df[col_name] = safe_divide(df[feat] - group_mean, group_std)

        return df

    def _compute_cross_sectional_percentile(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Percentile rank of each stock within the daily cross-section.

        A value of 10 means the stock is in the bottom 10 % of the
        universe on that metric — a stronger mean reversion candidate.
        """
        features_to_rank = [
            col for col in [
                "zscore", "rsi", "relative_weakness",
            ]
            if col in df.columns
        ]

        for feat in features_to_rank:
            col_name = f"cs_percentile_{feat}"
            df[col_name] = df.groupby("date")[feat].rank(pct=True) * 100

        return df

    def _compute_volume_exhaustion(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Volume exhaustion = high volume + negative return.

        When a stock drops on high volume, it often signals seller
        exhaustion — a classic mean reversion precursor.

        Score = volume_spike × |negative_return|  (zero if return > 0)
        """
        if "volume_spike" not in df.columns or "return_1d" not in df.columns:
            logger.debug(
                "Skipping volume_exhaustion — missing volume_spike or return_1d"
            )
            df["volume_exhaustion"] = np.nan
            return df

        # Only count as exhaustion when the stock dropped
        neg_return = df["return_1d"].clip(upper=0).abs()
        df["volume_exhaustion"] = df["volume_spike"] * neg_return

        return df

    def _compute_mean_reversion_score(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Composite mean reversion score (higher = stronger candidate).

        Components (inverted where necessary so lower-is-better becomes
        higher-is-better):
          - RSI (inverted)       : lower RSI → higher score
          - Z-score (inverted)   : more negative Z → higher score
          - Relative weakness (inverted) : more negative → higher score
          - Volume exhaustion    : higher → higher score
          - ATR normalised       : moderate ATR preferred
          - Distance from MA (inverted) : further below MA → higher score

        Each component is min-max normalised within the daily cross-
        section to [0, 1] before weighting.
        """
        weights = self.rank_cfg.weights

        # Map component names to actual columns and whether to invert
        component_map = {
            "rsi_score":               ("rsi", True),
            "zscore_score":            ("zscore", True),
            "relative_weakness_score": ("relative_weakness", True),
            "volume_exhaustion_score": ("volume_exhaustion", False),
            "atr_norm_score":          ("atr", True),   # lower ATR = less risky
            "dist_from_ma_score":      ("distance_from_sma_20", True),
        }

        for score_name, (col, invert) in component_map.items():
            if col not in df.columns:
                df[score_name] = np.nan
                continue

            values = df[col].copy()
            if invert:
                values = -values  # Flip so that lower original → higher score

            # Normalise within daily cross-section [0, 1]
            df[score_name] = df.groupby("date")[col].transform(
                lambda s: _safe_min_max(-s if invert else s)
            )

        # Weighted composite score
        available = [k for k in weights if k in df.columns and df[k].notna().any()]
        if available:
            w_sum = sum(weights[k] for k in available)
            df["mean_reversion_score"] = sum(
                df[k] * (weights[k] / w_sum) for k in available
            )
        else:
            df["mean_reversion_score"] = np.nan

        return df


# ─────────────────────────────────────────────────────────────────────
# Module-level helper
# ─────────────────────────────────────────────────────────────────────

def _safe_min_max(series: pd.Series) -> pd.Series:
    """Min-max normalise, returning NaN for constant-valued groups."""
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(0.5, index=series.index)
    return (series - lo) / (hi - lo)

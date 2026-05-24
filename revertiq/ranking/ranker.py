"""
RevertIQ — Stock Ranking Engine
================================

Ranks stocks daily by their mean reversion potential using a
weighted composite score.

Scoring Formula
---------------
::

    score = Σ  w_i × normalise(component_i)

Components (inverted where lower = better for mean reversion):

    ┌─────────────────────┬────────┬────────────────────────────────────┐
    │ Component           │ Weight │ Interpretation                     │
    ├─────────────────────┼────────┼────────────────────────────────────┤
    │ RSI (inv)           │ 0.20   │ Lower RSI → more oversold          │
    │ Z-score (inv)       │ 0.25   │ More negative → more stretched     │
    │ Relative weakness   │ 0.20   │ Underperforming market → reversion │
    │ Volume exhaustion   │ 0.15   │ High-volume drop → seller exhaust  │
    │ ATR normalised (inv)│ 0.10   │ Lower volatility → safer bet       │
    │ Distance from MA    │ 0.10   │ Further below MA → stretched       │
    └─────────────────────┴────────┴────────────────────────────────────┘

All normalisations are **min-max within the daily cross-section** so
that scores are comparable across dates.

Usage
-----
>>> from revertiq.ranking.ranker import StockRanker
>>> ranker = StockRanker()
>>> ranked = ranker.rank_all_dates(features_df, signals_df)
>>> today = ranker.get_top_candidates(ranked, n=5)
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from revertiq.config.settings import RankingConfig, get_default_config
from revertiq.utils.logger import get_logger

logger = get_logger(__name__)


class StockRanker:
    """Rank stocks by mean reversion potential on each trading day.

    Parameters
    ----------
    ranking_cfg : RankingConfig, optional
        Weights for composite score.  Falls back to defaults.
    """

    # Map score-name → (source column, invert?)
    _COMPONENT_MAP = {
        "rsi_score":               ("rsi",                True),
        "zscore_score":            ("zscore",             True),
        "relative_weakness_score": ("relative_weakness",  True),
        "volume_exhaustion_score": ("volume_exhaustion",  False),
        "atr_norm_score":          ("atr",                True),
        "dist_from_ma_score":      ("distance_from_sma_20", True),
    }

    def __init__(self, ranking_cfg: Optional[RankingConfig] = None) -> None:
        cfg = get_default_config()
        self.cfg = ranking_cfg or cfg.ranking
        self.cfg.validate()

    # ─────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────

    def rank_stocks(
        self,
        features_df: pd.DataFrame,
        signals_df: pd.DataFrame,
        date: pd.Timestamp,
    ) -> pd.DataFrame:
        """
        Rank all stocks with active BUY signals on a single date.

        Parameters
        ----------
        features_df : DataFrame
            Multi-stock features (must have ``date`` and ``ticker`` cols).
        signals_df : DataFrame
            Must have ``date``, ``ticker``, ``buy_signal``,
            ``signal_strength`` columns.
        date : Timestamp
            The date to rank.

        Returns
        -------
        DataFrame
            Rows sorted by composite score descending, with columns:
            rank, ticker, date, composite_score, signal_strength,
            plus individual component scores.
        """
        # Filter to given date
        feat_day = features_df[features_df["date"] == date].copy()
        sig_day = signals_df[
            (signals_df["date"] == date) & (signals_df["buy_signal"] == True)
        ]

        if sig_day.empty:
            return pd.DataFrame()

        # Only keep stocks with active buy signals
        active_tickers = set(sig_day["ticker"])
        candidates = feat_day[feat_day["ticker"].isin(active_tickers)].copy()

        if candidates.empty:
            return pd.DataFrame()

        # Compute component scores
        candidates = self._compute_component_scores(candidates)

        # Weighted composite
        candidates["composite_score"] = self._weighted_sum(candidates)

        # Merge signal strength
        strength_map = sig_day.set_index("ticker")["signal_strength"]
        candidates["signal_strength"] = (
            candidates["ticker"].map(strength_map).fillna(0)
        )

        # Rank
        candidates = candidates.sort_values(
            "composite_score", ascending=False
        ).reset_index(drop=True)
        candidates["rank"] = range(1, len(candidates) + 1)
        candidates["date"] = date

        # Select output columns
        out_cols = ["rank", "ticker", "date", "composite_score",
                    "signal_strength"]
        out_cols += [k for k in self._COMPONENT_MAP if k in candidates.columns]
        existing = [c for c in out_cols if c in candidates.columns]

        return candidates[existing]

    def rank_all_dates(
        self,
        features_df: pd.DataFrame,
        signals_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Rank stocks for every date that has at least one BUY signal.

        Returns
        -------
        DataFrame
            Concatenated rankings for all dates, sorted by
            (date, rank).
        """
        # Dates with at least one buy signal
        buy_dates = (
            signals_df[signals_df["buy_signal"] == True]["date"]
            .unique()
        )
        buy_dates = sorted(buy_dates)
        logger.info("Ranking stocks across %d dates with signals", len(buy_dates))

        chunks: List[pd.DataFrame] = []
        for dt in buy_dates:
            ranked = self.rank_stocks(features_df, signals_df, dt)
            if not ranked.empty:
                chunks.append(ranked)

        if not chunks:
            logger.warning("No ranked candidates found on any date.")
            return pd.DataFrame()

        result = pd.concat(chunks, ignore_index=True)
        result = result.sort_values(["date", "rank"]).reset_index(drop=True)

        logger.info(
            "Ranking complete: %d candidate-days across %d dates",
            len(result), len(buy_dates),
        )
        return result

    def get_top_candidates(
        self,
        rankings_df: pd.DataFrame,
        n: int = 5,
        date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """
        Return top-N candidates for a given date (or latest date).

        Parameters
        ----------
        rankings_df : DataFrame
            Output of ``rank_all_dates``.
        n : int
            Number of top candidates.
        date : Timestamp, optional
            If None, uses the most recent date in the DataFrame.

        Returns
        -------
        DataFrame
            Top-N candidates sorted by composite_score.
        """
        if rankings_df.empty:
            return pd.DataFrame()

        if date is None:
            date = rankings_df["date"].max()

        day = rankings_df[rankings_df["date"] == date]
        return day.head(n).reset_index(drop=True)

    # ─────────────────────────────────────────────────────────────────
    # Internal
    # ─────────────────────────────────────────────────────────────────

    def _compute_component_scores(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:
        """Min-max normalise each component within the day's cross-section."""
        for score_name, (col, invert) in self._COMPONENT_MAP.items():
            if col not in df.columns:
                df[score_name] = 0.0
                continue

            values = df[col].copy()

            if invert:
                values = -values  # lower original → higher score

            # Min-max to [0, 1]
            vmin, vmax = values.min(), values.max()
            if vmax == vmin:
                df[score_name] = 0.5
            else:
                df[score_name] = (values - vmin) / (vmax - vmin)

        return df

    def _weighted_sum(self, df: pd.DataFrame) -> pd.Series:
        """Compute weighted composite score from component columns."""
        weights = self.cfg.weights
        total = 0.0
        weight_sum = 0.0

        for score_name, w in weights.items():
            if score_name in df.columns:
                total = total + df[score_name] * w
                weight_sum += w

        # Re-normalise if not all components present
        if weight_sum > 0 and weight_sum < 1.0:
            total = total / weight_sum

        return total

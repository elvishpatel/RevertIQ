"""
RevertIQ — Signal Generator
============================

Multi-condition BUY/SELL signal engine for cross-sectional mean
reversion swing trading.

BUY Signal Logic
----------------
A BUY fires when **≥ min_conditions** of these are true:

1. RSI(2) < rsi_buy_threshold       (extreme oversold)
2. Z-score < zscore_buy_threshold    (statistically stretched)
3. Price < lower Bollinger Band      (below statistical envelope)
4. Relative weakness < threshold     (underperforming market)
5. Volume spike > threshold          (exhaustion selling)
6. Market regime is bullish          (macro filter)
7. ATR within 2× of 60-day median   (not extreme volatility)

EXIT Signal Logic
-----------------
Exit when **ANY** of:

1. Z-score > 0       (mean reversion achieved)
2. RSI(2) > 70       (momentum restored)
3. Holding > N days   (time stop)
4. ATR stop triggered  (risk stop)

Anti-Lookahead
--------------
Signals generated at close of day **T** → execution at open of **T+1**.
All input features must already be shifted.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from revertiq.config.settings import SignalConfig, get_default_config
from revertiq.utils.logger import get_logger

logger = get_logger(__name__)


class SignalGenerator:
    """Generate vectorized BUY / SELL signals for mean reversion.

    Parameters
    ----------
    signal_cfg : SignalConfig, optional
        Thresholds for signal conditions.  Falls back to defaults.
    """

    def __init__(self, signal_cfg: Optional[SignalConfig] = None) -> None:
        cfg = get_default_config()
        self.cfg = signal_cfg or cfg.signal

    # ─────────────────────────────────────────────────────────────────
    # Main API
    # ─────────────────────────────────────────────────────────────────

    def generate_all_signals(
        self,
        features_df: pd.DataFrame,
        regime_series: pd.Series,
    ) -> pd.DataFrame:
        """
        Generate both buy and exit signals for all stocks.

        Parameters
        ----------
        features_df : DataFrame
            Must contain columns: rsi, zscore, bb_lower, close,
            relative_weakness, volume_spike, atr, ticker, date.
        regime_series : Series
            Daily regime labels indexed by date.

        Returns
        -------
        DataFrame
            Original data with added columns:
            ``buy_signal``, ``sell_signal``, ``signal_strength``,
            ``conditions_met``.
        """
        df = features_df.copy()

        # Ensure 'date' is a column
        if "date" not in df.columns and (
            df.index.name == "date"
            or isinstance(df.index, pd.DatetimeIndex)
        ):
            df = df.reset_index()

        buy_result = self.generate_buy_signals(df, regime_series)
        df["buy_signal"] = buy_result["buy_signal"]
        df["signal_strength"] = buy_result["signal_strength"]
        df["buy_conditions_met"] = buy_result["conditions_met"]

        exit_result = self.generate_exit_signals(df)
        df["sell_signal"] = exit_result["sell_signal"]
        df["exit_reason"] = exit_result["exit_reason"]

        n_buy = df["buy_signal"].sum()
        n_sell = df["sell_signal"].sum()
        logger.info(
            "Signals generated: %d buy, %d sell across %d rows",
            n_buy, n_sell, len(df),
        )

        return df

    # ─────────────────────────────────────────────────────────────────
    # BUY Signals
    # ─────────────────────────────────────────────────────────────────

    def generate_buy_signals(
        self,
        df: pd.DataFrame,
        regime_series: pd.Series,
    ) -> Dict[str, pd.Series]:
        """
        Evaluate buy conditions and produce a composite signal.

        Returns dict with keys: ``buy_signal``, ``signal_strength``,
        ``conditions_met``.
        """
        c = self.cfg
        n = len(df)

        # ── Individual conditions (boolean masks) ──
        conditions: Dict[str, pd.Series] = {}

        # 1. RSI oversold
        if "rsi" in df.columns:
            conditions["rsi_oversold"] = df["rsi"] < c.rsi_buy_threshold
        else:
            conditions["rsi_oversold"] = pd.Series(False, index=df.index)

        # 2. Z-score stretched
        if "zscore" in df.columns:
            conditions["zscore_stretched"] = df["zscore"] < c.zscore_buy_threshold
        else:
            conditions["zscore_stretched"] = pd.Series(False, index=df.index)

        # 3. Price below lower Bollinger Band
        if "bb_lower" in df.columns and "close" in df.columns:
            conditions["below_bb"] = df["close"] < df["bb_lower"]
        else:
            conditions["below_bb"] = pd.Series(False, index=df.index)

        # 4. Relative weakness
        if "relative_weakness" in df.columns:
            conditions["rel_weak"] = (
                df["relative_weakness"] < c.relative_weakness_threshold
            )
        else:
            conditions["rel_weak"] = pd.Series(False, index=df.index)

        # 5. Volume spike
        if "volume_spike" in df.columns:
            conditions["vol_spike"] = (
                df["volume_spike"] > c.volume_spike_threshold
            )
        else:
            conditions["vol_spike"] = pd.Series(False, index=df.index)

        # 6. Market regime bullish
        if "date" in df.columns:
            regime_mapped = df["date"].map(regime_series).fillna("unknown")
        else:
            regime_mapped = pd.Series("unknown", index=df.index)
        conditions["regime_ok"] = regime_mapped.isin(
            ["strong_bull", "mild_bull"]
        )

        # 7. ATR within acceptable range (not extreme volatility)
        if "atr" in df.columns:
            atr_median = df.groupby("ticker")["atr"].transform(
                lambda s: s.rolling(60, min_periods=20).median()
            )
            conditions["atr_ok"] = df["atr"] < (atr_median * 2.0)
        else:
            conditions["atr_ok"] = pd.Series(True, index=df.index)

        # ── Aggregate: count conditions met ──
        cond_matrix = pd.DataFrame(conditions)
        conditions_count = cond_matrix.sum(axis=1)

        # BUY signal when enough conditions are satisfied
        buy_signal = conditions_count >= c.min_conditions_for_signal

        # ── Signal strength (0–100) ──
        max_possible = len(conditions)
        base_strength = (conditions_count / max_possible * 100).clip(0, 100)

        # Severity bonus: deeper Z-score / lower RSI → higher strength
        severity_bonus = pd.Series(0.0, index=df.index)
        if "zscore" in df.columns:
            severity_bonus += (
                df["zscore"].clip(upper=0).abs()
                .clip(upper=3.0)
                / 3.0
                * 10
            )
        if "rsi" in df.columns:
            severity_bonus += (
                (c.rsi_buy_threshold - df["rsi"])
                .clip(lower=0, upper=c.rsi_buy_threshold)
                / c.rsi_buy_threshold
                * 10
            )

        signal_strength = (base_strength + severity_bonus).clip(0, 100)
        signal_strength = signal_strength.where(buy_signal, 0.0)

        # ── Conditions met as comma-separated string ──
        conditions_met = cond_matrix.apply(
            lambda row: ",".join(
                name for name, val in row.items() if val
            ),
            axis=1,
        )

        return {
            "buy_signal": buy_signal,
            "signal_strength": signal_strength,
            "conditions_met": conditions_met,
        }

    # ─────────────────────────────────────────────────────────────────
    # EXIT / SELL Signals
    # ─────────────────────────────────────────────────────────────────

    def generate_exit_signals(
        self, df: pd.DataFrame
    ) -> Dict[str, pd.Series]:
        """
        Evaluate exit conditions.  ANY single condition triggers a sell.

        Returns dict with ``sell_signal`` (bool) and ``exit_reason`` (str).
        """
        c = self.cfg

        reasons: Dict[str, pd.Series] = {}

        # 1. Mean reversion achieved (Z-score normalised)
        if "zscore" in df.columns:
            reasons["mean_reversion"] = df["zscore"] > c.zscore_exit_threshold

        # 2. RSI normalised
        if "rsi" in df.columns:
            reasons["rsi_exit"] = df["rsi"] > c.rsi_exit_threshold

        # Build sell signal (any reason fires)
        if reasons:
            reasons_matrix = pd.DataFrame(reasons)
            sell_signal = reasons_matrix.any(axis=1)
            # Pick first matching reason
            exit_reason = reasons_matrix.apply(
                lambda row: next(
                    (name for name, val in row.items() if val), ""
                ),
                axis=1,
            )
        else:
            sell_signal = pd.Series(False, index=df.index)
            exit_reason = pd.Series("", index=df.index)

        # NOTE: Time-stop and ATR-stop are handled by the backtest engine
        # (they depend on entry date/price, which signals don't track).

        return {
            "sell_signal": sell_signal,
            "exit_reason": exit_reason,
        }

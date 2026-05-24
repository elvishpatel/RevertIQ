"""
RevertIQ — Backtest Engine
===========================

Day-by-day portfolio simulation for cross-sectional mean reversion.

Execution Model
---------------
1. Signal generated at close of day **T**
2. Trade executed at open of day **T+1**
3. Transaction costs and slippage applied on every trade
4. Multiple simultaneous positions (up to ``max_positions``)

Daily Loop
----------
For each trading day:

  1. Mark-to-market all open positions
  2. Check stop losses → close triggered positions
  3. Check exit signals → close mean-reverted / timed-out positions
  4. Check regime → skip new entries if bearish
  5. Get top-ranked BUY candidates
  6. Open new positions (up to capacity)
  7. Record daily equity snapshot

Usage
-----
>>> from revertiq.backtesting.engine import BacktestEngine
>>> engine = BacktestEngine(config)
>>> result = engine.run(price_data, signals, rankings, regime)
>>> print(result.metrics)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

from revertiq.backtesting.metrics import PerformanceMetrics
from revertiq.config.settings import RevertIQConfig, get_default_config
from revertiq.portfolio.manager import PortfolioManager
from revertiq.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class BacktestResult:
    """Container for all backtest outputs."""

    equity_curve: pd.DataFrame
    trade_log: pd.DataFrame
    daily_returns: pd.Series
    metrics: Dict
    config: RevertIQConfig
    benchmark_equity: Optional[pd.DataFrame] = None


class BacktestEngine:
    """Core portfolio simulation engine.

    Parameters
    ----------
    config : RevertIQConfig, optional
        Full system configuration.  Falls back to defaults.
    """

    def __init__(self, config: Optional[RevertIQConfig] = None) -> None:
        self.config = config or get_default_config()
        self.portfolio: Optional[PortfolioManager] = None

    # ─────────────────────────────────────────────────────────────────
    # Main entry point
    # ─────────────────────────────────────────────────────────────────

    def run(
        self,
        price_data: pd.DataFrame,
        signals_df: pd.DataFrame,
        rankings_df: pd.DataFrame,
        regime_series: pd.Series,
    ) -> BacktestResult:
        """
        Run the full backtest simulation.

        Parameters
        ----------
        price_data : DataFrame
            Combined OHLCV for all stocks. Must have columns:
            ``date``, ``ticker``, ``open``, ``high``, ``low``,
            ``close``, ``volume``, ``atr``.
        signals_df : DataFrame
            Output of ``SignalGenerator`` with ``date``, ``ticker``,
            ``buy_signal``, ``sell_signal``.
        rankings_df : DataFrame
            Output of ``StockRanker`` with ``date``, ``ticker``,
            ``rank``, ``composite_score``.
        regime_series : Series
            Daily regime labels indexed by date.

        Returns
        -------
        BacktestResult
        """
        self.portfolio = PortfolioManager(self.config.backtest)

        # ── Build lookup structures ──
        # Price dict: (date, ticker) → {open, high, low, close, atr}
        price_lookup = self._build_price_lookup(price_data)

        # Signal dict: (date, ticker) → {buy, sell, exit_reason}
        signal_lookup = self._build_signal_lookup(signals_df)

        # Ranking dict: date → [(ticker, score), ...] sorted descending
        ranking_lookup = self._build_ranking_lookup(rankings_df)

        # ── Determine trading dates ──
        all_dates = sorted(price_data["date"].unique())
        logger.info(
            "Starting backtest: %s → %s  (%d trading days)",
            all_dates[0], all_dates[-1], len(all_dates),
        )

        # ── Day-by-day simulation ──
        for i, date in enumerate(all_dates):
            self._simulate_day(
                date=date,
                next_date=all_dates[i + 1] if i + 1 < len(all_dates) else None,
                price_lookup=price_lookup,
                signal_lookup=signal_lookup,
                ranking_lookup=ranking_lookup,
                regime_series=regime_series,
            )

        # ── Compile results ──
        equity_curve = self.portfolio.get_equity_curve()
        trade_log = self.portfolio.get_trade_log()

        if equity_curve.empty:
            daily_returns = pd.Series(dtype=float)
        else:
            daily_returns = equity_curve["total_equity"].pct_change().dropna()

        metrics = PerformanceMetrics.compute_all(
            equity_curve, trade_log, daily_returns
        )

        logger.info(
            "Backtest complete: %d trades, CAGR=%.2f%%, Sharpe=%.2f, MaxDD=%.2f%%",
            len(trade_log),
            metrics.get("cagr", 0) * 100,
            metrics.get("sharpe_ratio", 0),
            metrics.get("max_drawdown_pct", 0) * 100,
        )

        return BacktestResult(
            equity_curve=equity_curve,
            trade_log=trade_log,
            daily_returns=daily_returns,
            metrics=metrics,
            config=self.config,
        )

    def run_with_benchmark(
        self,
        price_data: pd.DataFrame,
        signals_df: pd.DataFrame,
        rankings_df: pd.DataFrame,
        regime_series: pd.Series,
        benchmark_data: pd.DataFrame,
    ) -> BacktestResult:
        """
        Run backtest and also compute buy-and-hold benchmark equity.

        Parameters
        ----------
        benchmark_data : DataFrame
            NIFTY 50 index data with ``date`` and ``close`` columns.

        Returns
        -------
        BacktestResult
            With ``benchmark_equity`` populated.
        """
        result = self.run(price_data, signals_df, rankings_df, regime_series)

        # Benchmark: invest full capital at day 1
        bm = benchmark_data.copy()
        if "date" not in bm.columns:
            bm = bm.reset_index()
        bm = bm.sort_values("date")
        initial = self.config.backtest.initial_capital
        start_price = bm["close"].iloc[0]
        bm["benchmark_equity"] = initial * (bm["close"] / start_price)
        result.benchmark_equity = bm[["date", "benchmark_equity"]]

        return result

    # ─────────────────────────────────────────────────────────────────
    # Day simulation
    # ─────────────────────────────────────────────────────────────────

    def _simulate_day(
        self,
        date: pd.Timestamp,
        next_date: Optional[pd.Timestamp],
        price_lookup: Dict,
        signal_lookup: Dict,
        ranking_lookup: Dict,
        regime_series: pd.Series,
    ) -> None:
        """Simulate one trading day."""
        pm = self.portfolio

        # ── 1. Mark-to-market ──
        current_prices = {}
        current_lows = {}
        for ticker in list(pm.positions.keys()):
            key = (date, ticker)
            if key in price_lookup:
                current_prices[ticker] = price_lookup[key]["close"]
                current_lows[ticker] = price_lookup[key]["low"]
            else:
                # Missing data — use last known price
                current_prices[ticker] = pm.positions[ticker].entry_price

        pm.update_equity(date, current_prices)

        # ── 2. Check stop losses ──
        stopped = pm.check_stop_losses(date, current_lows)
        for ticker in stopped:
            price = current_prices.get(ticker, pm.positions[ticker].entry_price)
            pm.close_position(ticker, date, price, reason="atr_stop")

        # ── 3. Check exit signals + time stops ──
        for ticker in list(pm.positions.keys()):
            pos = pm.positions[ticker]
            key = (date, ticker)

            # Time stop
            if date > pos.entry_date:
                holding_days = np.busday_count(
                    pos.entry_date.date(), date.date()
                )
            else:
                holding_days = 0

            if holding_days >= self.config.signal.max_holding_days:
                price = current_prices.get(ticker, pos.entry_price)
                pm.close_position(ticker, date, price, reason="time_stop")
                continue

            # Signal-based exit
            sig = signal_lookup.get(key, {})
            if sig.get("sell_signal", False):
                price = current_prices.get(ticker, pos.entry_price)
                reason = sig.get("exit_reason", "mean_reversion")
                pm.close_position(ticker, date, price, reason=reason)

        # ── 4. Check regime ──
        regime = regime_series.get(date, "unknown")
        allowed = regime in self.config.regime.allowed_regimes

        if not allowed:
            return  # Skip new entries in unfavorable regime

        # ── 5. Open new positions from ranked candidates ──
        if next_date is None:
            return  # No next day to execute trades

        ranked = ranking_lookup.get(date, [])
        already_held: Set[str] = set(pm.positions.keys())

        for ticker, _score in ranked:
            if not pm.can_open_position():
                break

            if ticker in already_held:
                continue  # Don't double up

            # Execute at NEXT day's open (anti-lookahead)
            next_key = (next_date, ticker)
            if next_key not in price_lookup:
                continue  # No data for next day

            open_price = price_lookup[next_key]["open"]
            atr = price_lookup.get((date, ticker), {}).get("atr", open_price * 0.02)

            if open_price <= 0 or np.isnan(open_price):
                continue

            pm.open_position(ticker, next_date, open_price, atr)

    # ─────────────────────────────────────────────────────────────────
    # Lookup builders
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_price_lookup(
        price_data: pd.DataFrame,
    ) -> Dict:
        """Build (date, ticker) → price dict for O(1) access."""
        lookup = {}
        for row in price_data.itertuples(index=False):
            key = (row.date, row.ticker)
            lookup[key] = {
                "open": getattr(row, "open", np.nan),
                "high": getattr(row, "high", np.nan),
                "low": getattr(row, "low", np.nan),
                "close": getattr(row, "close", np.nan),
                "atr": getattr(row, "atr", np.nan),
            }
        return lookup

    @staticmethod
    def _build_signal_lookup(signals_df: pd.DataFrame) -> Dict:
        """Build (date, ticker) → signal dict."""
        lookup = {}
        for row in signals_df.itertuples(index=False):
            key = (row.date, row.ticker)
            lookup[key] = {
                "buy_signal": getattr(row, "buy_signal", False),
                "sell_signal": getattr(row, "sell_signal", False),
                "exit_reason": getattr(row, "exit_reason", ""),
            }
        return lookup

    @staticmethod
    def _build_ranking_lookup(rankings_df: pd.DataFrame) -> Dict:
        """Build date → [(ticker, score), ...] dict sorted by rank."""
        lookup: Dict = {}
        if rankings_df.empty:
            return lookup
        for date, group in rankings_df.groupby("date"):
            sorted_group = group.sort_values("rank")
            lookup[date] = list(
                zip(sorted_group["ticker"], sorted_group["composite_score"])
            )
        return lookup

"""
RevertIQ — Backtesting Performance Metrics
============================================

A comprehensive suite of performance metrics for evaluating mean-reversion
strategy backtests on Indian equity markets.

All calculations use **252 trading days/year** and an Indian risk-free rate
of **6 %** (approximate RBI repo rate) unless overridden.

Usage
-----
>>> from revertiq.backtesting.metrics import PerformanceMetrics
>>> metrics = PerformanceMetrics.compute_all(equity_curve, trade_log, daily_returns)
>>> print(f"CAGR: {metrics['cagr']:.2%}")
>>> print(f"Sharpe: {metrics['sharpe_ratio']:.2f}")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from revertiq.utils.logger import get_logger

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────

TRADING_DAYS_PER_YEAR: int = 252
INDIAN_RISK_FREE_RATE: float = 0.06  # 6 % — approximate RBI repo rate


class PerformanceMetrics:
    """
    Static-method library for computing portfolio performance metrics.

    All methods are pure functions — they take data in and return results
    without side effects.  This makes them easy to test and compose.
    """

    # ─────────────────────────────────────────────────────────────────
    # 1. CAGR — Compound Annual Growth Rate
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def cagr(equity_curve: pd.Series) -> float:
        """
        Compound Annual Growth Rate.

        Formula:
            CAGR = (V_final / V_initial) ^ (252 / N_days) - 1

        Parameters
        ----------
        equity_curve : pd.Series
            Daily portfolio equity values indexed by date.

        Returns
        -------
        float
            Annualized return as a decimal (e.g. 0.15 = 15 %).
        """
        # Drop NaNs to get clean start/end values
        clean = equity_curve.dropna()
        if len(clean) < 2:
            logger.warning("CAGR: equity_curve has fewer than 2 data points")
            return 0.0

        initial = clean.iloc[0]
        final = clean.iloc[-1]

        if initial <= 0:
            logger.warning("CAGR: initial equity is zero or negative")
            return 0.0

        n_days = len(clean)
        # Exponent converts to annualized: (252 trading days / total days)
        annualized_return = (final / initial) ** (TRADING_DAYS_PER_YEAR / n_days) - 1
        return float(annualized_return)

    # ─────────────────────────────────────────────────────────────────
    # 2. Sharpe Ratio
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def sharpe_ratio(
        daily_returns: pd.Series,
        risk_free: float = INDIAN_RISK_FREE_RATE,
    ) -> float:
        """
        Annualized Sharpe Ratio.

        Formula:
            Sharpe = (mean(R) - Rf/252) / std(R) × sqrt(252)

        Parameters
        ----------
        daily_returns : pd.Series
            Daily percentage returns (as decimals).
        risk_free : float
            Annual risk-free rate (default 6 % for India).

        Returns
        -------
        float
            Annualized Sharpe ratio.
        """
        clean = daily_returns.dropna()
        if len(clean) < 2:
            return 0.0

        # Convert annual risk-free rate to daily
        daily_rf = risk_free / TRADING_DAYS_PER_YEAR

        # Excess return over risk-free rate
        excess_returns = clean - daily_rf
        std = excess_returns.std()

        if std == 0 or np.isnan(std):
            return 0.0

        # Annualize: multiply by sqrt(252) to scale daily ratio to annual
        sharpe = (excess_returns.mean() / std) * np.sqrt(TRADING_DAYS_PER_YEAR)
        return float(sharpe)

    # ─────────────────────────────────────────────────────────────────
    # 3. Sortino Ratio
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def sortino_ratio(
        daily_returns: pd.Series,
        risk_free: float = INDIAN_RISK_FREE_RATE,
    ) -> float:
        """
        Annualized Sortino Ratio — like Sharpe but penalizes only downside volatility.

        Formula:
            Sortino = (mean(R) - Rf/252) / downside_std × sqrt(252)

            downside_std = std of min(R - Rf/252, 0)

        Parameters
        ----------
        daily_returns : pd.Series
            Daily percentage returns (as decimals).
        risk_free : float
            Annual risk-free rate (default 6 % for India).

        Returns
        -------
        float
            Annualized Sortino ratio.
        """
        clean = daily_returns.dropna()
        if len(clean) < 2:
            return 0.0

        daily_rf = risk_free / TRADING_DAYS_PER_YEAR
        excess = clean - daily_rf

        # Downside returns: only keep negative excess returns, set positive to 0
        downside = excess.clip(upper=0)
        downside_std = downside.std()

        if downside_std == 0 or np.isnan(downside_std):
            return 0.0

        sortino = (excess.mean() / downside_std) * np.sqrt(TRADING_DAYS_PER_YEAR)
        return float(sortino)

    # ─────────────────────────────────────────────────────────────────
    # 4. Maximum Drawdown
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def max_drawdown(
        equity_curve: pd.Series,
    ) -> Tuple[float, Optional[pd.Timestamp], Optional[pd.Timestamp], Optional[pd.Timestamp]]:
        """
        Maximum peak-to-trough drawdown.

        Returns the worst percentage drawdown along with the dates of the
        peak, trough, and recovery (if recovery occurred).

        Formula:
            drawdown_t = (equity_t - running_max_t) / running_max_t
            max_dd     = min(drawdown_t)

        Parameters
        ----------
        equity_curve : pd.Series
            Daily portfolio equity values indexed by date.

        Returns
        -------
        tuple[float, Timestamp | None, Timestamp | None, Timestamp | None]
            (max_drawdown_pct, peak_date, trough_date, recovery_date)
            max_drawdown_pct is negative (e.g. -0.25 = 25 % drawdown).
            recovery_date is None if the drawdown has not recovered.
        """
        clean = equity_curve.dropna()
        if len(clean) < 2:
            return (0.0, None, None, None)

        # Running peak: the highest equity value seen up to each point
        running_max = clean.cummax()

        # Drawdown series: percentage decline from running peak
        drawdown = (clean - running_max) / running_max

        # Worst drawdown
        max_dd = float(drawdown.min())

        if max_dd == 0.0:
            # No drawdown occurred — monotonically increasing equity
            return (0.0, None, None, None)

        # Trough date: where drawdown is deepest
        trough_date = drawdown.idxmin()

        # Peak date: the running peak just before the trough
        # It's the last date when equity was at its running-max before trough
        peak_mask = clean[:trough_date] == running_max[:trough_date]
        peak_dates = clean[:trough_date][peak_mask].index
        peak_date = peak_dates[-1] if len(peak_dates) > 0 else None

        # Recovery date: first date after trough where equity >= peak equity
        recovery_date = None
        if peak_date is not None:
            peak_value = clean[peak_date]
            post_trough = clean[trough_date:]
            recovered = post_trough[post_trough >= peak_value]
            if len(recovered) > 1:
                # First date after trough that matches or exceeds peak
                recovery_date = recovered.index[1] if recovered.index[0] == trough_date and len(recovered) > 1 else recovered.index[0]
                # Ensure recovery is strictly after trough
                if recovery_date <= trough_date and len(recovered) > 1:
                    recovery_date = recovered.index[1]
                elif recovery_date <= trough_date:
                    recovery_date = None

        return (max_dd, peak_date, trough_date, recovery_date)

    # ─────────────────────────────────────────────────────────────────
    # 5. Calmar Ratio
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def calmar_ratio(cagr_value: float, max_dd_value: float) -> float:
        """
        Calmar Ratio = CAGR / |Max Drawdown|.

        A risk-adjusted return metric that rewards high returns relative
        to the worst peak-to-trough loss.

        Parameters
        ----------
        cagr_value : float
            Compound annual growth rate (decimal).
        max_dd_value : float
            Maximum drawdown as a negative decimal (e.g. -0.25).

        Returns
        -------
        float
            Calmar ratio.  Returns 0 if drawdown is zero.
        """
        abs_dd = abs(max_dd_value)
        if abs_dd < 1e-10:
            return 0.0
        return float(cagr_value / abs_dd)

    # ─────────────────────────────────────────────────────────────────
    # 6. Win Rate
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def win_rate(trades_df: pd.DataFrame) -> float:
        """
        Percentage of trades that were profitable.

        Parameters
        ----------
        trades_df : pd.DataFrame
            Trade log with a ``pnl`` column (profit/loss per trade).

        Returns
        -------
        float
            Win rate as a decimal (e.g. 0.65 = 65 %).
        """
        if trades_df is None or len(trades_df) == 0 or "pnl" not in trades_df.columns:
            return 0.0

        total = len(trades_df)
        # A trade is a "win" if pnl > 0 (breakeven = 0 is not a win)
        winners = (trades_df["pnl"] > 0).sum()
        return float(winners / total)

    # ─────────────────────────────────────────────────────────────────
    # 7. Profit Factor
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def profit_factor(trades_df: pd.DataFrame) -> float:
        """
        Profit Factor = Gross Profit / Gross Loss.

        A value > 1 means the strategy is profitable overall.
        A value of 2.0 means you make ₹2 for every ₹1 you lose.

        Parameters
        ----------
        trades_df : pd.DataFrame
            Trade log with a ``pnl`` column.

        Returns
        -------
        float
            Profit factor.  Returns 0 if no losing trades (infinite PF
            is capped at 0 to signal edge case).
        """
        if trades_df is None or len(trades_df) == 0 or "pnl" not in trades_df.columns:
            return 0.0

        gross_profit = trades_df.loc[trades_df["pnl"] > 0, "pnl"].sum()
        gross_loss = abs(trades_df.loc[trades_df["pnl"] < 0, "pnl"].sum())

        if gross_loss < 1e-10:
            # No losses — return large number to indicate unbounded PF
            return float("inf") if gross_profit > 0 else 0.0

        return float(gross_profit / gross_loss)

    # ─────────────────────────────────────────────────────────────────
    # 8. Average Holding Period
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def avg_holding_period(trades_df: pd.DataFrame) -> float:
        """
        Mean holding duration across all closed trades.

        Parameters
        ----------
        trades_df : pd.DataFrame
            Trade log with ``entry_date`` and ``exit_date`` columns.

        Returns
        -------
        float
            Average holding period in calendar days.
        """
        if trades_df is None or len(trades_df) == 0:
            return 0.0

        required_cols = {"entry_date", "exit_date"}
        if not required_cols.issubset(trades_df.columns):
            logger.warning(
                "avg_holding_period: trade_log missing columns %s",
                required_cols - set(trades_df.columns),
            )
            return 0.0

        # Compute holding period as difference between exit and entry
        entry = pd.to_datetime(trades_df["entry_date"])
        exit_ = pd.to_datetime(trades_df["exit_date"])
        holding_days = (exit_ - entry).dt.days

        return float(holding_days.mean())

    # ─────────────────────────────────────────────────────────────────
    # 9. Exposure
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def exposure(equity_curve: pd.DataFrame) -> float:
        """
        Percentage of trading days with at least one open position.

        Parameters
        ----------
        equity_curve : pd.DataFrame
            Must contain a ``n_positions`` column indicating the number
            of open positions each day.  If not present, checks for an
            ``invested`` column (bool).

        Returns
        -------
        float
            Exposure as a decimal (e.g. 0.40 = 40 % of time invested).
        """
        if equity_curve is None or len(equity_curve) == 0:
            return 0.0

        if "n_positions" in equity_curve.columns:
            invested_days = (equity_curve["n_positions"] > 0).sum()
        elif "invested" in equity_curve.columns:
            invested_days = equity_curve["invested"].sum()
        else:
            logger.warning(
                "exposure: equity_curve lacks 'n_positions' or 'invested' column; "
                "returning 0"
            )
            return 0.0

        total_days = len(equity_curve)
        return float(invested_days / total_days) if total_days > 0 else 0.0

    # ─────────────────────────────────────────────────────────────────
    # 10. Compute All — master aggregator
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def compute_all(
        equity_curve: pd.DataFrame,
        trade_log: pd.DataFrame,
        daily_returns: pd.Series,
    ) -> Dict[str, Any]:
        """
        Compute every performance metric and return as a dictionary.

        This is the primary entry-point — call once at the end of a
        backtest to get a full performance snapshot.

        Parameters
        ----------
        equity_curve : pd.DataFrame
            Daily equity DataFrame with at least an ``equity`` column.
            Optionally includes ``n_positions`` for exposure calculation.
        trade_log : pd.DataFrame
            Closed-trade log with ``pnl``, ``entry_date``, ``exit_date``.
        daily_returns : pd.Series
            Daily portfolio percentage returns.

        Returns
        -------
        dict[str, Any]
            Dictionary of all performance metrics.
        """
        # Extract equity series from DataFrame
        if isinstance(equity_curve, pd.DataFrame):
            equity_series = equity_curve["equity"] if "equity" in equity_curve.columns else equity_curve.iloc[:, 0]
        else:
            equity_series = equity_curve

        # ── Core return metrics ──
        cagr_val = PerformanceMetrics.cagr(equity_series)
        sharpe_val = PerformanceMetrics.sharpe_ratio(daily_returns)
        sortino_val = PerformanceMetrics.sortino_ratio(daily_returns)
        max_dd_val, peak_date, trough_date, recovery_date = PerformanceMetrics.max_drawdown(equity_series)
        calmar_val = PerformanceMetrics.calmar_ratio(cagr_val, max_dd_val)

        # ── Trade-level metrics ──
        win_rate_val = PerformanceMetrics.win_rate(trade_log)
        profit_factor_val = PerformanceMetrics.profit_factor(trade_log)
        avg_hold_val = PerformanceMetrics.avg_holding_period(trade_log)
        exposure_val = PerformanceMetrics.exposure(equity_curve)

        # ── Total return ──
        total_return = 0.0
        if len(equity_series.dropna()) >= 2:
            total_return = float(
                equity_series.dropna().iloc[-1] / equity_series.dropna().iloc[0] - 1
            )

        # ── Trade counts ──
        n_trades = len(trade_log) if trade_log is not None else 0
        n_winners = int((trade_log["pnl"] > 0).sum()) if n_trades > 0 and "pnl" in trade_log.columns else 0
        n_losers = int((trade_log["pnl"] < 0).sum()) if n_trades > 0 and "pnl" in trade_log.columns else 0

        # ── Average trade PnL ──
        avg_pnl = float(trade_log["pnl"].mean()) if n_trades > 0 and "pnl" in trade_log.columns else 0.0
        avg_win = float(trade_log.loc[trade_log["pnl"] > 0, "pnl"].mean()) if n_winners > 0 else 0.0
        avg_loss = float(trade_log.loc[trade_log["pnl"] < 0, "pnl"].mean()) if n_losers > 0 else 0.0

        # ── Annualized volatility ──
        annual_vol = float(daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)) if len(daily_returns.dropna()) > 1 else 0.0

        # ── Days in market ──
        n_trading_days = len(equity_series.dropna())

        metrics: Dict[str, Any] = {
            # Return metrics
            "total_return": total_return,
            "cagr": cagr_val,
            "annual_volatility": annual_vol,
            # Risk-adjusted metrics
            "sharpe_ratio": sharpe_val,
            "sortino_ratio": sortino_val,
            "calmar_ratio": calmar_val,
            # Drawdown
            "max_drawdown": max_dd_val,
            "max_drawdown_peak_date": peak_date,
            "max_drawdown_trough_date": trough_date,
            "max_drawdown_recovery_date": recovery_date,
            # Trade statistics
            "total_trades": n_trades,
            "winning_trades": n_winners,
            "losing_trades": n_losers,
            "win_rate": win_rate_val,
            "profit_factor": profit_factor_val,
            "avg_pnl_per_trade": avg_pnl,
            "avg_winning_trade": avg_win,
            "avg_losing_trade": avg_loss,
            "avg_holding_period_days": avg_hold_val,
            # Exposure
            "exposure": exposure_val,
            "n_trading_days": n_trading_days,
        }

        logger.info(
            "Metrics computed — CAGR: %.2f%%, Sharpe: %.2f, MaxDD: %.2f%%, "
            "Win Rate: %.1f%%, Trades: %d",
            cagr_val * 100,
            sharpe_val,
            max_dd_val * 100,
            win_rate_val * 100,
            n_trades,
        )

        return metrics

    # ─────────────────────────────────────────────────────────────────
    # 11. Rolling Sharpe
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def rolling_sharpe(
        daily_returns: pd.Series,
        window: int = 63,
        risk_free: float = INDIAN_RISK_FREE_RATE,
    ) -> pd.Series:
        """
        Rolling annualized Sharpe ratio over a given window.

        Useful for diagnosing regime-dependent strategy performance —
        a steady rolling Sharpe indicates robustness.

        Formula:
            rolling_sharpe_t = (mean(R_{t-w:t}) - Rf/252) / std(R_{t-w:t}) × sqrt(252)

        Parameters
        ----------
        daily_returns : pd.Series
            Daily percentage returns.
        window : int
            Rolling window size in trading days (default: 63 ≈ 3 months).
        risk_free : float
            Annual risk-free rate.

        Returns
        -------
        pd.Series
            Rolling Sharpe ratio (first ``window - 1`` values will be NaN).
        """
        daily_rf = risk_free / TRADING_DAYS_PER_YEAR
        excess = daily_returns - daily_rf

        # Rolling mean and std of excess returns
        rolling_mean = excess.rolling(window=window, min_periods=window).mean()
        rolling_std = excess.rolling(window=window, min_periods=window).std()

        # Annualize and handle division by zero
        rolling_sharpe = (rolling_mean / rolling_std.replace(0, np.nan)) * np.sqrt(TRADING_DAYS_PER_YEAR)

        return rolling_sharpe

    # ─────────────────────────────────────────────────────────────────
    # 12. Monthly Returns Table
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def monthly_returns(daily_returns: pd.Series) -> pd.DataFrame:
        """
        Build a Year × Month matrix of monthly compounded returns.

        This is the standard "monthly returns heatmap table" found in
        most quant tear sheets.

        Parameters
        ----------
        daily_returns : pd.Series
            Daily percentage returns indexed by date.

        Returns
        -------
        pd.DataFrame
            Rows = years, Columns = months (1–12), Values = monthly returns.
            An extra ``YTD`` column contains year-to-date compounded returns.
        """
        clean = daily_returns.dropna()
        if len(clean) == 0:
            return pd.DataFrame()

        # Ensure the index is a DatetimeIndex
        clean.index = pd.to_datetime(clean.index)

        # Compound daily returns into monthly returns
        # (1 + r1) × (1 + r2) × ... - 1 for each month
        monthly = clean.groupby(
            [clean.index.year, clean.index.month]
        ).apply(lambda x: (1 + x).prod() - 1)

        # Reshape into Year × Month matrix
        monthly.index.names = ["year", "month"]
        monthly = monthly.reset_index()
        monthly.columns = ["year", "month", "return"]
        pivot = monthly.pivot(index="year", columns="month", values="return")

        # Rename month columns to calendar abbreviations for readability
        month_names = {
            1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
            5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
            9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
        }
        pivot = pivot.rename(columns=month_names)

        # Add YTD column: compound all monthly returns within each year
        pivot["YTD"] = pivot.apply(
            lambda row: (1 + row.dropna()).prod() - 1, axis=1
        )

        return pivot

    # ─────────────────────────────────────────────────────────────────
    # 13. Drawdown Series
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def drawdown_series(equity_curve: pd.Series) -> pd.Series:
        """
        Daily drawdown percentage from the running peak.

        Formula:
            dd_t = (equity_t - running_max_t) / running_max_t

        This is always ≤ 0.  A value of -0.10 means the portfolio is
        10 % below its all-time high at that point.

        Parameters
        ----------
        equity_curve : pd.Series
            Daily portfolio equity values indexed by date.

        Returns
        -------
        pd.Series
            Daily drawdown percentage (non-positive values).
        """
        clean = equity_curve.dropna()
        if len(clean) == 0:
            return pd.Series(dtype=float)

        running_max = clean.cummax()
        dd = (clean - running_max) / running_max
        dd.name = "drawdown"
        return dd

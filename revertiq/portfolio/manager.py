"""
RevertIQ — Portfolio Manager
=============================

Tracks positions, manages risk, and records trade history for the
backtesting engine.  All position sizing, stop-loss logic, and
transaction-cost adjustments live here so the backtest loop stays clean.

Key Design Decisions
--------------------
* **Equal-weight allocation** — each new position receives
  ``total_equity / max_positions``, but never more than available cash
  and never more than ``risk_per_trade_pct`` of total equity.
* **Slippage + costs baked into prices** — ``open_position`` inflates
  the entry price by ``buy_cost_pct + slippage_pct``; ``close_position``
  deflates exit price by ``sell_cost_pct + slippage_pct``.
* **ATR-based stop-loss** — ``entry_price - atr_stop_multiplier × ATR``
  is computed at entry and checked each bar via ``check_stop_losses``.

Usage
-----
>>> from revertiq.config.settings import get_default_config
>>> from revertiq.portfolio.manager import PortfolioManager
>>> cfg = get_default_config()
>>> pm = PortfolioManager(cfg.backtest, cfg.signal)
>>> pm.can_open_position()
True
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from revertiq.config.settings import BacktestConfig, SignalConfig
from revertiq.utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────────────────────────────

@dataclass
class Position:
    """An open (active) position in a single stock.

    Attributes:
        ticker: NSE ticker symbol (e.g. ``"RELIANCE.NS"``).
        entry_date: Date the position was opened.
        entry_price: Cost-adjusted entry price (includes slippage + fees).
        shares: Number of whole shares held.
        stop_loss: Price level that triggers an ATR-based stop-loss exit.
        atr_at_entry: ATR value on the entry date, used for risk maths.
    """

    ticker: str
    entry_date: pd.Timestamp
    entry_price: float
    shares: int
    stop_loss: float
    atr_at_entry: float


@dataclass
class Trade:
    """A completed (closed) round-trip trade.

    Attributes:
        ticker: NSE ticker symbol.
        entry_date: Date the position was opened.
        exit_date: Date the position was closed.
        entry_price: Cost-adjusted entry price.
        exit_price: Cost-adjusted exit price.
        shares: Number of shares traded.
        pnl: Absolute profit/loss in rupees.
        pnl_pct: Percentage return on capital deployed.
        holding_days: Calendar days between entry and exit.
        exit_reason: One of ``'mean_reversion'``, ``'rsi_exit'``,
                     ``'time_stop'``, ``'atr_stop'``.
    """

    ticker: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    shares: int
    pnl: float
    pnl_pct: float
    holding_days: int
    exit_reason: str  # 'mean_reversion', 'rsi_exit', 'time_stop', 'atr_stop'


@dataclass
class EquityPoint:
    """A single daily snapshot of the portfolio's financial state.

    Attributes:
        date: The snapshot date.
        cash: Uninvested cash available.
        invested_value: Mark-to-market value of all open positions.
        total_equity: ``cash + invested_value``.
        num_positions: Number of currently open positions.
        exposure_pct: ``invested_value / total_equity`` (0–1 scale).
    """

    date: pd.Timestamp
    cash: float
    invested_value: float
    total_equity: float
    num_positions: int
    exposure_pct: float


# ─────────────────────────────────────────────────────────────────────
# Portfolio Manager
# ─────────────────────────────────────────────────────────────────────

class PortfolioManager:
    """Tracks positions, manages risk, and records trade history.

    The manager owns three mutable collections that grow over the
    course of a backtest:

    * ``positions`` — currently open positions (keyed by ticker).
    * ``trade_history`` — closed round-trip trades.
    * ``equity_curve`` — daily mark-to-market snapshots.

    Parameters:
        backtest_cfg: BacktestConfig with capital, cost, and sizing params.
        signal_cfg: SignalConfig with ``atr_stop_multiplier`` and exit rules.

    Example:
        >>> cfg = get_default_config()
        >>> pm = PortfolioManager(cfg.backtest, cfg.signal)
        >>> pos = pm.open_position("INFY.NS", pd.Timestamp("2024-01-15"),
        ...                        1500.0, atr=35.0)
        >>> pm.cash  # reduced by cost of shares purchased
    """

    def __init__(
        self,
        backtest_cfg: BacktestConfig,
        signal_cfg: SignalConfig,
    ) -> None:
        # ── Store configuration ──
        self._bt_cfg = backtest_cfg
        self._sig_cfg = signal_cfg

        # ── Portfolio state ──
        self._initial_capital: float = backtest_cfg.initial_capital
        self.cash: float = backtest_cfg.initial_capital
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[Trade] = []
        self.equity_curve: List[EquityPoint] = []

        logger.info(
            "PortfolioManager initialised | capital=₹%.2f | "
            "max_positions=%d | risk_per_trade=%.2f%%",
            self.cash,
            backtest_cfg.max_positions,
            backtest_cfg.risk_per_trade_pct * 100,
        )

    # ── helpers ──────────────────────────────────────────────────────

    @property
    def total_equity(self) -> float:
        """Current total equity (cash + mark-to-market of open positions).

        Returns:
            Total portfolio equity as a float. When no current prices
            are available, uses entry prices for the estimate.
        """
        invested = sum(
            pos.entry_price * pos.shares for pos in self.positions.values()
        )
        return self.cash + invested

    @property
    def num_positions(self) -> int:
        """Number of currently open positions.

        Returns:
            Integer count of open positions.
        """
        return len(self.positions)

    def _effective_buy_price(self, raw_price: float) -> float:
        """Apply buy-side transaction costs and slippage to a raw price.

        The effective price a buyer pays is *higher* than the market
        price because of brokerage, STT, and market-impact slippage.

        Parameters:
            raw_price: Unadjusted market price.

        Returns:
            Adjusted price reflecting costs:
            ``raw_price × (1 + buy_cost_pct + slippage_pct)``.
        """
        cost_factor = 1.0 + self._bt_cfg.buy_cost_pct + self._bt_cfg.slippage_pct
        return raw_price * cost_factor

    def _effective_sell_price(self, raw_price: float) -> float:
        """Apply sell-side transaction costs and slippage to a raw price.

        The effective price a seller receives is *lower* than the market
        price.

        Parameters:
            raw_price: Unadjusted market price.

        Returns:
            Adjusted price reflecting costs:
            ``raw_price × (1 - sell_cost_pct - slippage_pct)``.
        """
        cost_factor = 1.0 - self._bt_cfg.sell_cost_pct - self._bt_cfg.slippage_pct
        return raw_price * cost_factor

    # ── 1. can_open_position ─────────────────────────────────────────

    def can_open_position(self) -> bool:
        """Check whether a new position can be opened.

        Two conditions must both be true:

        1. The number of open positions is below ``max_positions``.
        2. There is enough cash to buy at least one share at a
           reasonable price (heuristic: cash > ₹1 000).

        Returns:
            ``True`` if a new position is allowed, ``False`` otherwise.
        """
        # Check position count cap
        if self.num_positions >= self._bt_cfg.max_positions:
            return False

        # Minimum cash threshold to avoid dust positions
        _MIN_CASH_THRESHOLD = 1_000.0
        if self.cash < _MIN_CASH_THRESHOLD:
            return False

        return True

    # ── 2. open_position ─────────────────────────────────────────────

    def open_position(
        self,
        ticker: str,
        date: pd.Timestamp,
        price: float,
        atr: float,
    ) -> Optional[Position]:
        """Open a new long position in *ticker*.

        **Position-sizing logic (equal-weight with risk cap)**:

        1. ``target_alloc = total_equity / max_positions`` — equal-weight
           allocation per slot.
        2. ``risk_alloc = total_equity × risk_per_trade_pct`` — maximum
           capital risked per trade.
        3. ``alloc = min(target_alloc, risk_alloc, available_cash)`` —
           final capital to deploy.
        4. ``shares = floor(alloc / effective_entry_price)`` — round down
           to whole shares (no fractional trading on NSE).

        Parameters:
            ticker: NSE ticker symbol.
            date: Entry date.
            price: Raw market price at entry (before costs).
            atr: Average True Range value on entry date, used to set
                 the stop-loss level.

        Returns:
            The newly created ``Position``, or ``None`` if the position
            could not be opened (not enough capital, already holding,
            etc.).
        """
        # Guard: already in this stock
        if ticker in self.positions:
            logger.warning("SKIP %s — already holding a position", ticker)
            return None

        # Guard: capacity
        if not self.can_open_position():
            logger.warning(
                "SKIP %s — max positions (%d) reached or insufficient cash",
                ticker,
                self._bt_cfg.max_positions,
            )
            return None

        # ── Compute effective entry price (inflated by costs) ──
        effective_price = self._effective_buy_price(price)

        # ── Determine capital allocation ──
        equity = self.total_equity

        # Equal-weight allocation: spread equity across max_positions slots
        target_alloc = equity / self._bt_cfg.max_positions

        # Risk-cap: never risk more than risk_per_trade_pct of equity
        risk_alloc = equity * self._bt_cfg.risk_per_trade_pct

        # Use the smaller of the two, and never exceed available cash
        alloc = min(target_alloc, risk_alloc, self.cash)

        # ── Compute whole shares ──
        shares = math.floor(alloc / effective_price)

        if shares <= 0:
            logger.warning(
                "SKIP %s — allocation ₹%.2f too small for price ₹%.2f",
                ticker,
                alloc,
                effective_price,
            )
            return None

        # ── Compute stop-loss level ──
        stop_loss = effective_price - self._sig_cfg.atr_stop_multiplier * atr

        # ── Deduct cash ──
        cost = shares * effective_price
        self.cash -= cost

        # ── Create and store the position ──
        position = Position(
            ticker=ticker,
            entry_date=pd.Timestamp(date),
            entry_price=effective_price,
            shares=shares,
            stop_loss=stop_loss,
            atr_at_entry=atr,
        )
        self.positions[ticker] = position

        logger.info(
            "OPEN  %-15s | date=%s | price=₹%.2f (raw ₹%.2f) | "
            "shares=%d | cost=₹%.2f | stop=₹%.2f | cash_left=₹%.2f",
            ticker,
            date.strftime("%Y-%m-%d"),
            effective_price,
            price,
            shares,
            cost,
            stop_loss,
            self.cash,
        )

        return position

    # ── 3. close_position ────────────────────────────────────────────

    def close_position(
        self,
        ticker: str,
        date: pd.Timestamp,
        price: float,
        reason: str,
    ) -> Optional[Trade]:
        """Close an open position and record the round-trip trade.

        Parameters:
            ticker: NSE ticker symbol of the position to close.
            date: Exit date.
            price: Raw market price at exit (before costs).
            reason: Why the position was closed — one of
                    ``'mean_reversion'``, ``'rsi_exit'``,
                    ``'time_stop'``, ``'atr_stop'``.

        Returns:
            The completed ``Trade`` object, or ``None`` if *ticker*
            is not in the current positions.
        """
        if ticker not in self.positions:
            logger.warning("CLOSE FAILED — %s not in positions", ticker)
            return None

        position = self.positions.pop(ticker)

        # ── Compute effective exit price (deflated by costs) ──
        effective_exit = self._effective_sell_price(price)

        # ── PnL calculation ──
        pnl = (effective_exit - position.entry_price) * position.shares
        # Percentage PnL relative to capital deployed
        capital_deployed = position.entry_price * position.shares
        pnl_pct = (pnl / capital_deployed) if capital_deployed > 0 else 0.0

        # ── Holding period ──
        holding_days = (pd.Timestamp(date) - position.entry_date).days

        # ── Return cash proceeds ──
        proceeds = effective_exit * position.shares
        self.cash += proceeds

        # ── Record the trade ──
        trade = Trade(
            ticker=ticker,
            entry_date=position.entry_date,
            exit_date=pd.Timestamp(date),
            entry_price=position.entry_price,
            exit_price=effective_exit,
            shares=position.shares,
            pnl=pnl,
            pnl_pct=pnl_pct,
            holding_days=holding_days,
            exit_reason=reason,
        )
        self.trade_history.append(trade)

        logger.info(
            "CLOSE %-15s | date=%s | exit=₹%.2f (raw ₹%.2f) | "
            "PnL=₹%.2f (%.2f%%) | days=%d | reason=%s | cash=₹%.2f",
            ticker,
            date.strftime("%Y-%m-%d"),
            effective_exit,
            price,
            pnl,
            pnl_pct * 100,
            holding_days,
            reason,
            self.cash,
        )

        return trade

    # ── 4. update_equity ─────────────────────────────────────────────

    def update_equity(
        self,
        date: pd.Timestamp,
        current_prices: Dict[str, float],
    ) -> EquityPoint:
        """Mark-to-market all open positions and record a daily snapshot.

        Parameters:
            date: The snapshot date.
            current_prices: Mapping of ticker → current market price for
                            every held position.  Tickers not in the dict
                            are valued at their entry price (fallback).

        Returns:
            An ``EquityPoint`` snapshot appended to ``self.equity_curve``.
        """
        # ── Sum up mark-to-market invested value ──
        invested_value = 0.0
        for ticker, pos in self.positions.items():
            # Use current price if available, else fall back to entry price
            mtm_price = current_prices.get(ticker, pos.entry_price)
            invested_value += mtm_price * pos.shares

        total_equity = self.cash + invested_value

        # Exposure as a fraction; handle zero-equity edge case
        exposure_pct = (
            (invested_value / total_equity) if total_equity > 0 else 0.0
        )

        point = EquityPoint(
            date=pd.Timestamp(date),
            cash=self.cash,
            invested_value=invested_value,
            total_equity=total_equity,
            num_positions=self.num_positions,
            exposure_pct=exposure_pct,
        )
        self.equity_curve.append(point)

        return point

    # ── 5. check_stop_losses ─────────────────────────────────────────

    def check_stop_losses(
        self,
        date: pd.Timestamp,
        current_prices: Dict[str, float],
    ) -> List[str]:
        """Identify positions whose stop-loss has been breached.

        A stop-loss is breached when the current market price drops
        *at or below* the position's pre-computed ``stop_loss`` level.

        Parameters:
            date: Current bar date (used for logging).
            current_prices: Mapping of ticker → current market price.

        Returns:
            List of ticker symbols that should be closed via
            ``close_position(ticker, date, price, 'atr_stop')``.
        """
        triggered: List[str] = []

        for ticker, pos in self.positions.items():
            current_price = current_prices.get(ticker)

            if current_price is None:
                # No price data for this ticker today — skip
                continue

            if current_price <= pos.stop_loss:
                logger.info(
                    "STOP-LOSS %-15s | price=₹%.2f <= stop=₹%.2f | "
                    "date=%s",
                    ticker,
                    current_price,
                    pos.stop_loss,
                    date.strftime("%Y-%m-%d"),
                )
                triggered.append(ticker)

        return triggered

    # ── 6. get_portfolio_summary ─────────────────────────────────────

    def get_portfolio_summary(self) -> Dict:
        """Return a dictionary summarising the current portfolio state.

        Returns:
            Dict with keys: ``cash``, ``total_equity``, ``num_positions``,
            ``positions`` (list of dicts), ``total_trades``,
            ``winning_trades``, ``losing_trades``, ``win_rate``,
            ``total_pnl``, ``avg_pnl_pct``.
        """
        equity = self.total_equity

        # ── Position details ──
        pos_details = [
            {
                "ticker": pos.ticker,
                "entry_date": pos.entry_date.strftime("%Y-%m-%d"),
                "entry_price": round(pos.entry_price, 2),
                "shares": pos.shares,
                "stop_loss": round(pos.stop_loss, 2),
                "invested": round(pos.entry_price * pos.shares, 2),
            }
            for pos in self.positions.values()
        ]

        # ── Trade statistics ──
        total_trades = len(self.trade_history)
        winning = [t for t in self.trade_history if t.pnl > 0]
        losing = [t for t in self.trade_history if t.pnl <= 0]
        win_rate = (len(winning) / total_trades) if total_trades > 0 else 0.0
        total_pnl = sum(t.pnl for t in self.trade_history)
        avg_pnl_pct = (
            np.mean([t.pnl_pct for t in self.trade_history])
            if total_trades > 0
            else 0.0
        )

        return {
            "cash": round(self.cash, 2),
            "total_equity": round(equity, 2),
            "num_positions": self.num_positions,
            "positions": pos_details,
            "total_trades": total_trades,
            "winning_trades": len(winning),
            "losing_trades": len(losing),
            "win_rate": round(win_rate, 4),
            "total_pnl": round(total_pnl, 2),
            "avg_pnl_pct": round(float(avg_pnl_pct), 4),
        }

    # ── 7. get_trade_log ─────────────────────────────────────────────

    def get_trade_log(self) -> pd.DataFrame:
        """Return all completed trades as a DataFrame.

        Returns:
            DataFrame with columns matching the ``Trade`` dataclass
            fields.  Returns an empty DataFrame (with correct columns)
            if no trades have been recorded yet.
        """
        columns = [
            "ticker",
            "entry_date",
            "exit_date",
            "entry_price",
            "exit_price",
            "shares",
            "pnl",
            "pnl_pct",
            "holding_days",
            "exit_reason",
        ]

        if not self.trade_history:
            return pd.DataFrame(columns=columns)

        records = [
            {
                "ticker": t.ticker,
                "entry_date": t.entry_date,
                "exit_date": t.exit_date,
                "entry_price": round(t.entry_price, 2),
                "exit_price": round(t.exit_price, 2),
                "shares": t.shares,
                "pnl": round(t.pnl, 2),
                "pnl_pct": round(t.pnl_pct, 6),
                "holding_days": t.holding_days,
                "exit_reason": t.exit_reason,
            }
            for t in self.trade_history
        ]

        return pd.DataFrame(records, columns=columns)

    # ── 8. get_equity_curve ──────────────────────────────────────────

    def get_equity_curve(self) -> pd.DataFrame:
        """Return the daily equity curve as a DataFrame.

        Returns:
            DataFrame with columns matching the ``EquityPoint``
            dataclass fields, indexed by date.  Returns an empty
            DataFrame if no snapshots have been recorded.
        """
        columns = [
            "date",
            "cash",
            "invested_value",
            "total_equity",
            "num_positions",
            "exposure_pct",
        ]

        if not self.equity_curve:
            return pd.DataFrame(columns=columns)

        records = [
            {
                "date": ep.date,
                "cash": round(ep.cash, 2),
                "invested_value": round(ep.invested_value, 2),
                "total_equity": round(ep.total_equity, 2),
                "num_positions": ep.num_positions,
                "exposure_pct": round(ep.exposure_pct, 6),
            }
            for ep in self.equity_curve
        ]

        df = pd.DataFrame(records, columns=columns)
        df = df.set_index("date")
        return df

    # ── 9. reset ─────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset the portfolio to its initial state.

        Clears all positions, trade history, and equity snapshots.
        Restores cash to ``initial_capital``.
        """
        self.cash = self._initial_capital
        self.positions.clear()
        self.trade_history.clear()
        self.equity_curve.clear()

        logger.info(
            "Portfolio RESET | cash=₹%.2f | positions=%d | trades=%d",
            self.cash,
            self.num_positions,
            len(self.trade_history),
        )

    # ── dunder ───────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"PortfolioManager(cash=₹{self.cash:,.2f}, "
            f"positions={self.num_positions}, "
            f"trades={len(self.trade_history)}, "
            f"equity=₹{self.total_equity:,.2f})"
        )

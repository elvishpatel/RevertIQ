"""
RevertIQ — Walk-Forward Optimizer
==================================

Prevents overfitting by using proper out-of-sample validation.

Walk-Forward Process
--------------------
::

    |──── In-Sample (252d) ────|── OOS (63d) ──|
              |──── In-Sample (252d) ────|── OOS (63d) ──|
                        |──── In-Sample (252d) ────|── OOS (63d) ──|

1. Optimise parameters on in-sample window
2. Test with *fixed* parameters on out-of-sample window
3. Roll forward and repeat
4. Concatenate all OOS results → realistic performance estimate

Overfitting Detection
---------------------
``overfitting_ratio = mean(IS_sharpe) / mean(OOS_sharpe)``

- Ratio ~ 1.0 → robust strategy
- Ratio > 2.0 → likely overfitting
- Ratio > 3.0 → definitely overfitting

Usage
-----
>>> from revertiq.backtesting.optimizer import WalkForwardOptimizer
>>> opt = WalkForwardOptimizer(config)
>>> result = opt.optimize(price_data, market_data)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from revertiq.config.settings import (
    OptimizationConfig,
    RevertIQConfig,
    SignalConfig,
    get_default_config,
)
from revertiq.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class WindowResult:
    """Result for a single walk-forward window."""

    window_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    best_params: Dict[str, Any]
    in_sample_sharpe: float
    out_of_sample_sharpe: float
    out_of_sample_returns: pd.Series = field(repr=False)


@dataclass
class WFOResult:
    """Aggregated walk-forward optimisation results."""

    windows: List[WindowResult]
    concatenated_oos_returns: pd.Series
    overall_metrics: Dict[str, float]
    param_stability: pd.DataFrame
    overfitting_ratio: float

    def print_summary(self) -> None:
        """Print a formatted summary of walk-forward results."""
        print("\n" + "=" * 60)
        print("  WALK-FORWARD OPTIMISATION RESULTS")
        print("=" * 60)
        print(f"  Windows evaluated   : {len(self.windows)}")
        print(f"  Overfitting ratio   : {self.overfitting_ratio:.2f}")

        status = (
            "✅ ROBUST"
            if self.overfitting_ratio < 2.0
            else "⚠️  MARGINAL"
            if self.overfitting_ratio < 3.0
            else "❌ OVERFITTING"
        )
        print(f"  Status              : {status}")

        print("\n  OOS Performance:")
        for k, v in self.overall_metrics.items():
            if isinstance(v, float):
                print(f"    {k:22s}: {v:>10.4f}")

        print("\n  Parameter Stability:")
        print(self.param_stability.to_string(index=True))

        print("\n  Per-Window Sharpe:")
        for w in self.windows:
            print(
                f"    Window {w.window_id:2d}: "
                f"IS={w.in_sample_sharpe:>6.2f}  "
                f"OOS={w.out_of_sample_sharpe:>6.2f}  "
                f"Params={w.best_params}"
            )
        print("=" * 60)


class WalkForwardOptimizer:
    """Walk-forward parameter optimiser.

    Parameters
    ----------
    config : RevertIQConfig, optional
        System configuration (uses optimisation sub-config).
    """

    def __init__(self, config: Optional[RevertIQConfig] = None) -> None:
        self.config = config or get_default_config()
        self.opt_cfg: OptimizationConfig = self.config.optimization

    def optimize(
        self,
        price_data: pd.DataFrame,
        market_data: pd.DataFrame,
        backtest_fn=None,
    ) -> WFOResult:
        """
        Run full walk-forward optimisation.

        Parameters
        ----------
        price_data : DataFrame
            Combined OHLCV for all stocks (date, ticker, OHLCV).
        market_data : DataFrame
            NIFTY index data (date, close) for regime filter.
        backtest_fn : callable, optional
            Function ``(price_data, market_data, params) → (sharpe, daily_returns)``.
            If None, uses a built-in simplified backtest.

        Returns
        -------
        WFOResult
        """
        dates = sorted(price_data["date"].unique())
        n_dates = len(dates)

        is_days = self.opt_cfg.in_sample_days
        oos_days = self.opt_cfg.out_of_sample_days
        step_days = self.opt_cfg.step_days

        # Build parameter grid
        param_grid = self._expand_param_grid(self.opt_cfg.param_grid)
        logger.info(
            "Walk-forward: IS=%d, OOS=%d, step=%d, grid size=%d",
            is_days, oos_days, step_days, len(param_grid),
        )

        windows: List[WindowResult] = []
        window_id = 0
        start_idx = 0

        while start_idx + is_days + oos_days <= n_dates:
            train_dates = dates[start_idx : start_idx + is_days]
            test_dates = dates[start_idx + is_days : start_idx + is_days + oos_days]

            train_start = train_dates[0]
            train_end = train_dates[-1]
            test_start = test_dates[0]
            test_end = test_dates[-1]

            logger.info(
                "Window %d: Train [%s → %s], Test [%s → %s]",
                window_id, train_start.date(), train_end.date(),
                test_start.date(), test_end.date(),
            )

            # ── In-sample optimisation ──
            train_price = price_data[
                price_data["date"].isin(train_dates)
            ]
            train_market = market_data[
                market_data.index.isin(train_dates)
                if isinstance(market_data.index, pd.DatetimeIndex)
                else market_data["date"].isin(train_dates)
            ] if "date" not in market_data.columns else market_data[
                market_data["date"].isin(train_dates)
            ]

            best_params, is_sharpe = self._grid_search(
                train_price, train_market, param_grid, backtest_fn
            )

            # ── Out-of-sample test ──
            test_price = price_data[
                price_data["date"].isin(test_dates)
            ]
            test_market = market_data[
                market_data.index.isin(test_dates)
                if isinstance(market_data.index, pd.DatetimeIndex)
                else market_data["date"].isin(test_dates)
            ] if "date" not in market_data.columns else market_data[
                market_data["date"].isin(test_dates)
            ]

            if backtest_fn:
                oos_sharpe, oos_returns = backtest_fn(
                    test_price, test_market, best_params
                )
            else:
                oos_sharpe, oos_returns = self._simple_backtest(
                    test_price, test_market, best_params
                )

            windows.append(WindowResult(
                window_id=window_id,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                best_params=best_params,
                in_sample_sharpe=is_sharpe,
                out_of_sample_sharpe=oos_sharpe,
                out_of_sample_returns=oos_returns,
            ))

            window_id += 1
            start_idx += step_days

        if not windows:
            logger.warning("No walk-forward windows could be constructed.")
            return WFOResult(
                windows=[],
                concatenated_oos_returns=pd.Series(dtype=float),
                overall_metrics={},
                param_stability=pd.DataFrame(),
                overfitting_ratio=float("inf"),
            )

        # ── Aggregate OOS results ──
        oos_returns = pd.concat(
            [w.out_of_sample_returns for w in windows]
        ).sort_index()

        # Overall OOS metrics
        from revertiq.backtesting.metrics import PerformanceMetrics

        oos_equity = (1 + oos_returns).cumprod() * self.config.backtest.initial_capital
        oos_equity_df = pd.DataFrame({
            "date": oos_equity.index,
            "total_equity": oos_equity.values,
        })

        overall_metrics = {
            "oos_sharpe": PerformanceMetrics.sharpe_ratio(oos_returns),
            "oos_sortino": PerformanceMetrics.sortino_ratio(oos_returns),
            "oos_total_return": float((1 + oos_returns).prod() - 1),
            "oos_annual_vol": float(oos_returns.std() * np.sqrt(252)),
        }

        # ── Parameter stability ──
        param_stability = pd.DataFrame(
            [w.best_params for w in windows]
        )

        # ── Overfitting ratio ──
        mean_is = np.mean([w.in_sample_sharpe for w in windows])
        mean_oos = np.mean([w.out_of_sample_sharpe for w in windows])
        overfitting_ratio = (
            mean_is / mean_oos if mean_oos != 0 else float("inf")
        )

        result = WFOResult(
            windows=windows,
            concatenated_oos_returns=oos_returns,
            overall_metrics=overall_metrics,
            param_stability=param_stability,
            overfitting_ratio=overfitting_ratio,
        )

        result.print_summary()
        return result

    # ─────────────────────────────────────────────────────────────────
    # Grid search
    # ─────────────────────────────────────────────────────────────────

    def _grid_search(
        self,
        price_data: pd.DataFrame,
        market_data: pd.DataFrame,
        param_grid: List[Dict],
        backtest_fn=None,
    ) -> tuple:
        """Find best params by Sharpe ratio on in-sample data."""
        best_sharpe = -np.inf
        best_params = param_grid[0] if param_grid else {}

        for params in param_grid:
            try:
                if backtest_fn:
                    sharpe, _ = backtest_fn(price_data, market_data, params)
                else:
                    sharpe, _ = self._simple_backtest(
                        price_data, market_data, params
                    )
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = params
            except Exception:
                continue

        return best_params, best_sharpe

    def _simple_backtest(
        self,
        price_data: pd.DataFrame,
        market_data: pd.DataFrame,
        params: Dict,
    ) -> tuple:
        """
        Simplified mean reversion backtest for optimisation speed.

        Uses a fast vectorised approach (not the full BacktestEngine)
        to enable rapid parameter sweeps.
        """
        if price_data.empty:
            return 0.0, pd.Series(dtype=float)

        rsi_thresh = params.get("rsi_buy_threshold", 10)
        zscore_thresh = params.get("zscore_buy_threshold", -1.5)
        hold_days = params.get("max_holding_days", 10)
        stop_mult = params.get("atr_stop_multiplier", 2.0)

        # Simple aggregate daily return of stocks meeting entry conditions
        daily_returns = []
        dates = sorted(price_data["date"].unique())

        for i, date in enumerate(dates[:-hold_days]):
            day_data = price_data[price_data["date"] == date]

            # Simple entry filter
            mask = pd.Series(True, index=day_data.index)
            if "rsi" in day_data.columns:
                mask &= day_data["rsi"] < rsi_thresh
            if "zscore" in day_data.columns:
                mask &= day_data["zscore"] < zscore_thresh

            candidates = day_data[mask]
            if candidates.empty:
                daily_returns.append(0.0)
                continue

            # Forward return over hold_days (simplified)
            future_dates = dates[i + 1 : i + 1 + hold_days]
            if not future_dates:
                daily_returns.append(0.0)
                continue

            tickers = candidates["ticker"].tolist()[:5]  # Top 5
            future = price_data[
                (price_data["date"].isin(future_dates))
                & (price_data["ticker"].isin(tickers))
            ]

            if future.empty:
                daily_returns.append(0.0)
                continue

            # Average return across selected stocks
            entry_prices = candidates.set_index("ticker")["close"]
            exit_data = price_data[
                (price_data["date"] == future_dates[-1])
                & (price_data["ticker"].isin(tickers))
            ].set_index("ticker")["close"]

            common = entry_prices.index.intersection(exit_data.index)
            if common.empty:
                daily_returns.append(0.0)
                continue

            ret = ((exit_data[common] / entry_prices[common]) - 1).mean()
            # Spread over hold_days
            daily_ret = ret / hold_days
            for _ in range(min(hold_days, len(dates) - i - 1)):
                daily_returns.append(daily_ret)

        if not daily_returns:
            return 0.0, pd.Series(dtype=float)

        returns = pd.Series(daily_returns, index=dates[: len(daily_returns)])
        # Apply costs
        cost = self.config.backtest.buy_cost_pct + self.config.backtest.sell_cost_pct
        # Approximate: deduct cost proportionally
        returns = returns - cost / 10  # Amortised

        sharpe = (
            returns.mean() / returns.std() * np.sqrt(252)
            if returns.std() > 0
            else 0.0
        )

        return sharpe, returns

    @staticmethod
    def _expand_param_grid(grid_spec: Dict[str, List]) -> List[Dict]:
        """Expand parameter grid into list of all combinations."""
        if not grid_spec:
            return [{}]
        keys = list(grid_spec.keys())
        values = list(grid_spec.values())
        return [dict(zip(keys, combo)) for combo in product(*values)]

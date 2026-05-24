"""
RevertIQ — Centralized Configuration
=====================================

All tunable parameters in one place.  Every module reads from these
dataclasses so that a single config change propagates everywhere.

Usage
-----
>>> from revertiq.config.settings import get_default_config
>>> cfg = get_default_config()
>>> cfg.signal.rsi_buy_threshold       # 10
>>> cfg.backtest.initial_capital       # 1_000_000

Override for experiments:
>>> from dataclasses import replace
>>> custom = replace(cfg.signal, rsi_buy_threshold=15)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────
# Data Configuration
# ─────────────────────────────────────────────────────────────────────

@dataclass
class DataConfig:
    """Settings for the data pipeline (downloading, storage, cleaning)."""

    # Date range for historical data
    start_date: str = "2018-01-01"
    end_date: str = "2026-05-23"

    # Storage paths (relative to project root)
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    results_dir: str = "data/results"

    # Download settings
    chunk_size: int = 10              # Tickers per batch (avoid rate limits)
    delay_between_chunks: float = 2.0 # Seconds between batches
    max_retries: int = 3              # Retry failed downloads
    retry_delay: float = 5.0          # Seconds before retry

    # Cleaning settings
    max_missing_pct: float = 0.20     # Drop tickers with >20 % missing data
    max_ffill_days: int = 5           # Forward-fill gaps up to 5 days


# ─────────────────────────────────────────────────────────────────────
# Technical Indicator Configuration
# ─────────────────────────────────────────────────────────────────────

@dataclass
class IndicatorConfig:
    """Parameters for technical indicator calculations."""

    # RSI — short-period for mean reversion (Connors-style)
    rsi_period: int = 2

    # Moving Averages
    sma_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 50, 200])
    ema_periods: List[int] = field(default_factory=lambda: [9, 21, 50])

    # Average True Range
    atr_period: int = 14

    # Bollinger Bands
    bb_period: int = 20
    bb_std_dev: float = 2.0

    # Z-score
    zscore_lookback: int = 20

    # Rolling return / volatility lookbacks
    return_periods: List[int] = field(default_factory=lambda: [1, 5, 10, 20])
    volatility_lookback: int = 20

    # Volume analysis
    volume_ma_period: int = 20


# ─────────────────────────────────────────────────────────────────────
# Signal Configuration
# ─────────────────────────────────────────────────────────────────────

@dataclass
class SignalConfig:
    """Thresholds for generating BUY / SELL signals."""

    # ── BUY conditions ──
    rsi_buy_threshold: float = 10.0       # RSI(2) < 10
    zscore_buy_threshold: float = -1.5    # Z-score < -1.5
    relative_weakness_threshold: float = -0.02  # Stock underperforming market by >2 %
    volume_spike_threshold: float = 1.5   # Volume > 1.5× 20-day avg
    bb_buy_condition: bool = True         # Price below lower Bollinger Band

    # ── SELL conditions ──
    zscore_exit_threshold: float = 0.0    # Mean reversion achieved
    rsi_exit_threshold: float = 70.0      # RSI(2) normalized
    max_holding_days: int = 10            # Time-stop
    atr_stop_multiplier: float = 2.0      # Stop-loss = entry - 2×ATR

    # ── Signal quality ──
    min_conditions_for_signal: int = 3    # Need at least N conditions true


# ─────────────────────────────────────────────────────────────────────
# Market Regime Configuration
# ─────────────────────────────────────────────────────────────────────

@dataclass
class RegimeConfig:
    """Market regime filter settings."""

    # Primary filter: NIFTY vs long-term moving average
    regime_ma_period: int = 200
    regime_buffer_pct: float = 1.0        # 1 % buffer to reduce whipsaw
    confirmation_days: int = 3            # N consecutive days to confirm regime

    # Secondary filter: volatility
    vix_threshold_percentile: float = 0.75  # VIX below 75th percentile
    realized_vol_lookback: int = 20
    realized_vol_benchmark: int = 252       # Annual lookback for percentile

    # Allowed regimes for trading (long side)
    allowed_regimes: List[str] = field(
        default_factory=lambda: ["strong_bull", "mild_bull"]
    )


# ─────────────────────────────────────────────────────────────────────
# Backtest Configuration
# ─────────────────────────────────────────────────────────────────────

@dataclass
class BacktestConfig:
    """Settings for the backtesting engine."""

    # Capital
    initial_capital: float = 1_000_000.0  # ₹10 lakhs

    # Transaction costs (Indian equity delivery)
    buy_cost_pct: float = 0.0015          # 0.15 % per buy (STT + charges)
    sell_cost_pct: float = 0.0015         # 0.15 % per sell
    slippage_pct: float = 0.0005          # 0.05 % per side

    # Position sizing
    max_positions: int = 5
    risk_per_trade_pct: float = 0.02      # Max 2 % of portfolio per trade

    # Rebalancing
    rebalance_frequency: str = "daily"    # 'daily' or 'weekly'


# ─────────────────────────────────────────────────────────────────────
# Ranking Configuration
# ─────────────────────────────────────────────────────────────────────

@dataclass
class RankingConfig:
    """Weights for composite ranking score.  Must sum to 1.0."""

    weights: Dict[str, float] = field(default_factory=lambda: {
        "rsi_score":               0.20,
        "zscore_score":            0.25,
        "relative_weakness_score": 0.20,
        "volume_exhaustion_score": 0.15,
        "atr_norm_score":          0.10,
        "dist_from_ma_score":      0.10,
    })

    def validate(self) -> None:
        total = sum(self.weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Ranking weights must sum to 1.0, got {total:.4f}"
            )


# ─────────────────────────────────────────────────────────────────────
# Walk-Forward Optimization Configuration
# ─────────────────────────────────────────────────────────────────────

@dataclass
class OptimizationConfig:
    """Settings for walk-forward parameter optimization."""

    # Window sizes (in trading days)
    in_sample_days: int = 252      # ~1 year
    out_of_sample_days: int = 63   # ~3 months
    step_days: int = 63            # Roll forward by OOS period

    # Objective function
    optimization_metric: str = "sharpe"  # 'sharpe', 'sortino', 'calmar'

    # Parameter search space (used by grid search)
    param_grid: Dict[str, List] = field(default_factory=lambda: {
        "rsi_buy_threshold":   [5, 10, 15, 20],
        "zscore_buy_threshold": [-2.5, -2.0, -1.5, -1.0],
        "max_holding_days":    [5, 7, 10],
        "atr_stop_multiplier": [1.5, 2.0, 2.5, 3.0],
    })


# ─────────────────────────────────────────────────────────────────────
# Master Configuration
# ─────────────────────────────────────────────────────────────────────

@dataclass
class RevertIQConfig:
    """Top-level configuration aggregating all sub-configs."""

    data: DataConfig = field(default_factory=DataConfig)
    indicator: IndicatorConfig = field(default_factory=IndicatorConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    ranking: RankingConfig = field(default_factory=RankingConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)

    # Project root (auto-detected)
    project_root: str = field(default_factory=lambda: _find_project_root())

    def validate(self) -> None:
        """Run all validation checks."""
        self.ranking.validate()

    def resolve_path(self, relative_path: str) -> str:
        """Resolve a relative path against the project root."""
        return os.path.join(self.project_root, relative_path)


def _find_project_root() -> str:
    """
    Walk up from this file to find the project root (directory
    containing ``setup.py`` or ``revertiq/``).

    Falls back to current working directory (Colab-friendly).
    """
    current = os.path.dirname(os.path.abspath(__file__))
    for _ in range(5):
        if os.path.exists(os.path.join(current, "setup.py")):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return os.getcwd()


def get_default_config() -> RevertIQConfig:
    """Return a validated default configuration."""
    cfg = RevertIQConfig()
    cfg.validate()
    return cfg

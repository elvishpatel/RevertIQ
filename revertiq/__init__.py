"""
RevertIQ — Cross-Sectional Mean Reversion Quant Research Platform
=================================================================

A modular quantitative trading research system for Indian equity
markets (NSE / NIFTY 50) focused on cross-sectional mean reversion
swing trading.

Quick Start
-----------
>>> from revertiq.config.settings import get_default_config
>>> from revertiq.data.downloader import DataDownloader
>>> from revertiq.indicators.technical import TechnicalIndicators
>>> from revertiq.signals.generator import SignalGenerator
>>> from revertiq.ranking.ranker import StockRanker
>>> from revertiq.backtesting.engine import BacktestEngine

Modules
-------
- config     : Centralized configuration
- data       : Data downloading, cleaning, and universe management
- indicators : Technical indicator calculations (RSI, ATR, BB, etc.)
- features   : Cross-sectional feature engineering
- signals    : Signal generation and market regime filtering
- ranking    : Cross-sectional stock ranking
- backtesting: Portfolio simulation and performance metrics
- portfolio  : Position sizing and risk management
- visualization: Charts, reports, and dashboards
- utils      : Logging, helpers, and utilities
"""

__version__ = "0.1.0"
__author__ = "RevertIQ Team"

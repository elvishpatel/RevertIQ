"""
RevertIQ — Data Downloader
===========================

Downloads OHLCV data from Yahoo Finance for NIFTY 50 constituents and
key market indices (NIFTY 50 index, India VIX).

Design decisions
----------------
* **Chunked downloads** — tickers are split into batches (default 10)
  with a configurable sleep between batches to avoid yfinance rate-limits.
* **Exponential backoff** — each ticker retries up to ``max_retries``
  times with delay doubling on every failure.
* **Flat CSV storage** — one file per ticker in ``data/raw/``, making it
  easy to update individual stocks without re-downloading everything.
* **MultiIndex handling** — yfinance may return MultiIndex columns
  ``(Price, Ticker)``; we flatten them to lowercase scalar names.
* **auto_adjust** — recent yfinance (≥ 0.2.36) defaults to
  ``auto_adjust=True`` so the ``Close`` column is already adjusted.
  We detect and handle both adjusted and unadjusted layouts gracefully.

Usage
-----
>>> from revertiq.data.downloader import DataDownloader
>>> dl = DataDownloader()
>>> data = dl.download_all()   # dict[str, pd.DataFrame]
"""

from __future__ import annotations

import os
import re
import time
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf
from tqdm import tqdm

from revertiq.config.settings import DataConfig, get_default_config
from revertiq.data.universe import get_index_tickers, get_nifty50_tickers
from revertiq.utils.helpers import ensure_dir, flatten_yf_columns
from revertiq.utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _safe_filename(ticker: str) -> str:
    """Convert a ticker symbol to a filesystem-safe filename.

    The ``&`` character in **M&M.NS** (Mahindra & Mahindra) breaks on
    Windows paths, so we replace it with ``_``.

    Parameters
    ----------
    ticker : str
        Raw yfinance ticker symbol (e.g. ``"M&M.NS"``).

    Returns
    -------
    str
        Safe filename string (e.g. ``"M_M.NS"``).
    """
    # Replace characters that are unsafe in filenames: & \ / : * ? " < > |
    return re.sub(r'[&\\/:*?"<>|]', "_", ticker)


def _flatten_single_ticker_df(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Flatten yfinance output for a *single* ticker download.

    yfinance can return several column layouts:
    1. Simple columns: ``['Open', 'High', 'Low', 'Close', 'Volume']``
    2. MultiIndex columns: ``(Price, Ticker)`` — happens even for single
       tickers in some yfinance versions.

    We normalise everything to lowercase scalar column names.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame from ``yf.download``.
    ticker : str
        Ticker symbol (used only for logging).

    Returns
    -------
    pd.DataFrame
        DataFrame with lowercase column names.
    """
    if df.empty:
        return df

    # Case 1: MultiIndex columns — drop the ticker level
    if isinstance(df.columns, pd.MultiIndex):
        # yfinance MultiIndex is typically (Price, Ticker)
        df.columns = df.columns.get_level_values(0)

    # Lowercase and replace spaces with underscores
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]

    # Remove any duplicate column names that can arise after flattening
    df = df.loc[:, ~df.columns.duplicated()]

    return df


# ─────────────────────────────────────────────────────────────────────
# DataDownloader
# ─────────────────────────────────────────────────────────────────────

class DataDownloader:
    """Download and persist OHLCV data for NIFTY 50 stocks and indices.

    Parameters
    ----------
    config : DataConfig, optional
        Data-pipeline configuration.  Falls back to the project default.
    project_root : str, optional
        Project root directory.  Used to resolve relative paths in the
        config (e.g. ``data/raw``).  Detected automatically when *None*.

    Examples
    --------
    >>> dl = DataDownloader()
    >>> data = dl.download_all()
    >>> dl.download_index_data()
    """

    def __init__(
        self,
        config: Optional[DataConfig] = None,
        project_root: Optional[str] = None,
    ) -> None:
        cfg = get_default_config()
        self.config: DataConfig = config or cfg.data
        self.project_root: str = project_root or cfg.project_root

        # Resolve absolute path for raw data storage
        self.raw_dir: str = os.path.join(self.project_root, self.config.raw_data_dir)
        ensure_dir(self.raw_dir)

        logger.info(
            "DataDownloader initialised — raw_dir=%s, range=%s→%s",
            self.raw_dir,
            self.config.start_date,
            self.config.end_date,
        )

    # ─────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────

    def download_all(self) -> Dict[str, pd.DataFrame]:
        """Download OHLCV data for every NIFTY 50 constituent.

        Tickers are processed in chunks of ``config.chunk_size`` with a
        ``config.delay_between_chunks`` pause between batches.

        Returns
        -------
        dict[str, pd.DataFrame]
            Mapping of ``{ticker: ohlcv_dataframe}``.  Tickers that
            failed after all retries are excluded.
        """
        tickers: List[str] = get_nifty50_tickers(suffix=True)
        logger.info(
            "Starting download for %d NIFTY 50 tickers (chunk=%d, delay=%.1fs)",
            len(tickers),
            self.config.chunk_size,
            self.config.delay_between_chunks,
        )

        results: Dict[str, pd.DataFrame] = {}
        chunks = self._chunk_list(tickers, self.config.chunk_size)

        for chunk_idx, chunk in enumerate(
            tqdm(chunks, desc="Downloading chunks", unit="chunk")
        ):
            for ticker in chunk:
                df = self.download_ticker(ticker)
                if df is not None and not df.empty:
                    results[ticker] = df

            # Delay between chunks (skip after the last chunk)
            if chunk_idx < len(chunks) - 1:
                time.sleep(self.config.delay_between_chunks)

        logger.info(
            "Download complete — %d / %d tickers succeeded",
            len(results),
            len(tickers),
        )
        return results

    def download_ticker(self, ticker: str) -> Optional[pd.DataFrame]:
        """Download OHLCV data for a single ticker with retry logic.

        Implements exponential backoff: on the *k*-th retry the delay is
        ``retry_delay × 2^k`` seconds.

        Parameters
        ----------
        ticker : str
            Yahoo Finance ticker symbol (e.g. ``"RELIANCE.NS"``).

        Returns
        -------
        pd.DataFrame or None
            OHLCV DataFrame indexed by date, or *None* if all retries
            fail.
        """
        delay = self.config.retry_delay

        for attempt in range(1, self.config.max_retries + 1):
            try:
                logger.debug(
                    "Downloading %s (attempt %d/%d)",
                    ticker,
                    attempt,
                    self.config.max_retries,
                )

                # yfinance ≥ 0.2.36 uses auto_adjust=True by default,
                # so 'Close' is already the adjusted close price.
                df: pd.DataFrame = yf.download(
                    tickers=ticker,
                    start=self.config.start_date,
                    end=self.config.end_date,
                    auto_adjust=True,
                    progress=False,
                )

                if df.empty:
                    logger.warning("Empty DataFrame for %s — skipping", ticker)
                    return None

                # Flatten potential MultiIndex columns
                df = _flatten_single_ticker_df(df, ticker)

                # Ensure we have the core OHLCV columns
                expected = {"open", "high", "low", "close", "volume"}
                available = set(df.columns)
                missing = expected - available
                if missing:
                    logger.warning(
                        "%s is missing columns %s — available: %s",
                        ticker,
                        missing,
                        available,
                    )

                # Persist to CSV
                self._save_csv(df, ticker)

                logger.debug(
                    "%s downloaded — %d rows, %s → %s",
                    ticker,
                    len(df),
                    df.index.min().strftime("%Y-%m-%d") if len(df) else "N/A",
                    df.index.max().strftime("%Y-%m-%d") if len(df) else "N/A",
                )
                return df

            except Exception as exc:
                logger.warning(
                    "Attempt %d/%d failed for %s: %s",
                    attempt,
                    self.config.max_retries,
                    ticker,
                    exc,
                )
                if attempt < self.config.max_retries:
                    logger.info("Retrying %s in %.1f s …", ticker, delay)
                    time.sleep(delay)
                    delay *= 2  # exponential backoff
                else:
                    logger.error(
                        "All %d retries exhausted for %s", self.config.max_retries, ticker
                    )
                    return None

    def download_index_data(self) -> Dict[str, pd.DataFrame]:
        """Download data for market indices (NIFTY 50, India VIX, etc.).

        Uses the same retry logic as :meth:`download_ticker`.

        Returns
        -------
        dict[str, pd.DataFrame]
            Mapping of ``{friendly_name: ohlcv_dataframe}`` — e.g.
            ``{'nifty50': df, 'india_vix': df}``.
        """
        index_map: Dict[str, str] = get_index_tickers()
        results: Dict[str, pd.DataFrame] = {}

        for name, ticker in tqdm(
            index_map.items(), desc="Downloading indices", unit="idx"
        ):
            logger.info("Downloading index: %s (%s)", name, ticker)
            df = self.download_ticker(ticker)
            if df is not None and not df.empty:
                results[name] = df
            # Small delay between index downloads
            time.sleep(1.0)

        logger.info(
            "Index download complete — %d / %d succeeded",
            len(results),
            len(index_map),
        )
        return results

    # ─────────────────────────────────────────────────────────────────
    # Private Helpers
    # ─────────────────────────────────────────────────────────────────

    def _save_csv(self, df: pd.DataFrame, ticker: str) -> None:
        """Save a ticker DataFrame to ``raw_dir/<safe_ticker>.csv``.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV data with a DatetimeIndex.
        ticker : str
            Original ticker symbol.
        """
        safe = _safe_filename(ticker)
        filepath = os.path.join(self.raw_dir, f"{safe}.csv")
        df.to_csv(filepath)
        logger.debug("Saved %s → %s (%d rows)", ticker, filepath, len(df))

    @staticmethod
    def _chunk_list(lst: List[str], size: int) -> List[List[str]]:
        """Split a list into sublists of at most *size* elements.

        Parameters
        ----------
        lst : list[str]
            Items to chunk.
        size : int
            Maximum items per chunk.

        Returns
        -------
        list[list[str]]
            List of chunks.
        """
        return [lst[i : i + size] for i in range(0, len(lst), size)]

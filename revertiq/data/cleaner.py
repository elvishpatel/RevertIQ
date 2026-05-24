"""
RevertIQ — Data Cleaner
========================

Loads raw CSV data, validates OHLCV integrity, fills small gaps, removes
low-quality tickers, and outputs cleaned per-ticker CSVs plus a combined
MultiIndex DataFrame.

Cleaning pipeline (per ticker)
------------------------------
1. **Load** — read CSV from ``data/raw/``, parse dates as index.
2. **Standardise** — lowercase column names, add ``ticker`` column.
3. **Forward-fill** — fill NaN gaps up to ``max_ffill_days`` trading
   days.  Longer gaps are left as NaN so they get caught by the
   missing-data check.
4. **Validate** — enforce ``High ≥ Low``, ``Volume ≥ 0``, no negative
   prices.  Invalid rows are dropped.
5. **Quality gate** — if the remaining missing-data percentage exceeds
   ``max_missing_pct`` the ticker is rejected entirely.
6. **Save** — write the cleaned DataFrame to ``data/processed/``.

Usage
-----
>>> from revertiq.data.cleaner import DataCleaner
>>> cl = DataCleaner()
>>> stats = cl.clean_all()       # returns per-ticker stats
>>> combined = cl.get_combined_df()  # MultiIndex (date, ticker)
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from revertiq.config.settings import DataConfig, get_default_config
from revertiq.utils.helpers import ensure_dir
from revertiq.utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Cleaning Statistics
# ─────────────────────────────────────────────────────────────────────

@dataclass
class CleaningStats:
    """Summary statistics for a single ticker's cleaning run.

    Attributes
    ----------
    ticker : str
        Ticker symbol.
    raw_rows : int
        Number of rows in the raw CSV.
    gaps_filled : int
        Number of NaN values forward-filled.
    invalid_rows_removed : int
        Rows dropped due to integrity violations.
    final_rows : int
        Rows remaining after cleaning.
    missing_pct : float
        Fraction of NaN values in the *raw* data (before fill).
    accepted : bool
        Whether the ticker passed the quality gate.
    rejection_reason : str
        Human-readable reason if the ticker was rejected.
    """

    ticker: str = ""
    raw_rows: int = 0
    gaps_filled: int = 0
    invalid_rows_removed: int = 0
    final_rows: int = 0
    missing_pct: float = 0.0
    accepted: bool = True
    rejection_reason: str = ""


# ─────────────────────────────────────────────────────────────────────
# DataCleaner
# ─────────────────────────────────────────────────────────────────────

class DataCleaner:
    """Clean and validate raw OHLCV CSVs for the NIFTY 50 universe.

    Parameters
    ----------
    config : DataConfig, optional
        Data-pipeline configuration.  Defaults to project settings.
    project_root : str, optional
        Project root for resolving relative paths.

    Examples
    --------
    >>> cl = DataCleaner()
    >>> stats = cl.clean_all()
    >>> df = cl.get_combined_df()
    """

    # Core OHLCV columns that every file must contain after cleaning
    _REQUIRED_COLS: List[str] = ["open", "high", "low", "close", "volume"]

    def __init__(
        self,
        config: Optional[DataConfig] = None,
        project_root: Optional[str] = None,
    ) -> None:
        cfg = get_default_config()
        self.config: DataConfig = config or cfg.data
        self.project_root: str = project_root or cfg.project_root

        # Resolve absolute paths
        self.raw_dir: str = os.path.join(self.project_root, self.config.raw_data_dir)
        self.processed_dir: str = os.path.join(
            self.project_root, self.config.processed_data_dir
        )
        ensure_dir(self.processed_dir)

        logger.info(
            "DataCleaner initialised — raw=%s, processed=%s, "
            "max_missing=%.0f%%, max_ffill=%dd",
            self.raw_dir,
            self.processed_dir,
            self.config.max_missing_pct * 100,
            self.config.max_ffill_days,
        )

    # ─────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────

    def clean_all(self) -> Dict[str, CleaningStats]:
        """Clean every raw CSV found in the raw data directory.

        Returns
        -------
        dict[str, CleaningStats]
            ``{ticker: stats}`` for every file processed.
        """
        csv_files = self._list_raw_csvs()
        if not csv_files:
            logger.warning("No CSV files found in %s", self.raw_dir)
            return {}

        logger.info("Cleaning %d raw CSV files …", len(csv_files))
        all_stats: Dict[str, CleaningStats] = {}

        accepted_count = 0
        rejected_count = 0

        for filename in tqdm(csv_files, desc="Cleaning tickers", unit="file"):
            ticker = self._ticker_from_filename(filename)
            stats = self.clean_ticker(ticker, filename)
            all_stats[ticker] = stats

            if stats.accepted:
                accepted_count += 1
            else:
                rejected_count += 1

        # Print summary report
        self._log_summary(all_stats, accepted_count, rejected_count)
        return all_stats

    def clean_ticker(
        self,
        ticker: str,
        filename: Optional[str] = None,
    ) -> CleaningStats:
        """Clean a single ticker's raw CSV.

        Parameters
        ----------
        ticker : str
            Ticker symbol (used for the ``ticker`` column and logging).
        filename : str, optional
            CSV filename inside ``raw_dir``.  If *None*, we derive it
            from the ticker.

        Returns
        -------
        CleaningStats
            Summary of what happened during cleaning.
        """
        stats = CleaningStats(ticker=ticker)

        # ── 1. Load raw CSV ──────────────────────────────────────────
        if filename is None:
            filename = self._filename_from_ticker(ticker)

        filepath = os.path.join(self.raw_dir, filename)
        if not os.path.exists(filepath):
            stats.accepted = False
            stats.rejection_reason = f"File not found: {filepath}"
            logger.warning("Skipping %s — file not found", ticker)
            return stats

        df = self._load_csv(filepath)
        if df.empty:
            stats.accepted = False
            stats.rejection_reason = "Empty DataFrame after loading"
            logger.warning("Skipping %s — empty CSV", ticker)
            return stats

        stats.raw_rows = len(df)

        # ── 2. Standardise column names ──────────────────────────────
        df = self._standardise_columns(df)

        # Check required columns exist
        missing_cols = set(self._REQUIRED_COLS) - set(df.columns)
        if missing_cols:
            stats.accepted = False
            stats.rejection_reason = f"Missing columns: {missing_cols}"
            logger.warning("Skipping %s — missing columns %s", ticker, missing_cols)
            return stats

        # ── 3. Measure missing data (before fill) ───────────────────
        ohlcv = df[self._REQUIRED_COLS]
        total_cells = ohlcv.size
        nan_cells = int(ohlcv.isna().sum().sum())
        stats.missing_pct = nan_cells / total_cells if total_cells > 0 else 0.0

        # ── 4. Forward-fill gaps (up to max_ffill_days) ──────────────
        nan_before = int(df[self._REQUIRED_COLS].isna().sum().sum())
        df[self._REQUIRED_COLS] = df[self._REQUIRED_COLS].ffill(
            limit=self.config.max_ffill_days
        )
        nan_after = int(df[self._REQUIRED_COLS].isna().sum().sum())
        stats.gaps_filled = nan_before - nan_after

        # ── 5. Drop remaining NaN rows ───────────────────────────────
        rows_before_drop = len(df)
        df = df.dropna(subset=self._REQUIRED_COLS)
        rows_dropped_nan = rows_before_drop - len(df)

        # ── 6. Validate OHLCV integrity ──────────────────────────────
        rows_before_valid = len(df)
        df = self._validate_ohlcv(df)
        rows_dropped_invalid = rows_before_valid - len(df)
        stats.invalid_rows_removed = rows_dropped_nan + rows_dropped_invalid

        # ── 7. Quality gate ──────────────────────────────────────────
        if stats.raw_rows > 0:
            # Effective missing = cells we could NOT recover
            effective_missing = 1.0 - (len(df) / stats.raw_rows)
        else:
            effective_missing = 1.0

        if effective_missing > self.config.max_missing_pct:
            stats.accepted = False
            stats.rejection_reason = (
                f"Too much missing data: {effective_missing:.1%} "
                f"(threshold {self.config.max_missing_pct:.0%})"
            )
            logger.warning(
                "Rejecting %s — %.1f%% effective missing data",
                ticker,
                effective_missing * 100,
            )
            return stats

        # ── 8. Add ticker column ─────────────────────────────────────
        df["ticker"] = ticker

        stats.final_rows = len(df)
        stats.accepted = True

        # ── 9. Save cleaned CSV ──────────────────────────────────────
        self._save_processed(df, ticker)

        logger.debug(
            "%s cleaned — %d→%d rows, %d gaps filled, %d invalid removed",
            ticker,
            stats.raw_rows,
            stats.final_rows,
            stats.gaps_filled,
            stats.invalid_rows_removed,
        )
        return stats

    def load_all_processed(self) -> Dict[str, pd.DataFrame]:
        """Load all cleaned CSVs from the processed directory.

        Returns
        -------
        dict[str, pd.DataFrame]
            ``{ticker: cleaned_dataframe}`` for every file found.
        """
        csv_files = self._list_processed_csvs()
        if not csv_files:
            logger.warning("No processed CSVs in %s", self.processed_dir)
            return {}

        result: Dict[str, pd.DataFrame] = {}
        for filename in tqdm(csv_files, desc="Loading processed data", unit="file"):
            ticker = self._ticker_from_filename(filename)
            filepath = os.path.join(self.processed_dir, filename)
            df = self._load_csv(filepath)
            if not df.empty:
                result[ticker] = df

        logger.info("Loaded %d processed ticker files", len(result))
        return result

    def get_combined_df(self) -> pd.DataFrame:
        """Load all processed tickers into a single MultiIndex DataFrame.

        The resulting DataFrame has a ``(date, ticker)`` MultiIndex,
        making it straightforward to do cross-sectional operations::

            combined.loc["2024-01-15"]          # all tickers on that date
            combined.xs("RELIANCE.NS", level=1) # single stock time-series

        Returns
        -------
        pd.DataFrame
            Combined DataFrame with MultiIndex ``(date, ticker)``.
        """
        all_data = self.load_all_processed()
        if not all_data:
            logger.warning("No processed data to combine")
            return pd.DataFrame()

        frames: List[pd.DataFrame] = []
        for ticker, df in all_data.items():
            # Ensure 'ticker' column exists
            if "ticker" not in df.columns:
                df = df.copy()
                df["ticker"] = ticker
            frames.append(df)

        # Concatenate all tickers vertically
        combined = pd.concat(frames, ignore_index=False)

        # Ensure the index is named 'date'
        combined.index.name = "date"

        # Set MultiIndex (date, ticker)
        if "ticker" in combined.columns:
            combined = combined.set_index("ticker", append=True)

        combined = combined.sort_index()

        logger.info(
            "Combined DataFrame: %d rows × %d cols, %d unique tickers",
            len(combined),
            len(combined.columns),
            combined.index.get_level_values("ticker").nunique(),
        )
        return combined

    # ─────────────────────────────────────────────────────────────────
    # Private Helpers
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _load_csv(filepath: str) -> pd.DataFrame:
        """Read a CSV with date parsing and index setting.

        Parameters
        ----------
        filepath : str
            Absolute path to the CSV file.

        Returns
        -------
        pd.DataFrame
            DataFrame with DatetimeIndex, or empty DataFrame on error.
        """
        try:
            df = pd.read_csv(filepath, parse_dates=True, index_col=0)
            # Ensure the index is actually datetime
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index, errors="coerce")
                df = df[df.index.notna()]
            df = df.sort_index()
            return df
        except Exception as exc:
            logger.error("Failed to load %s: %s", filepath, exc)
            return pd.DataFrame()

    @staticmethod
    def _standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Lowercase and normalise column names.

        Parameters
        ----------
        df : pd.DataFrame
            Raw DataFrame.

        Returns
        -------
        pd.DataFrame
            Same DataFrame with cleaned column names.
        """
        # Flatten MultiIndex if present (shouldn't be for processed CSVs,
        # but defensive coding).
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
        return df

    @staticmethod
    def _validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
        """Validate OHLCV integrity and drop invalid rows.

        Rules
        -----
        * ``high >= low``
        * ``open, high, low, close > 0``  (no negative prices)
        * ``volume >= 0``

        Parameters
        ----------
        df : pd.DataFrame
            Must contain ``open``, ``high``, ``low``, ``close``,
            ``volume`` columns.

        Returns
        -------
        pd.DataFrame
            Filtered DataFrame with invalid rows removed.
        """
        # Boolean mask: True → row is VALID
        valid = (
            (df["high"] >= df["low"])         # high must be ≥ low
            & (df["open"] > 0)                # no negative / zero prices
            & (df["high"] > 0)
            & (df["low"] > 0)
            & (df["close"] > 0)
            & (df["volume"] >= 0)             # volume can be zero on holidays
        )

        n_invalid = int((~valid).sum())
        if n_invalid > 0:
            logger.debug(
                "Dropping %d rows with invalid OHLCV values", n_invalid
            )
        return df.loc[valid].copy()

    def _save_processed(self, df: pd.DataFrame, ticker: str) -> None:
        """Save cleaned DataFrame to the processed directory.

        Parameters
        ----------
        df : pd.DataFrame
            Cleaned OHLCV data.
        ticker : str
            Ticker symbol.
        """
        safe = self._safe_filename(ticker)
        filepath = os.path.join(self.processed_dir, f"{safe}.csv")
        df.to_csv(filepath)
        logger.debug("Saved processed %s → %s (%d rows)", ticker, filepath, len(df))

    @staticmethod
    def _safe_filename(ticker: str) -> str:
        """Convert ticker to a filesystem-safe filename.

        Parameters
        ----------
        ticker : str
            Raw ticker symbol.

        Returns
        -------
        str
            Safe filename string.
        """
        return re.sub(r'[&\\/:*?"<>|]', "_", ticker)

    def _filename_from_ticker(self, ticker: str) -> str:
        """Derive the expected CSV filename from a ticker symbol.

        Parameters
        ----------
        ticker : str
            Ticker symbol.

        Returns
        -------
        str
            Expected filename (e.g. ``"RELIANCE.NS.csv"``).
        """
        safe = self._safe_filename(ticker)
        return f"{safe}.csv"

    @staticmethod
    def _ticker_from_filename(filename: str) -> str:
        """Extract ticker symbol from a CSV filename.

        We strip the ``.csv`` extension.  Note that tickers with ``&``
        have been stored with ``_`` so ``M_M.NS`` maps back; we keep
        the safe version as the canonical name for consistency.

        Parameters
        ----------
        filename : str
            CSV filename (e.g. ``"RELIANCE.NS.csv"``).

        Returns
        -------
        str
            Ticker symbol (e.g. ``"RELIANCE.NS"``).
        """
        return filename.replace(".csv", "")

    def _list_raw_csvs(self) -> List[str]:
        """List all ``.csv`` files in the raw data directory.

        Returns
        -------
        list[str]
            Sorted list of filenames.
        """
        if not os.path.isdir(self.raw_dir):
            return []
        return sorted(
            f for f in os.listdir(self.raw_dir) if f.lower().endswith(".csv")
        )

    def _list_processed_csvs(self) -> List[str]:
        """List all ``.csv`` files in the processed data directory.

        Returns
        -------
        list[str]
            Sorted list of filenames.
        """
        if not os.path.isdir(self.processed_dir):
            return []
        return sorted(
            f for f in os.listdir(self.processed_dir) if f.lower().endswith(".csv")
        )

    @staticmethod
    def _log_summary(
        all_stats: Dict[str, CleaningStats],
        accepted: int,
        rejected: int,
    ) -> None:
        """Log a human-readable summary of the cleaning run.

        Parameters
        ----------
        all_stats : dict[str, CleaningStats]
            Per-ticker statistics.
        accepted : int
            Number of tickers that passed.
        rejected : int
            Number of tickers that were rejected.
        """
        total_raw = sum(s.raw_rows for s in all_stats.values())
        total_final = sum(s.final_rows for s in all_stats.values() if s.accepted)
        total_gaps = sum(s.gaps_filled for s in all_stats.values())
        total_invalid = sum(s.invalid_rows_removed for s in all_stats.values())

        logger.info("=" * 60)
        logger.info("CLEANING SUMMARY")
        logger.info("=" * 60)
        logger.info("Tickers processed : %d", len(all_stats))
        logger.info("  Accepted        : %d", accepted)
        logger.info("  Rejected        : %d", rejected)
        logger.info("Total raw rows    : %d", total_raw)
        logger.info("Total final rows  : %d", total_final)
        logger.info("Gaps filled       : %d", total_gaps)
        logger.info("Invalid removed   : %d", total_invalid)
        logger.info("=" * 60)

        # List rejected tickers with reasons
        rejected_tickers = [
            s for s in all_stats.values() if not s.accepted
        ]
        if rejected_tickers:
            logger.info("Rejected tickers:")
            for s in rejected_tickers:
                logger.info("  ✗ %s — %s", s.ticker, s.rejection_reason)

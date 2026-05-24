"""
RevertIQ — Stock Universe Management
=====================================

Manages the NIFTY 50 constituent list and related index tickers.

The primary list is hardcoded for reproducibility in backtesting.
A Wikipedia scraper is provided as a secondary validation tool.

Usage
-----
>>> from revertiq.data.universe import get_nifty50_tickers
>>> tickers = get_nifty50_tickers()            # ['RELIANCE.NS', ...]
>>> symbols = get_nifty50_tickers(suffix=False) # ['RELIANCE', ...]
"""

from __future__ import annotations

from typing import Dict, List

from revertiq.utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Hardcoded NIFTY 50 Constituents (as of May 2026)
# ─────────────────────────────────────────────────────────────────────
# NOTE: This list is updated semi-annually by NSE.  For backtesting,
# using a fixed list introduces mild *survivorship bias* — acceptable
# for research but should be noted in reports.
# ─────────────────────────────────────────────────────────────────────

_NIFTY50_SYMBOLS: List[str] = [
    "ADANIENT",
    "ADANIPORTS",
    "APOLLOHOSP",
    "ASIANPAINT",
    "AXISBANK",
    "BAJAJ-AUTO",
    "BAJFINANCE",
    "BAJAJFINSV",
    "BEL",
    "BHARTIARTL",
    "CIPLA",
    "COALINDIA",
    "DRREDDY",
    "EICHERMOT",
    "GRASIM",
    "HCLTECH",
    "HDFCBANK",
    "HDFCLIFE",
    "HINDALCO",
    "HINDUNILVR",
    "ICICIBANK",
    "INFY",
    "INDIGO",
    "ITC",
    "JSWSTEEL",
    "JIOFIN",
    "KOTAKBANK",
    "LT",
    "M&M",
    "MARUTI",
    "MAXHEALTH",
    "NTPC",
    "NESTLEIND",
    "ONGC",
    "POWERGRID",
    "RELIANCE",
    "SBILIFE",
    "SHRIRAMFIN",
    "SBIN",
    "SUNPHARMA",
    "TCS",
    "TATACONSUM",
    "TATAMOTORS",
    "TATASTEEL",
    "TECHM",
    "TITAN",
    "TRENT",
    "ULTRACEMCO",
    "WIPRO",
    "HEROMOTOCO",
]


def get_nifty50_tickers(suffix: bool = True) -> List[str]:
    """
    Return NIFTY 50 constituent tickers.

    Parameters
    ----------
    suffix : bool
        If *True* (default), append ``.NS`` for yfinance compatibility.

    Returns
    -------
    list[str]
        Sorted list of ticker symbols.
    """
    symbols = sorted(_NIFTY50_SYMBOLS)
    if suffix:
        return [f"{s}.NS" for s in symbols]
    return symbols


def get_index_tickers() -> Dict[str, str]:
    """
    Return yfinance tickers for key Indian market indices.

    Returns
    -------
    dict
        ``{'nifty50': '^NSEI', 'india_vix': '^INDIAVIX', ...}``
    """
    return {
        "nifty50": "^NSEI",
        "nifty_bank": "^NSEBANK",
        "india_vix": "^INDIAVIX",
    }


# ─────────────────────────────────────────────────────────────────────
# Wikipedia Scraper (validation / update tool)
# ─────────────────────────────────────────────────────────────────────

def fetch_nifty50_from_wikipedia() -> List[str]:
    """
    Scrape the current NIFTY 50 constituents from Wikipedia.

    Returns
    -------
    list[str]
        Ticker symbols (without ``.NS`` suffix).

    Raises
    ------
    RuntimeError
        If scraping fails.
    """
    try:
        import pandas as pd

        url = "https://en.wikipedia.org/wiki/NIFTY_50"
        tables = pd.read_html(url)

        # The constituents table contains a 'Symbol' column
        for tbl in tables:
            cols_lower = [c.lower() for c in tbl.columns]
            if "symbol" in cols_lower:
                idx = cols_lower.index("symbol")
                col_name = tbl.columns[idx]
                symbols = tbl[col_name].dropna().tolist()
                symbols = [str(s).strip() for s in symbols if str(s).strip()]
                if len(symbols) >= 40:  # sanity check
                    logger.info(
                        "Fetched %d NIFTY 50 symbols from Wikipedia", len(symbols)
                    )
                    return sorted(symbols)

        raise RuntimeError("Could not locate NIFTY 50 table on Wikipedia page")

    except Exception as exc:
        logger.warning("Wikipedia scrape failed: %s — using hardcoded list", exc)
        raise RuntimeError(f"Wikipedia scrape failed: {exc}") from exc


def validate_universe() -> Dict[str, List[str]]:
    """
    Compare hardcoded list against Wikipedia and report differences.

    Returns
    -------
    dict
        ``{'added': [...], 'removed': [...], 'common': [...]}``
    """
    hardcoded = set(get_nifty50_tickers(suffix=False))
    try:
        wiki = set(fetch_nifty50_from_wikipedia())
    except RuntimeError:
        return {"added": [], "removed": [], "common": list(hardcoded)}

    added = sorted(wiki - hardcoded)
    removed = sorted(hardcoded - wiki)
    common = sorted(hardcoded & wiki)

    if added:
        logger.info("New in NIFTY 50 (not in hardcoded): %s", added)
    if removed:
        logger.info("Removed from NIFTY 50 (still in hardcoded): %s", removed)

    return {"added": added, "removed": removed, "common": common}

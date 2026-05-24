"""Data module — downloading, cleaning, and universe management."""
from revertiq.data.universe import get_nifty50_tickers, get_index_tickers
from revertiq.data.downloader import DataDownloader
from revertiq.data.cleaner import DataCleaner

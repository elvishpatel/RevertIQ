"""
RevertIQ — Report Generator
=============================

Generates HTML backtest reports, CSV exports, and formatted
console output.

Usage
-----
>>> from revertiq.visualization.reports import ReportGenerator
>>> rg = ReportGenerator()
>>> rg.generate_backtest_report(result, output_dir="data/results")
>>> rg.print_metrics_summary(result.metrics)
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import pandas as pd

from revertiq.utils.helpers import ensure_dir
from revertiq.utils.logger import get_logger

logger = get_logger(__name__)


class ReportGenerator:
    """Generate reports, exports, and formatted summaries."""

    # ─────────────────────────────────────────────────────────────────
    # HTML Backtest Report
    # ─────────────────────────────────────────────────────────────────

    def generate_backtest_report(
        self,
        backtest_result,
        output_dir: str = "data/results",
        filename: str = "backtest_report.html",
    ) -> str:
        """
        Generate a standalone HTML report with charts and metrics.

        Parameters
        ----------
        backtest_result : BacktestResult
            Output from ``BacktestEngine.run()``.
        output_dir : str
            Directory to save the report.
        filename : str
            Output filename.

        Returns
        -------
        str
            Path to the generated HTML file.
        """
        from revertiq.backtesting.metrics import PerformanceMetrics
        from revertiq.visualization.charts import ChartEngine

        ensure_dir(output_dir)
        filepath = os.path.join(output_dir, filename)

        ce = ChartEngine()
        metrics = backtest_result.metrics

        # ── Generate charts ──
        charts_html = []

        # Equity curve
        eq_fig = ce.equity_curve(
            backtest_result.equity_curve,
            backtest_result.benchmark_equity,
        )
        charts_html.append(eq_fig.to_html(full_html=False, include_plotlyjs=False))

        # Drawdown
        if not backtest_result.daily_returns.empty:
            dd = PerformanceMetrics.drawdown_series(backtest_result.equity_curve)
            if dd is not None and not dd.empty:
                dd_fig = ce.drawdown_chart(dd)
                charts_html.append(dd_fig.to_html(full_html=False, include_plotlyjs=False))

            # Rolling Sharpe
            rs = PerformanceMetrics.rolling_sharpe(backtest_result.daily_returns)
            if rs is not None and not rs.empty:
                rs_fig = ce.rolling_sharpe_chart(rs)
                charts_html.append(rs_fig.to_html(full_html=False, include_plotlyjs=False))

            # Return distribution
            dist_fig = ce.return_distribution(backtest_result.daily_returns)
            charts_html.append(dist_fig.to_html(full_html=False, include_plotlyjs=False))

            # Monthly heatmap
            monthly = PerformanceMetrics.monthly_returns(backtest_result.daily_returns)
            if monthly is not None and not monthly.empty:
                hm_fig = ce.monthly_returns_heatmap(monthly)
                charts_html.append(hm_fig.to_html(full_html=False, include_plotlyjs=False))

        # ── Metrics table HTML ──
        metrics_html = self._metrics_to_html(metrics)

        # ── Trade summary HTML ──
        trade_html = ""
        if not backtest_result.trade_log.empty:
            trade_html = (
                "<h2>Trade Log (last 50)</h2>"
                + backtest_result.trade_log.tail(50).to_html(
                    index=False, classes="trade-table"
                )
            )

        # ── Assemble HTML ──
        html = _HTML_TEMPLATE.format(
            title="RevertIQ — Backtest Report",
            metrics_html=metrics_html,
            charts_html="\n".join(charts_html),
            trade_html=trade_html,
        )

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html)

        logger.info("Backtest report saved → %s", filepath)
        return filepath

    # ─────────────────────────────────────────────────────────────────
    # CSV Exports
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def export_trade_log(
        trade_log: pd.DataFrame,
        filepath: str = "data/results/trade_log.csv",
    ) -> str:
        """Export trade log to CSV."""
        ensure_dir(os.path.dirname(filepath))
        trade_log.to_csv(filepath, index=False)
        logger.info("Trade log exported → %s (%d trades)", filepath, len(trade_log))
        return filepath

    @staticmethod
    def export_daily_watchlist(
        rankings_df: pd.DataFrame,
        date: Optional[pd.Timestamp] = None,
        filepath: str = "data/results/watchlist.csv",
    ) -> str:
        """Export top candidates for a given date (or latest)."""
        ensure_dir(os.path.dirname(filepath))

        df = rankings_df.copy()
        if date is not None:
            df = df[df["date"] == date]
        elif "date" in df.columns and not df.empty:
            df = df[df["date"] == df["date"].max()]

        df = df.head(10)
        df.to_csv(filepath, index=False)
        logger.info("Watchlist exported → %s (%d candidates)", filepath, len(df))
        return filepath

    @staticmethod
    def export_equity_curve(
        equity_df: pd.DataFrame,
        filepath: str = "data/results/equity_curve.csv",
    ) -> str:
        """Export daily equity curve to CSV."""
        ensure_dir(os.path.dirname(filepath))
        equity_df.to_csv(filepath, index=False)
        logger.info("Equity curve exported → %s", filepath)
        return filepath

    # ─────────────────────────────────────────────────────────────────
    # Console Summary
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def print_metrics_summary(metrics: Dict) -> None:
        """Print a formatted metrics summary to console."""
        print()
        print("╔" + "═" * 50 + "╗")
        print("║  RevertIQ — Backtest Performance Summary       ║")
        print("╠" + "═" * 50 + "╣")

        _FORMATS = {
            "cagr":               ("  CAGR",               "{:>10.2%}"),
            "sharpe_ratio":       ("  Sharpe Ratio",       "{:>10.2f}"),
            "sortino_ratio":      ("  Sortino Ratio",      "{:>10.2f}"),
            "calmar_ratio":       ("  Calmar Ratio",       "{:>10.2f}"),
            "max_drawdown_pct":   ("  Max Drawdown",       "{:>10.2%}"),
            "win_rate":           ("  Win Rate",           "{:>10.2%}"),
            "profit_factor":      ("  Profit Factor",      "{:>10.2f}"),
            "total_trades":       ("  Total Trades",       "{:>10d}"),
            "avg_holding_days":   ("  Avg Holding (days)", "{:>10.1f}"),
            "avg_pnl_pct":        ("  Avg PnL per Trade",  "{:>10.2%}"),
            "exposure":           ("  Exposure",           "{:>10.2%}"),
            "total_return":       ("  Total Return",       "{:>10.2%}"),
        }

        for key, (label, fmt) in _FORMATS.items():
            val = metrics.get(key)
            if val is not None:
                try:
                    formatted = fmt.format(val)
                except (ValueError, TypeError):
                    formatted = f"{val:>10}"
                print(f"║{label:.<38s}{formatted} ║")

        print("╚" + "═" * 50 + "╝")
        print()

    # ─────────────────────────────────────────────────────────────────
    # Internal
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _metrics_to_html(metrics: Dict) -> str:
        """Convert metrics dict to styled HTML table."""
        rows = ""
        for key, val in metrics.items():
            if isinstance(val, float):
                if "pct" in key or key in ("cagr", "win_rate", "exposure", "total_return"):
                    display = f"{val:.2%}"
                else:
                    display = f"{val:.4f}"
            elif isinstance(val, int):
                display = str(val)
            else:
                display = str(val)

            label = key.replace("_", " ").title()
            rows += f"<tr><td>{label}</td><td>{display}</td></tr>\n"

        return f"""
        <h2>Performance Metrics</h2>
        <table class="metrics-table">
            <thead><tr><th>Metric</th><th>Value</th></tr></thead>
            <tbody>{rows}</tbody>
        </table>
        """


# ─────────────────────────────────────────────────────────────────────
# HTML Template
# ─────────────────────────────────────────────────────────────────────

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    background: #0d1117;
    color: #c9d1d9;
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
    padding: 2rem;
  }}
  h1 {{
    color: #58a6ff;
    font-size: 2rem;
    margin-bottom: 1rem;
    border-bottom: 1px solid #21262d;
    padding-bottom: 0.5rem;
  }}
  h2 {{
    color: #58a6ff;
    font-size: 1.3rem;
    margin: 2rem 0 1rem 0;
  }}
  .metrics-table, .trade-table {{
    border-collapse: collapse;
    width: 100%;
    max-width: 600px;
    margin: 1rem 0;
  }}
  .metrics-table th, .trade-table th {{
    background: #161b22;
    color: #58a6ff;
    padding: 0.6rem 1rem;
    text-align: left;
    border-bottom: 2px solid #21262d;
  }}
  .metrics-table td, .trade-table td {{
    padding: 0.5rem 1rem;
    border-bottom: 1px solid #21262d;
  }}
  .metrics-table tr:hover, .trade-table tr:hover {{
    background: #161b22;
  }}
  .chart-container {{
    margin: 2rem 0;
    border: 1px solid #21262d;
    border-radius: 8px;
    overflow: hidden;
  }}
</style>
</head>
<body>
<h1>⚡ {title}</h1>
{metrics_html}
<div class="chart-container">{charts_html}</div>
{trade_html}
<footer style="margin-top:3rem;color:#8b949e;font-size:0.85rem;">
  Generated by RevertIQ — Cross-Sectional Mean Reversion Research Platform
</footer>
</body>
</html>
"""

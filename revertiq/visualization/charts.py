"""
RevertIQ — Chart Engine
========================

Professional interactive Plotly charts for quantitative analysis.

All charts use a dark theme with consistent styling, are interactive
(hover, zoom, pan) and work natively in Google Colab.

Usage
-----
>>> from revertiq.visualization.charts import ChartEngine
>>> ce = ChartEngine()
>>> fig = ce.equity_curve(equity_df, benchmark_df)
>>> fig.show()
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from revertiq.utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Theme
# ─────────────────────────────────────────────────────────────────────

_COLORS = {
    "bg":          "#0d1117",
    "paper":       "#161b22",
    "grid":        "#21262d",
    "text":        "#c9d1d9",
    "muted":       "#8b949e",
    "accent":      "#58a6ff",
    "green":       "#3fb950",
    "red":         "#f85149",
    "orange":      "#d29922",
    "purple":      "#bc8cff",
    "teal":        "#39d353",
    "gradient_hi": "#26a641",
    "gradient_lo": "#da3633",
}


def _theme(fig: go.Figure, title: str = "") -> go.Figure:
    """Apply consistent dark theme to any Plotly figure."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=_COLORS["paper"],
        plot_bgcolor=_COLORS["bg"],
        font=dict(family="Inter, system-ui, sans-serif", color=_COLORS["text"]),
        title=dict(
            text=title,
            font=dict(size=20, color=_COLORS["accent"]),
            x=0.5,
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor=_COLORS["grid"],
            borderwidth=1,
            font=dict(size=11),
        ),
        margin=dict(l=60, r=30, t=60, b=40),
        xaxis=dict(gridcolor=_COLORS["grid"], showgrid=True),
        yaxis=dict(gridcolor=_COLORS["grid"], showgrid=True),
        hovermode="x unified",
    )
    return fig


class ChartEngine:
    """Create professional interactive charts for quant analysis."""

    # ─────────────────────────────────────────────────────────────────
    # 1. Equity Curve
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def equity_curve(
        equity_df: pd.DataFrame,
        benchmark_df: Optional[pd.DataFrame] = None,
        title: str = "Portfolio Equity Curve",
    ) -> go.Figure:
        """
        Portfolio equity over time with optional benchmark overlay.

        Parameters
        ----------
        equity_df : DataFrame
            Must have ``date`` and ``total_equity`` columns.
        benchmark_df : DataFrame, optional
            Must have ``date`` and ``benchmark_equity`` columns.
        """
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=equity_df["date"],
            y=equity_df["total_equity"],
            name="Portfolio",
            line=dict(color=_COLORS["accent"], width=2.5),
            fill="tozeroy",
            fillcolor="rgba(88,166,255,0.08)",
        ))

        if benchmark_df is not None:
            fig.add_trace(go.Scatter(
                x=benchmark_df["date"],
                y=benchmark_df["benchmark_equity"],
                name="NIFTY 50 (B&H)",
                line=dict(color=_COLORS["muted"], width=1.5, dash="dot"),
            ))

        fig.update_yaxes(title_text="₹ Equity")
        return _theme(fig, title)

    # ─────────────────────────────────────────────────────────────────
    # 2. Drawdown Chart
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def drawdown_chart(
        drawdown_series: pd.Series,
        title: str = "Portfolio Drawdown",
    ) -> go.Figure:
        """Filled area chart showing drawdown percentage."""
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=drawdown_series.index,
            y=drawdown_series.values * 100,
            name="Drawdown %",
            fill="tozeroy",
            fillcolor="rgba(248,81,73,0.25)",
            line=dict(color=_COLORS["red"], width=1.5),
        ))

        # Annotate max drawdown
        max_dd_idx = drawdown_series.idxmin()
        max_dd_val = drawdown_series.min() * 100

        fig.add_annotation(
            x=max_dd_idx,
            y=max_dd_val,
            text=f"Max DD: {max_dd_val:.1f}%",
            showarrow=True,
            arrowhead=2,
            arrowcolor=_COLORS["red"],
            font=dict(color=_COLORS["red"], size=12),
        )

        fig.update_yaxes(title_text="Drawdown %")
        return _theme(fig, title)

    # ─────────────────────────────────────────────────────────────────
    # 3. Signal Chart (Candlestick + Signals)
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def signal_chart(
        ticker: str,
        price_df: pd.DataFrame,
        buy_signals: Optional[pd.Series] = None,
        sell_signals: Optional[pd.Series] = None,
        title: Optional[str] = None,
    ) -> go.Figure:
        """
        Candlestick chart with Bollinger Bands and trade markers.

        Parameters
        ----------
        price_df : DataFrame
            Must have date index/column and OHLCV + bb_upper, bb_lower.
        buy_signals, sell_signals : Series[bool]
            Boolean series aligned with price_df index.
        """
        df = price_df.copy()
        if "date" in df.columns:
            df = df.set_index("date")

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            row_heights=[0.75, 0.25],
            vertical_spacing=0.03,
        )

        # Candlestick
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df["open"], high=df["high"],
            low=df["low"], close=df["close"],
            name=ticker,
            increasing_line_color=_COLORS["green"],
            decreasing_line_color=_COLORS["red"],
        ), row=1, col=1)

        # Bollinger Bands
        if "bb_upper" in df.columns and "bb_lower" in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df["bb_upper"],
                name="BB Upper", line=dict(color=_COLORS["muted"], width=1, dash="dash"),
                showlegend=False,
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=df.index, y=df["bb_lower"],
                name="BB Lower", line=dict(color=_COLORS["muted"], width=1, dash="dash"),
                fill="tonexty", fillcolor="rgba(139,148,158,0.08)",
                showlegend=False,
            ), row=1, col=1)

        # Buy signals
        if buy_signals is not None and buy_signals.any():
            buy_dates = buy_signals[buy_signals].index
            buy_prices = df.loc[buy_dates, "low"] * 0.98
            fig.add_trace(go.Scatter(
                x=buy_dates, y=buy_prices,
                mode="markers", name="BUY",
                marker=dict(
                    symbol="triangle-up", size=12,
                    color=_COLORS["green"], line=dict(width=1, color="white"),
                ),
            ), row=1, col=1)

        # Sell signals
        if sell_signals is not None and sell_signals.any():
            sell_dates = sell_signals[sell_signals].index
            sell_prices = df.loc[sell_dates, "high"] * 1.02
            fig.add_trace(go.Scatter(
                x=sell_dates, y=sell_prices,
                mode="markers", name="SELL",
                marker=dict(
                    symbol="triangle-down", size=12,
                    color=_COLORS["red"], line=dict(width=1, color="white"),
                ),
            ), row=1, col=1)

        # Volume bars
        if "volume" in df.columns:
            colors = [
                _COLORS["green"] if c >= o else _COLORS["red"]
                for o, c in zip(df["open"], df["close"])
            ]
            fig.add_trace(go.Bar(
                x=df.index, y=df["volume"],
                marker_color=colors, opacity=0.5,
                name="Volume", showlegend=False,
            ), row=2, col=1)

        fig.update_xaxes(rangeslider_visible=False)
        fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)

        return _theme(fig, title or f"{ticker} — Signals & Bollinger Bands")

    # ─────────────────────────────────────────────────────────────────
    # 4. Rolling Sharpe
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def rolling_sharpe_chart(
        rolling_sharpe: pd.Series,
        title: str = "Rolling Sharpe Ratio (63-day)",
    ) -> go.Figure:
        """Rolling Sharpe ratio with conditional green/red coloring."""
        fig = go.Figure()

        pos = rolling_sharpe.clip(lower=0)
        neg = rolling_sharpe.clip(upper=0)

        fig.add_trace(go.Scatter(
            x=pos.index, y=pos, name="Positive",
            fill="tozeroy", fillcolor="rgba(63,185,80,0.2)",
            line=dict(color=_COLORS["green"], width=1.5),
        ))
        fig.add_trace(go.Scatter(
            x=neg.index, y=neg, name="Negative",
            fill="tozeroy", fillcolor="rgba(248,81,73,0.2)",
            line=dict(color=_COLORS["red"], width=1.5),
        ))

        fig.add_hline(y=0, line_dash="dash", line_color=_COLORS["muted"])
        fig.update_yaxes(title_text="Sharpe Ratio")
        return _theme(fig, title)

    # ─────────────────────────────────────────────────────────────────
    # 5. Monthly Returns Heatmap
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def monthly_returns_heatmap(
        monthly_df: pd.DataFrame,
        title: str = "Monthly Returns (%)",
    ) -> go.Figure:
        """
        Year × Month heatmap of returns.

        Parameters
        ----------
        monthly_df : DataFrame
            Index = years, Columns = month names (Jan–Dec),
            Values = returns as floats (0.05 = 5 %).
        """
        values_pct = monthly_df.values * 100  # Convert to %

        fig = go.Figure(data=go.Heatmap(
            z=values_pct,
            x=monthly_df.columns.tolist(),
            y=[str(y) for y in monthly_df.index],
            colorscale=[
                [0.0, _COLORS["gradient_lo"]],
                [0.5, _COLORS["bg"]],
                [1.0, _COLORS["gradient_hi"]],
            ],
            zmid=0,
            text=np.round(values_pct, 1),
            texttemplate="%{text:.1f}%",
            textfont=dict(size=11),
            hoverongaps=False,
        ))

        fig.update_yaxes(autorange="reversed")
        return _theme(fig, title)

    # ─────────────────────────────────────────────────────────────────
    # 6. Return Distribution
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def return_distribution(
        daily_returns: pd.Series,
        title: str = "Daily Return Distribution",
    ) -> go.Figure:
        """Histogram with mean/median lines and stats annotation."""
        fig = go.Figure()

        fig.add_trace(go.Histogram(
            x=daily_returns * 100,
            nbinsx=80,
            marker_color=_COLORS["accent"],
            opacity=0.7,
            name="Returns",
        ))

        mean_val = daily_returns.mean() * 100
        median_val = daily_returns.median() * 100
        std_val = daily_returns.std() * 100
        skew_val = daily_returns.skew()
        kurt_val = daily_returns.kurtosis()

        fig.add_vline(x=mean_val, line_dash="dash",
                      line_color=_COLORS["green"],
                      annotation_text=f"Mean: {mean_val:.3f}%")
        fig.add_vline(x=median_val, line_dash="dot",
                      line_color=_COLORS["orange"],
                      annotation_text=f"Median: {median_val:.3f}%")

        fig.add_annotation(
            x=0.98, y=0.95, xref="paper", yref="paper",
            text=(
                f"σ = {std_val:.3f}%<br>"
                f"Skew = {skew_val:.2f}<br>"
                f"Kurt = {kurt_val:.2f}"
            ),
            showarrow=False,
            bgcolor="rgba(0,0,0,0.6)",
            font=dict(size=12, color=_COLORS["text"]),
            align="left",
        )

        fig.update_xaxes(title_text="Daily Return (%)")
        fig.update_yaxes(title_text="Frequency")
        return _theme(fig, title)

    # ─────────────────────────────────────────────────────────────────
    # 7. Ranked Stocks Table
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def ranked_stocks_table(
        rankings_df: pd.DataFrame,
        date: Optional[pd.Timestamp] = None,
        title: str = "Top Mean Reversion Candidates",
    ) -> go.Figure:
        """Interactive table showing top ranked candidates."""
        df = rankings_df.copy()
        if date is not None:
            df = df[df["date"] == date]
        elif "date" in df.columns:
            df = df[df["date"] == df["date"].max()]

        df = df.head(10)

        if df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No candidates", showarrow=False)
            return _theme(fig, title)

        show_cols = ["rank", "ticker", "composite_score", "signal_strength"]
        show_cols = [c for c in show_cols if c in df.columns]

        # Color signal strength
        strength = df.get("signal_strength", pd.Series([50] * len(df)))
        cell_colors = [
            _COLORS["green"] if v >= 60
            else _COLORS["orange"] if v >= 40
            else _COLORS["red"]
            for v in strength
        ]

        fig = go.Figure(data=[go.Table(
            header=dict(
                values=[c.replace("_", " ").title() for c in show_cols],
                fill_color=_COLORS["paper"],
                font=dict(color=_COLORS["accent"], size=13),
                align="left",
                line_color=_COLORS["grid"],
            ),
            cells=dict(
                values=[df[c].tolist() for c in show_cols],
                fill_color=_COLORS["bg"],
                font=dict(color=_COLORS["text"], size=12),
                align="left",
                line_color=_COLORS["grid"],
                format=[None, None, ".3f", ".1f"][:len(show_cols)],
            ),
        )])

        return _theme(fig, title)

    # ─────────────────────────────────────────────────────────────────
    # 8. Regime Chart
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def regime_chart(
        nifty_data: pd.DataFrame,
        regime_series: pd.Series,
        title: str = "Market Regime — NIFTY 50",
    ) -> go.Figure:
        """NIFTY price with 200-DMA and regime-colored background."""
        df = nifty_data.copy()
        if "date" in df.columns:
            df = df.set_index("date")

        fig = go.Figure()

        # NIFTY price
        fig.add_trace(go.Scatter(
            x=df.index, y=df["close"],
            name="NIFTY 50", line=dict(color=_COLORS["accent"], width=2),
        ))

        # 200-DMA
        sma200 = df["close"].rolling(200).mean()
        fig.add_trace(go.Scatter(
            x=df.index, y=sma200,
            name="200-DMA",
            line=dict(color=_COLORS["orange"], width=1.5, dash="dash"),
        ))

        # Regime background shading
        regime_colors = {
            "strong_bull": "rgba(63,185,80,0.12)",
            "mild_bull":   "rgba(63,185,80,0.06)",
            "cautious":    "rgba(210,153,34,0.10)",
            "bearish":     "rgba(248,81,73,0.12)",
        }

        aligned = regime_series.reindex(df.index).fillna("unknown")
        prev = None
        start = None

        for date, regime in aligned.items():
            if regime != prev:
                if prev is not None and start is not None:
                    color = regime_colors.get(prev, "rgba(0,0,0,0)")
                    fig.add_vrect(
                        x0=start, x1=date,
                        fillcolor=color, layer="below", line_width=0,
                    )
                start = date
                prev = regime

        # Last segment
        if prev is not None and start is not None:
            color = regime_colors.get(prev, "rgba(0,0,0,0)")
            fig.add_vrect(
                x0=start, x1=df.index[-1],
                fillcolor=color, layer="below", line_width=0,
            )

        fig.update_yaxes(title_text="NIFTY 50 Level")
        return _theme(fig, title)

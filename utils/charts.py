"""
utils/charts.py — Reusable Plotly Chart Components
===================================================
All charts use the FoodCast dark glassmorphism design system.
Fonts matched to Syne (titles) + DM Sans (body) from style.css.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Optional

# ── Design tokens ──────────────────────────────
PRIMARY   = "#00C897"
SECONDARY = "#FFC857"
ALERT     = "#FF5A5F"
BG        = "#0B1020"
BG2       = "#111827"
GLASS     = "rgba(255,255,255,0.04)"
MUTED     = "rgba(255,255,255,0.5)"
GRID      = "rgba(255,255,255,0.06)"

# Font matched to style.css
FONT_TITLE = "Syne, sans-serif"
FONT_BODY  = "DM Sans, sans-serif"

BASE_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family=FONT_BODY, color="#FFFFFF", size=12),
    margin=dict(l=12, r=12, t=44, b=12),
    xaxis=dict(
        gridcolor=GRID,
        zerolinecolor="rgba(255,255,255,0.08)",
        tickfont=dict(color=MUTED, size=11, family=FONT_BODY),
        linecolor="rgba(255,255,255,0.08)",
        showspikes=True,
        spikecolor="rgba(0,200,151,0.3)",
        spikethickness=1,
        spikedash="dot",
    ),
    yaxis=dict(
        gridcolor=GRID,
        zerolinecolor="rgba(255,255,255,0.08)",
        tickfont=dict(color=MUTED, size=11, family=FONT_BODY),
        linecolor="rgba(255,255,255,0.08)",
    ),
    legend=dict(
        bgcolor="rgba(255,255,255,0.04)",
        bordercolor="rgba(255,255,255,0.1)",
        borderwidth=1,
        font=dict(size=11, family=FONT_BODY),
        itemsizing="constant",
    ),
    hoverlabel=dict(
        bgcolor="#111827",
        bordercolor="rgba(0,200,151,0.45)",
        font=dict(family=FONT_BODY, size=12, color="#ffffff"),
        align="left",
    ),
    hovermode="x unified",
    transition=dict(duration=400, easing="cubic-in-out"),
)


def _apply_base(fig: go.Figure, title: str = "", height: int = 380) -> go.Figure:
    layout = dict(**BASE_LAYOUT, height=height)
    if title:
        layout["title"] = dict(
            text=title,
            font=dict(size=14, color="#FFFFFF", family=FONT_TITLE, weight=700),
            x=0.02,
            xanchor="left",
            pad=dict(b=8),
        )
    fig.update_layout(**layout)
    return fig


# ── 1. Donation Forecast Chart ─────────────────

def donation_forecast_chart(
    historical: pd.DataFrame,
    forecast: pd.DataFrame,
    title: str = "Donation Forecast"
) -> go.Figure:
    fig = go.Figure()

    # Confidence band
    if "upper" in forecast.columns and "lower" in forecast.columns:
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast["date"], forecast["date"][::-1]]),
            y=pd.concat([forecast["upper"], forecast["lower"][::-1]]),
            fill="toself",
            fillcolor="rgba(0,200,151,0.07)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Confidence Band",
            showlegend=True,
            hoverinfo="skip",
        ))

    # Historical line
    fig.add_trace(go.Scatter(
        x=historical["date"], y=historical["amount"],
        mode="lines",
        line=dict(color="rgba(255,255,255,0.35)", width=2, dash="dot"),
        name="Historical",
        hovertemplate="<b>%{x|%d %b %Y}</b><br>₹%{y:,.0f}<extra>Historical</extra>"
    ))

    # Forecast line — glowing green
    fig.add_trace(go.Scatter(
        x=forecast["date"], y=forecast["forecast"],
        mode="lines+markers",
        line=dict(color=PRIMARY, width=3),
        marker=dict(
            size=7,
            color=PRIMARY,
            line=dict(color="#0b0f14", width=2),
            symbol="circle",
        ),
        name="Forecast",
        hovertemplate="<b>%{x|%d %b %Y}</b><br>₹%{y:,.0f}<extra>Forecast</extra>"
    ))

    # Connector from historical end to forecast start
    if len(historical) > 0 and len(forecast) > 0:
        fig.add_trace(go.Scatter(
            x=[historical["date"].iloc[-1], forecast["date"].iloc[0]],
            y=[historical["amount"].iloc[-1], forecast["forecast"].iloc[0]],
            mode="lines",
            line=dict(color=PRIMARY, width=2, dash="dash"),
            showlegend=False,
            hoverinfo="skip",
        ))

    fig = _apply_base(fig, title)
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True, tickprefix="₹", tickformat=",.0f")
    return fig


# ── 2. Monthly Bar Chart ───────────────────────

def monthly_donations_bar(df: pd.DataFrame, title: str = "Monthly Donations") -> go.Figure:
    monthly = df.copy()
    monthly["month"] = monthly["date"].dt.strftime("%b %Y")
    avg = monthly["amount"].mean()
    monthly["color"] = monthly["amount"].apply(
        lambda x: PRIMARY if x >= avg else "rgba(0,200,151,0.35)"
    )

    fig = go.Figure(go.Bar(
        x=monthly["month"],
        y=monthly["amount"],
        marker=dict(
            color=monthly["color"],
            line=dict(color="rgba(0,0,0,0)"),
            cornerradius=6,
        ),
        hovertemplate="<b>%{x}</b><br>₹%{y:,.0f}<extra></extra>",
    ))

    fig.add_hline(
        y=avg,
        line=dict(color=SECONDARY, dash="dash", width=1.5),
        annotation_text="  Avg",
        annotation_font_color=SECONDARY,
        annotation_font_size=11,
    )

    fig = _apply_base(fig, title)
    fig.update_yaxes(tickprefix="₹", tickformat=",.0f")
    fig.update_layout(hovermode="x")
    return fig


# ── 3. Donor Trend Line ────────────────────────

def donor_trend_chart(df: pd.DataFrame, title: str = "Donor Trends") -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["date"], y=df["donors"],
        fill="tozeroy",
        fillcolor="rgba(255,200,87,0.06)",
        line=dict(color=SECONDARY, width=2.5),
        mode="lines",
        name="Donors",
        hovertemplate="<b>%{x|%d %b %Y}</b><br>%{y:,} donors<extra></extra>",
    ))

    # Rolling average overlay
    if len(df) >= 4:
        rolling = df["donors"].rolling(4, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=df["date"], y=rolling,
            mode="lines",
            line=dict(color="rgba(255,200,87,0.5)", width=1.5, dash="dot"),
            name="4-week avg",
            hoverinfo="skip",
        ))

    fig = _apply_base(fig, title)
    fig.update_yaxes(tickformat=",")
    return fig


# ── 4. Model Comparison Radar ─────────────────

def model_comparison_radar(comparison_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    colors = [PRIMARY, SECONDARY, ALERT, "rgba(130,80,255,0.9)", "rgba(80,160,255,0.9)"]

    cols = ["RMSE", "MAE", "MAPE"]
    normalised = comparison_df.copy()
    for c in cols:
        max_v = comparison_df[c].max()
        min_v = comparison_df[c].min()
        rng = max_v - min_v if max_v != min_v else 1
        normalised[c] = 1 - (comparison_df[c] - min_v) / rng

    categories = ["RMSE ↓", "MAE ↓", "MAPE ↓"]

    for i, row in normalised.iterrows():
        vals = [row["RMSE"], row["MAE"], row["MAPE"]]
        vals += vals[:1]
        color = colors[i % len(colors)]

        fig.add_trace(go.Scatterpolar(
            r=vals,
            theta=categories + [categories[0]],
            fill="toself",
            fillcolor=color.replace("0.9", "0.10"),
            line=dict(color=color, width=2),
            name=row["Model"],
            hovertemplate="<b>%{theta}</b><br>Score: %{r:.2f}<extra>" + row["Model"] + "</extra>",
        ))

    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                gridcolor=GRID,
                tickfont=dict(color=MUTED, family=FONT_BODY),
                tickvals=[0.25, 0.5, 0.75, 1.0],
            ),
            angularaxis=dict(
                gridcolor=GRID,
                tickfont=dict(color="#FFFFFF", size=12, family=FONT_TITLE),
            )
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_BODY, color="#FFFFFF"),
        height=360,
        legend=dict(
            bgcolor=GLASS,
            bordercolor="rgba(255,255,255,0.1)",
            borderwidth=1,
            font=dict(family=FONT_BODY),
        ),
        margin=dict(l=30, r=30, t=20, b=20),
        transition=dict(duration=400, easing="cubic-in-out"),
    )
    return fig


# ── 5. RMSE Bar Comparison ─────────────────────

def model_rmse_bar(comparison_df: pd.DataFrame) -> go.Figure:
    colors = [PRIMARY if row["Best"] else "rgba(0,200,151,0.28)"
              for _, row in comparison_df.iterrows()]

    fig = go.Figure(go.Bar(
        x=comparison_df["Model"],
        y=comparison_df["RMSE"],
        marker=dict(
            color=colors,
            cornerradius=8,
            line=dict(color="rgba(0,0,0,0)"),
        ),
        text=comparison_df["RMSE"].apply(lambda x: f"₹{x:,.0f}"),
        textposition="outside",
        textfont=dict(color="#FFFFFF", size=11, family=FONT_BODY),
        hovertemplate="<b>%{x}</b><br>RMSE: ₹%{y:,.0f}<extra></extra>",
    ))

    fig = _apply_base(fig, "RMSE Comparison  (lower = better)", 300)
    fig.update_yaxes(tickprefix="₹", tickformat=",.0f")
    fig.update_layout(hovermode="x")
    return fig


# ── 6. Seasonal Heatmap ────────────────────────

def seasonal_heatmap(df: pd.DataFrame) -> go.Figure:
    df2 = df.copy()
    df2["year"]  = df2["date"].dt.year
    df2["month"] = df2["date"].dt.month

    pivot = df2.pivot_table(values="amount", index="year", columns="month", aggfunc="sum")
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]
    cols = [month_labels[c - 1] for c in pivot.columns]

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=cols,
        y=[str(y) for y in pivot.index],
        colorscale=[
            [0.0,  "rgba(0,200,151,0.04)"],
            [0.35, "rgba(0,200,151,0.25)"],
            [0.7,  "rgba(0,200,151,0.6)"],
            [1.0,  PRIMARY],
        ],
        hoverongaps=False,
        hovertemplate="<b>%{x} %{y}</b><br>₹%{z:,.0f}<extra></extra>",
        colorbar=dict(
            tickfont=dict(color=MUTED, family=FONT_BODY),
            tickprefix="₹",
            outlinewidth=0,
            thickness=12,
        ),
        xgap=2,
        ygap=2,
    ))
    fig = _apply_base(fig, "Donation Heatmap — Month × Year", 280)
    return fig


# ── 7. Donut Chart ─────────────────────────────

def category_donut(df: pd.DataFrame, col: str = "category") -> go.Figure:
    if col not in df.columns:
        return go.Figure()

    breakdown = df.groupby(col)["amount"].sum().reset_index()
    colors = [PRIMARY, SECONDARY, ALERT,
              "rgba(130,80,255,0.9)", "rgba(80,200,255,0.9)"]

    fig = go.Figure(go.Pie(
        labels=breakdown[col],
        values=breakdown["amount"],
        hole=0.62,
        marker=dict(
            colors=colors[:len(breakdown)],
            line=dict(color="#0b0f14", width=3),
        ),
        textfont=dict(family=FONT_BODY, size=12),
        hovertemplate="<b>%{label}</b><br>₹%{value:,.0f}<br>%{percent}<extra></extra>",
        pull=[0.04] + [0] * (len(breakdown) - 1),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_BODY, color="#FFFFFF"),
        showlegend=True,
        legend=dict(
            bgcolor=GLASS,
            bordercolor="rgba(255,255,255,0.1)",
            borderwidth=1,
            font=dict(family=FONT_BODY),
        ),
        margin=dict(l=10, r=10, t=10, b=10),
        height=300,
        transition=dict(duration=400, easing="cubic-in-out"),
    )
    return fig


# ── 8. Alert / Drought Timeline ───────────────

def drought_timeline(monthly_df: pd.DataFrame, z_scores: list, mean_baseline: float) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.09,
        row_heights=[0.65, 0.35],
    )

    colors = [ALERT if z < -1.5 else (SECONDARY if z < -0.5 else PRIMARY)
              for z in z_scores]

    fig.add_trace(go.Bar(
        x=monthly_df["date"],
        y=monthly_df["amount"],
        marker=dict(color=colors, cornerradius=6),
        name="Monthly Donations",
        hovertemplate="<b>%{x|%b %Y}</b><br>₹%{y:,.0f}<extra></extra>",
    ), row=1, col=1)

    fig.add_hline(
        y=mean_baseline,
        line=dict(color=SECONDARY, dash="dash", width=1.5),
        annotation_text="  Baseline",
        annotation_font_color=SECONDARY,
        row=1, col=1,
    )

    fig.add_trace(go.Scatter(
        x=monthly_df["date"],
        y=z_scores,
        mode="lines+markers",
        line=dict(color=ALERT, width=2),
        marker=dict(
            color=[ALERT if z < -1.5 else MUTED for z in z_scores],
            size=7,
            line=dict(color="#0b0f14", width=1.5),
        ),
        name="Z-Score",
        hovertemplate="<b>%{x|%b %Y}</b><br>Z: %{y:.2f}<extra></extra>",
    ), row=2, col=1)

    fig.add_hline(
        y=-1.5,
        line=dict(color=ALERT, dash="dot", width=1),
        annotation_text="  Alert threshold",
        annotation_font_color=ALERT,
        row=2, col=1,
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_BODY, color="#FFFFFF"),
        height=420,
        margin=dict(l=12, r=12, t=20, b=12),
        legend=dict(
            bgcolor=GLASS,
            bordercolor="rgba(255,255,255,0.1)",
            borderwidth=1,
            font=dict(family=FONT_BODY),
        ),
        xaxis2=dict(gridcolor=GRID, tickfont=dict(color=MUTED, family=FONT_BODY)),
        yaxis=dict(
            gridcolor=GRID,
            tickprefix="₹",
            tickformat=",.0f",
            tickfont=dict(color=MUTED, family=FONT_BODY),
        ),
        yaxis2=dict(
            gridcolor=GRID,
            tickfont=dict(color=MUTED, family=FONT_BODY),
            title=dict(text="Z-Score", font=dict(color=MUTED, family=FONT_BODY)),
        ),
        transition=dict(duration=400, easing="cubic-in-out"),
    )
    return fig


# ── 9. Campaign Success Gauge ─────────────────

def success_gauge(probability: float) -> go.Figure:
    color = PRIMARY if probability >= 70 else (SECONDARY if probability >= 45 else ALERT)

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability,
        number=dict(
            suffix="%",
            font=dict(size=40, color=color, family=FONT_TITLE),
        ),
        delta=dict(
            reference=60,
            increasing=dict(color=PRIMARY),
            decreasing=dict(color=ALERT),
            font=dict(family=FONT_BODY),
        ),
        gauge=dict(
            axis=dict(
                range=[0, 100],
                tickcolor=MUTED,
                tickfont=dict(color=MUTED, family=FONT_BODY),
                dtick=25,
            ),
            bar=dict(color=color, thickness=0.72),
            bgcolor="rgba(255,255,255,0.03)",
            borderwidth=0,
            steps=[
                dict(range=[0,  45],  color="rgba(255,90,95,0.10)"),
                dict(range=[45, 70],  color="rgba(255,200,87,0.10)"),
                dict(range=[70, 100], color="rgba(0,200,151,0.10)"),
            ],
            threshold=dict(
                line=dict(color=color, width=3),
                thickness=0.82,
                value=probability,
            ),
        ),
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_BODY, color="#FFFFFF"),
        height=270,
        margin=dict(l=24, r=24, t=24, b=24),
        transition=dict(duration=500, easing="cubic-in-out"),
    )
    return fig

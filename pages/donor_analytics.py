"""
pages/donor_analytics.py — Donor Analytics
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils.helpers import format_currency, kpi_card_html, page_header
from utils.charts import donor_trend_chart, category_donut, BASE_LAYOUT, PRIMARY, SECONDARY, ALERT, MUTED, GRID
from predict import preprocess_donations, predict_donor_churn, forecast_arima, aggregate_monthly


def render():
    page_header("👥 Donor Analytics", "Churn prediction, retention forecasting, and donor growth intelligence")

    if "df" not in st.session_state:
        st.warning("⚠️ No data loaded. Go to **Upload Dashboard** first.")
        return

    df      = preprocess_donations(st.session_state["df"])
    churn   = predict_donor_churn(df)
    monthly = churn.get("monthly_donors", aggregate_monthly(df))

    # ── KPI row ───────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    for col, label, val, icon in [
        (c1, "Churn Rate",       f"{churn['churn_rate']}%",     "🔴"),
        (c2, "Retention Rate",   f"{churn['retention_rate']}%", "💚"),
        (c3, "Donors at Risk",   f"{churn['at_risk_donors']:,}", "⚠️"),
        (c4, "Donor Trend",      churn.get("trend","—"),         "📊"),
    ]:
        with col:
            st.markdown(kpi_card_html(label, val, "", icon), unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # ── Churn gauge ───────────────────────────────
    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.markdown("#### Churn Risk Meter")
        _churn_gauge(churn["churn_rate"])
        risk_color = ALERT if churn["churn_rate"] > 25 else (SECONDARY if churn["churn_rate"] > 15 else PRIMARY)
        st.markdown(f"""
        <div style="text-align:center">
          <span class="alert-badge {'danger' if churn['churn_rate']>25 else 'warning' if churn['churn_rate']>15 else 'success'}">
            {'🔴 High Risk' if churn['churn_rate']>25 else '🟡 Moderate Risk' if churn['churn_rate']>15 else '🟢 Low Risk'}
          </span>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("#### Donor Growth Trend")
        st.plotly_chart(donor_trend_chart(df), use_container_width=True)

    st.markdown("---")

    # ── Donor retention forecast ───────────────────
    st.markdown("#### 📅 Retention Forecast (Next 12 Weeks)")
    if "donors" in df.columns:
        fc_df, _ = forecast_arima(df[["date", "amount"]].rename(columns={"amount": "amount"}), horizon=12, freq="W")
        # Use donors column scaled
        donor_ratio = df["donors"].mean() / (df["amount"].mean() + 1e-9)
        fc_donors   = fc_df.copy()
        fc_donors["forecast"] = fc_df["forecast"] * donor_ratio
        fc_donors["lower"]    = fc_df["lower"] * donor_ratio
        fc_donors["upper"]    = fc_df["upper"] * donor_ratio

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pd.concat([fc_donors["date"], fc_donors["date"][::-1]]),
            y=pd.concat([fc_donors["upper"], fc_donors["lower"][::-1]]),
            fill="toself", fillcolor="rgba(255,200,87,0.08)",
            line=dict(color="rgba(0,0,0,0)"), name="CI"
        ))
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["donors"],
            mode="lines", line=dict(color=MUTED, width=1.5, dash="dot"), name="Historical"
        ))
        fig.add_trace(go.Scatter(
            x=fc_donors["date"], y=fc_donors["forecast"],
            mode="lines+markers", line=dict(color=SECONDARY, width=2.5),
            marker=dict(size=5, color=SECONDARY), name="Forecast"
        ))
        fig.update_layout(**BASE_LAYOUT, height=300,
                          yaxis=dict(gridcolor=GRID, tickfont=dict(color=MUTED)),
                          xaxis=dict(gridcolor=GRID, tickfont=dict(color=MUTED)))
        st.plotly_chart(fig, use_container_width=True)

    # ── Breakdown charts ──────────────────────────
    col_x, col_y = st.columns(2)
    with col_x:
        st.markdown("#### By Category")
        if "category" in df.columns:
            st.plotly_chart(category_donut(df, "category"), use_container_width=True)
        else:
            st.info("No `category` column found in dataset")

    with col_y:
        st.markdown("#### By Region")
        if "region" in df.columns:
            st.plotly_chart(category_donut(df, "region"), use_container_width=True)
        else:
            st.info("No `region` column found in dataset")

    # ── Retention recommendations ─────────────────
    st.markdown("---")
    st.markdown("#### 💡 Retention Recommendations")
    recs = [
        ("Send personalised thank-you emails within 48h of each donation", "Boosts repeat donation rate by ~20%"),
        ("Launch re-engagement campaign targeting donors inactive for 60+ days", f"~{churn['at_risk_donors']:,} donors qualify"),
        ("Introduce recurring donation option with incentive", "Recurring donors have 5× higher LTV"),
        ("Monthly impact reports to existing donors", "Increases donor retention by 15–25%"),
    ]
    for rec, impact in recs:
        st.markdown(f"""
        <div style="display:flex;align-items:flex-start;gap:12px;padding:12px 16px;
                    background:rgba(255,255,255,0.03);border-radius:10px;margin-bottom:8px;
                    border:1px solid rgba(255,255,255,0.07)">
          <span style="color:#00C897;font-size:1.1rem">→</span>
          <div>
            <div style="font-weight:600;font-size:0.9rem">{rec}</div>
            <div style="font-size:0.78rem;color:rgba(255,255,255,0.5);margin-top:2px">{impact}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)


def _churn_gauge(churn_rate: float):
    color = ALERT if churn_rate > 25 else (SECONDARY if churn_rate > 15 else PRIMARY)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=churn_rate,
        number=dict(suffix="%", font=dict(size=32, color=color, family="Space Grotesk")),
        gauge=dict(
            axis=dict(range=[0, 50], tickcolor=MUTED, tickfont=dict(color=MUTED)),
            bar=dict(color=color, thickness=0.7),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            steps=[
                dict(range=[0, 15],  color="rgba(0,200,151,0.1)"),
                dict(range=[15, 25], color="rgba(255,200,87,0.1)"),
                dict(range=[25, 50], color="rgba(255,90,95,0.1)")
            ]
        )
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Space Grotesk", color="#FFFFFF"),
        height=220, margin=dict(l=10, r=10, t=10, b=10)
    )
    st.plotly_chart(fig, use_container_width=True)

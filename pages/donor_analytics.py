"""
pages/donor_analytics.py — Donor Analytics
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils.helpers import format_currency, kpi_card_html, page_header, summary_metrics, cur
from utils.charts import donor_trend_chart, category_donut, BASE_LAYOUT, PRIMARY, SECONDARY, ALERT, MUTED, GRID
from predict import preprocess_donations, predict_donor_churn, forecast_arima, aggregate_monthly

FONT_TITLE = "Syne, sans-serif"
FONT_BODY  = "DM Sans, sans-serif"


def render():
    page_header("👥 Donor Analytics", "Churn prediction, retention forecasting & donor growth intelligence")

    if "df" not in st.session_state:
        st.markdown("""
        <div style="text-align:center;padding:64px 24px;background:rgba(255,255,255,0.02);
                    border:1px dashed rgba(255,255,255,0.1);border-radius:16px;margin-top:24px">
          <div style="font-size:3rem;margin-bottom:16px">👥</div>
          <div style="font-size:1.1rem;font-weight:700;color:#fff;margin-bottom:8px">No Data Loaded</div>
          <div style="color:rgba(255,255,255,0.45);font-size:0.9rem">
            Go to <strong style="color:#00C897">Upload Dashboard</strong> or load demo data from the sidebar.
          </div>
        </div>
        """, unsafe_allow_html=True)
        return

    df    = preprocess_donations(st.session_state["df"])
    churn = predict_donor_churn(df)

    # FIX #8 — show a clear notice when donor data is inferred, not real
    metrics = summary_metrics(st.session_state["df"])
    if metrics.get("donors_estimated"):
        st.info(
            "ℹ️ No **donors** column found in your data — donor counts are estimated "
            "from transaction volume. Churn and retention figures are approximate. "
            "Add a `donors` column to your CSV for accurate analytics."
        )

    churn_rate     = churn["churn_rate"]
    retention_rate = churn["retention_rate"]
    at_risk        = churn["at_risk_donors"]
    trend_label    = churn.get("trend", "—")

    churn_color     = ALERT   if churn_rate > 25 else (SECONDARY if churn_rate > 15 else PRIMARY)
    retention_color = PRIMARY if retention_rate > 75 else (SECONDARY if retention_rate > 60 else ALERT)

    kpi_data = [
        ("🔴", "Churn Rate",      f"{churn_rate}%",      churn_color,     0),
        ("💚", "Retention Rate",  f"{retention_rate}%",  retention_color, 0.06),
        ("⚠️", "Donors at Risk", f"{at_risk:,}",          SECONDARY,       0.12),
        ("📊", "Donor Trend",     trend_label,             PRIMARY,         0.18),
    ]

    cols = st.columns(4)
    for i, (icon, label, val, color, delay) in enumerate(kpi_data):
        with cols[i]:
            st.markdown(f"""
            <div class="kpi-card" style="animation-delay:{delay}s">
              <div style="font-size:1.5rem;margin-bottom:6px">{icon}</div>
              <div class="kpi-value" style="color:{color}">{val}</div>
              <div class="kpi-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    col_a, col_b = st.columns([1, 2])

    with col_a:
        st.markdown("""
        <div style="font-size:0.72rem;color:rgba(255,255,255,0.35);
                    text-transform:uppercase;letter-spacing:0.1em;margin-bottom:10px">
          Churn Risk Meter
        </div>
        """, unsafe_allow_html=True)
        _churn_gauge(churn_rate)

        risk_label  = "🔴 High Risk"      if churn_rate > 25 else \
                      "🟡 Moderate Risk"  if churn_rate > 15 else \
                      "🟢 Low Risk"
        risk_bg     = "rgba(255,90,95,0.1)"  if churn_rate > 25 else \
                      "rgba(255,200,87,0.1)" if churn_rate > 15 else \
                      "rgba(0,200,151,0.1)"
        risk_border = "rgba(255,90,95,0.3)"  if churn_rate > 25 else \
                      "rgba(255,200,87,0.3)" if churn_rate > 15 else \
                      "rgba(0,200,151,0.3)"

        st.markdown(f"""
        <div style="text-align:center;padding:8px 16px;border-radius:99px;
                    background:{risk_bg};border:1px solid {risk_border};
                    font-size:0.85rem;font-weight:600;color:#fff;margin-top:4px">
          {risk_label}
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div style="font-size:0.72rem;color:rgba(255,255,255,0.35);
                    text-transform:uppercase;letter-spacing:0.1em;margin-bottom:10px">
          Donor Growth Trend
        </div>
        """, unsafe_allow_html=True)
        st.plotly_chart(donor_trend_chart(df), use_container_width=True)

    st.markdown('<hr style="border-color:rgba(255,255,255,0.07);margin:24px 0">', unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size:0.72rem;color:rgba(255,255,255,0.35);
                text-transform:uppercase;letter-spacing:0.1em;margin-bottom:12px">
      📅 Retention Forecast — Next 12 Weeks
    </div>
    """, unsafe_allow_html=True)

    if "donors" in df.columns:
        fc_df, _ = forecast_arima(df[["date", "amount"]], horizon=12, freq="W")
        donor_ratio           = df["donors"].mean() / (df["amount"].mean() + 1e-9)
        fc_donors             = fc_df.copy()
        fc_donors["forecast"] = fc_df["forecast"] * donor_ratio
        fc_donors["lower"]    = fc_df["lower"]    * donor_ratio
        fc_donors["upper"]    = fc_df["upper"]    * donor_ratio

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pd.concat([fc_donors["date"], fc_donors["date"][::-1]]),
            y=pd.concat([fc_donors["upper"], fc_donors["lower"][::-1]]),
            fill="toself", fillcolor="rgba(255,200,87,0.07)",
            line=dict(color="rgba(0,0,0,0)"), name="CI", hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["donors"],
            mode="lines", line=dict(color="rgba(255,255,255,0.3)", width=1.5, dash="dot"),
            name="Historical",
            hovertemplate="<b>%{x|%d %b %Y}</b><br>%{y:,} donors<extra>Historical</extra>",
        ))
        fig.add_trace(go.Scatter(
            x=fc_donors["date"], y=fc_donors["forecast"],
            mode="lines+markers",
            line=dict(color=SECONDARY, width=2.5),
            marker=dict(size=6, color=SECONDARY, line=dict(color="#0b0f14", width=1.5)),
            name="Forecast",
            hovertemplate="<b>%{x|%d %b %Y}</b><br>%{y:,.0f} donors<extra>Forecast</extra>",
        ))
        layout = {k: v for k, v in BASE_LAYOUT.items() if k not in ('font', 'transition')}
        fig.update_layout(**layout, height=300,
        
        
                          font=dict(family=FONT_BODY, color="#fff"),
                          transition=dict(duration=400, easing="cubic-in-out"))
        fig.update_yaxes(gridcolor=GRID, tickfont=dict(color=MUTED, family=FONT_BODY))
        fig.update_xaxes(gridcolor=GRID, tickfont=dict(color=MUTED, family=FONT_BODY))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.markdown("""
        <div style="padding:20px;background:rgba(255,200,87,0.06);border:1px solid rgba(255,200,87,0.2);
                    border-radius:10px;color:rgba(255,255,255,0.55);font-size:0.88rem">
          ⚠️ Add a <code>donors</code> column to your CSV to enable retention forecasting.
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr style="border-color:rgba(255,255,255,0.07);margin:24px 0">', unsafe_allow_html=True)

    col_x, col_y = st.columns(2)
    with col_x:
        st.markdown("""
        <div style="font-size:0.72rem;color:rgba(255,255,255,0.35);
                    text-transform:uppercase;letter-spacing:0.1em;margin-bottom:10px">
          By Category
        </div>
        """, unsafe_allow_html=True)
        if "category" in df.columns:
            st.plotly_chart(category_donut(df, "category"), use_container_width=True)
        else:
            st.markdown("""
            <div style="padding:32px;text-align:center;background:rgba(255,255,255,0.02);
                        border:1px dashed rgba(255,255,255,0.1);border-radius:12px;
                        color:rgba(255,255,255,0.35);font-size:0.85rem">
              No <code>category</code> column in dataset
            </div>
            """, unsafe_allow_html=True)

    with col_y:
        st.markdown("""
        <div style="font-size:0.72rem;color:rgba(255,255,255,0.35);
                    text-transform:uppercase;letter-spacing:0.1em;margin-bottom:10px">
          By Region
        </div>
        """, unsafe_allow_html=True)
        if "region" in df.columns:
            st.plotly_chart(category_donut(df, "region"), use_container_width=True)
        else:
            st.markdown("""
            <div style="padding:32px;text-align:center;background:rgba(255,255,255,0.02);
                        border:1px dashed rgba(255,255,255,0.1);border-radius:12px;
                        color:rgba(255,255,255,0.35);font-size:0.85rem">
              No <code>region</code> column in dataset
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<hr style="border-color:rgba(255,255,255,0.07);margin:24px 0">', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.72rem;color:rgba(255,255,255,0.35);
                text-transform:uppercase;letter-spacing:0.1em;margin-bottom:16px">
      💡 Retention Recommendations
    </div>
    """, unsafe_allow_html=True)

    recs = [
        ("✉️", "Personalised Thank-You Emails",
         "Send within 48h of each donation.",
         "Boosts repeat donation rate by ~20%", PRIMARY),
        ("🎯", "Re-engagement Campaign",
         f"Target donors inactive for 60+ days — ~{at_risk:,} donors qualify.",
         "Recovers 15–30% of at-risk donors", SECONDARY),
        ("🔁", "Recurring Donation Option",
         "Introduce a monthly giving option with a small incentive.",
         "Recurring donors have 5× higher lifetime value", PRIMARY),
        ("📊", "Monthly Impact Reports",
         "Send data-driven impact stories to existing donors.",
         "Increases retention by 15–25%", SECONDARY),
    ]

    for i, (icon, title, body, impact, color) in enumerate(recs):
        st.markdown(f"""
        <div style="display:flex;gap:16px;align-items:flex-start;padding:14px 18px;
                    background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);
                    border-radius:12px;margin-bottom:8px;
                    animation:fadeInUp 0.4s {i*0.07}s ease both">
          <div style="font-size:1.4rem;min-width:30px">{icon}</div>
          <div style="flex:1">
            <div style="font-weight:700;font-size:0.9rem;color:#fff;margin-bottom:3px">{title}</div>
            <div style="font-size:0.82rem;color:rgba(255,255,255,0.55);line-height:1.6">{body}</div>
          </div>
          <div style="font-size:0.75rem;font-weight:600;color:{color};
                      background:rgba(0,200,151,0.08);border:1px solid rgba(0,200,151,0.2);
                      border-radius:99px;padding:4px 10px;white-space:nowrap;align-self:center">
            {impact}
          </div>
        </div>
        """, unsafe_allow_html=True)


def _churn_gauge(churn_rate: float):
    color = ALERT if churn_rate > 25 else (SECONDARY if churn_rate > 15 else PRIMARY)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=churn_rate,
        number=dict(suffix="%", font=dict(size=34, color=color, family=FONT_TITLE)),
        gauge=dict(
            axis=dict(range=[0, 50], tickcolor=MUTED,
                      tickfont=dict(color=MUTED, family=FONT_BODY), dtick=10),
            bar=dict(color=color, thickness=0.72),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            steps=[
                dict(range=[0,  15], color="rgba(0,200,151,0.08)"),
                dict(range=[15, 25], color="rgba(255,200,87,0.08)"),
                dict(range=[25, 50], color="rgba(255,90,95,0.08)"),
            ],
            threshold=dict(line=dict(color=color, width=3), thickness=0.8, value=churn_rate),
        ),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_BODY, color="#FFFFFF"),
        height=220,
        margin=dict(l=10, r=10, t=10, b=10),
        transition=dict(duration=400, easing="cubic-in-out"),
    )
    st.plotly_chart(fig, use_container_width=True)

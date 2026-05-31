"""
pages/seasonal.py — Seasonal Campaign Insights
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils.helpers import format_currency, kpi_card_html, page_header, cur
from utils.charts import seasonal_heatmap, BASE_LAYOUT, PRIMARY, SECONDARY, ALERT, MUTED, GRID
from predict import preprocess_donations, analyze_seasonal_spikes


def render():
    page_header("🎉 Seasonal Campaign Insights", "Festival spike detection, campaign ROI windows, and peak donation periods")

    if "df" not in st.session_state:
        st.warning("⚠️ No data loaded. Go to **Upload Dashboard** first.")
        return

    df     = preprocess_donations(st.session_state["df"])
    spikes = analyze_seasonal_spikes(df)

    c1, c2, c3, c4 = st.columns(4)
    spike_list = spikes["spikes"]
    for col, label, val, icon in [
        (c1, "Peak Month",      spikes.get("peak_month", "N/A"),         "🏆"),
        (c2, "Peak Uplift",     f"+{spikes.get('peak_uplift', 0):.1f}%",  "📈"),
        (c3, "Spike Periods",   str(len(spike_list)),                      "⚡"),
        (c4, "Best ROI Window", spike_list[0]["festival"] if spike_list else "—", "💡"),
    ]:
        with col:
            st.markdown(kpi_card_html(label, val, "", icon), unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    st.markdown("#### 📊 Monthly Average Donations vs Baseline")
    _monthly_spike_chart(df, spikes)

    st.markdown("---")

    st.markdown("#### 🎊 Detected Donation Spikes")
    if not spike_list:
        st.info("No significant spikes detected. Load more data covering multiple years for better analysis.")
    else:
        cols = st.columns(min(len(spike_list), 3))
        for i, spike in enumerate(spike_list[:6]):
            with cols[i % 3]:
                uplift_color = PRIMARY if spike["uplift_pct"] > 30 else SECONDARY
                sym = "$"
                st.markdown(f"""
                <div class="glass-card" style="margin-bottom:12px">
                  <div style="font-size:1.4rem;margin-bottom:6px">🎉</div>
                  <div style="font-weight:700;font-size:1rem">{spike['name']}</div>
                  <div style="color:rgba(255,255,255,0.5);font-size:0.78rem;margin:2px 0 10px">{spike['festival']}</div>
                  <div style="font-size:1.5rem;font-weight:700;color:{uplift_color}">
                    +{spike['uplift_pct']}%
                  </div>
                  <div style="font-size:0.75rem;color:rgba(255,255,255,0.45)">above annual average</div>
                  <div style="margin-top:10px;padding-top:10px;border-top:1px solid rgba(255,255,255,0.08);
                              font-size:0.85rem;font-weight:600">
                    {sym}{spike['avg']:,.0f} avg
                  </div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("#### 🗓️ Donation Heatmap — Month × Year")
    st.plotly_chart(seasonal_heatmap(df), use_container_width=True)

    st.markdown("---")

    st.markdown("#### 💰 Campaign ROI Forecast by Season")
    _roi_forecast_chart(df, spikes)

    st.markdown("---")

    st.markdown("#### 🗓️ Campaign Calendar Recommendations")
    calendar_tips = [
        ("Q4 (Oct–Dec)", "Diwali + Christmas + Year-End",              "🔥 Highest ROI", "danger"),
        ("Q1 (Jan–Mar)", "New Year + Holi + Budget Season",            "✅ Strong",       "success"),
        ("Q2 (Apr–Jun)", "Baisakhi + Eid + Summer Giving",             "🟡 Moderate",     "warning"),
        ("Q3 (Jul–Sep)", "Monsoon Aid + Independence Day + Navratri Prep", "📈 Building", "success"),
    ]
    for quarter, events, rating, badge in calendar_tips:
        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;align-items:center;
                    padding:14px 18px;background:rgba(255,255,255,0.03);
                    border:1px solid rgba(255,255,255,0.07);border-radius:12px;margin-bottom:8px">
          <div>
            <div style="font-weight:700;font-size:0.95rem">{quarter}</div>
            <div style="font-size:0.78rem;color:rgba(255,255,255,0.5);margin-top:3px">{events}</div>
          </div>
          <span class="alert-badge {badge}">{rating}</span>
        </div>
        """, unsafe_allow_html=True)


def _monthly_spike_chart(df: pd.DataFrame, spikes: dict):
    avg_by_month = spikes["avg_by_month"]
    overall_mean = df["amount"].mean()
    spike_months = {s["month"] for s in spikes["spikes"]}

    months      = list(range(1, 13))
    month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    values      = [float(avg_by_month.get(m, overall_mean)) for m in months]
    colors      = [PRIMARY if m in spike_months else "rgba(0,200,151,0.35)" for m in months]

    sym = "$"
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=month_names, y=values,
        marker=dict(color=colors, cornerradius=8, line=dict(color="rgba(0,0,0,0)")),
        hovertemplate=f"<b>%{{x}}</b><br>{sym}%{{y:,.0f}}<extra></extra>"
    ))
    fig.add_hline(
        y=overall_mean,
        line=dict(color=SECONDARY, dash="dash", width=1.5),
        annotation_text="Annual Average",
        annotation_font_color=SECONDARY
    )
    fig.update_layout(**BASE_LAYOUT, height=300)
    fig.update_yaxes(gridcolor=GRID, tickprefix=sym, tickformat=",.0f", tickfont=dict(color=MUTED))
    fig.update_xaxes(gridcolor=GRID, tickfont=dict(color=MUTED))
    st.plotly_chart(fig, use_container_width=True)


def _roi_forecast_chart(df: pd.DataFrame, spikes: dict):
    avg_by_month = spikes["avg_by_month"]
    overall_mean = df["amount"].mean() or 1
    months       = list(range(1, 13))
    month_names  = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    roi_pct = [(float(avg_by_month.get(m, overall_mean)) - overall_mean) / overall_mean * 100 for m in months]
    colors  = [PRIMARY if r > 15 else (SECONDARY if r > 0 else ALERT) for r in roi_pct]

    fig = go.Figure(go.Bar(
        x=month_names, y=roi_pct,
        marker=dict(color=colors, cornerradius=6, line=dict(color="rgba(0,0,0,0)")),
        text=[f"{r:+.0f}%" for r in roi_pct],
        textposition="outside",
        textfont=dict(color="#FFFFFF", size=10),
        hovertemplate="<b>%{x}</b><br>ROI: %{y:+.1f}%<extra></extra>"
    ))
    fig.add_hline(y=0, line=dict(color=MUTED, width=1))
    fig.update_layout(**BASE_LAYOUT, height=300)
    fig.update_yaxes(gridcolor=GRID, ticksuffix="%", tickfont=dict(color=MUTED))
    fig.update_xaxes(gridcolor=GRID, tickfont=dict(color=MUTED))
    st.plotly_chart(fig, use_container_width=True)

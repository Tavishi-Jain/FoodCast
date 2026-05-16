"""
pages/campaign_predictor.py — Campaign Success Predictor
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from utils.helpers import format_currency, kpi_card_html, page_header, summary_metrics
from utils.charts import success_gauge, PRIMARY, SECONDARY, ALERT
from predict import preprocess_donations, predict_campaign_success


def render():
    page_header("🎯 Campaign Success Predictor", "Random Forest model to forecast campaign outcomes before you launch")

    # ── Campaign input form ───────────────────────
    st.markdown("#### ⚙️ Configure Your Campaign")

    if "df" in st.session_state:
        df      = preprocess_donations(st.session_state["df"])
        metrics = summary_metrics(df)
        default_budget = float(metrics["avg_weekly"] * 4)
        default_donors = float(metrics["total_donors"] / max(len(df) / 4, 1))
    else:
        default_budget = 50000.0
        default_donors = 80.0

    col1, col2 = st.columns(2)
    with col1:
        budget   = st.number_input("Campaign Budget (₹)", min_value=1000.0, max_value=10_000_000.0,
                                   value=default_budget, step=5000.0, format="%.0f")
        duration = st.slider("Campaign Duration (days)", 7, 90, 21)
        month    = st.selectbox("Launch Month", list(range(1, 13)),
                                index=datetime.now().month - 1,
                                format_func=lambda m: datetime(2024, m, 1).strftime("%B"))

    with col2:
        hist_avg     = st.number_input("Historical Monthly Average (₹)", min_value=1000.0,
                                       max_value=5_000_000.0, value=default_budget * 0.9, step=5000.0)
        past_campaigns = st.slider("Past Campaigns Run", 0, 50, 5)
        avg_donors   = st.number_input("Expected Donors", min_value=1.0, max_value=10000.0,
                                       value=default_donors, step=10.0)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    predict_btn = st.button("🔮 Predict Campaign Success", use_container_width=True)

    if predict_btn or "campaign_result" in st.session_state:
        if predict_btn:
            with st.spinner("Running Random Forest model…"):
                result = predict_campaign_success(
                    campaign_budget=budget,
                    campaign_duration_days=duration,
                    historical_avg=hist_avg,
                    month=month,
                    num_past_campaigns=past_campaigns,
                    avg_donors=avg_donors
                )
                st.session_state["campaign_result"] = result

        result = st.session_state.get("campaign_result", {})
        prob   = result.get("success_probability", 50)
        risk   = result.get("risk_level", "Medium")

        st.markdown("---")
        st.markdown("#### 📊 Prediction Results")

        col_g, col_r = st.columns([1, 2])
        with col_g:
            st.markdown("**Success Probability**")
            st.plotly_chart(success_gauge(prob), use_container_width=True)

        with col_r:
            risk_color = {"Low": PRIMARY, "Medium": SECONDARY, "High": ALERT}.get(risk, SECONDARY)
            risk_badge = {"Low": "success", "Medium": "warning", "High": "danger"}.get(risk, "warning")

            st.markdown(f"""
            <div style="margin-top:16px">
              <div style="font-size:0.8rem;color:rgba(255,255,255,0.5);margin-bottom:6px">RISK LEVEL</div>
              <span class="alert-badge {risk_badge}" style="font-size:1rem;padding:8px 20px">
                {'🔴' if risk=='High' else '🟡' if risk=='Medium' else '🟢'} {risk} Risk
              </span>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

            # Factor breakdown
            factors = [
                ("Budget vs Historical", "✅ Strong" if budget > hist_avg else "⚠️ Below average",
                 budget > hist_avg),
                ("Campaign Duration",    "✅ Sufficient" if duration >= 14 else "⚠️ Short",
                 duration >= 14),
                ("Festival Season",      "✅ Peak window" if month in [10,11,12,3] else "🔵 Off-peak",
                 month in [10,11,12,3]),
                ("Experience Level",     "✅ Experienced" if past_campaigns >= 5 else "🆕 New",
                 past_campaigns >= 5),
            ]
            for factor, status, good in factors:
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;padding:8px 0;
                            border-bottom:1px solid rgba(255,255,255,0.06);font-size:0.88rem">
                  <span style="color:rgba(255,255,255,0.7)">{factor}</span>
                  <span style="color:{'#00C897' if good else '#FFC857'};font-weight:600">{status}</span>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### 💡 AI Recommendations")
        recs = result.get("recommendations", [])
        for rec in recs:
            st.markdown(f"""
            <div style="padding:12px 16px;background:rgba(255,255,255,0.03);
                        border-left:3px solid #00C897;border-radius:0 10px 10px 0;
                        margin-bottom:8px;font-size:0.9rem">{rec}</div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### 📋 Campaign Summary")
        summary_data = {
            "Parameter":    ["Budget", "Duration", "Launch Month", "Historical Avg", "Past Campaigns", "Expected Donors"],
            "Value":        [f"₹{budget:,.0f}", f"{duration} days",
                            datetime(2024, month, 1).strftime("%B"),
                            f"₹{hist_avg:,.0f}", str(past_campaigns), f"{avg_donors:.0f}"],
            "Status":       ["✅" if budget >= hist_avg else "⚠️",
                            "✅" if duration >= 14 else "⚠️",
                            "🔥" if month in [10,11,12,3] else "🔵",
                            "—", "✅" if past_campaigns >= 5 else "🆕", "—"]
        }
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

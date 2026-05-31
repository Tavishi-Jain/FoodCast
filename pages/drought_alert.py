"""
pages/drought_alert.py — Drought Alert System
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io
from utils.helpers import format_currency, kpi_card_html, page_header, generate_report_csv, cur
from utils.charts import drought_timeline, PRIMARY, SECONDARY, ALERT, MUTED
from predict import preprocess_donations, detect_donation_droughts, aggregate_monthly


def render():
    page_header("🚨 Drought Alert System", "Detect donation drops, red-flag periods, and download risk reports")

    if "df" not in st.session_state:
        st.warning("⚠️ No data loaded. Go to **Upload Dashboard** first.")
        return

    df = preprocess_donations(st.session_state["df"])

    with st.sidebar:
        st.markdown("### ⚙️ Alert Settings")
        z_thresh = st.slider("Alert Sensitivity (Z-threshold)", 0.5, 3.0, 1.5, 0.1,
                             help="Lower = more alerts. Standard: 1.5σ below mean")
        st.caption("1.5 = Warning · 2.5 = Critical")

    drought  = detect_donation_droughts(df, z_threshold=z_thresh)
    alerts   = drought["alerts"]
    severity = drought["severity"]

    sev_config = {
        "Critical": ("🔴", ALERT,    "rgba(255,90,95,0.12)",  "rgba(255,90,95,0.4)",  "CRITICAL — Immediate action required"),
        "Warning":  ("🟡", SECONDARY,"rgba(255,200,87,0.10)", "rgba(255,200,87,0.4)", "WARNING — Monitor closely"),
        "Normal":   ("🟢", PRIMARY,  "rgba(0,200,151,0.08)",  "rgba(0,200,151,0.3)",  "NORMAL — Donation flow healthy"),
    }
    icon, color, bg, border, message = sev_config.get(severity, sev_config["Normal"])

    st.markdown(f"""
    <div style="background:{bg};border:1.5px solid {border};border-radius:14px;
                padding:20px 24px;display:flex;align-items:center;gap:16px;margin-bottom:24px">
      <div style="font-size:2.2rem">{icon}</div>
      <div>
        <div style="font-weight:700;font-size:1.05rem;color:{color}">System Status: {message}</div>
        <div style="font-size:0.82rem;color:rgba(255,255,255,0.5);margin-top:3px">
          {len(alerts)} alert period(s) detected · Baseline: {format_currency(drought['mean_baseline'])} / month
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # FIX #14 — "all clear" shown prominently before charts when no alerts
    if severity == "Normal" and not alerts:
        st.success(
            "✅ **All clear!** All donation periods are within the normal range. "
            "No drought conditions detected at the current sensitivity level."
        )

    monthly = drought.get("monthly", aggregate_monthly(df))
    worst   = min(alerts, key=lambda x: x["pct_drop"], default=None)

    sym = cur()
    c1, c2, c3, c4 = st.columns(4)
    for col, label, val, icon_str in [
        (c1, "Alert Periods",    str(len(alerts)),                                   "🚨"),
        (c2, "Worst Drop",       f"{worst['pct_drop']}%" if worst else "—",          "📉"),
        (c3, "Worst Period",     worst["period"] if worst else "—",                  "📅"),
        (c4, "Baseline Monthly", format_currency(drought["mean_baseline"]),          "📊"),
    ]:
        with col:
            st.markdown(kpi_card_html(label, val, "", icon_str), unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    st.markdown("#### 📉 Donation Drought Timeline")
    if len(monthly) >= 3:
        z_scores = drought.get("z_scores", [0] * len(monthly))
        z_scores = z_scores[:len(monthly)]
        if len(z_scores) < len(monthly):
            z_scores += [0] * (len(monthly) - len(z_scores))
        st.plotly_chart(drought_timeline(monthly, z_scores, drought["mean_baseline"]),
                        use_container_width=True)
    else:
        st.info("Need at least 3 months of data for drought timeline.")

    st.markdown("---")

    st.markdown("#### 🚨 Alert Log")
    if not alerts:
        st.markdown(f"""
        <div style="text-align:center;padding:40px;background:rgba(0,200,151,0.05);
                    border:1px solid rgba(0,200,151,0.2);border-radius:14px">
          <div style="font-size:2rem;margin-bottom:8px">✅</div>
          <div style="font-weight:600;color:#00C897">No drought periods detected</div>
          <div style="font-size:0.82rem;color:rgba(255,255,255,0.45);margin-top:4px">
            All periods are within {z_thresh}σ of the baseline
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        for alert in alerts:
            sev_color = ALERT if alert["severity"] == "Critical" else SECONDARY
            sev_badge = "danger" if alert["severity"] == "Critical" else "warning"
            # FIX #2 — pre-compute background colour before the f-string
            alert_bg = "rgba(255,90,95,0.05)" if alert["severity"] == "Critical" else "rgba(255,200,87,0.05)"
            alert_icon = "🔴" if alert["severity"] == "Critical" else "🟡"
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;align-items:center;
                        padding:14px 18px;margin-bottom:8px;
                        background:{alert_bg};
                        border:1px solid {sev_color}33;border-radius:12px">
              <div style="display:flex;align-items:center;gap:14px">
                <div style="font-size:1.5rem">{alert_icon}</div>
                <div>
                  <div style="font-weight:700;font-size:0.95rem">{alert['period']}</div>
                  <div style="font-size:0.78rem;color:rgba(255,255,255,0.5)">
                    Collected {format_currency(alert['amount'])} · {alert['pct_drop']}% below baseline
                  </div>
                </div>
              </div>
              <span class="alert-badge {sev_badge}">{alert['severity']}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("#### 📋 Drought Response Playbook")
    playbook = [
        ("Week 1–2",  "Immediate", "Send personalised outreach to lapsed donors (email + WhatsApp)"),
        ("Week 2–4",  "Short-term","Launch emergency matching campaign or matching donor challenge"),
        ("Month 2",   "Medium",   "Host a live fundraising event or virtual walkathon"),
        ("Month 3+",  "Recovery", "Audit campaign messaging and redesign donor journey"),
    ]
    for timeframe, urgency, action in playbook:
        urgency_color = ALERT if urgency == "Immediate" else (SECONDARY if urgency == "Short-term" else PRIMARY)
        st.markdown(f"""
        <div style="display:flex;gap:16px;padding:12px 16px;margin-bottom:8px;
                    background:rgba(255,255,255,0.03);border-radius:10px;
                    border:1px solid rgba(255,255,255,0.07)">
          <div style="min-width:80px;font-size:0.75rem;color:rgba(255,255,255,0.45);padding-top:2px">{timeframe}</div>
          <div style="min-width:90px">
            <span style="font-size:0.72rem;font-weight:700;color:{urgency_color};
                         background:{urgency_color}18;padding:2px 8px;border-radius:100px">{urgency}</span>
          </div>
          <div style="font-size:0.88rem">{action}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("#### ⬇️ Download Drought Report")
    report_df = monthly.copy()
    report_df["z_score"]  = (drought.get("z_scores", [0]*len(monthly)))[:len(monthly)]
    report_df["alert"]    = report_df["z_score"] < -z_thresh
    report_df["severity"] = report_df["z_score"].apply(
        lambda z: "Critical" if z < -2.5 else ("Warning" if z < -z_thresh else "Normal")
    )

    buf = io.BytesIO()
    report_df.to_csv(buf, index=False)
    st.download_button(
        "⬇️ Download Full Drought Report (CSV)",
        data=buf.getvalue(),
        file_name="foodcast_drought_report.csv",
        mime="text/csv",
        use_container_width=True
    )

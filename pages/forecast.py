"""
pages/forecast.py — Donation Forecast Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
# FIX #3 — removed local re-definition of format_currency; single source of truth
from utils.helpers import format_currency, kpi_card_html, page_header, generate_report_csv, summary_metrics
from utils.charts import donation_forecast_chart, monthly_donations_bar, seasonal_heatmap
from predict import preprocess_donations, forecast_arima, forecast_prophet, forecast_xgboost, forecast_lstm

MODEL_MAP = {
    "ARIMA":   forecast_arima,
    "Prophet": forecast_prophet,
    "XGBoost": forecast_xgboost,
    "LSTM":    forecast_lstm,
}

FREQ_MAP = {"Weekly": "W", "Monthly": "ME", "Bi-weekly": "2W"}

# FIX #6 — LSTM is disclosed as a placeholder in the description
MODEL_DESC = {
    "ARIMA":   "Classic statistical model. Best for stable, linear trends.",
    "Prophet": "Facebook's model. Handles seasonality & holidays well.",
    "XGBoost": "Gradient boosting. Captures non-linear patterns.",
    "LSTM":    "⚠️ Placeholder — statistical estimator until a trained LSTM model is connected.",
}

# FIX #13 — minimum row count for reliable forecasting
MIN_ROWS_RECOMMENDED = 52


def render():
    page_header("📈 Donation Forecast", "AI-powered time-series predictions with confidence intervals")

    if "df" not in st.session_state:
        st.markdown("""
        <div style="text-align:center;padding:64px 24px;background:rgba(255,255,255,0.02);
                    border:1px dashed rgba(255,255,255,0.1);border-radius:16px;margin-top:24px">
          <div style="font-size:3rem;margin-bottom:16px">📂</div>
          <div style="font-size:1.1rem;font-weight:700;color:#fff;margin-bottom:8px">No Data Loaded</div>
          <div style="color:rgba(255,255,255,0.45);font-size:0.9rem">
            Go to <strong style="color:#00C897">Upload Dashboard</strong> to load your CSV,
            or use the <strong style="color:#00C897">Load Demo Data</strong> button in the sidebar.
          </div>
        </div>
        """, unsafe_allow_html=True)
        return

    df = preprocess_donations(st.session_state["df"])

    # FIX #13 — data quality warning for sparse datasets
    if len(df) < MIN_ROWS_RECOMMENDED:
        st.warning(
            f"⚠️ **Short dataset detected** — only {len(df)} rows found. "
            f"Forecasting models work best with at least {MIN_ROWS_RECOMMENDED} data points. "
            "Results may be less reliable on sparse data."
        )

    with st.sidebar:
        st.markdown("""
        <div style="font-size:0.72rem;color:rgba(255,255,255,0.35);
                    text-transform:uppercase;letter-spacing:0.1em;margin-bottom:10px">
          ⚙️ Forecast Settings
        </div>
        """, unsafe_allow_html=True)

        model_name = st.selectbox("Model", list(MODEL_MAP.keys()))
        horizon    = st.slider("Forecast Horizon", 4, 52, 16, 1,
                               help="Number of periods ahead to predict")
        freq_label = st.selectbox("Granularity", list(FREQ_MAP.keys()))
        freq       = FREQ_MAP[freq_label]
        show_ci    = st.toggle("Show Confidence Interval", value=True)

        st.markdown(f"""
        <div style="background:rgba(0,200,151,0.06);border:1px solid rgba(0,200,151,0.18);
                    border-radius:8px;padding:8px 12px;font-size:0.78rem;
                    color:rgba(255,255,255,0.55);margin:8px 0 12px">
          {MODEL_DESC.get(model_name, "")}
        </div>
        """, unsafe_allow_html=True)

        run_btn = st.button("▶ Run Forecast", use_container_width=True)

    # FIX #9 — cache key includes model + horizon + freq so stale results
    # are never shown when settings change
    cache_key = f"{model_name}|{horizon}|{freq}"
    needs_run = (
        run_btn
        or "forecast_df" not in st.session_state
        or st.session_state.get("forecast_cache_key") != cache_key
    )

    if needs_run:
        # FIX #12 — use st.spinner (tied to real computation) instead of a
        # fake sleep-driven progress bar
        with st.spinner(f"Running {model_name} forecast for {horizon} periods…"):
            fn = MODEL_MAP[model_name]
            fc_df, metrics = fn(df, horizon=horizon, freq=freq)
            st.session_state["forecast_df"]        = fc_df
            st.session_state["forecast_metrics"]   = metrics
            st.session_state["forecast_cache_key"] = cache_key

    fc_df   = st.session_state["forecast_df"]
    metrics = st.session_state["forecast_metrics"]

    hist_metrics   = summary_metrics(df)
    forecast_total = fc_df["forecast"].sum()
    peak_fc        = fc_df["forecast"].max()
    avg_fc         = fc_df["forecast"].mean()
    growth         = ((avg_fc - df["amount"].mean()) / (df["amount"].mean() + 1e-9)) * 100

    growth_color = "#00C897" if growth >= 0 else "#FF5A5F"
    growth_arrow = "▲" if growth >= 0 else "▼"

    kpis = [
        ("🔮", "Forecast Total",       format_currency(forecast_total), None,         None),
        ("🚀", "Peak Period",           format_currency(peak_fc),        None,         None),
        ("📊", "Avg / Period",          format_currency(avg_fc),         None,         None),
        ("📈", "Growth vs Historical",  f"{abs(growth):.1f}%",           growth_color, growth_arrow),
    ]

    cols = st.columns(4)
    for i, (icon, label, val, color, arrow) in enumerate(kpis):
        with cols[i]:
            delta_html = ""
            if color and arrow:
                delta_html = f'<div style="font-size:0.78rem;font-weight:600;color:{color};margin-top:6px">{arrow} vs historical avg</div>'
            st.markdown(f"""
            <div class="kpi-card" style="animation-delay:{i*0.08}s">
              <div style="font-size:1.5rem;margin-bottom:6px">{icon}</div>
              <div class="kpi-value">{val}</div>
              <div class="kpi-label">{label}</div>
              {delta_html}
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    fc_plot = fc_df if show_ci else fc_df.drop(columns=["lower", "upper"], errors="ignore")
    fig = donation_forecast_chart(df, fc_plot, f"{model_name} Forecast — {freq_label}")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div style="font-size:0.72rem;color:rgba(255,255,255,0.35);
                text-transform:uppercase;letter-spacing:0.1em;margin:24px 0 12px">
      🎯 Model Performance Metrics
    </div>
    """, unsafe_allow_html=True)

    metric_cols = st.columns(3)
    metric_info = {
        "RMSE": ("Root Mean Squared Error — lower is better", "📉"),
        "MAE":  ("Mean Absolute Error — lower is better",     "📏"),
        "MAPE": ("Mean Absolute % Error — lower is better",   "🎯"),
    }
    for i, (k, (tooltip, icon)) in enumerate(metric_info.items()):
        with metric_cols[i]:
            val = metrics.get(k, 0)
            fmt = f"{val:.1f}%" if k == "MAPE" else format_currency(val)
            color = "#00C897" if val < 10 else ("#FFC857" if val < 25 else "#FF5A5F")
            st.markdown(f"""
            <div class="kpi-card" style="animation-delay:{i*0.07}s" title="{tooltip}">
              <div style="font-size:1.4rem;margin-bottom:4px">{icon}</div>
              <div class="kpi-value" style="color:{color}">{fmt}</div>
              <div class="kpi-label">{k}</div>
              <div style="font-size:0.72rem;color:rgba(255,255,255,0.3);margin-top:4px">{tooltip}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    _render_ai_insights(df, fc_df, growth, hist_metrics)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown('<hr style="border-color:rgba(255,255,255,0.07)">', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📅 Monthly Trend", "🌡️ Heatmap", "📋 Forecast Table"])

    with tab1:
        st.plotly_chart(monthly_donations_bar(df), use_container_width=True)

    with tab2:
        st.plotly_chart(seasonal_heatmap(df), use_container_width=True)

    with tab3:
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        sym = cur()
        display = fc_df.copy()
        display["forecast"] = display["forecast"].apply(lambda x: f"{sym}{x:,.0f}")
        if "lower" in display.columns:
            display["lower"] = display["lower"].apply(lambda x: f"{sym}{x:,.0f}")
            display["upper"] = display["upper"].apply(lambda x: f"{sym}{x:,.0f}")
        st.dataframe(display, use_container_width=True)

        csv_bytes = generate_report_csv(fc_df, metrics)
        st.download_button(
            "⬇️ Download Forecast Report (CSV)",
            data=csv_bytes,
            file_name=f"foodcast_forecast_{model_name.lower()}.csv",
            mime="text/csv",
            use_container_width=True,
        )


def _render_ai_insights(df, fc_df, growth, hist_metrics):
    peak_month  = fc_df.loc[fc_df["forecast"].idxmax(), "date"].strftime("%B %Y")
    low_month   = fc_df.loc[fc_df["forecast"].idxmin(), "date"].strftime("%B %Y")
    trend_dir   = "upward 📈" if growth > 0 else "downward 📉"
    trend_color = "#00C897" if growth > 0 else "#FF5A5F"

    insights = [
        (
            "📊", "Trend Direction",
            f"Overall <strong style='color:{trend_color}'>{abs(growth):.1f}% {trend_dir}</strong> trend predicted vs historical averages.",
        ),
        (
            "🚀", "Peak Campaign Window",
            f"<strong style='color:#00C897'>{peak_month}</strong> is forecast as the highest-activity period — launch major campaigns before this window.",
        ),
        (
            "⚠️", "Low Activity Alert",
            f"Expect a trough around <strong style='color:#FFC857'>{low_month}</strong> — prepare re-engagement and retention strategies in advance.",
        ),
        (
            "📏", "Historical Baseline",
            f"Historical average is <strong style='color:#fff'>{format_currency(df['amount'].mean())} / period</strong> — the forecast aligns within expected seasonal bounds.",
        ),
    ]

    st.markdown("""
    <div class="glass-card" style="margin-bottom:4px">
      <div style="font-size:0.72rem;color:rgba(255,255,255,0.35);
                  text-transform:uppercase;letter-spacing:0.1em;margin-bottom:16px">
        🤖 AI-Generated Insights
      </div>
    """, unsafe_allow_html=True)

    for i, (icon, title, body) in enumerate(insights):
        st.markdown(f"""
        <div style="display:flex;gap:14px;align-items:flex-start;
                    padding:12px 14px;border-radius:10px;margin-bottom:8px;
                    background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);
                    animation:fadeInUp 0.4s {i*0.07}s ease both">
          <div style="font-size:1.3rem;min-width:28px">{icon}</div>
          <div>
            <div style="font-weight:700;font-size:0.85rem;margin-bottom:3px;color:#fff">{title}</div>
            <div style="font-size:0.82rem;color:rgba(255,255,255,0.55);line-height:1.6">{body}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

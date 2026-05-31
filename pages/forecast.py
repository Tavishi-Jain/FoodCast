"""
pages/forecast.py — Donation Forecast Dashboard

FIXES APPLIED:
  [UX-1] Stale forecast cache invalidated when model or frequency changes.
  [UX-2] Duplicate format_currency definition removed.
  [C1]   LSTM placeholder warning shown; metrics section hidden for placeholders.
  [W3]   Fallback warning shown if the real model silently failed.
"""

import streamlit as st
import pandas as pd
import numpy as np
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


def render():
    page_header("📈 Donation Forecast", "AI-powered time-series predictions with confidence intervals")

    if "df" not in st.session_state:
        st.warning("⚠️ No data loaded. Go to **Upload Dashboard** first.")
        return

    df = preprocess_donations(st.session_state["df"])

    with st.sidebar:
        st.markdown("### ⚙️ Forecast Settings")
        model_name = st.selectbox("Model", list(MODEL_MAP.keys()))
        horizon    = st.slider("Forecast Horizon", 4, 52, 16, 1,
                               help="Number of periods ahead to predict")
        freq_label = st.selectbox("Granularity", list(FREQ_MAP.keys()))
        freq       = FREQ_MAP[freq_label]
        show_ci    = st.toggle("Show Confidence Interval", value=True)
        run_btn    = st.button("▶ Run Forecast", use_container_width=True)

    # [UX-1 FIX] Invalidate cache when controls change
    cache_key  = f"{model_name}|{horizon}|{freq}"
    cache_stale = st.session_state.get("forecast_cache_key") != cache_key

    if "forecast_df" not in st.session_state or run_btn or cache_stale:
        with st.spinner(f"Running {model_name} model…"):
            fn = MODEL_MAP[model_name]
            fc_df, metrics = fn(df, horizon=horizon, freq=freq)
            st.session_state["forecast_df"]        = fc_df
            st.session_state["forecast_metrics"]   = metrics
            st.session_state["forecast_cache_key"] = cache_key

    fc_df   = st.session_state["forecast_df"]
    metrics = st.session_state["forecast_metrics"]

    # [C1-FIX] Banner for placeholder models
    if model_name == "LSTM" or metrics.get("is_placeholder"):
        st.warning(
            "⚠️ **LSTM is a statistical placeholder** (exponential smoothing + jitter). "
            "Metrics are not shown. Connect a trained LSTM model in `predict.py → forecast_lstm()` "
            "for real deep-learning results."
        )

    # [W3-FIX] Banner if real model silently fell back
    if metrics.get("fallback_reason"):
        st.warning(
            f"⚠️ **{model_name} model failed** and fell back to exponential smoothing. "
            f"Reason: `{metrics['fallback_reason']}`"
        )

    # ── Top KPIs ─────────────────────────────────
    forecast_total = fc_df["forecast"].sum()
    peak_fc        = fc_df["forecast"].max()
    avg_fc         = fc_df["forecast"].mean()
    growth         = ((avg_fc - df["amount"].mean()) / (df["amount"].mean() + 1e-9)) * 100

    c1, c2, c3, c4 = st.columns(4)
    for col, label, val, icon in [
        (c1, "Forecast Total",       format_currency(forecast_total), "🔮"),
        (c2, "Peak Period",          format_currency(peak_fc),        "🚀"),
        (c3, "Avg Forecast",         format_currency(avg_fc),         "📊"),
        (c4, "Growth vs Historical", f"{growth:+.1f}%",              "📈"),
    ]:
        with col:
            st.markdown(kpi_card_html(label, val, "", icon), unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # ── Forecast chart ────────────────────────────
    fc_plot = fc_df if show_ci else fc_df.drop(columns=["lower", "upper"], errors="ignore")
    fig = donation_forecast_chart(df, fc_plot, f"{model_name} Donation Forecast — {freq_label}")
    st.plotly_chart(fig, use_container_width=True)

    # ── Model metrics ─────────────────────────────
    st.markdown("### 🎯 Model Performance")
    if metrics.get("is_placeholder"):
        st.info(
            "Metrics are not available for placeholder models. "
            "Use **Model Comparison** to benchmark real models side-by-side."
        )
    else:
        eval_note = metrics.get("eval", "")
        if eval_note:
            st.caption(f"📐 Evaluated on: {eval_note}")
        m1, m2, m3 = st.columns(3)
        for col, k, label in [(m1, "RMSE", "RMSE"), (m2, "MAE", "MAE"), (m3, "MAPE", "MAPE %")]:
            with col:
                val = metrics.get(k, None)
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    fmt = "—"
                else:
                    fmt = f"{val:.1f}%" if k == "MAPE" else format_currency(val)
                st.markdown(kpi_card_html(label, fmt, "", "📉"), unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    _render_ai_insights(df, fc_df, growth)

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["Monthly Trend", "Heatmap", "Forecast Table"])

    with tab1:
        st.plotly_chart(monthly_donations_bar(df), use_container_width=True)

    with tab2:
        st.plotly_chart(seasonal_heatmap(df), use_container_width=True)

    with tab3:
        display = fc_df.copy()
        display["forecast"] = display["forecast"].apply(lambda x: f"₹{x:,.0f}")
        if "lower" in display.columns:
            display["lower"] = display["lower"].apply(lambda x: f"₹{x:,.0f}")
            display["upper"] = display["upper"].apply(lambda x: f"₹{x:,.0f}")
        st.dataframe(display, use_container_width=True)

        csv_bytes = generate_report_csv(fc_df, metrics)
        st.download_button(
            "⬇️ Download Forecast Report",
            data=csv_bytes,
            file_name=f"foodcast_forecast_{model_name.lower()}.csv",
            mime="text/csv",
            use_container_width=True
        )


def _render_ai_insights(df, fc_df, growth):
    peak_month = fc_df.loc[fc_df["forecast"].idxmax(), "date"].strftime("%B %Y")
    low_month  = fc_df.loc[fc_df["forecast"].idxmin(), "date"].strftime("%B %Y")
    trend_dir  = "upward 📈" if growth > 0 else "downward 📉"

    insights = [
        f"The model predicts an overall **{abs(growth):.1f}% {trend_dir}** trend vs historical averages.",
        f"**Peak donation period** is forecast for **{peak_month}** — consider launching major campaigns before this window.",
        f"**Lowest activity** expected around **{low_month}** — prepare donor re-engagement strategies in advance.",
        f"Historical average donation is **{format_currency(df['amount'].mean())} / period** — forecast aligns within expected seasonality.",
    ]

    st.markdown("""
    <div class="glass-card">
    <div style="font-size:1rem;font-weight:700;margin-bottom:14px">🤖 AI-Generated Insights</div>
    """, unsafe_allow_html=True)
    for ins in insights:
        st.markdown(f"→ {ins}")
    st.markdown("</div>", unsafe_allow_html=True)
         
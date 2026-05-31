"""
pages/model_comparison.py — Model Comparison

FIXES APPLIED:
  [C1] LSTM row greyed out; NaN metrics shown as '—'; never ranked Best.
  [W1] Eval method shown per row (hold-out test split).
  [W3] Fallback rows flagged with a warning badge.
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
from utils.helpers import format_currency, kpi_card_html, page_header
from utils.charts import model_comparison_radar, model_rmse_bar, PRIMARY, SECONDARY, ALERT, MUTED, BASE_LAYOUT, GRID
from predict import preprocess_donations, compare_all_models
import plotly.graph_objects as go

FREQ_MAP = {"Weekly": "W", "Monthly": "ME", "Bi-weekly": "2W"}
ROW_WARN_THRESHOLD = 5_000

MODEL_INFO = {
    "ARIMA": {
        "icon": "📈", "color": PRIMARY,
        "desc": "AutoRegressive Integrated Moving Average. Order auto-selected by AIC grid-search.",
        "strength": "Interpretable, fast, great for short horizons",
        "weakness": "Struggles with complex seasonality & non-linearity"
    },
    "Prophet": {
        "icon": "🔮", "color": "#8250ff",
        "desc": "Facebook's additive model with built-in seasonality, holidays, and trend changepoints.",
        "strength": "Excellent seasonality + holiday detection",
        "weakness": "Slower; may overfit with sparse data"
    },
    "XGBoost": {
        "icon": "⚡", "color": SECONDARY,
        "desc": "Gradient-boosted tree model using lag features, rolling stats, and seasonal indicators.",
        "strength": "Handles non-linear trends, very accurate",
        "weakness": "Requires careful feature engineering"
    },
    "LSTM": {
        "icon": "🧠", "color": ALERT,
        "desc": "⚠️ Statistical placeholder — connect a trained LSTM artefact for real deep-learning results.",
        "strength": "Framework-ready for real LSTM model drop-in",
        "weakness": "Currently uses exponential-smoothing fallback — metrics not comparable"
    },
}


def _fmt(val):
    if val is None:
        return "—"
    try:
        if np.isnan(float(val)):
            return "—"
    except (TypeError, ValueError):
        return "—"
    return val


def render():
    page_header("🏆 Model Comparison", "Benchmark all ML models side-by-side — find the best fit for your data")

    if "df" not in st.session_state:
        st.warning("⚠️ No data loaded. Go to **Upload Dashboard** first.")
        return

    df = preprocess_donations(st.session_state["df"])

    if len(df) > ROW_WARN_THRESHOLD:
        st.warning(
            f"⚠️ **Large dataset detected** — {len(df):,} rows. Running all 4 models may take "
            "60–120 seconds on this hardware. Consider using a sampled subset for exploration."
        )

    st.info(
        "ℹ️ **About these metrics:** ARIMA, Prophet, and XGBoost metrics are computed on a "
        "hold-out test split (last 20% of data) — not on the training set. "
        "**LSTM** is a statistical placeholder (exponential smoothing); its metrics are excluded from ranking."
    )

    with st.sidebar:
        st.markdown("### ⚙️ Comparison Settings")
        horizon    = st.slider("Forecast Horizon", 4, 52, 12)
        freq_label = st.selectbox("Granularity", list(FREQ_MAP.keys()))
        freq       = FREQ_MAP[freq_label]
        run_btn    = st.button("▶ Run All Models", use_container_width=True)

    if "comparison_df" not in st.session_state or run_btn:
        with st.spinner("Running ARIMA · Prophet · XGBoost · LSTM — this may take a moment…"):
            comp = compare_all_models(df, horizon=horizon, freq=freq)
            st.session_state["comparison_df"] = comp

    comp      = st.session_state["comparison_df"]
    best_rows = comp[comp["Best"] == True]

    if best_rows.empty:
        st.warning("No non-placeholder model could be ranked. Check your data.")
        return

    best_row    = best_rows.iloc[0]
    model_color = MODEL_INFO.get(best_row["Model"], {}).get("color", PRIMARY)
    rmse_val    = best_row["RMSE"]
    mae_val     = best_row["MAE"]
    mape_val    = best_row["MAPE"]

    st.markdown(f"""
    <div style="background:linear-gradient(135deg, {model_color}15, rgba(0,0,0,0));
    border:1.5px solid {model_color}40;border-radius:16px;
    padding:20px 28px;display:flex;align-items:center;gap:20px;margin-bottom:24px">
    <div style="font-size:2.8rem">{MODEL_INFO.get(best_row['Model'], {}).get('icon','🏆')}</div>
    <div>
      <div style="font-size:0.75rem;color:rgba(255,255,255,0.45);letter-spacing:0.1em;
      text-transform:uppercase;margin-bottom:4px">Best Model (hold-out test set)</div>
      <div style="font-size:1.5rem;font-weight:700;color:{model_color}">{best_row['Model']}</div>
      <div style="font-size:0.82rem;color:rgba(255,255,255,0.55);margin-top:4px">
        RMSE {format_currency(rmse_val)} · MAE {format_currency(mae_val)} · MAPE {mape_val:.1f}%
      </div>
    </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 📊 Performance Metrics")
    _render_metrics_table(comp)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    comp_real = comp[~comp["Is Placeholder"]].copy()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### RMSE Comparison")
        if not comp_real.empty:
            st.plotly_chart(model_rmse_bar(comp_real), use_container_width=True)
    with col2:
        st.markdown("#### Performance Radar")
        if not comp_real.empty:
            st.plotly_chart(model_comparison_radar(comp_real), use_container_width=True)

    st.markdown("---")

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("#### MAE Comparison")
        if not comp_real.empty:
            st.plotly_chart(_metric_bar(comp_real, "MAE", ""), use_container_width=True)
    with col4:
        st.markdown("#### MAPE Comparison")
        if not comp_real.empty:
            st.plotly_chart(_metric_bar(comp_real, "MAPE", "", "%"), use_container_width=True)

    st.markdown("---")

    st.markdown("#### 📚 Model Encyclopedia")
    cols = st.columns(2)
    for i, (name, info) in enumerate(MODEL_INFO.items()):
        with cols[i % 2]:
            row          = comp[comp["Model"] == name]
            rmse         = row["RMSE"].values[0] if len(row) else float("nan")
            is_best      = name == best_row["Model"]
            is_ph        = len(row) > 0 and bool(row["Is Placeholder"].values[0])
            has_fallback = len(row) > 0 and bool(row["Fallback"].values[0]) if "Fallback" in row.columns else False
            border       = f"border:1.5px solid {info['color']}60" if is_best else "border:1px solid rgba(255,255,255,0.08)"
            opacity      = "opacity:0.6;" if is_ph else ""
            rmse_display = format_currency(rmse) if not (isinstance(rmse, float) and np.isnan(rmse)) else "—"
            ph_note      = '<div style="margin-top:4px;font-size:0.72rem;color:#FFC857">⚠️ Placeholder — not ranked</div>' if is_ph else ""
            fb_note      = '<div style="margin-top:4px;font-size:0.72rem;color:#FF5A5F">⚠️ Real model failed — showing fallback</div>' if has_fallback else ""

            st.markdown(f"""
            <div class="glass-card" style="{border};margin-bottom:12px;{opacity}">
              <div style="display:flex;justify-content:space-between;align-items:flex-start">
                <div>
                  <div style="font-size:1.3rem">{info['icon']}</div>
                  <div style="font-weight:700;font-size:1rem;color:{info['color']};margin-top:4px">{name}</div>
                  {'<span class="alert-badge success" style="font-size:0.7rem;padding:2px 8px;margin-top:4px">★ BEST</span>' if is_best else ''}
                  {ph_note}{fb_note}
                </div>
                <div style="text-align:right">
                  <div style="font-size:0.72rem;color:rgba(255,255,255,0.4)">RMSE</div>
                  <div style="font-weight:700;color:#fff">{rmse_display}</div>
                </div>
              </div>
              <div style="font-size:0.82rem;color:rgba(255,255,255,0.55);margin:10px 0">{info['desc']}</div>
              <div style="display:flex;gap:8px;flex-wrap:wrap">
                <div style="font-size:0.75rem;color:{PRIMARY};background:rgba(0,200,151,0.1);
                padding:3px 10px;border-radius:100px">✅ {info['strength']}</div>
              </div>
              <div style="margin-top:6px;font-size:0.75rem;color:rgba(255,90,95,0.8)">⚠️ {info['weakness']}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    buf = io.BytesIO()
    comp.to_csv(buf, index=False)
    st.download_button(
        "⬇️ Download Comparison Report",
        data=buf.getvalue(),
        file_name="foodcast_model_comparison.csv",
        mime="text/csv",
        use_container_width=True
    )


def _render_metrics_table(comp: pd.DataFrame):
    for _, row in comp.iterrows():
        info         = MODEL_INFO.get(row["Model"], {"icon": "🤖", "color": PRIMARY})
        is_best      = row["Best"]
        is_ph        = bool(row.get("Is Placeholder", False))
        has_fallback = bool(row.get("Fallback", False))
        color_val    = info["color"]
        opacity      = "opacity:0.55;" if is_ph else ""

        try:
            if color_val.startswith("#") and len(color_val) == 7:
                r = int(color_val[1:3], 16)
                g = int(color_val[3:5], 16)
                b = int(color_val[5:7], 16)
                bg = f"rgba({r},{g},{b},0.08)" if is_best else "rgba(255,255,255,0.03)"
            else:
                bg = "rgba(255,255,255,0.06)" if is_best else "rgba(255,255,255,0.03)"
        except Exception:
            bg = "rgba(255,255,255,0.03)"

        border = f"1.5px solid {color_val}50" if is_best else "1px solid rgba(255,255,255,0.07)"

        rmse_val     = row["RMSE"]
        mae_val      = row["MAE"]
        mape_val     = row["MAPE"]
        rmse_display = format_currency(rmse_val) if _fmt(rmse_val) != "—" else "—"
        mae_display  = format_currency(mae_val)  if _fmt(mae_val)  != "—" else "—"
        mape_display = f"{mape_val:.1f}%"        if _fmt(mape_val) != "—" else "—"

        if is_ph or _fmt(rmse_val) == "—":
            rmse_bar  = 0
            bar_color = "rgba(255,255,255,0.15)"
        else:
            valid_rmse = comp.dropna(subset=["RMSE"])["RMSE"]
            rmse_bar   = min(float(rmse_val) / (valid_rmse.max() + 1e-9) * 100, 100)
            bar_color  = color_val

        ph_tag = '<span style="font-size:0.68rem;color:#FFC857;margin-left:6px">⚠ placeholder</span>' if is_ph else ""
        fb_tag = '<span style="font-size:0.68rem;color:#FF5A5F;margin-left:6px">⚠ fallback</span>' if has_fallback else ""
        eval_note = row.get("Eval Method", "")

        st.markdown(f"""
        <div style="background:{bg};border:{border};border-radius:12px;{opacity}
        padding:14px 18px;margin-bottom:8px;display:flex;align-items:center;gap:16px">
          <div style="font-size:1.5rem;min-width:36px">{info['icon']}</div>
          <div style="flex:1">
            <div style="display:flex;justify-content:space-between;margin-bottom:6px">
              <span style="font-weight:700;color:{color_val}">{row['Model']}{ph_tag}{fb_tag}</span>
              {'<span class="alert-badge success" style="font-size:0.7rem;padding:2px 8px">★ Best</span>' if is_best else ''}
            </div>
            <div style="background:rgba(0,200,151,0.1);border-radius:100px;height:4px">
              <div style="background:{bar_color};height:4px;border-radius:100px;width:{100 - rmse_bar:.0f}%"></div>
            </div>
            <div style="font-size:0.68rem;color:rgba(255,255,255,0.3);margin-top:4px">{eval_note}</div>
          </div>
          <div style="display:flex;gap:24px;text-align:right">
            <div><div style="font-size:0.7rem;color:rgba(255,255,255,0.4)">RMSE</div>
            <div style="font-weight:700">{rmse_display}</div></div>
            <div><div style="font-size:0.7rem;color:rgba(255,255,255,0.4)">MAE</div>
            <div style="font-weight:700">{mae_display}</div></div>
            <div><div style="font-size:0.7rem;color:rgba(255,255,255,0.4)">MAPE</div>
            <div style="font-weight:700">{mape_display}</div></div>
          </div>
        </div>
        """, unsafe_allow_html=True)


def _metric_bar(comp: pd.DataFrame, metric: str, prefix: str = "", suffix: str = "") -> go.Figure:
    colors = [PRIMARY if row["Best"] else "rgba(0,200,151,0.3)" for _, row in comp.iterrows()]
    fig = go.Figure(go.Bar(
        x=comp["Model"], y=comp[metric],
        marker=dict(color=colors, cornerradius=8, line=dict(color="rgba(0,0,0,0)")),
        text=[f"{prefix}{v:,.0f}{suffix}" for v in comp[metric]],
        textposition="outside",
        textfont=dict(color="#FFFFFF", size=11),
        hovertemplate=f"<b>%{{x}}</b><br>{metric}: {prefix}%{{y:,.1f}}{suffix}<extra></extra>"
    ))
    fig.update_layout(**BASE_LAYOUT, height=280)
    fig.update_yaxes(gridcolor=GRID, tickprefix=prefix, ticksuffix=suffix, tickfont=dict(color=MUTED))
    fig.update_xaxes(gridcolor=GRID, tickfont=dict(color=MUTED))
    return fig

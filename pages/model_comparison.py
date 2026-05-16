"""
pages/model_comparison.py — Model Comparison
"""

import streamlit as st
import pandas as pd
import numpy as np
from utils.helpers import format_currency, kpi_card_html, page_header
from utils.charts import model_comparison_radar, model_rmse_bar, PRIMARY, SECONDARY, ALERT, MUTED, BASE_LAYOUT, GRID
from predict import preprocess_donations, compare_all_models
import plotly.graph_objects as go


FREQ_MAP = {"Weekly": "W", "Monthly": "ME", "Bi-weekly": "2W"}

MODEL_INFO = {
    "ARIMA": {
        "icon": "📈", "color": PRIMARY,
        "desc": "AutoRegressive Integrated Moving Average. Best for stationary time-series with clear autocorrelation.",
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
        "desc": "Long Short-Term Memory neural network for deep sequence modelling.",
        "strength": "Captures complex long-range dependencies",
        "weakness": "Needs large datasets; compute-intensive"
    },
}


def render():
    page_header("🏆 Model Comparison", "Benchmark all ML models side-by-side — find the best fit for your data")

    if "df" not in st.session_state:
        st.warning("⚠️ No data loaded. Go to **Upload Dashboard** first.")
        return

    df = preprocess_donations(st.session_state["df"])

    with st.sidebar:
        st.markdown("### ⚙️ Comparison Settings")
        horizon    = st.slider("Forecast Horizon", 4, 52, 12)
        freq_label = st.selectbox("Granularity", list(FREQ_MAP.keys()))
        freq       = FREQ_MAP[freq_label]
        run_btn    = st.button("▶ Run All Models", use_container_width=True)

    # Run comparison
    if "comparison_df" not in st.session_state or run_btn:
        with st.spinner("Running ARIMA · Prophet · XGBoost · LSTM…"):
            comp = compare_all_models(df, horizon=horizon, freq=freq)
            st.session_state["comparison_df"] = comp

    comp = st.session_state["comparison_df"]
    best_row = comp[comp["Best"] == True].iloc[0]

    # ── Winner banner ──────────────────────────────
    model_color = MODEL_INFO.get(best_row["Model"], {}).get("color", PRIMARY)
    st.markdown(f"""
    <div style="background:linear-gradient(135deg, {model_color}15, rgba(0,0,0,0));
                border:1.5px solid {model_color}40;border-radius:16px;
                padding:20px 28px;display:flex;align-items:center;gap:20px;margin-bottom:24px">
      <div style="font-size:2.8rem">{MODEL_INFO.get(best_row['Model'], {}).get('icon','🏆')}</div>
      <div>
        <div style="font-size:0.75rem;color:rgba(255,255,255,0.45);letter-spacing:0.1em;
                    text-transform:uppercase;margin-bottom:4px">Best Model Recommendation</div>
        <div style="font-size:1.5rem;font-weight:700;color:{model_color}">{best_row['Model']}</div>
        <div style="font-size:0.82rem;color:rgba(255,255,255,0.55);margin-top:4px">
          RMSE ₹{best_row['RMSE']:,.0f} · MAE ₹{best_row['MAE']:,.0f} · MAPE {best_row['MAPE']:.1f}%
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Metrics table ──────────────────────────────
    st.markdown("#### 📊 Performance Metrics")
    _render_metrics_table(comp)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # ── Charts ─────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### RMSE Comparison")
        st.plotly_chart(model_rmse_bar(comp), use_container_width=True)
    with col2:
        st.markdown("#### Performance Radar")
        st.plotly_chart(model_comparison_radar(comp), use_container_width=True)

    st.markdown("---")

    # ── MAE / MAPE bars ────────────────────────────
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("#### MAE Comparison")
        st.plotly_chart(_metric_bar(comp, "MAE", "₹"), use_container_width=True)
    with col4:
        st.markdown("#### MAPE Comparison")
        st.plotly_chart(_metric_bar(comp, "MAPE", "", "%"), use_container_width=True)

    st.markdown("---")

    # ── Model cards ────────────────────────────────
    st.markdown("#### 📚 Model Encyclopedia")
    cols = st.columns(2)
    for i, (name, info) in enumerate(MODEL_INFO.items()):
        with cols[i % 2]:
            row  = comp[comp["Model"] == name]
            rmse = row["RMSE"].values[0] if len(row) else 0
            is_best = name == best_row["Model"]
            border  = f"border:1.5px solid {info['color']}60" if is_best else "border:1px solid rgba(255,255,255,0.08)"
            st.markdown(f"""
            <div class="glass-card" style="{border};margin-bottom:12px">
              <div style="display:flex;justify-content:space-between;align-items:flex-start">
                <div>
                  <div style="font-size:1.3rem">{info['icon']}</div>
                  <div style="font-weight:700;font-size:1rem;color:{info['color']};margin-top:4px">{name}</div>
                  {'<span class="alert-badge success" style="font-size:0.7rem;padding:2px 8px;margin-top:4px">★ BEST</span>' if is_best else ''}
                </div>
                <div style="text-align:right">
                  <div style="font-size:0.72rem;color:rgba(255,255,255,0.4)">RMSE</div>
                  <div style="font-weight:700;color:#fff">₹{rmse:,.0f}</div>
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

    # ── Download ──────────────────────────────────
    import io
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
        info     = MODEL_INFO.get(row["Model"], {"icon": "🤖", "color": PRIMARY})
        is_best  = row["Best"]
        bg       = f"rgba({','.join(str(int(info['color'].lstrip('#')[i:i+2],16)) for i in (0,2,4))},0.08)" if is_best else "rgba(255,255,255,0.03)"
        border   = f"1.5px solid {info['color']}50" if is_best else "1px solid rgba(255,255,255,0.07)"

        rmse_bar = min(row["RMSE"] / (comp["RMSE"].max() + 1e-9) * 100, 100)
        st.markdown(f"""
        <div style="background:{bg};border:{border};border-radius:12px;
                    padding:14px 18px;margin-bottom:8px;display:flex;align-items:center;gap:16px">
          <div style="font-size:1.5rem;min-width:36px">{info['icon']}</div>
          <div style="flex:1">
            <div style="display:flex;justify-content:space-between;margin-bottom:6px">
              <span style="font-weight:700;color:{info['color']}">{row['Model']}</span>
              {'<span class="alert-badge success" style="font-size:0.7rem;padding:2px 8px">★ Best</span>' if is_best else ''}
            </div>
            <div style="background:rgba(0,200,151,0.1);border-radius:100px;height:4px">
              <div style="background:{info['color']};height:4px;border-radius:100px;width:{100-rmse_bar:.0f}%"></div>
            </div>
          </div>
          <div style="display:flex;gap:24px;text-align:right">
            <div><div style="font-size:0.7rem;color:rgba(255,255,255,0.4)">RMSE</div>
                 <div style="font-weight:700">₹{row['RMSE']:,.0f}</div></div>
            <div><div style="font-size:0.7rem;color:rgba(255,255,255,0.4)">MAE</div>
                 <div style="font-weight:700">₹{row['MAE']:,.0f}</div></div>
            <div><div style="font-size:0.7rem;color:rgba(255,255,255,0.4)">MAPE</div>
                 <div style="font-weight:700">{row['MAPE']:.1f}%</div></div>
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
    fig.update_layout(**BASE_LAYOUT, height=280,
                      yaxis=dict(gridcolor=GRID, tickprefix=prefix, ticksuffix=suffix, tickfont=dict(color=MUTED)),
                      xaxis=dict(gridcolor=GRID, tickfont=dict(color=MUTED)))
    return fig

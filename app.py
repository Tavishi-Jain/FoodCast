"""
app.py — FoodCast: AI Donation Forecasting Platform
====================================================
Entry point. Renders landing page + sidebar navigation.
Run:  streamlit run app.py
"""

import streamlit as st
import sys
from pathlib import Path

# ── Path setup ────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ── Page config (MUST be first Streamlit call) ────
st.set_page_config(
    page_title="FoodCast — AI Donation Forecasting",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load CSS ──────────────────────────────────────
from utils.helpers import load_css, load_demo_data
load_css()

# ── Page imports ──────────────────────────────────
from pages import upload, forecast, donor_analytics, seasonal, drought_alert, campaign_predictor, model_comparison

# ── Navigation config ──────────────────────────────
NAV_ITEMS = {
    "🏠  Home":                  "home",
    "📂  Upload Dashboard":       "upload",
    "📈  Donation Forecast":      "forecast",
    "👥  Donor Analytics":        "donor_analytics",
    "🎉  Seasonal Insights":      "seasonal",
    "🚨  Drought Alert":          "drought_alert",
    "🎯  Campaign Predictor":     "campaign_predictor",
    "🏆  Model Comparison":       "model_comparison",
}

# ── Sidebar ───────────────────────────────────────
with st.sidebar:
    # Logo / brand
    st.markdown("""
    <div style="padding: 8px 0 24px">
      <div style="display:flex;align-items:center;gap:10px">
        <div style="font-size:1.8rem">🌱</div>
        <div>
          <div style="font-size:1.1rem;font-weight:700;letter-spacing:-0.02em;color:#fff">FoodCast</div>
          <div style="font-size:0.7rem;color:rgba(255,255,255,0.4);letter-spacing:0.06em">AI DONATION FORECASTING</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="font-size:0.7rem;color:rgba(255,255,255,0.35);text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px">Navigation</div>', unsafe_allow_html=True)

    page_label = st.radio(
        "nav", list(NAV_ITEMS.keys()),
        label_visibility="collapsed",
        key="nav_radio"
    )

    st.markdown("---")

    # Data status indicator
    if "df" in st.session_state:
        df_loaded = st.session_state["df"]
        src = st.session_state.get("data_source", "Uploaded")
        st.markdown(f"""
        <div style="background:rgba(0,200,151,0.08);border:1px solid rgba(0,200,151,0.25);
                    border-radius:10px;padding:10px 14px;font-size:0.8rem">
          <div style="color:#00C897;font-weight:700;margin-bottom:2px">✅ Data Loaded</div>
          <div style="color:rgba(255,255,255,0.5)">{src}</div>
          <div style="color:rgba(255,255,255,0.4);font-size:0.72rem">{len(df_loaded):,} rows</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:rgba(255,200,87,0.07);border:1px solid rgba(255,200,87,0.25);
                    border-radius:10px;padding:10px 14px;font-size:0.8rem">
          <div style="color:#FFC857;font-weight:700;margin-bottom:2px">⚠️ No Data</div>
          <div style="color:rgba(255,255,255,0.45)">Upload CSV or load demo</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    if st.button("🗂️ Load Demo Data", use_container_width=True, key="sidebar_demo"):
        st.session_state["df"] = load_demo_data()
        st.session_state["data_source"] = "Demo Dataset"
        st.rerun()

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.72rem;color:rgba(255,255,255,0.3);line-height:1.6">
      <div>Models: ARIMA · Prophet · XGBoost · LSTM · RF</div>
      <div style="margin-top:4px">v1.0.0 · Built for NGOs & CSR Teams</div>
    </div>
    """, unsafe_allow_html=True)


# ── Landing Page ──────────────────────────────────
def _render_landing():
    # Hero
    st.markdown("""
    <div class="hero-gradient">
      <div class="hero-badge">🌱 AI-Powered Social Impact Analytics</div>
      <h1 class="hero-title">Predict. Plan.<br>Maximise Impact.</h1>
      <p class="hero-sub">
        FoodCast uses machine learning to help NGOs, CSR teams, and fundraisers
        forecast donations, detect droughts, and time campaigns for maximum ROI.
      </p>
    </div>
    """, unsafe_allow_html=True)

    # CTA buttons
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        if st.button("📂 Upload Your Data", use_container_width=True):
            st.session_state["nav_radio"] = "📂  Upload Dashboard"
            st.rerun()
    with c2:
        if st.button("🗂️ Try Demo Dataset", use_container_width=True):
            st.session_state["df"] = load_demo_data()
            st.session_state["data_source"] = "Demo Dataset"
            st.session_state["nav_radio"] = "📈  Donation Forecast"
            st.rerun()
    with c3:
        if st.button("🏆 Compare Models", use_container_width=True):
            st.session_state["nav_radio"] = "🏆  Model Comparison"
            st.rerun()

    st.markdown("<div style='height:48px'></div>", unsafe_allow_html=True)

    # Feature cards
    st.markdown('<div style="text-align:center;margin-bottom:32px"><span style="font-size:0.75rem;color:rgba(255,255,255,0.4);text-transform:uppercase;letter-spacing:0.12em">Platform Capabilities</span></div>', unsafe_allow_html=True)

    features = [
        ("📈", "Donation Forecasting",     "ARIMA, Prophet, XGBoost & LSTM models predict weekly/monthly donation volumes with confidence intervals."),
        ("🚨", "Drought Alert System",      "Automatic Z-score detection flags donation dry spells before they hurt your campaigns."),
        ("🎉", "Seasonal Intelligence",     "Festival spike detection & campaign calendar to maximise ROI during Diwali, Christmas, Holi & more."),
        ("👥", "Donor Churn Prediction",    "Identify at-risk donors before they lapse. Retention forecasting with personalised outreach triggers."),
        ("🎯", "Campaign Predictor",        "Random Forest model scores your campaign setup before launch — budget, timing, duration & more."),
        ("🏆", "Model Benchmarking",        "Compare RMSE, MAE & MAPE across all models. Always deploy the best-performing algorithm."),
    ]

    cols = st.columns(3)
    for i, (icon, title, desc) in enumerate(features):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="glass-card" style="margin-bottom:16px;min-height:160px">
              <div style="font-size:1.8rem;margin-bottom:10px">{icon}</div>
              <div style="font-weight:700;font-size:0.95rem;margin-bottom:8px">{title}</div>
              <div style="font-size:0.82rem;color:rgba(255,255,255,0.55);line-height:1.6">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:48px'></div>", unsafe_allow_html=True)

    # Stats row
    st.markdown("""
    <div style="display:flex;justify-content:center;gap:60px;padding:32px;
                background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);
                border-radius:16px;text-align:center;flex-wrap:wrap">
      <div><div style="font-size:2rem;font-weight:700;color:#00C897">5</div>
           <div style="font-size:0.78rem;color:rgba(255,255,255,0.45);margin-top:4px">ML Models</div></div>
      <div><div style="font-size:2rem;font-weight:700;color:#FFC857">7</div>
           <div style="font-size:0.78rem;color:rgba(255,255,255,0.45);margin-top:4px">Dashboard Pages</div></div>
      <div><div style="font-size:2rem;font-weight:700;color:#00C897">∞</div>
           <div style="font-size:0.78rem;color:rgba(255,255,255,0.45);margin-top:4px">Forecast Horizon</div></div>
      <div><div style="font-size:2rem;font-weight:700;color:#FFC857">100%</div>
           <div style="font-size:0.78rem;color:rgba(255,255,255,0.45);margin-top:4px">Open Source</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:48px'></div>", unsafe_allow_html=True)

    # Target users
    st.markdown('<div style="text-align:center;margin-bottom:24px"><span style="font-size:0.75rem;color:rgba(255,255,255,0.4);text-transform:uppercase;letter-spacing:0.12em">Who Is This For?</span></div>', unsafe_allow_html=True)

    users = [
        ("🏛️", "NGOs & Non-profits",        "Forecast grant cycles, donor campaigns, and annual fund drives"),
        ("🏢", "CSR Teams",                 "Report giving trends to leadership with data-backed projections"),
        ("🕌", "Religious Organisations",    "Plan Zakat, tithe, and festival collection campaigns with precision"),
        ("🚀", "Individual Fundraisers",     "Know when to push your campaign for maximum donor engagement"),
        ("💻", "Crowdfunding Platforms",     "Integrate FoodCast APIs to power smart campaign recommendations"),
    ]

    for icon, title, desc in users:
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:16px;padding:14px 18px;
                    background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);
                    border-radius:12px;margin-bottom:8px">
          <div style="font-size:1.5rem;min-width:36px">{icon}</div>
          <div>
            <div style="font-weight:700;font-size:0.9rem">{title}</div>
            <div style="font-size:0.8rem;color:rgba(255,255,255,0.5);margin-top:2px">{desc}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div style="text-align:center;padding:48px 0 24px;color:rgba(255,255,255,0.25);font-size:0.78rem">
      Built with ❤️ for social impact · FoodCast v1.0.0 · Hugging Face Spaces Ready
    </div>
    """, unsafe_allow_html=True)


# ── Router ────────────────────────────────────────
page_key = NAV_ITEMS[page_label]

if page_key == "home":
    _render_landing()
elif page_key == "upload":
    upload.render()
elif page_key == "forecast":
    forecast.render()
elif page_key == "donor_analytics":
    donor_analytics.render()
elif page_key == "seasonal":
    seasonal.render()
elif page_key == "drought_alert":
    drought_alert.render()
elif page_key == "campaign_predictor":
    campaign_predictor.render()
elif page_key == "model_comparison":
    model_comparison.render()

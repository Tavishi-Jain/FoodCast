"""
app.py — FoodCast: AI Donation Forecasting Platform
====================================================
Entry point. Cinematic landing + sidebar navigation.
Pure CSS animations — no JavaScript required.
"""

import streamlit as st
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

st.set_page_config(
    page_title="FoodCast — AI Donation Forecasting",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

from utils.helpers import load_css, load_demo_data
load_css()

# ── Kill Streamlit's default padding/gaps ─────────
st.markdown("""
<style>
/* Remove Streamlit top padding on the main block */
.main .block-container {
    padding-top: 0rem !important;
    padding-bottom: 0rem !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
    max-width: 100% !important;
}
/* Remove gap above first element */
.main .block-container > div:first-child { margin-top: 0 !important; }
/* Remove Streamlit element wrapper bottom margin */
div[data-testid="element-container"] { margin-bottom: 0 !important; }
/* Hero: tight padding, no min-height */
.hero-wrap {
    min-height: unset !important;
    padding: 40px 24px 32px !important;
}
</style>
""", unsafe_allow_html=True)

from pages import upload, forecast, donor_analytics, seasonal, drought_alert, campaign_predictor, model_comparison

NAV_ITEMS = {
    "🏠 Home":               "home",
    "📂 Upload Dashboard":   "upload",
    "📈 Donation Forecast":  "forecast",
    "👥 Donor Analytics":    "donor_analytics",
    "🎉 Seasonal Insights":  "seasonal",
    "🚨 Drought Alert":      "drought_alert",
    "🎯 Campaign Predictor": "campaign_predictor",
    "🏆 Model Comparison":   "model_comparison",
}

# ── Sidebar ───────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 8px 0 24px">
      <div style="display:flex;align-items:center;gap:10px">
        <div style="font-size:1.8rem">🌱</div>
        <div>
          <div style="font-family:'Playfair Display',Georgia,serif;font-size:1.15rem;
                      font-weight:700;letter-spacing:-0.01em;color:#fff">FoodCast</div>
          <div style="font-size:0.65rem;color:rgba(255,255,255,0.35);
                      letter-spacing:0.1em;text-transform:uppercase">AI Donation Forecasting</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="font-size:0.65rem;color:rgba(255,255,255,0.3);'
                'text-transform:uppercase;letter-spacing:0.12em;margin-bottom:8px">Navigate</div>',
                unsafe_allow_html=True)

    if "nav_target" in st.session_state:
        target = st.session_state.pop("nav_target")
        if target in NAV_ITEMS:
            st.session_state["nav_radio"] = target

    page_label = st.radio(
        "nav", list(NAV_ITEMS.keys()),
        label_visibility="collapsed",
        key="nav_radio"
    )

    st.markdown("---")

    if "df" in st.session_state:
        df_loaded = st.session_state["df"]
        src = st.session_state.get("data_source", "Uploaded")
        st.markdown(f"""
        <div style="background:rgba(0,200,151,0.07);border:1px solid rgba(0,200,151,0.22);
             border-radius:10px;padding:10px 14px;font-size:0.8rem">
          <div style="color:#00C897;font-weight:700;margin-bottom:2px">✅ Data Loaded</div>
          <div style="color:rgba(255,255,255,0.5)">{src}</div>
          <div style="color:rgba(255,255,255,0.35);font-size:0.7rem">{len(df_loaded):,} rows</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:rgba(255,200,87,0.06);border:1px solid rgba(255,200,87,0.22);
             border-radius:10px;padding:10px 14px;font-size:0.8rem">
          <div style="color:#FFC857;font-weight:700;margin-bottom:2px">⚠️ No Data</div>
          <div style="color:rgba(255,255,255,0.4)">Upload CSV or load demo</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    if st.button("🗂️ Load Demo Data", use_container_width=True, key="sidebar_demo"):
        st.session_state["df"] = load_demo_data()
        st.session_state["data_source"] = "Demo Dataset"
        st.rerun()

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.7rem;color:rgba(255,255,255,0.28);line-height:1.7">
      <div>Models: ARIMA · Prophet · XGBoost · LSTM · RF</div>
      <div style="margin-top:4px">v1.0.0 · Built for NGOs &amp; CSR Teams</div>
    </div>
    """, unsafe_allow_html=True)


# ── Landing Page ──────────────────────────────────
def _render_landing():

    # HERO
    st.markdown("""
    <div class="hero-wrap">
      <span class="hero-badge">🌱 AI-Powered Social Impact Analytics</span>
      <h1 class="hero-title">
        Predict.<br>
        <span class="accent-green">Plan.</span><br>
        <span class="accent-amber">Maximise Impact.</span>
      </h1>
      <p class="hero-sub">
        FoodCast uses machine learning to help NGOs, CSR teams, and fundraisers
        forecast donations, detect droughts, and time campaigns for maximum ROI.
      </p>
      <div class="scroll-cue">
        <span>Scroll to explore</span>
        <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
          <path d="M9 4v10M5 10l4 4 4-4" stroke="currentColor" stroke-width="1.5"
                stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # CTA buttons
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("📂 Upload Your Data", use_container_width=True):
            st.session_state["nav_target"] = "📂 Upload Dashboard"
            st.rerun()
    with c2:
        if st.button("🗂️ Try Demo Dataset", use_container_width=True):
            st.session_state["df"] = load_demo_data()
            st.session_state["data_source"] = "Demo Dataset"
            st.session_state["nav_target"] = "📈 Donation Forecast"
            st.rerun()
    with c3:
        if st.button("🏆 Compare Models", use_container_width=True):
            st.session_state["nav_target"] = "🏆 Model Comparison"
            st.rerun()

    st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)

    # STATS STRIP
    st.markdown("""
    <div class="stats-strip">
      <div>
        <div class="stat-num" style="color:#00C897">5</div>
        <div class="stat-label">ML Models</div>
      </div>
      <div>
        <div class="stat-num" style="color:#FFC857">7</div>
        <div class="stat-label">Dashboard Pages</div>
      </div>
      <div>
        <div class="stat-num" style="color:#00C897">∞</div>
        <div class="stat-label">Forecast Horizon</div>
      </div>
      <div>
        <div class="stat-num" style="color:#FFC857">100%</div>
        <div class="stat-label">Open Source</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)

    # FEATURES HEADER
    st.markdown("""
    <div class="section-header">
      <span class="section-eyebrow">Platform Capabilities</span>
      <h2 class="section-title">Everything your fundraising team needs.</h2>
    </div>
    """, unsafe_allow_html=True)

    # FEATURE CARDS — wrapped in a single div so nth-child stagger works
    features = [
        ("📈", "Donation Forecasting",   "ARIMA, Prophet, XGBoost & LSTM models predict weekly/monthly donation volumes with confidence intervals."),
        ("🚨", "Drought Alert System",   "Automatic Z-score detection flags donation dry spells before they hurt your campaigns."),
        ("🎉", "Seasonal Intelligence",  "Festival spike detection & campaign calendar to maximise ROI during Diwali, Christmas, Holi & more."),
        ("👥", "Donor Churn Prediction", "Identify at-risk donors before they lapse — retention forecasting with personalised outreach triggers."),
        ("🎯", "Campaign Predictor",     "Random Forest model scores your campaign setup before launch — budget, timing, duration & more."),
        ("🏆", "Model Benchmarking",     "Compare RMSE, MAE & MAPE across all models. Always deploy the best-performing algorithm."),
    ]

    cols = st.columns(3)
    for i, (icon, title, desc) in enumerate(features):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="feature-card">
              <span class="feature-icon">{icon}</span>
              <div class="feature-title">{title}</div>
              <div class="feature-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)

    # POWERED BY AI — gradient halo section
    st.markdown("""
    <div class="ai-section">
      <span class="section-eyebrow" style="color:rgba(255,255,255,0.35)">Under the hood</span>
      <h2 class="section-title" style="margin-bottom:16px">
        Powered by AI<span style="color:#00C897">*</span>
      </h2>
      <p style="color:rgba(255,255,255,0.55);max-width:520px;margin:0 auto 28px;
                font-size:0.92rem;line-height:1.7">
        Five production-grade time-series models — ARIMA, ETS, Prophet, XGBoost, and LSTM —
        benchmarked on real NGO donation patterns. Best model: XGBoost at
        <strong style="color:#00C897">13.81% MAPE</strong>.
      </p>
      <div style="display:flex;justify-content:center;gap:10px;flex-wrap:wrap">
        <span class="metric-pill">ARIMA</span>
        <span class="metric-pill">Prophet</span>
        <span class="metric-pill">XGBoost</span>
        <span class="metric-pill">LSTM</span>
        <span class="metric-pill">Random Forest</span>
      </div>
      <p style="color:rgba(255,255,255,0.2);font-size:0.68rem;margin:24px 0 0;
                letter-spacing:0.06em;text-transform:uppercase">
        *AI refers to machine learning models trained on donation time-series data
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)

    # WHO IS THIS FOR
    st.markdown("""
    <div class="section-header">
      <span class="section-eyebrow">Designed for</span>
      <h2 class="section-title">Who is FoodCast for?</h2>
    </div>
    """, unsafe_allow_html=True)

    # All rows in one HTML block so nth-child stagger works
    st.markdown("""
    <div>
      <div class="user-row">
        <div style="font-size:1.5rem;min-width:36px">🏛️</div>
        <div>
          <div style="font-weight:700;font-size:0.92rem">NGOs &amp; Non-profits</div>
          <div style="font-size:0.8rem;color:rgba(255,255,255,0.5);margin-top:2px">Forecast grant cycles, donor campaigns, and annual fund drives</div>
        </div>
      </div>
      <div class="user-row">
        <div style="font-size:1.5rem;min-width:36px">🏢</div>
        <div>
          <div style="font-weight:700;font-size:0.92rem">CSR Teams</div>
          <div style="font-size:0.8rem;color:rgba(255,255,255,0.5);margin-top:2px">Report giving trends to leadership with data-backed projections</div>
        </div>
      </div>
      <div class="user-row">
        <div style="font-size:1.5rem;min-width:36px">🕌</div>
        <div>
          <div style="font-weight:700;font-size:0.92rem">Religious Organisations</div>
          <div style="font-size:0.8rem;color:rgba(255,255,255,0.5);margin-top:2px">Plan Zakat, tithe, and festival collection campaigns with precision</div>
        </div>
      </div>
      <div class="user-row">
        <div style="font-size:1.5rem;min-width:36px">🚀</div>
        <div>
          <div style="font-weight:700;font-size:0.92rem">Individual Fundraisers</div>
          <div style="font-size:0.8rem;color:rgba(255,255,255,0.5);margin-top:2px">Know when to push your campaign for maximum donor engagement</div>
        </div>
      </div>
      <div class="user-row">
        <div style="font-size:1.5rem;min-width:36px">💻</div>
        <div>
          <div style="font-weight:700;font-size:0.92rem">Crowdfunding Platforms</div>
          <div style="font-size:0.8rem;color:rgba(255,255,255,0.5);margin-top:2px">Integrate FoodCast APIs to power smart campaign recommendations</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # FOOTER
    st.markdown("""
    <div class="site-footer">
      Built with ❤️ for social impact &nbsp;·&nbsp; FoodCast v1.0.0 &nbsp;·&nbsp; Hugging Face Spaces
    </div>
    """, unsafe_allow_html=True)


# ── Router ────────────────────────────────────────
page_key = NAV_ITEMS[page_label]

if   page_key == "home":               _render_landing()
elif page_key == "upload":             upload.render()
elif page_key == "forecast":           forecast.render()
elif page_key == "donor_analytics":    donor_analytics.render()
elif page_key == "seasonal":           seasonal.render()
elif page_key == "drought_alert":      drought_alert.render()
elif page_key == "campaign_predictor": campaign_predictor.render()
elif page_key == "model_comparison":   model_comparison.render()


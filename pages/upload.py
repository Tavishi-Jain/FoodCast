"""
pages/upload.py — Upload Dashboard
"""

import streamlit as st
import pandas as pd
from utils.helpers import validate_csv, load_demo_data, summary_metrics, format_currency, kpi_card_html, page_header


def render():
    page_header("📂 Upload Dashboard", "Import your donation data or explore with our sample dataset")

    # ── Upload zone ───────────────────────────────
    col1, col2 = st.columns([3, 1], gap="large")

    with col1:
        st.markdown("""
        <div style="background:rgba(0,200,151,0.04);border:2px dashed rgba(0,200,151,0.25);
                    border-radius:16px;padding:32px;text-align:center;margin-bottom:16px">
          <div style="font-size:2.5rem;margin-bottom:8px">📁</div>
          <div style="font-size:1rem;font-weight:600;color:#fff;margin-bottom:6px">Drop your CSV here</div>
          <div style="font-size:0.82rem;color:rgba(255,255,255,0.5)">
            Required columns: <code>date</code>, <code>amount</code><br>
            Optional: <code>donors</code>, <code>campaign</code>, <code>category</code>, <code>region</code>
          </div>
        </div>
        """, unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Upload CSV", type=["csv"],
            label_visibility="collapsed"
        )

    with col2:
        st.markdown("<div style='height:80px'></div>", unsafe_allow_html=True)
        st.markdown("<div style='text-align:center;color:rgba(255,255,255,0.4);font-size:0.85rem;margin-bottom:12px'>or</div>", unsafe_allow_html=True)
        if st.button("🗂️ Load Demo Data", use_container_width=True):
            st.session_state["df"] = load_demo_data()
            st.session_state["data_source"] = "Demo Dataset"
            st.success("Demo data loaded!")

    # ── Process upload ────────────────────────────
    if uploaded:
    try:
        df = pd.read_csv(uploaded)
        # Auto-detect date column (case-insensitive)
        date_col = next((c for c in df.columns if c.strip().lower() == "date"), None)
        if date_col:
            df.rename(columns={date_col: "date"}, inplace=True)
            df["date"] = pd.to_datetime(df["date"])
        is_valid, errors = validate_csv(df)
        if not is_valid:
            for e in errors:
                st.error(e)
            return
        st.session_state["df"] = df
        st.session_state["data_source"] = uploaded.name
        st.success(f"✅ **{uploaded.name}** loaded — {len(df):,} rows")
    except Exception as ex:
        st.error(f"Failed to parse file: {ex}")
        return

    # ── Show data if available ────────────────────
    if "df" not in st.session_state:
        st.markdown("""
        <div style="text-align:center;padding:60px;color:rgba(255,255,255,0.35)">
          <div style="font-size:3rem">📊</div>
          <div style="margin-top:12px">Upload a file or load demo data to begin</div>
        </div>
        """, unsafe_allow_html=True)
        return

    df = st.session_state["df"]

    st.markdown("---")

    # ── KPI summary cards ─────────────────────────
    metrics = summary_metrics(df)
    c1, c2, c3, c4 = st.columns(4)
    cards = [
        (c1, "Total Raised",     format_currency(metrics["total"]),        "💰", ""),
        (c2, "Total Donors",     f'{int(metrics["total_donors"]):,}',       "👥", ""),
        (c3, "Avg per Donor",    format_currency(metrics["avg_donation"]),  "🎯", ""),
        (c4, "Peak Week",        format_currency(metrics["peak_amount"]),   "🚀", ""),
    ]
    for col, label, val, icon, delta in cards:
        with col:
            st.markdown(kpi_card_html(label, val, delta, icon), unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # ── Validation checks ─────────────────────────
    with st.expander("✅ Validation Report", expanded=False):
        checks = {
            "Date column present":   "date" in df.columns,
            "Amount column present": "amount" in df.columns,
            "No null dates":         df["date"].isnull().sum() == 0,
            "No null amounts":       df["amount"].isnull().sum() == 0,
            "Positive amounts":      (df["amount"] > 0).all(),
            "Sufficient rows (≥ 10)":len(df) >= 10,
        }
        for check, passed in checks.items():
            icon = "✅" if passed else "❌"
            st.markdown(f"{icon} {check}")

    # ── Data preview ─────────────────────────────
    st.markdown("""
    <div class="section-title" style="margin-top:24px">Dataset Preview</div>
    <div class="section-sub">Showing first 50 rows · Source: <strong>{}</strong> · {} rows total</div>
    """.format(st.session_state.get("data_source", "unknown"), len(df)), unsafe_allow_html=True)

    # Column selector
    display_cols = st.multiselect(
        "Columns to display",
        options=df.columns.tolist(),
        default=df.columns.tolist()[:min(7, len(df.columns))]
    )
    st.dataframe(
        df[display_cols].head(50).style.format(
            {"amount": "₹{:,.0f}", "donors": "{:,.0f}"}
            if "amount" in display_cols else {}
        ),
        use_container_width=True, height=320
    )

    # ── Column stats ──────────────────────────────
    st.markdown("### 📊 Column Statistics")
    st.dataframe(df.describe().style.format("{:.1f}"), use_container_width=True)

"""
pages/upload.py — Upload Dashboard
"""

import streamlit as st
import pandas as pd
from utils.helpers import load_demo_data, summary_metrics, format_currency, kpi_card_html, page_header


def render():
    page_header("📂 Upload Dashboard", "Import your donation data or explore with our sample dataset")

    col1, col2 = st.columns([3, 1], gap="large")

    with col1:
        st.markdown("""
        <div style="background:rgba(0,200,151,0.04);border:2px dashed rgba(0,200,151,0.25);
        border-radius:16px;padding:32px;text-align:center;margin-bottom:16px">
        <div style="font-size:2.5rem;margin-bottom:8px">📁</div>
        <div style="font-size:1rem;font-weight:600;color:#fff;margin-bottom:6px">Drop your CSV here</div>
        <div style="font-size:0.82rem;color:rgba(255,255,255,0.5)">
            Any CSV works — you'll map your columns after upload.
        </div>
        </div>
        """, unsafe_allow_html=True)

        uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

    with col2:
        st.markdown("<div style='height:80px'></div>", unsafe_allow_html=True)
        st.markdown("<div style='text-align:center;color:rgba(255,255,255,0.4);font-size:0.85rem;margin-bottom:12px'>or</div>", unsafe_allow_html=True)
        if st.button("🗂️ Load Demo Data", use_container_width=True):
            st.session_state["df"] = load_demo_data()
            st.session_state["data_source"] = "Demo Dataset"
            st.session_state.pop("raw_df", None)
            st.session_state.pop("upload_name", None)
            st.rerun()

    # FIX #15 — compare against "upload_name" (the actual filename stored on
    # successful mapping), not "data_source" which can be "Demo Dataset" or
    # any other string unrelated to the filename.
    if uploaded:
        stored_upload_name = st.session_state.get("upload_name", "")
        if uploaded.name != stored_upload_name or "raw_df" not in st.session_state:
            try:
                raw_df = pd.read_csv(uploaded)
                st.session_state["raw_df"]      = raw_df
                st.session_state["upload_name"] = uploaded.name
                # Clear any previously loaded df so column mapping is shown
                st.session_state.pop("df", None)
            except Exception as ex:
                st.error(f"Failed to read file: {ex}")
                return

    if "raw_df" in st.session_state and "df" not in st.session_state:
        raw_df = st.session_state["raw_df"]
        cols   = raw_df.columns.tolist()

        st.markdown("---")
        st.markdown("### 🗂️ Map Your Columns")
        st.markdown("Tell us which columns contain the **date** and **amount** data:")

        m1, m2, m3, m4 = st.columns(4)

        with m1:
            date_col = st.selectbox("📅 Date column", options=cols,
                index=next((i for i, c in enumerate(cols) if any(k in c.lower() for k in ["date", "time", "ds", "period", "week", "month", "day"])), 0))
        with m2:
            amount_col = st.selectbox("💰 Amount column", options=cols,
                index=next((i for i, c in enumerate(cols) if any(k in c.lower() for k in ["amount", "sales", "revenue", "donation", "value", "price", "total"])), min(1, len(cols)-1)))
        with m3:
            donor_col_options = ["(none)"] + cols
            donor_col = st.selectbox("👥 Donors column (optional)", options=donor_col_options,
                index=next((i+1 for i, c in enumerate(cols) if any(k in c.lower() for k in ["donor", "count", "num"])), 0))
        with m4:
            category_col_options = ["(none)"] + cols
            category_col = st.selectbox("🏷️ Category column (optional)", options=category_col_options,
                index=next((i+1 for i, c in enumerate(cols) if any(k in c.lower() for k in ["category", "segment", "type", "campaign"])), 0))

        if st.button("✅ Confirm Mapping & Load", use_container_width=True):
            try:
                df = raw_df.copy()

                rename_map = {date_col: "date", amount_col: "amount"}
                if donor_col != "(none)":
                    rename_map[donor_col] = "donors"
                if category_col != "(none)" and category_col not in rename_map:
                    rename_map[category_col] = "category"
                df = df.rename(columns=rename_map)

                # FIX #5 — removed deprecated infer_datetime_format
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                df["amount"] = pd.to_numeric(
                    df["amount"].astype(str).str.replace(r"[^\d.]", "", regex=True),
                    errors="coerce"
                )

                before = len(df)
                df = df.dropna(subset=["date", "amount"])
                df = df[df["amount"] > 0]
                after = len(df)

                if after < 10:
                    st.error(f"Need at least 10 valid rows. Only {after} rows had parseable date + amount values.")
                    return

                df = df.sort_values("date").reset_index(drop=True)
                st.session_state["df"]          = df
                st.session_state["data_source"] = st.session_state.get("upload_name", "Uploaded CSV")
                st.session_state.pop("raw_df", None)

                if before != after:
                    st.warning(f"⚠️ {before - after} rows dropped. {after:,} rows loaded.")
                else:
                    st.success(f"✅ **{st.session_state['data_source']}** loaded — {after:,} rows")

                st.rerun()

            except Exception as ex:
                st.error(f"Mapping failed: {ex}")
                return

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

    metrics = summary_metrics(df)

    # FIX #8 — show notice when donor count is estimated, not from real data
    if metrics.get("donors_estimated"):
        st.info("ℹ️ No **donors** column found — donor counts are estimated (rows × 35). "
                "Add a `donors` column to your CSV for accurate donor analytics.")

    c1, c2, c3, c4 = st.columns(4)
    cards = [
        (c1, "Total Raised",   format_currency(metrics["total"]),          "💰", ""),
        (c2, "Total Donors",   f'{int(metrics["total_donors"]):,}{"*" if metrics.get("donors_estimated") else ""}', "👥", ""),
        (c3, "Avg per Donor",  format_currency(metrics["avg_donation"]),   "🎯", ""),
        (c4, "Peak Value",     format_currency(metrics["peak_amount"]),    "🚀", ""),
    ]
    for col, label, val, icon, delta in cards:
        with col:
            st.markdown(kpi_card_html(label, val, delta, icon), unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="section-title" style="margin-top:24px">Dataset Preview</div>
    <div class="section-sub">Showing first 50 rows · Source: <strong>{}</strong> · {} rows total</div>
    """.format(st.session_state.get("data_source", "unknown"), len(df)), unsafe_allow_html=True)

    display_cols = st.multiselect(
        "Columns to display",
        options=df.columns.tolist(),
        default=df.columns.tolist()[:min(7, len(df.columns))]
    )
    st.dataframe(
        df[display_cols].head(50).style.format(
            {"amount": "{:,.2f}", "donors": "{:,.0f}"} if "amount" in display_cols else {}
        ),
        use_container_width=True, height=320
    )

    st.markdown("### 📊 Column Statistics")
    st.dataframe(df.describe().style.format("{:.2f}"), use_container_width=True)

    if st.button("🔄 Re-map columns / Upload new file"):
        st.session_state.pop("df", None)
        st.session_state.pop("raw_df", None)
        st.session_state.pop("upload_name", None)
        st.rerun()

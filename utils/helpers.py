"""
utils/helpers.py — Shared Utilities
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
from pathlib import Path

def cur(value):
    """Shorthand currency formatter."""
    return format_currency(value)


def load_css() -> None:
    """Inject custom CSS from assets/style.css."""
    css_path = Path(__file__).parent.parent / "assets" / "style.css"
    if css_path.exists():
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def load_demo_data() -> pd.DataFrame:
    """Load bundled sample dataset."""
    data_path = Path(__file__).parent.parent / "data" / "sample_donations.csv"
    df = pd.read_csv(data_path, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


# ── Column name aliases ────────────────────────────────────────────────────────
# Maps internal name → list of possible column names in uploaded CSVs
COLUMN_ALIASES = {
    "date": [
        "date", "order date", "transaction date", "donation date",
        "ds", "time", "timestamp", "period", "week", "month", "day",
        "order_date", "transaction_date", "donation_date",
    ],
    "amount": [
        "amount", "sales", "revenue", "donation", "donations",
        "value", "total", "sum", "price", "quantity", "qty",
        "sales amount", "total amount", "donation amount",
        "sales_amount", "total_amount", "donation_amount",
    ],
    "donors": [
        "donors", "donor count", "num donors", "number of donors",
        "donor_count", "num_donors",
    ],
    "campaign": [
        "campaign", "campaign name", "fund", "initiative",
        "campaign_name",
    ],
    "category": [
        "category", "segment", "type", "sub-category", "subcategory",
        "product category", "category name",
    ],
    "region": [
        "region", "state", "city", "location", "area", "zone",
        "country", "geography",
    ],
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename uploaded CSV columns to internal standard names.
    Case-insensitive, strips whitespace.
    """
    col_map = {}
    lowered = {c: c.strip().lower() for c in df.columns}

    for internal_name, aliases in COLUMN_ALIASES.items():
        if internal_name in df.columns:
            continue  # already correct
        for col, col_lower in lowered.items():
            if col_lower in aliases and internal_name not in col_map.values():
                col_map[col] = internal_name
                break

    if col_map:
        df = df.rename(columns=col_map)

    # Parse date column if found
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], infer_datetime_format=True, errors="coerce")

    # Parse amount column if found
    if "amount" in df.columns and not pd.api.types.is_numeric_dtype(df["amount"]):
        df["amount"] = pd.to_numeric(df["amount"].astype(str).str.replace(r"[^\d.]", "", regex=True), errors="coerce")

    return df


def validate_csv(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """
    Validate uploaded CSV.
    Returns (is_valid, list_of_errors).
    """
    errors = []
    if "date" not in df.columns:
        errors.append("Missing required column: **date** (also tried: order date, transaction date, ds, time...)")
    if "amount" not in df.columns:
        errors.append("Missing required column: **amount** (also tried: sales, revenue, donation, value...)")
    if "date" in df.columns:
        bad_dates = df["date"].isnull().sum()
        if bad_dates > 0:
            errors.append(f"Column **date** has {bad_dates} unparseable values")
    if "amount" in df.columns:
        if not pd.api.types.is_numeric_dtype(df["amount"]):
            errors.append("Column **amount** must be numeric")
        elif (df["amount"] <= 0).all():
            errors.append("Column **amount** has no positive values")
    if len(df) < 10:
        errors.append("Need at least 10 rows for forecasting")
    return len(errors) == 0, errors


def format_currency(value: float, symbol: str = "₹") -> str:
    if value >= 1_00_00_000:
        return f"{symbol}{value/1_00_00_000:.1f}Cr"
    elif value >= 1_00_000:
        return f"{symbol}{value/1_00_000:.1f}L"
    elif value >= 1000:
        return f"{symbol}{value/1000:.1f}K"
    return f"{symbol}{value:,.0f}"


def delta_arrow(current: float, previous: float) -> str:
    if previous == 0:
        return "—"
    pct = (current - previous) / abs(previous) * 100
    arrow = "▲" if pct > 0 else "▼"
    color_class = "positive" if pct > 0 else "negative"
    return f'<span class="kpi-delta {color_class}">{arrow} {abs(pct):.1f}%</span>'


def summary_metrics(df: pd.DataFrame) -> dict:
    """Compute top-level KPIs from the donation DataFrame."""
    df2 = df.copy()
    df2["date"] = pd.to_datetime(df2["date"])

    total = df2["amount"].sum()
    avg_weekly = df2.groupby(pd.Grouper(key="date", freq="W"))["amount"].sum().mean()
    total_donors = df2["donors"].sum() if "donors" in df2.columns else len(df2) * 35
    avg_donation = total / max(total_donors, 1)
    peak_amount = df2["amount"].max()
    peak_date = df2.loc[df2["amount"].idxmax(), "date"]

    monthly = df2.set_index("date").resample("ME")["amount"].sum()
    if len(monthly) >= 2:
        mom_growth = (monthly.iloc[-1] - monthly.iloc[-2]) / (monthly.iloc[-2] + 1e-9) * 100
    else:
        mom_growth = 0.0

    return {
        "total": total,
        "avg_weekly": avg_weekly,
        "total_donors": total_donors,
        "avg_donation": avg_donation,
        "peak_amount": peak_amount,
        "peak_date": peak_date.strftime("%d %b %Y"),
        "mom_growth": mom_growth,
        "num_weeks": len(df2),
    }


def generate_report_csv(forecast_df: pd.DataFrame, metrics: dict) -> bytes:
    """Generate a downloadable CSV report."""
    buf = io.BytesIO()
    report = forecast_df.copy()
    report["RMSE"] = metrics.get("RMSE", "N/A")
    report["MAE"] = metrics.get("MAE", "N/A")
    report["MAPE"] = metrics.get("MAPE", "N/A")
    report["Model"] = metrics.get("model", "N/A")
    report.to_csv(buf, index=False)
    return buf.getvalue()


def kpi_card_html(label: str, value: str, delta_html: str = "", icon: str = "") -> str:
    return f"""
    <div class="kpi-card">
        <div style="font-size:1.6rem;margin-bottom:4px">{icon}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-label">{label}</div>
        {delta_html}
    </div>
    """


def page_header(title: str, subtitle: str = "") -> None:
    st.markdown(f"""
    <div style="margin-bottom: 24px; padding-bottom: 16px; border-bottom: 1px solid rgba(255,255,255,0.07);">
        <h1 style="font-size:1.8rem;font-weight:700;letter-spacing:-0.02em;margin:0;color:#fff">{title}</h1>
        {"" if not subtitle else f'<p style="color:rgba(255,255,255,0.55);margin:6px 0 0;font-size:0.9rem">{subtitle}</p>'}
    </div>
    """, unsafe_allow_html=True)
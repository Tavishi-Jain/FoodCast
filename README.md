---
title: FoodCast
emoji: 🌱
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
license: mit
---

# 🌱 FoodCast — AI Donation Forecasting Platform

> \*\*Predict. Plan. Maximise Impact.\*\*  
> ML-powered forecasting for NGOs, CSR teams, fundraisers \& religious organisations.

!\[Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
!\[Streamlit](https://img.shields.io/badge/Streamlit-1.32-red?style=flat-square)
!\[License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

\---

## ✨ Features

|Page|Description|
|-|-|
|🏠 **Landing Page**|Hero section with animated background, mission CTAs|
|📂 **Upload Dashboard**|CSV drag-and-drop, validation, KPI summary cards|
|📈 **Donation Forecast**|ARIMA / Prophet / XGBoost / LSTM with confidence intervals|
|👥 **Donor Analytics**|Churn prediction, retention forecast, regional breakdown|
|🎉 **Seasonal Insights**|Festival spike detection, ROI calendar, heatmap|
|🚨 **Drought Alert**|Z-score anomaly detection, red alerts, downloadable reports|
|🎯 **Campaign Predictor**|Random Forest success probability + recommendations|
|🏆 **Model Comparison**|RMSE / MAE / MAPE benchmarking + radar chart|

\---

## 🚀 Quick Start

### 1\. Clone \& Install

```bash
git clone https://github.com/your-org/foodcast.git
cd foodcast
pip install -r requirements.txt
```

### 2\. Run Locally

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

### 3\. Load Sample Data

Click **"Load Demo Data"** in the sidebar or on the landing page to explore with 18 months of synthetic donation records.

\---

## 📁 Project Structure

```
foodcast/
├── app.py                    # Main entry point + landing page
├── predict.py                # ML model hooks (ARIMA, Prophet, XGBoost, LSTM, RF)
├── requirements.txt
├── README.md
│
├── pages/
│   ├── upload.py             # Upload Dashboard
│   ├── forecast.py           # Donation Forecast
│   ├── donor\_analytics.py    # Donor Analytics
│   ├── seasonal.py           # Seasonal Campaign Insights
│   ├── drought\_alert.py      # Drought Alert System
│   ├── campaign\_predictor.py # Campaign Success Predictor
│   └── model\_comparison.py   # Model Comparison
│
├── utils/
│   ├── helpers.py            # Shared utilities (CSS, metrics, formatters)
│   └── charts.py             # Reusable Plotly chart components
│
├── assets/
│   └── style.css             # Dark glassmorphism design system
│
└── data/
    └── sample\_donations.csv  # Bundled demo dataset
```

\---

## 📊 CSV Format

Your donation CSV must include at minimum:

|Column|Type|Example|Required|
|-|-|-|-|
|`date`|date|2024-01-15|✅ Yes|
|`amount`|numeric|12500|✅ Yes|
|`donors`|integer|45|Optional|
|`campaign`|string|Diwali Drive|Optional|
|`category`|string|Food|Optional|
|`region`|string|North|Optional|

\---

## 🤖 ML Models

|Model|Purpose|Library|
|-|-|-|
|**ARIMA**|Stationary time-series forecasting|`statsmodels`|
|**Prophet**|Seasonality + Indian holidays|`prophet`|
|**XGBoost**|Lag-feature trend prediction|`xgboost`|
|**LSTM**|Deep learning sequence model|`torch` (hook ready)|
|**Random Forest**|Campaign success classifier|`scikit-learn`|

All models in `predict.py` are modular — replace function bodies with your trained model artefacts for production.

\---

## 🌐 Deploy to Hugging Face Spaces

1. Create a new Space → **Streamlit** SDK
2. Push this repo to the Space's git remote
3. Set Python version to **3.10** in `README.md` metadata (see below)
4. Hugging Face auto-installs `requirements.txt` and runs `app.py`

Add to the top of your `README.md` for Hugging Face:



\---

## 🎨 Design System

* **Theme:** Dark mode glassmorphism
* **Primary:** `#00C897` (teal-green)
* **Secondary:** `#FFC857` (amber)
* **Alert:** `#FF5A5F` (coral-red)
* **Background:** `#0B1020` (deep navy)
* **Font:** Space Grotesk

\---

## 📄 License

MIT © 2024 FoodCast. Built with ❤️ for social impact.
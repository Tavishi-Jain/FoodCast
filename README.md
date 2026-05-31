---
title: FoodCast
emoji: 🍕
colorFrom: green
colorTo: yellow
sdk: docker
pinned: false
license: mit
short_description: AI-powered donation forecasting platform for food banks
---




# 🍱 FoodCast — AI Donation Forecasting Platform

FoodCast helps food banks and humanitarian organizations predict incoming food donations, spot seasonal trends, and plan distribution more effectively using machine learning.

## Features

- **Donation Forecast** — Prophet, XGBoost, and LSTM models with configurable horizon
- **Seasonal Analysis** — identify recurring donation patterns across months and weekdays
- **Drought Alert System** — flag donation shortfalls before they become crises
- **Donor Analytics** — summarize donor behavior and estimate retention
- **Campaign Predictor** — estimate the impact of upcoming fundraising campaigns
- **Model Comparison** — benchmark all models head-to-head on your data

## Usage

1. Upload your own donation CSV (date + amount columns), or use the built-in demo dataset
2. Choose a forecasting model and horizon from the sidebar
3. Explore forecasts, alerts, and analytics across the navigation pages

## Stack

- [Streamlit](https://streamlit.io/) — UI framework
- [Prophet](https://facebook.github.io/prophet/) — time-series forecasting
- [XGBoost](https://xgboost.readthedocs.io/) — gradient boosting
- [Plotly](https://plotly.com/) — interactive charts
- [Drizzle / pandas](https://pandas.pydata.org/) — data processing

## Notes

- LSTM model is currently a placeholder — results are illustrative
- Donor count estimates are synthetic when individual donor IDs are not present in the uploaded data
- Forecasts are probabilistic; always combine with domain expertise

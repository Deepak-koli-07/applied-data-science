# applied-ml-apps

Personal ML learning repo. A mix of analysis notebooks and two production-style pipelines — built to understand the full data science workflow from EDA to deployment.

---

## End-to-End Pipelines

| Folder | What it does |
|--------|-------------|
| [`aqi-mlops`](aqi-mlops/) | Real-time AQI prediction — MLflow tracking, automated retraining, model promotion, deployed to Hugging Face Space (`aqi-mlops/hf-aqi-space/`) |
| [`cvd-risk-score-predictor`](cvd-risk-score-predictor/) | Cardiovascular risk prediction — scikit-learn pipeline, Streamlit app |

---

## Analysis Notebooks

| Folder | What it covers |
|--------|---------------|
| [`california-housing-eda`](california-housing-eda/) | EDA on 1990 census data, log transforms, Linear Regression vs Random Forest |
| [`time-series`](time-series/) | ARIMA forecasting — stationarity testing, ACF/PACF, auto_arima on synthetic sales + Nifty 50 |

---

## Stack

Python · scikit-learn · statsmodels · pmdarima · pandas · matplotlib · seaborn · MLflow · Streamlit · Gradio · Hugging Face

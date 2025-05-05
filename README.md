# 48-Hour-Air-Quality-Forecast-CO-GT-Prediction

# 🌫️ 48-Hour Air Quality Forecast: CO(GT) Prediction

This project predicts **Carbon Monoxide concentration (`CO(GT)`)** for the next 48 hours using **RandomForest** and **XGBoost** models. It includes data preprocessing, model training, feature importance analysis, and residual evaluation.

---

## 📁 Repository Structure

forecast-48h-air-quality/
│
├── data/
│ ├── raw/ # Original input file (e.g., forecast_48_hours.xls)
│ └── processed/ # Cleaned data used in modeling
│
├── notebooks/
│ ├── 01_data_exploration.ipynb # EDA: visualization, missing values
│ ├── 02_model_rf_xgb.ipynb # Model training: RandomForest and XGBoost
│ └── 03_evaluation_residuals.ipynb# Residual ACF + metrics (RMSE/MAE)
│
├── src/
│ ├── preprocessing.py # Data cleaning functions
│ ├── models.py # Training functions for RF/XGB
│ └── utils.py # Plotting feature importance, ACF
│
├── outputs/
│ ├── figures/ # Feature importance & ACF plots
│ └── models/ # Saved model files (.pkl/.joblib)
│
├── README.md # You are here!
├── requirements.txt # Python dependencies
├── .gitignore # Ignore unnecessary files
└── LICENSE # MIT License


---

## 🔍 Project Highlights

- 📊 Forecasts CO(GT) 48 hours into the future using ensemble ML
- 📌 Uses past pollutant readings as features
- 📈 Includes feature importance graphs for model explainability
- 🔁 Residual analysis via **Autocorrelation Function (ACF)**
- 📉 Model performance evaluation using **RMSE** and **MAE**

---

## 🧪 Model Summary

| Model        | RMSE   | MAE    | Residual ACF |
|--------------|--------|--------|---------------|
| RandomForest | XX.XX  | XX.XX  | Some autocorrelation |
| XGBoost      | XX.XX  | XX.XX  | Mostly white noise ✅ |

> Residual ACF plots are saved in `outputs/figures/`.

---

## 📦 Installation

```bash
git clone https://github.com/yourusername/forecast-48h-air-quality.git
cd forecast-48h-air-quality
pip install -r requirements.txt

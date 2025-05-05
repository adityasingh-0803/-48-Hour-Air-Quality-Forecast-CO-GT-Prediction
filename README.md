# 48-Hour-Air-Quality-Forecast-CO-GT-Prediction

# 🌫️ 48-Hour Air Quality Forecast: CO(GT) Prediction

This project predicts **Carbon Monoxide concentration (`CO(GT)`)** for the next 48 hours using **RandomForest** and **XGBoost** models. It includes data preprocessing, model training, feature importance analysis, and residual evaluation.

---

## 📁 Repository Structure

```plaintext
forecast-48h-air-quality/
├── data/                     # Data files
│   ├── raw/                 # Raw data (e.g., forecast_48_hours.xls)
│   └── processed/           # Cleaned or transformed data
├── notebooks/               # Jupyter Notebooks
│   ├── 01_data_exploration.ipynb      # EDA and visualization
│   ├── 02_model_rf_xgb.ipynb          # Model training (RandomForest & XGBoost)
│   └── 03_evaluation_residuals.ipynb  # Evaluation, ACF analysis, metrics
├── src/                     # Source code
│   ├── preprocessing.py     # Data cleaning and transformation functions
│   ├── models.py            # Model training and saving
│   └── utils.py             # Helper functions (e.g., plots)
├── outputs/                 # Generated outputs
│   ├── figures/             # Plots (feature importance, ACF)
│   └── models/              # Trained model files (.joblib/.pkl)
├── README.md                # Project overview and instructions
├── requirements.txt         # Project dependencies
├── .gitignore               # Files/folders to ignore in Git
└── LICENSE                  # License file (e.g., MIT)
```

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

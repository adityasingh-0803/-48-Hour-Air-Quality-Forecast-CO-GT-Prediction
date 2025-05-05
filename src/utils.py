import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.graphics.tsaplots import plot_acf
import os

def plot_feature_importance(model, feature_names, model_name, output_path):
    importances = model.feature_importances_
    sorted_indices = np.argsort(importances)
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_indices)), importances[sorted_indices])
    plt.yticks(range(len(sorted_indices)), [feature_names[i] for i in sorted_indices])
    plt.title(f"Feature Importance for {model_name}")
    plt.tight_layout()
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(os.path.join(output_path, f"feature_importance_{model_name}.png"))
    plt.close()

def plot_residual_acf(y_true, y_pred, model_name, output_path):
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_acf(residuals, ax=ax, lags=24, alpha=0.05)
    ax.set_title(f"Autocorrelation of Residuals for {model_name}")
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(os.path.join(output_path, f"residuals_acf_{model_name}.png"))
    plt.close()

def evaluate_model(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    return {"RMSE": rmse, "R2": r2}

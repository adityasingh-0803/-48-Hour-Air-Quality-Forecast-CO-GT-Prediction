from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from joblib import dump, load
import os

def train_random_forest(X_train, y_train, n_estimators=100, max_depth=None):
    rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    rf.fit(X_train, y_train)
    return rf

def train_xgboost(X_train, y_train, n_estimators=100, max_depth=3):
    xgb = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    xgb.fit(X_train, y_train)
    return xgb

def save_model(model, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    dump(model, filename)

def load_model(filename):
    return load(filename)

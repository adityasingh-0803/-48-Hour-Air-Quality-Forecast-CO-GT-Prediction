import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    df = pd.read_excel(filepath)
    return df

def preprocess_data(df, target_column, drop_cols=[]):
    df = df.dropna()  # Drop rows with missing values
    if drop_cols:
        df = df.drop(columns=drop_cols)
    X = df.drop(columns=[target_column])
    y = df[target_column]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

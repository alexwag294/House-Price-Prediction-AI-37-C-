"""
data_loader.py
──────────────
Loads train.csv and returns ALL 262 features + target.
Target: SalePrice (log-transformed)
"""

import pandas as pd

TARGET = "SalePrice"


def load_data(filepath: str = "data/train.csv"):
    df = pd.read_csv(filepath)

    if TARGET not in df.columns:
        raise ValueError(f"'{TARGET}' column not found in dataset.")

    y = df[TARGET].copy()
    X = df.drop(columns=[TARGET]).copy()

    # Convert boolean columns to int
    bool_cols = X.select_dtypes(include="bool").columns
    X[bool_cols] = X[bool_cols].astype(int)

    # Fill missing values with median
    X = X.fillna(X.median(numeric_only=True))

    print(f"[DataLoader] Loaded  →  {X.shape[0]} rows  |  {X.shape[1]} features")
    return X, y
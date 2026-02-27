"""
data_loader.py
──────────────
Loads train.csv and returns ALL features + log-transformed target.
Target: SalePrice (log-transformed)
"""

import numpy as np
import pandas as pd

TARGET = "SalePrice"


def load_data(filepath: str = "data/train.csv"):
    df = pd.read_csv(filepath)

    if TARGET not in df.columns:
        raise ValueError(f"'{TARGET}' column not found in dataset.")

    y = np.log(df[TARGET].copy())   # log-transform target
    X = df.drop(columns=[TARGET, "Id"], errors="ignore").copy()

    # One-hot encode categoricals
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    # Convert boolean columns to int
    bool_cols = X.select_dtypes(include="bool").columns
    X[bool_cols] = X[bool_cols].astype(int)

    # Fill missing values with median
    X = X.fillna(X.median(numeric_only=True))

    print(f"[DataLoader] Loaded  →  {X.shape[0]} rows  |  {X.shape[1]} features")
    return X, y
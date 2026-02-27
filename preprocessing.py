"""
preprocessing.py
────────────────
Handles all feature engineering and cleaning steps:
  - Boolean → int conversion
  - Median imputation for missing values
  - StandardScaler fit / transform
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


class Preprocessor:
    """
    Fits on training data and transforms any split consistently.

    Usage
    -----
        pre = Preprocessor()
        X_train_sc = pre.fit_transform(X_train)
        X_test_sc  = pre.transform(X_test)
    """

    def __init__(self):
        self.scaler      = StandardScaler()
        self.median_vals = None

    # ── internal cleaning (no data leakage) ──────────────────
    def _clean(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        # Convert boolean columns to integer
        bool_cols = X.select_dtypes(include="bool").columns
        X[bool_cols] = X[bool_cols].astype(int)
        return X

    # ── fit + transform on training data ─────────────────────
    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        X = self._clean(X)
        # Learn median from training data only
        self.median_vals = X.median(numeric_only=True)
        X = X.fillna(self.median_vals)
        X_sc = self.scaler.fit_transform(X)
        print(f"[Preprocessor] Fitted  →  {X_sc.shape[1]} features scaled")
        return X_sc

    # ── transform new data with learned parameters ────────────
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X = self._clean(X)
        X = X.fillna(self.median_vals)
        return self.scaler.transform(X)
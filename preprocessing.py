"""
preprocessing.py
────────────────
Scales all features using StandardScaler.
Stores median values from training set for use at inference time.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class Preprocessor:
    def __init__(self):
        self.scaler      = StandardScaler()
        self.median_vals = None
        self.columns_    = None

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        self.median_vals = X.median(numeric_only=True)
        self.columns_    = X.columns.tolist()
        X_filled = X.fillna(self.median_vals)
        X_sc = self.scaler.fit_transform(X_filled)
        print(f"[Preprocessor] Fitted  →  {X_sc.shape[1]} features scaled")
        return X_sc

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        # Align columns to training set — add missing cols as 0
        X = X.reindex(columns=self.columns_, fill_value=0)
        X = X.fillna(self.median_vals)
        return self.scaler.transform(X)
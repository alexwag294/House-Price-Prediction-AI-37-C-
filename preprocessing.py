"""
preprocessing.py
────────────────
Scales all 262 features using StandardScaler.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class Preprocessor:
    def __init__(self):
        self.scaler      = StandardScaler()
        self.median_vals = None

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        self.median_vals = X.median(numeric_only=True)
        X = X.fillna(self.median_vals)
        X_sc = self.scaler.fit_transform(X)
        print(f"[Preprocessor] Fitted  →  {X_sc.shape[1]} features scaled")
        return X_sc

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X = X.fillna(self.median_vals)
        return self.scaler.transform(X)
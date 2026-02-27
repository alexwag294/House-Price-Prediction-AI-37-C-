"""
training.py
───────────
Handles:
  - Train / test splitting
  - Model fitting
  - Extended K-Fold Cross-Validation  (3, 5, 10-fold)
  - Learning Curve computation + plot
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import (
    train_test_split, KFold,
    cross_validate, learning_curve
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


# ══════════════════════════════════════════════════════════
#  SPLIT
# ══════════════════════════════════════════════════════════
def split_data(X: pd.DataFrame, y: pd.Series,
               test_size: float = 0.2, random_state: int = 42):
    """
    Split into train / test sets.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"[Training] Split  →  Train: {len(X_train)}  |  Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test


# ══════════════════════════════════════════════════════════
#  FIT
# ══════════════════════════════════════════════════════════
def fit_model(model, X_train_sc: np.ndarray, y_train: pd.Series):
    """
    Fit the model on scaled training data.

    Returns
    -------
    model : fitted LinearRegression
    """
    model.fit(X_train_sc, y_train)
    print(f"[Training] Model fitted  →  {X_train_sc.shape[1]} features")
    return model


# ══════════════════════════════════════════════════════════
#  CROSS-VALIDATION
# ══════════════════════════════════════════════════════════
def run_cross_validation(X: pd.DataFrame, y: pd.Series,
                         cv_folds: list = [3, 5, 10]) -> dict:
    """
    Run K-Fold CV for multiple k values.
    Uses an internal pipeline (scaler + MLR) to avoid data leakage.

    Returns
    -------
    results : dict  { k: cross_validate output }
    """
    pipe = make_pipeline(StandardScaler(), LinearRegression(n_jobs=-1))

    # Fill + encode before CV
    X_clean = X.copy()
    bool_cols = X_clean.select_dtypes("bool").columns
    X_clean[bool_cols] = X_clean[bool_cols].astype(int)
    X_clean = X_clean.fillna(X_clean.median(numeric_only=True))

    print("\n" + "═"*56)
    print("   CROSS-VALIDATION RESULTS")
    print("═"*56)
    print(f"   {'k':>5}  │  {'R² Mean':>8}  {'±Std':>7}  │  "
          f"{'MAE':>8}  {'RMSE':>8}  │  {'TrainR²':>8}")
    print("   " + "─"*53)

    results = {}
    for k in cv_folds:
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        cv = cross_validate(
            pipe, X_clean, y, cv=kf,
            scoring=["r2", "neg_mean_absolute_error",
                     "neg_root_mean_squared_error"],
            return_train_score=True, n_jobs=-1
        )
        r2m   = cv["test_r2"].mean()
        r2s   = cv["test_r2"].std()
        mae   = (-cv["test_neg_mean_absolute_error"]).mean()
        rmse  = (-cv["test_neg_root_mean_squared_error"]).mean()
        tr_r2 = cv["train_r2"].mean()

        print(f"   {k:>5}-Fold │  {r2m:>8.4f}  {r2s:>7.4f}  │  "
              f"{mae:>8.4f}  {rmse:>8.4f}  │  {tr_r2:>8.4f}")
        results[k] = cv

    print("═"*56)
    return results


# ══════════════════════════════════════════════════════════
#  LEARNING CURVE
# ══════════════════════════════════════════════════════════
def compute_and_plot_learning_curve(X: pd.DataFrame, y: pd.Series,
                                    name: str = "Multiple Linear Regression"):
    """
    Compute and save a learning curve plot.
    Shows Train R² vs Validation R² as training size increases.
    """
    pipe = make_pipeline(StandardScaler(), LinearRegression(n_jobs=-1))

    X_clean = X.copy()
    bool_cols = X_clean.select_dtypes("bool").columns
    X_clean[bool_cols] = X_clean[bool_cols].astype(int)
    X_clean = X_clean.fillna(X_clean.median(numeric_only=True))

    print(f"\n[Training] Computing learning curve for '{name}' ...")

    train_sizes, train_scores, val_scores = learning_curve(
        pipe, X_clean, y,
        train_sizes=np.linspace(0.05, 1.0, 15),
        cv=5, scoring="r2", n_jobs=-1
    )

    train_mean = train_scores.mean(axis=1)
    val_mean   = val_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    val_std    = val_scores.std(axis=1)

    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, train_mean, "o-", color="blue",  label="Training R²")
    plt.plot(train_sizes, val_mean,   "o-", color="green", label="Validation R²")
    plt.fill_between(train_sizes, train_mean - train_std,
                     train_mean + train_std, alpha=0.15, color="blue")
    plt.fill_between(train_sizes, val_mean - val_std,
                     val_mean + val_std,     alpha=0.15, color="green")
    plt.title(f"Learning Curve — {name}")
    plt.xlabel("Training Set Size")
    plt.ylabel("R² Score")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"plots/learning_curve_{name.replace(' ', '_')}.png", dpi=150)
    plt.close()   # ← was plt.show()

    print(f"[Training] Learning curve saved → "
          f"plots/learning_curve_{name.replace(' ', '_')}.png")

    return train_sizes, train_scores, val_scores
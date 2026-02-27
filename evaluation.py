"""
evaluation.py
─────────────
Handles:
  - Hold-out test set metrics  (R², MAE, RMSE in log & $ scale)
  - predict_house_price()  — reusable prediction utility
  - Diagnostic plots  (Actual vs Predicted, Residuals, CV bar chart)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ══════════════════════════════════════════════════════════
#  METRICS
# ══════════════════════════════════════════════════════════
def evaluate_model(model, preprocessor,
                   X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Evaluate the trained model on the hold-out test set.

    Returns
    -------
    metrics : dict  with r2, mae, rmse, mae_usd, rmse_usd
    y_pred  : np.ndarray  (log-scale predictions)
    """
    X_sc   = preprocessor.transform(X_test)
    y_pred = model.predict(X_sc)

    r2      = r2_score(y_test, y_pred)
    mae     = mean_absolute_error(y_test, y_pred)
    rmse    = np.sqrt(mean_squared_error(y_test, y_pred))
    mae_usd = mean_absolute_error(np.exp(y_test), np.exp(y_pred))
    rmse_usd= np.sqrt(mean_squared_error(np.exp(y_test), np.exp(y_pred)))

    print("\n" + "═"*50)
    print("   HOLD-OUT TEST SET RESULTS")
    print("═"*50)
    print(f"   R²               : {r2:.4f}")
    print(f"   MAE  (log scale) : {mae:.4f}")
    print(f"   RMSE (log scale) : {rmse:.4f}")
    print(f"   MAE  ($)         : ${mae_usd:>10,.0f}")
    print(f"   RMSE ($)         : ${rmse_usd:>10,.0f}")
    print("═"*50)

    return dict(r2=r2, mae=mae, rmse=rmse,
                mae_usd=mae_usd, rmse_usd=rmse_usd), y_pred


# ══════════════════════════════════════════════════════════
#  PREDICTION UTILITY
# ══════════════════════════════════════════════════════════
def predict_house_price(model, preprocessor,
                        X_new: pd.DataFrame) -> pd.DataFrame:
    """
    Predict house prices for new / unseen data.

    Parameters
    ----------
    model        : fitted LinearRegression
    preprocessor : fitted Preprocessor
    X_new        : pd.DataFrame  (raw features, same columns as training)

    Returns
    -------
    pd.DataFrame with columns:
        predicted_log_price  — raw model output
        predicted_price_usd  — exponentiated back to dollars
        lower_bound_usd      — indicative −10 % interval
        upper_bound_usd      — indicative +10 % interval
    """
    X_sc     = preprocessor.transform(X_new)
    log_pred = model.predict(X_sc)
    price    = np.exp(log_pred)

    return pd.DataFrame({
        "predicted_log_price": np.round(log_pred, 4),
        "predicted_price_usd": price.astype(int),
        "lower_bound_usd":     (price * 0.90).astype(int),
        "upper_bound_usd":     (price * 1.10).astype(int),
    }, index=X_new.index)


# ══════════════════════════════════════════════════════════
#  DIAGNOSTIC PLOTS
# ══════════════════════════════════════════════════════════
def plot_predicted_vs_actual(y_test: pd.Series, y_pred: np.ndarray):
    r2 = r2_score(y_test, y_pred)
    y_true_usd = np.exp(y_test)
    y_pred_usd = np.exp(y_pred)

    plt.figure(figsize=(7, 6))
    plt.scatter(y_true_usd / 1000, y_pred_usd / 1000,
                alpha=0.45, s=30, color="#3A7DC9", edgecolors="white", lw=0.3)
    lo = min(y_true_usd.min(), y_pred_usd.min()) / 1000 * 0.95
    hi = max(y_true_usd.max(), y_pred_usd.max()) / 1000 * 1.05
    plt.plot([lo, hi], [lo, hi], "--", color="#E05252", lw=2, label="Perfect Fit")
    plt.xlim(lo, hi); plt.ylim(lo, hi)
    plt.xlabel("Actual Price ($K)")
    plt.ylabel("Predicted Price ($K)")
    plt.title(f"Predicted vs Actual  (R² = {r2:.4f})")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/predicted_vs_actual_MLR.png", dpi=150)
    plt.close()
    print("[Evaluation] Plot saved → plots/predicted_vs_actual_MLR.png")


def plot_residuals(y_test: pd.Series, y_pred: np.ndarray):
    residuals = y_test.values - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Residual Diagnostics — Multiple Linear Regression",
                 fontsize=13, fontweight="bold")

    axes[0].scatter(y_pred, residuals, alpha=0.4, s=30,
                    color="#4DB87A", edgecolors="white", lw=0.3)
    axes[0].axhline(0, color="#E05252", lw=1.8, linestyle="--")
    axes[0].set_xlabel("Fitted Value (log scale)")
    axes[0].set_ylabel("Residual")
    axes[0].set_title("Residuals vs Fitted")

    axes[1].hist(residuals, bins=40, color="#9B72CF",
                 edgecolor="white", linewidth=0.4)
    axes[1].axvline(0, color="#E05252", lw=1.8, linestyle="--")
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Residual Distribution")

    plt.tight_layout()
    plt.savefig("plots/residuals_MLR.png", dpi=150)
    plt.close()
    print("[Evaluation] Plot saved → plots/residuals_MLR.png")


def plot_cv_comparison(cv_results: dict):
    ks    = sorted(cv_results.keys())
    means = [cv_results[k]["test_r2"].mean() for k in ks]
    stds  = [cv_results[k]["test_r2"].std()  for k in ks]
    colors= ["#3A7DC9", "#4DB87A", "#F5A623"]

    plt.figure(figsize=(6, 5))
    bars = plt.bar([f"{k}-Fold" for k in ks], means,
                   color=colors, edgecolor="white", lw=0.5, width=0.5)
    plt.errorbar([f"{k}-Fold" for k in ks], means, yerr=stds,
                 fmt="none", color="black", capsize=7, lw=1.5)
    for bar, m in zip(bars, means):
        plt.text(bar.get_x() + bar.get_width() / 2, m + 0.003,
                 f"{m:.4f}", ha="center", va="bottom",
                 fontsize=10, fontweight="bold")
    plt.ylim(min(means) - 0.06, 1.0)
    plt.ylabel("Mean R²")
    plt.title("Cross-Validation R²  (3 / 5 / 10-Fold)")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/cv_comparison_MLR.png", dpi=150)
    plt.close()
    print("[Evaluation] Plot saved → plots/cv_comparison_MLR.png")
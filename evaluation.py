# evaluation.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance


# ─────────────────────────────────────────────
# 4.2  Results table (printed to console)
# ─────────────────────────────────────────────
def display_results(results):
    print("\n" + "=" * 65)
    print(f"{'Model':<22} {'R²':>8} {'RMSE':>12} {'CV R²':>8}  Best Params")
    print("=" * 65)
    for name, m in results.items():
        params_str = str(m["Params"])[:25]
        print(f"{name:<22} {m['R2']:>8.3f} {m['RMSE']:>12.2f} {m['CV']:>8.3f}  {params_str}")
    print("=" * 65)


# ─────────────────────────────────────────────
# 4.3a Bar charts — R², RMSE, CV R² comparison
# ─────────────────────────────────────────────
def plot_comparison(results):
    names       = list(results.keys())
    r2_scores   = [results[n]["R2"]   for n in names]
    rmse_scores = [results[n]["RMSE"] for n in names]
    cv_scores   = [results[n]["CV"]   for n in names]
    colors      = ["#4CAF50", "#FF9800", "#2196F3"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Model Comparison", fontsize=14, fontweight="bold")

    for ax, values, title, color, ylabel in zip(
        axes,
        [r2_scores, rmse_scores, cv_scores],
        ["R² Score", "RMSE ($)", "Cross-Validation R²"],
        colors, ["R²", "RMSE", "R²"]
    ):
        bars = ax.bar(names, values, color=color, edgecolor="black", width=0.5)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=15, ha="right")
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(values) * 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig("plots/model_comparison.png", dpi=150)
    plt.close()


# ─────────────────────────────────────────────
# 4.3b Predicted vs. Actual
# ─────────────────────────────────────────────
def plot_predicted_vs_actual(results_preds, name="Random Forest"):
    if name not in results_preds:
        return
    y_test, y_pred = results_preds[name]

    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, color="steelblue", edgecolors="none", s=20)
    lo, hi = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    plt.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="Perfect prediction")
    plt.title(f"Predicted vs. Actual — {name}")
    plt.xlabel("Actual SalePrice ($)")
    plt.ylabel("Predicted SalePrice ($)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/predicted_vs_actual_{name.replace(' ', '_')}.png", dpi=150)
    plt.close()


# ─────────────────────────────────────────────
# 4.3c Residual plot
# ─────────────────────────────────────────────
def plot_residuals(results_preds, name="Random Forest"):
    if name not in results_preds:
        return
    y_test, y_pred = results_preds[name]
    residuals = y_test - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Residual Analysis — {name}", fontsize=13, fontweight="bold")

    axes[0].scatter(y_test, residuals, alpha=0.5, color="tomato", s=20)
    axes[0].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Actual SalePrice ($)")
    axes[0].set_ylabel("Residual ($)")
    axes[0].set_title("Residuals vs. Actual Price")

    axes[1].hist(residuals, bins=40, color="steelblue", edgecolor="white")
    axes[1].axvline(0, color="red", linestyle="--")
    axes[1].set_xlabel("Residual ($)")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Residual Distribution")

    plt.tight_layout()
    plt.savefig(f"plots/residuals_{name.replace(' ', '_')}.png", dpi=150)
    plt.close()


# ─────────────────────────────────────────────
# 4.3d Feature importance
# ─────────────────────────────────────────────
def plot_feature_importance(best_model, X_test, y_test, top_n=15):
    reg          = best_model.named_steps["regressor"]
    preprocessor = best_model.named_steps["preprocessor"]

    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = [f"Feature {i}" for i in range(len(reg.feature_importances_))]

    feature_names = [
        n.replace("num__", "").replace("cat__", "")
        for n in feature_names
    ]

    if hasattr(reg, "feature_importances_"):
        importances = reg.feature_importances_
        indices     = np.argsort(importances)[-top_n:]

        plt.figure(figsize=(10, 6))
        plt.barh(range(len(indices)), importances[indices],
                 color="steelblue", edgecolor="black")
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel("Importance (Mean Decrease in Impurity)")
        plt.title(f"Top {top_n} Feature Importances — Random Forest")
        plt.tight_layout()
        plt.savefig("plots/feature_importance.png", dpi=150)
        plt.close()

    else:
        perm    = permutation_importance(best_model, X_test, y_test,
                                         n_repeats=10, random_state=42)
        indices = np.argsort(perm.importances_mean)[-top_n:]

        plt.figure(figsize=(10, 6))
        plt.barh(range(len(indices)), perm.importances_mean[indices],
                 color="steelblue")
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel("Permutation Importance (Mean R² decrease)")
        plt.title(f"Top {top_n} Feature Importances (Permutation)")
        plt.tight_layout()
        plt.savefig(f"plots/feature_importance_{best_model.named_steps['regressor'].__class__.__name__}.png", dpi=150)
        plt.close()
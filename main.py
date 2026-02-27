"""
main.py
───────
Runs the full MLR pipeline step by step.
Each result and plot is shown one at a time with a clear separator.

Run:
    python main.py
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
os.makedirs("plots", exist_ok=True)

from data_loader   import load_data
from preprocessing import Preprocessor
from models        import get_mlr_model
from training      import (split_data, fit_model,
                            run_cross_validation,
                            compute_and_plot_learning_curve)
from evaluation    import (evaluate_model, predict_house_price,
                            plot_predicted_vs_actual,
                            plot_residuals, plot_cv_comparison)
from plot_3d       import plot_3d


# ── Helper ────────────────────────────────────────────────
def section(title, step, total=10):
    print("\n" + "█" * 60)
    print(f"  STEP {step}/{total} — {title}")
    print("█" * 60)
    time.sleep(0.3)

def done(msg=""):
    print(f"  ✅  {msg}")
    time.sleep(0.3)


# ══════════════════════════════════════════════════════════
def main():

    print("\n" + "=" * 60)
    print("   🏠  HOUSE PRICE PREDICTOR")
    print("       Multiple Linear Regression — Full Pipeline")
    print("=" * 60)
    time.sleep(0.5)


    # ── STEP 1: Load Data ─────────────────────────────────
    section("LOAD DATA", 1)
    X, y = load_data("data/train.csv")
    print(f"\n  Rows     : {X.shape[0]}")
    print(f"  Features : {X.shape[1]}")
    print(f"  Target   : SalePrice (log-transformed)")
    print(f"  Price range: ${np.exp(y.min()):,.0f}  →  ${np.exp(y.max()):,.0f}")
    done("Data loaded successfully")


    # ── STEP 2: Split ─────────────────────────────────────
    section("TRAIN / TEST SPLIT", 2)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
    print(f"\n  Training set : {len(X_train)} houses  (80%)")
    print(f"  Test set     : {len(X_test)} houses  (20%)")
    done("Split complete")


    # ── STEP 3: Preprocess ────────────────────────────────
    section("PREPROCESSING", 3)
    preprocessor = Preprocessor()
    X_train_sc   = preprocessor.fit_transform(X_train)
    print(f"\n  Boolean columns   → converted to int")
    print(f"  Missing values    → filled with median")
    print(f"  Features scaled   → StandardScaler applied")
    done("Preprocessing complete")


    # ── STEP 4: Train Model ───────────────────────────────
    section("TRAINING MODEL", 4)
    model = get_mlr_model()
    model = fit_model(model, X_train_sc, y_train)
    print(f"\n  Algorithm   : Multiple Linear Regression")
    print(f"  Intercept   : {model.intercept_:.4f}")
    print(f"  Coefficients: {len(model.coef_)} (one per feature)")
    done("Model trained")


    # ── STEP 5: Evaluate ──────────────────────────────────
    section("HOLD-OUT TEST SET RESULTS", 5)
    metrics, y_pred = evaluate_model(model, preprocessor, X_test, y_test)
    done("Evaluation complete")


    # ── STEP 6: Cross-Validation ──────────────────────────
    section("CROSS-VALIDATION  (3, 5, 10-Fold)", 6)
    cv_results = run_cross_validation(X, y, cv_folds=[3, 5, 10])
    done("Cross-validation complete")


    # ── STEP 7: Learning Curve Plot ───────────────────────
    section("PLOT — LEARNING CURVE", 7)
    compute_and_plot_learning_curve(X, y, name="Multiple Linear Regression")
    _show_plot("plots/learning_curve_Multiple_Linear_Regression.png",
               "Learning Curve")
    done("Learning curve saved → plots/")


    # ── STEP 8: Diagnostic Plots ──────────────────────────
    section("PLOT — DIAGNOSTIC CHARTS", 8)

    print("\n  [8a] Actual vs Predicted...")
    plot_predicted_vs_actual(y_test, y_pred)
    _show_plot("plots/predicted_vs_actual_MLR.png", "Actual vs Predicted")

    print("\n  [8b] Residual Analysis...")
    plot_residuals(y_test, y_pred)
    _show_plot("plots/residuals_MLR.png", "Residuals")

    print("\n  [8c] Cross-Validation Comparison...")
    plot_cv_comparison(cv_results)
    _show_plot("plots/cv_comparison_MLR.png", "CV Comparison")

    done("All diagnostic plots saved → plots/")


    # ── STEP 9: 3D Plots ──────────────────────────────────
    section("PLOT — 3D VISUALISATION", 9)
    plot_3d(model, preprocessor, X, y)
    _show_plot("plots/3d_house_price_plot.png", "3D House Price Plot")
    done("3D plot saved → plots/")


    # ── STEP 10: Sample Predictions ───────────────────────
    section("SAMPLE PREDICTIONS FROM DATASET", 10)
    sample      = X_test.sample(5, random_state=99)
    predictions = predict_house_price(model, preprocessor, sample)
    actual_usd  = np.exp(y_test.loc[sample.index]).astype(int)
    predictions.insert(0, "actual_price_usd", actual_usd.values)
    predictions["error_%"] = (
        (predictions["predicted_price_usd"] - predictions["actual_price_usd"])
        / predictions["actual_price_usd"] * 100
    ).round(1)

    print()
    for i, (idx, row) in enumerate(predictions.iterrows(), 1):
        tick = "PASS" if abs(row["error_%"]) <= 10 else "MISS"
        print(f"  House {i}  (Row {idx})")
        print(f"    Actual    : ${int(row['actual_price_usd']):>10,}")
        print(f"    Predicted : ${int(row['predicted_price_usd']):>10,}  "
              f"({row['error_%']:+.1f}%)  [{tick}]")
        print(f"    Range     : ${int(row['lower_bound_usd']):,} – ${int(row['upper_bound_usd']):,}")
        print()
        time.sleep(0.2)

    done("Predictions complete")


    # ── FINAL SUMMARY ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE — SUMMARY")
    print("=" * 60)
    print(f"  R²  (test set)      : {metrics['r2']:.4f}")
    print(f"  MAE ($)             : ${metrics['mae_usd']:>10,.0f}")
    print(f"  RMSE ($)            : ${metrics['rmse_usd']:>10,.0f}")
    print(f"\n  Plots saved in → plots/")
    print(f"    • learning_curve_Multiple_Linear_Regression.png")
    print(f"    • predicted_vs_actual_MLR.png")
    print(f"    • residuals_MLR.png")
    print(f"    • cv_comparison_MLR.png")
    print(f"    • 3d_house_price_plot.png")
    print("\n  To predict a specific house → python predict.py")
    print("=" * 60 + "\n")


# ── Utility: show plot and wait until user closes the window ──
def _show_plot(path: str, title: str):
    """Open the saved PNG — pipeline continues when you close the window."""
    img = plt.imread(path)
    fig, ax = plt.subplots(figsize=(14, 6) if "3d" in path else (10, 5))
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    plt.tight_layout()
    print(f"  📊  Close the plot window to continue...")
    plt.show(block=True)   # blocks here until window is closed


if __name__ == "__main__":
    main()
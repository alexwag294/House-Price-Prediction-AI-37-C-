"""
main.py
───────
Full MLR pipeline with ALL 262 features. Run:  python main.py

Steps:
  1.  Load data          (262 features)
  2.  Train / Test split
  3.  Preprocess         (StandardScaler)
  4.  Train MLR          (Normal Equation)
  5.  Beta Coefficients  (β₀ + top 10 shown)
  6.  Evaluate           (R², MAE, RMSE)
  7.  Cross-Validation   (3, 5, 10-fold)
  8.  Plot — Learning Curve
  9.  Plot — Actual vs Predicted
  10. Plot — Residuals
  11. Plot — CV Comparison
  12. Plot — Top 15 Beta Coefficients
  13. Plot — 3D Visualisation
  14. Sample Predictions
"""

import os, time, numpy as np, matplotlib.pyplot as plt
os.makedirs("plots", exist_ok=True)

from data_loader   import load_data
from preprocessing import Preprocessor
from models        import get_mlr_model
from training      import (split_data, fit_model, print_betas,
                            run_cross_validation,
                            compute_and_plot_learning_curve)
from evaluation    import (evaluate_model, predict_house_price,
                            plot_predicted_vs_actual, plot_residuals,
                            plot_cv_comparison, plot_feature_importance)
from plot_3d       import plot_3d


def section(title, step, total=14):
    print(f"\n  STEP {step}/{total} — {title}")
    print("  " + "-"*50)

def done(msg):
    print(f"  {msg}")

def show_plot(path, title):
    """Show plot — close the window to continue to next step."""
    img = plt.imread(path)
    fig, ax = plt.subplots(figsize=(14 if "3d" in path else 10,
                                    7  if "3d" in path else 5))
    ax.imshow(img); ax.axis("off")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    plt.tight_layout()
    print(f"  📊  Close the plot window to continue...")
    plt.show(block=True)


def main():
    print("\n" + "="*60)
    print("   🏠  HOUSE PRICE PREDICTOR — MLR with 262 Features")
    print("="*60)

    # ── 1. Load ──────────────────────────────────────────────
    section("LOAD DATA", 1)
    X, y = load_data("data/train.csv")
    print(f"\n  Samples      : {X.shape[0]}")
    print(f"  Features     : {X.shape[1]}")
    print(f"  Price range  : ${np.exp(y.min()):,.0f}  →  ${np.exp(y.max()):,.0f}")
    done("Data loaded")

    # ── 2. Split ─────────────────────────────────────────────
    section("TRAIN / TEST SPLIT  (80% / 20%)", 2)
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"\n  Training : {len(X_train)} houses")
    print(f"  Test     : {len(X_test)} houses")
    done("Split complete")

    # ── 3. Preprocess ────────────────────────────────────────
    section("PREPROCESSING  (StandardScaler)", 3)
    preprocessor = Preprocessor()
    X_train_sc   = preprocessor.fit_transform(X_train)
    print(f"\n  Boolean columns → converted to int")
    print(f"  Missing values  → filled with median")
    print(f"  Features        → StandardScaler applied")
    done("Preprocessing complete")

    # ── 4. Train ─────────────────────────────────────────────
    section("TRAINING MODEL  (Normal Equation: β̂ = (XᵀX)⁻¹XᵀY)", 4)
    model = get_mlr_model()
    model = fit_model(model, X_train_sc, y_train)
    done("Model trained")

    # ── 5. Beta Coefficients ─────────────────────────────────
    section("BETA COEFFICIENTS  (top 10 of 262)", 5)
    print_betas(model, X.columns.tolist())
    done("Betas printed")

    # ── 6. Evaluate ──────────────────────────────────────────
    section("HOLD-OUT TEST SET RESULTS", 6)
    metrics, y_pred = evaluate_model(model, preprocessor, X_test, y_test)
    done("Evaluation complete")

    # ── 7. Cross-Validation ──────────────────────────────────
    section("CROSS-VALIDATION  (3, 5, 10-Fold)", 7)
    cv_results = run_cross_validation(X, y)
    done("Cross-validation complete")

    # ── 8. Learning Curve ────────────────────────────────────
    section("PLOT — LEARNING CURVE", 8)
    compute_and_plot_learning_curve(X, y)
    show_plot("plots/learning_curve_Multiple_Linear_Regression.png", "Learning Curve")
    done("Plot done")

    # ── 9. Actual vs Predicted ───────────────────────────────
    section("PLOT — ACTUAL vs PREDICTED", 9)
    plot_predicted_vs_actual(y_test, y_pred)
    show_plot("plots/predicted_vs_actual_MLR.png", "Actual vs Predicted")
    done("Plot done")

    # ── 10. Residuals ────────────────────────────────────────
    section("PLOT — RESIDUALS", 10)
    plot_residuals(y_test, y_pred)
    show_plot("plots/residuals_MLR.png", "Residual Analysis")
    done("Plot done")

    # ── 11. CV Comparison ────────────────────────────────────
    section("PLOT — CROSS-VALIDATION COMPARISON", 11)
    plot_cv_comparison(cv_results)
    show_plot("plots/cv_comparison_MLR.png", "CV Comparison")
    done("Plot done")

    # ── 12. Beta Coefficients Chart ──────────────────────────
    section("PLOT — TOP 15 BETA COEFFICIENTS", 12)
    plot_feature_importance(model, X.columns.tolist())
    show_plot("plots/feature_importance_MLR.png", "Top 15 Beta Coefficients")
    done("Plot done")

    # ── 13. 3D Plot ──────────────────────────────────────────
    section("PLOT — 3D VISUALISATION", 13)
    plot_3d(model, preprocessor, X, y)
    show_plot("plots/3d_house_price_plot.png", "3D House Price Plot")
    done("Plot done")

    # ── 14. Sample Predictions ───────────────────────────────
    section("SAMPLE PREDICTIONS FROM DATASET", 14)
    sample      = X_test.sample(5, random_state=99)
    predictions = predict_house_price(model, preprocessor, sample)
    actual_usd  = np.exp(y_test.loc[sample.index]).astype(int)
    predictions.insert(0, "actual_price_usd", actual_usd.values)
    predictions["error_%"] = (
        (predictions["predicted_price_usd"] - predictions["actual_price_usd"])
        / predictions["actual_price_usd"] * 100).round(1)

    print()
    for i, (idx, row) in enumerate(predictions.iterrows(), 1):
        tick = "PASS" if abs(row["error_%"]) <= 10 else "MISS"
        print(f"  House {i}  (Row {idx})")
        print(f"    Actual    : ${int(row['actual_price_usd']):>10,}")
        print(f"    Predicted : ${int(row['predicted_price_usd']):>10,}  ({row['error_%']:+.1f}%)  [{tick}]")
        print(f"    Range     : ${int(row['lower_bound_usd']):,} – ${int(row['upper_bound_usd']):,}")
        print()
        time.sleep(0.15)
    done("Predictions complete")

    # ── Summary ──────────────────────────────────────────────
    print("\n" + "="*60)
    print("  PIPELINE COMPLETE — SUMMARY")
    print("="*60)
    print(f"  Features used   : 262  (all features)")
    print(f"  R²  (test set)  : {metrics['r2']:.4f}")
    print(f"  MAE  ($)        : ${metrics['mae_usd']:>10,.0f}")
    print(f"  RMSE ($)        : ${metrics['rmse_usd']:>10,.0f}")
    print(f"\n  Plots saved in  → plots/")
    print(f"    • learning_curve_Multiple_Linear_Regression.png")
    print(f"    • predicted_vs_actual_MLR.png")
    print(f"    • residuals_MLR.png")
    print(f"    • cv_comparison_MLR.png")
    print(f"    • feature_importance_MLR.png")
    print(f"    • 3d_house_price_plot.png")
    print(f"\n  To predict a house → python predict.py")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
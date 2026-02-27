"""
main.py
───────
Entry point. Orchestrates the full MLR pipeline:

    data_loader   →  preprocessing  →  models
         ↓                                ↓
    training  (fit + CV + learning curve)
         ↓
    evaluation  (metrics + predict + plots)

Run:
    python main.py
"""

import os
os.makedirs("plots", exist_ok=True)

# ── Local modules ──────────────────────────────────────────
from data_loader   import load_data
from preprocessing import Preprocessor
from models        import get_mlr_model
from training      import (split_data, fit_model,
                            run_cross_validation,
                            compute_and_plot_learning_curve)
from evaluation    import (evaluate_model, predict_house_price,
                            plot_predicted_vs_actual,
                            plot_residuals, plot_cv_comparison)


def main():
    print("\n" + "═"*56)
    print("   HOUSE PRICE PREDICTOR — Multiple Linear Regression")
    print("═"*56 + "\n")

    # ── 1. Load ──────────────────────────────────────────────
    X, y = load_data("data/train.csv")

    # ── 2. Split ─────────────────────────────────────────────
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)

    # ── 3. Preprocess ────────────────────────────────────────
    preprocessor  = Preprocessor()
    X_train_sc    = preprocessor.fit_transform(X_train)

    # ── 4. Model ─────────────────────────────────────────────
    model = get_mlr_model()

    # ── 5. Train ─────────────────────────────────────────────
    model = fit_model(model, X_train_sc, y_train)

    # ── 6. Evaluate (hold-out) ───────────────────────────────
    metrics, y_pred = evaluate_model(model, preprocessor, X_test, y_test)

    # ── 7. Cross-Validation (3, 5, 10-fold) ──────────────────
    cv_results = run_cross_validation(X, y, cv_folds=[3, 5, 10])

    # ── 8. Learning Curve ────────────────────────────────────
    compute_and_plot_learning_curve(X, y, name="Multiple Linear Regression")

    # ── 9. Diagnostic Plots ──────────────────────────────────
    plot_predicted_vs_actual(y_test, y_pred)
    plot_residuals(y_test, y_pred)
    plot_cv_comparison(cv_results)

    # ── 10. Sample Predictions (5 random test houses) ────────
    sample      = X_test.sample(5, random_state=99)
    predictions = predict_house_price(model, preprocessor, sample)
    import numpy as np
    predictions.insert(0, "actual_price_usd",
                       np.exp(y_test.loc[sample.index]).astype(int).values)

    print("\n" + "═"*64)
    print("   SAMPLE PREDICTIONS  (5 test houses)")
    print("═"*64)
    print(predictions.to_string(index=False))
    print("═"*64)

    print("\n✅  All done!  Plots saved in → plots/\n")


if __name__ == "__main__":
    main()
# main.py
import os
from data_loader   import load_dataset
from preprocessing import create_preprocessor, NUMERIC_FEATURES
from models        import get_models
from training      import train_and_evaluate
from evaluation    import (display_results, plot_comparison,
                           plot_predicted_vs_actual, plot_residuals,
                           plot_feature_importance)

DATA_PATH = "data/train.csv"

os.makedirs("plots", exist_ok=True)

# 3.1 Load dataset
df = load_dataset(DATA_PATH)

X = df[NUMERIC_FEATURES]
y = df["SalePrice"]

# Train all models
models        = get_models()
results       = {}
results_preds = {}
best_model    = None
best_r2       = -1
best_Xtest    = None
best_ytest    = None

for name, model in models.items():
    print(f"\n{'='*50}\nTraining: {name}\n{'='*50}")

    trained_model, r2, rmse, cv, params, y_test, y_pred = train_and_evaluate(
        name, model, create_preprocessor(), X, y
    )

    results[name]       = {"R2": r2, "RMSE": rmse, "CV": cv, "Params": params}
    results_preds[name] = (y_test, y_pred)

    if r2 > best_r2:
        best_r2    = r2
        best_model = trained_model
        best_Xtest = X.loc[y_test.index]
        best_ytest = y_test

# 4.2 Results table
display_results(results)

# 4.3 Visualizations
plot_comparison(results)
plot_predicted_vs_actual(results_preds, "Random Forest")
plot_residuals(results_preds, "Random Forest")
plot_feature_importance(best_model, best_Xtest, best_ytest)

print("\n✅ All plots saved to /plots directory.")
print(f"✅ Best model: Random Forest  |  R² = {best_r2:.3f}")
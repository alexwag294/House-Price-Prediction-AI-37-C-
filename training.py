# training.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error


def train_and_evaluate(name, model, preprocessor, X, y):
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    best_params = "Default"

    # ── Random Forest: GridSearch + tuning table/plot ──────────────────────
    if name == "Random Forest":
        param_grid = {
            "regressor__n_estimators": [100, 200],
            "regressor__max_depth":    [None, 10, 20]
        }
        grid = GridSearchCV(pipeline, param_grid, cv=3, scoring="r2", n_jobs=-1)
        grid.fit(X_train, y_train)
        pipeline    = grid.best_estimator_
        best_params = grid.best_params_

        # ── Hyperparameter tuning results table ────────────────────────────
        print("\n===== RANDOM FOREST HYPERPARAMETER TUNING TABLE =====")
        cv_results = pd.DataFrame(grid.cv_results_)
        tuning_table = cv_results[[
            "param_regressor__n_estimators",
            "param_regressor__max_depth",
            "mean_test_score",
            "std_test_score",
            "rank_test_score"
        ]].sort_values("rank_test_score")
        tuning_table.columns = ["n_estimators", "max_depth", "Mean CV R²", "Std", "Rank"]
        print(tuning_table.to_string(index=False))

        # ── Hyperparameter tuning bar chart ────────────────────────────────
        labels = [
            f"n={r['n_estimators']}\ndepth={r['max_depth']}"
            for _, r in tuning_table.iterrows()
        ]
        scores = tuning_table["Mean CV R²"].values

        plt.figure(figsize=(10, 5))
        bars = plt.bar(labels, scores, color="steelblue", edgecolor="black")
        best_idx = np.argmax(scores)
        bars[best_idx].set_color("darkorange")
        plt.title("Random Forest: Hyperparameter Tuning Results (CV R²)")
        plt.ylabel("Mean CV R² Score")
        plt.xlabel("Parameter Combination")
        plt.ylim(min(scores) - 0.02, max(scores) + 0.02)
        plt.xticks(fontsize=8)
        for bar, score in zip(bars, scores):
            plt.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.002,
                     f"{score:.3f}", ha="center", va="bottom", fontsize=8)
        plt.tight_layout()
        plt.savefig("plots/hyperparameter_tuning.png", dpi=150)
        plt.show()

    else:
        pipeline.fit(X_train, y_train)

    # ── Learning curve (training progress / loss curve equivalent) ─────────
    plot_learning_curve(pipeline, name, X, y)

    y_pred = pipeline.predict(X_test)

    r2   = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    cv_score = cross_val_score(pipeline, X, y, cv=5, scoring="r2").mean()

    return pipeline, r2, rmse, cv_score, best_params, y_test, y_pred


def plot_learning_curve(pipeline, name, X, y):
    """Plots training vs. validation R² as training size increases (loss-curve equivalent)."""
    train_sizes, train_scores, val_scores = learning_curve(
        pipeline, X, y,
        cv=5,
        scoring="r2",
        train_sizes=np.linspace(0.1, 1.0, 8),
        n_jobs=-1
    )

    train_mean = train_scores.mean(axis=1)
    val_mean   = val_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    val_std    = val_scores.std(axis=1)

    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, train_mean, "o-", color="blue",  label="Training R²")
    plt.plot(train_sizes, val_mean,   "o-", color="green", label="Validation R²")
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color="blue")
    plt.fill_between(train_sizes, val_mean   - val_std,   val_mean   + val_std,   alpha=0.15, color="green")
    plt.title(f"Learning Curve — {name}")
    plt.xlabel("Training Set Size")
    plt.ylabel("R² Score")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"plots/learning_curve_{name.replace(' ', '_')}.png", dpi=150)
    plt.show()
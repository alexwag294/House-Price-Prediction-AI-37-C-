"""
models.py
─────────
Returns the Multiple Linear Regression model.

With 262 features the model equation becomes:
  Y = β₀ + β₁X₁ + β₂X₂ + ... + β₂₆₂X₂₆₂ + ε

Beta coefficients solved using the Normal Equation:
  β̂ = (XᵀX)⁻¹XᵀY
"""

from sklearn.linear_model import LinearRegression


def get_mlr_model() -> LinearRegression:
    model = LinearRegression(fit_intercept=True, n_jobs=-1)
    print("[Models] Multiple Linear Regression model created")
    print("         Y = β₀ + β₁X₁ + β₂X₂ + ... + β₂₆₂X₂₆₂ + ε")
    print("         β̂ = (XᵀX)⁻¹XᵀY")
    return model
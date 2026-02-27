"""
models.py
─────────
Returns the Multiple Linear Regression model.

Model equation:
  Y = β₀ + β₁X₁ + β₂X₂ + ... + βₙXₙ + ε

Beta coefficients solved via the Normal Equation:
  β̂ = (XᵀX)⁻¹XᵀY
"""

from sklearn.linear_model import LinearRegression


def get_mlr_model() -> LinearRegression:
    model = LinearRegression(fit_intercept=True, n_jobs=-1)
    print("[Models] Multiple Linear Regression model created")
    print("         Y = β₀ + β₁X₁ + β₂X₂ + ... + βₙXₙ + ε")
    print("         β̂ = (XᵀX)⁻¹XᵀY")
    return model
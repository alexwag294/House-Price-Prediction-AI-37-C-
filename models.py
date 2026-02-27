"""
models.py
─────────
Defines and returns the Multiple Linear Regression model.
Keeping model definitions here makes it easy to swap or
add models in the future without touching training logic.
"""

from sklearn.linear_model import LinearRegression


def get_mlr_model() -> LinearRegression:
    """
    Return a configured Multiple Linear Regression model.

    Returns
    -------
    model : LinearRegression
    """
    model = LinearRegression(
        fit_intercept=True,
        copy_X=True,
        n_jobs=-1,          # use all CPU cores
    )
    print("[Models] Multiple Linear Regression model created")
    return model
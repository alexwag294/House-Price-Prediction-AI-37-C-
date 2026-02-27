"""
plot_3d.py
──────────
3D visualisation of the MLR model predictions.

Three panels:
  1. Actual prices + regression surface  (Quality x Living Area x Price)
  2. Predicted prices + regression surface
  3. Actual vs Predicted coloured by % error

Called automatically by main.py — no need to run this directly.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


def plot_3d(model, preprocessor, X, y):
    """
    Generate and save the 3D house price visualisation.

    Parameters
    ----------
    model        : fitted LinearRegression
    preprocessor : fitted Preprocessor
    X            : full feature DataFrame
    y            : full target Series (log scale)
    """
    print("[3D Plot] Generating 3D visualisation...")

    # Get predictions for the full dataset
    X_clean   = X.copy()
    bool_cols = X_clean.select_dtypes("bool").columns
    X_clean[bool_cols] = X_clean[bool_cols].astype(int)
    X_clean   = X_clean.fillna(X_clean.median(numeric_only=True))
    X_sc_all  = preprocessor.scaler.transform(X_clean)
    y_pred_all = model.predict(X_sc_all)

    actual_price    = np.exp(y.values)
    predicted_price = np.exp(y_pred_all)

    x_vals = X["OverallQual"].values
    y_vals = X["GrLivArea"].values

    # Build regression surface (Quality + Area only, for visualisation)
    surf_X    = np.column_stack([x_vals, y_vals])
    sc_surf   = StandardScaler()
    surf_X_sc = sc_surf.fit_transform(surf_X)
    surf_model = LinearRegression()
    surf_model.fit(surf_X_sc, np.log(actual_price))

    x_range = np.linspace(x_vals.min(), x_vals.max(), 40)
    y_range = np.linspace(y_vals.min(), y_vals.max(), 40)
    xx, yy  = np.meshgrid(x_range, y_range)
    grid    = np.column_stack([xx.ravel(), yy.ravel()])
    zz      = np.exp(surf_model.predict(sc_surf.transform(grid))).reshape(xx.shape)

    # Figure with 3 panels
    fig = plt.figure(figsize=(20, 8))
    fig.suptitle("3D House Price Visualisation — Multiple Linear Regression",
                 fontsize=14, fontweight="bold", y=1.01)

    # Panel 1 — Actual prices + surface
    ax1 = fig.add_subplot(131, projection="3d")
    sc1 = ax1.scatter(x_vals, y_vals, actual_price / 1000,
                      c=actual_price / 1000, cmap="plasma",
                      s=12, alpha=0.5, depthshade=True)
    ax1.plot_surface(xx, yy, zz / 1000, alpha=0.25,
                     color="deepskyblue", edgecolor="none")
    ax1.set_xlabel("Overall Quality", labelpad=8)
    ax1.set_ylabel("Living Area (sqft)", labelpad=8)
    ax1.set_zlabel("Price ($K)", labelpad=8)
    ax1.set_title("Actual Prices\n+ Regression Surface", pad=12)
    fig.colorbar(sc1, ax=ax1, shrink=0.45, pad=0.1, label="Price ($K)")
    ax1.view_init(elev=22, azim=-55)

    # Panel 2 — Predicted prices + surface
    ax2 = fig.add_subplot(132, projection="3d")
    sc2 = ax2.scatter(x_vals, y_vals, predicted_price / 1000,
                      c=predicted_price / 1000, cmap="viridis",
                      s=12, alpha=0.5, depthshade=True)
    ax2.plot_surface(xx, yy, zz / 1000, alpha=0.25,
                     color="coral", edgecolor="none")
    ax2.set_xlabel("Overall Quality", labelpad=8)
    ax2.set_ylabel("Living Area (sqft)", labelpad=8)
    ax2.set_zlabel("Predicted Price ($K)", labelpad=8)
    ax2.set_title("Predicted Prices\n+ Regression Surface", pad=12)
    fig.colorbar(sc2, ax=ax2, shrink=0.45, pad=0.1, label="Price ($K)")
    ax2.view_init(elev=22, azim=-55)

    # Panel 3 — Actual vs Predicted coloured by % error
    ax3 = fig.add_subplot(133, projection="3d")
    error_pct = (predicted_price - actual_price) / actual_price * 100
    sc3 = ax3.scatter(actual_price / 1000, predicted_price / 1000,
                      error_pct, c=error_pct, cmap="RdYlGn_r",
                      vmin=-40, vmax=40, s=12, alpha=0.6, depthshade=True)
    lim = [min(actual_price.min(), predicted_price.min()) / 1000 * 0.95,
           max(actual_price.max(), predicted_price.max()) / 1000 * 1.05]
    ax3.plot(lim, lim, [0, 0], color="black", lw=1.5,
             linestyle="--", label="Perfect fit")
    ax3.set_xlabel("Actual Price ($K)", labelpad=8)
    ax3.set_ylabel("Predicted Price ($K)", labelpad=8)
    ax3.set_zlabel("Error (%)", labelpad=8)
    ax3.set_title("Actual vs Predicted\nColoured by % Error", pad=12)
    fig.colorbar(sc3, ax=ax3, shrink=0.45, pad=0.1, label="Error (%)")
    ax3.legend(fontsize=7, loc="upper left")
    ax3.view_init(elev=22, azim=45)

    plt.tight_layout()
    plt.savefig("plots/3d_house_price_plot.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[3D Plot] Saved -> plots/3d_house_price_plot.png")
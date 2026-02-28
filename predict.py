"""
predict.py
──────────
Predict house prices using 8 key features as user input.
No value restrictions — enter any number.

Run:  python predict.py
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

FEATURES = [
    ("OverallQual",  "Overall Quality      (1–10)",    6),
    ("GrLivArea",    "Above Ground Area    (sq ft)",   1462),
    ("TotalSF",      "Total Square Feet    (sq ft)",   2473),
    ("GarageCars",   "Garage Capacity      (cars)",    2),
    ("TotalBath",    "Total Bathrooms      (e.g 2.5)", 2),
    ("GarageArea",   "Garage Area          (sq ft)",   480),
    ("YearBuilt",    "Year Built",                     1973),
    ("TotalBsmtSF",  "Basement Area        (sq ft)",   991),
]

# ── Load & Train ──────────────────────────────────────────
print("\n" + "="*58)
print("   🏠  HOUSE PRICE PREDICTOR  —  MLR Model")
print("="*58)
print("\n  Loading data and training model...")

df    = pd.read_csv("data/train.csv")
y_log = df["SalePrice"]
X_raw = df.drop(columns=["SalePrice"])
bool_cols = X_raw.select_dtypes("bool").columns
X_raw[bool_cols] = X_raw[bool_cols].astype(int)
X_filled = X_raw.fillna(X_raw.median(numeric_only=True))
medians  = X_filled.median(numeric_only=True)
all_cols = X_filled.columns.tolist()

X_train, _, y_train, _ = train_test_split(
    X_filled, y_log, test_size=0.2, random_state=42)

scaler = StandardScaler()
model  = LinearRegression(n_jobs=-1)
model.fit(scaler.fit_transform(X_train), y_train)
print("  ✅  Model ready!\n")

# ── Predict Loop ──────────────────────────────────────────
go = True
while go:
    print("="*58)
    print("  Enter house details  (press Enter for default)")
    print("="*58)

    user = {}
    for feat, label, default in FEATURES:
        while True:
            try:
                raw = input(f"  {label} [{default}]: ").strip()
                user[feat] = float(raw) if raw else default
                break
            except ValueError:
                print("    ⚠  Please enter a number")

    # Build full feature row from medians, override the 8
    row = medians.to_dict()
    for k, v in user.items():
        if k in row:
            row[k] = v

    inp      = pd.DataFrame([row])[all_cols]
    log_pred = model.predict(scaler.transform(inp))[0]
    price    = int(np.exp(log_pred))
    low      = int(price * 0.90)
    high     = int(price * 1.10)
    per_sf   = price // int(user["TotalSF"]) if user["TotalSF"] > 0 else 0

    print("\n" + "="*58)
    print("   🏠  PREDICTION RESULT")
    print("="*58)
    print(f"\n  OverallQual : {int(user['OverallQual'])}/10"
          f"   |  GrLivArea  : {int(user['GrLivArea']):,} sq ft")
    print(f"  TotalSF     : {int(user['TotalSF']):,} sq ft"
          f"  |  GarageCars : {int(user['GarageCars'])} car(s)")
    print(f"  TotalBath   : {user['TotalBath']}"
          f"        |  GarageArea : {int(user['GarageArea']):,} sq ft")
    print(f"  YearBuilt   : {int(user['YearBuilt'])}"
          f"      |  TotalBsmtSF: {int(user['TotalBsmtSF']):,} sq ft")
    print()
    print(f"  {'─'*54}")
    print(f"  Predicted   : ${price:>12,}")
    print(f"  Range       : ${low:,}  –  ${high:,}")
    print(f"  Per sq ft   : ${per_sf:,}")
    print(f"  {'─'*54}\n")

    go = input("  Predict another house? (y/n) [n]: ").strip().lower() == "y"
    print()

print("="*58 + "\n")
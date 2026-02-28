"""
predict.py  —  STANDALONE VERSION
──────────────────────────────────
Does NOT depend on any other file in the project.
Just needs:  data/train.csv

Run:  python predict.py
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ══════════════════════════════════════════════════════════
#  STEP 1 — Load & prepare data
# ══════════════════════════════════════════════════════════
print("\n[predict.py] Loading data and training model...")

df     = pd.read_csv("data/train.csv")
TARGET = "SalePrice"

y_log  = df[TARGET]                 # log-scale target (~12.2 = $208,500)
X      = df.drop(columns=[TARGET])

# Convert booleans to int, fill missing
bool_cols    = X.select_dtypes("bool").columns
X[bool_cols] = X[bool_cols].astype(int)
X            = X.fillna(X.median(numeric_only=True))

# ══════════════════════════════════════════════════════════
#  STEP 2 — Train model
# ══════════════════════════════════════════════════════════
X_train, X_test, y_train, y_test = train_test_split(
    X, y_log, test_size=0.2, random_state=42)

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)

model = LinearRegression()
model.fit(X_train_sc, y_train)

print("[predict.py] Model trained!\n")

# ══════════════════════════════════════════════════════════
#  STEP 3 — Pick mode
# ══════════════════════════════════════════════════════════
MODE        = "single"
ROW_INDEX   = 0
ROW_INDICES = [0, 5, 42, 100, 500]
RANDOM_N    = 5

if MODE == "single":
    sel_X = X.iloc[[ROW_INDEX]]
    sel_y = y_log.iloc[[ROW_INDEX]]
elif MODE == "multiple":
    sel_X = X.iloc[ROW_INDICES]
    sel_y = y_log.iloc[ROW_INDICES]
elif MODE == "random":
    rng   = np.random.default_rng(42)
    idxs  = rng.choice(len(X), size=RANDOM_N, replace=False)
    sel_X = X.iloc[idxs]
    sel_y = y_log.iloc[idxs]

# ══════════════════════════════════════════════════════════
#  STEP 4 — Predict and convert log price → real dollars
# ══════════════════════════════════════════════════════════
X_sc      = scaler.transform(sel_X)
log_preds = model.predict(X_sc)

# np.exp() is the key — converts log(price) back to $
actual_usd    = np.exp(sel_y.values).astype(int)
predicted_usd = np.exp(log_preds).astype(int)
lower_usd     = (np.exp(log_preds) * 0.90).astype(int)
upper_usd     = (np.exp(log_preds) * 1.10).astype(int)
error_pct     = ((predicted_usd - actual_usd) / actual_usd * 100).round(1)

# ══════════════════════════════════════════════════════════
#  STEP 5 — Print results
# ══════════════════════════════════════════════════════════
print("=" * 58)
print("   HOUSE PRICE PREDICTION RESULTS")
print("=" * 58)

for i, (idx, actual, predicted, lower, upper, err) in enumerate(
        zip(sel_X.index, actual_usd, predicted_usd,
            lower_usd, upper_usd, error_pct), 1):

    tick  = "PASS" if abs(err) <= 10 else "MISS"
    house = sel_X.loc[idx]

    print(f"\n  House {i}  (Row {idx})")
    print(f"  TotalSF: {int(house['TotalSF']):,}  |  "
          f"Quality: {int(house['OverallQual'])}/10  |  "
          f"GrLivArea: {int(house['GrLivArea']):,}")
    print(f"  GarageCars: {int(house['GarageCars'])}  |  "
          f"TotalBath: {house['TotalBath']}  |  "
          f"GarageArea: {int(house['GarageArea'])}")
    print(f"  {'─' * 47}")
    print(f"  Actual    :  ${actual:>10,}")
    print(f"  Predicted :  ${predicted:>10,}  ({err:+.1f}%)  [{tick}]")
    print(f"  Range     :  ${lower:,}  --  ${upper:,}")

if len(sel_X) > 1:
    within = sum(abs(e) <= 10 for e in error_pct)
    print(f"\n{'=' * 58}")
    print(f"  Within +-10%  :  {within}/{len(sel_X)} houses")
    print(f"  Mean Error    :  {abs(error_pct).mean():.1f}%")

print("=" * 58 + "\n")
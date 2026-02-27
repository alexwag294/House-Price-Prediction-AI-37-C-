"""
predict.py
──────────
Predict house prices using real rows from train.csv.
Uses only the 8 selected features.

HOW TO USE:
    MODE = "single"    → set ROW_INDEX  (0 to 1457)
    MODE = "multiple"  → set ROW_INDICES list
    MODE = "random"    → set RANDOM_N

Run:  python predict.py
"""

import numpy as np, pandas as pd
from data_loader   import load_data
from preprocessing import Preprocessor
from models        import get_mlr_model
from training      import split_data, fit_model
from evaluation    import predict_house_price

# ── CONFIGURE ────────────────────────────────────────────
MODE        = "single"
ROW_INDEX   = 0
ROW_INDICES = [0, 5, 42, 100, 500]
RANDOM_N    = 5
RANDOM_SEED = 42

# ── Train ─────────────────────────────────────────────────
print("\n[predict.py] Training model on 8 features...")
X, y          = load_data("data/train.csv")
X_train, X_test, y_train, y_test = split_data(X, y)
preprocessor  = Preprocessor()
X_train_sc    = preprocessor.fit_transform(X_train)
model         = get_mlr_model()
model         = fit_model(model, X_train_sc, y_train)
print("[predict.py] Ready!\n")

# ── Select rows ───────────────────────────────────────────
if MODE == "single":
    sel_X = X.iloc[[ROW_INDEX]]; sel_y = y.iloc[[ROW_INDEX]]
elif MODE == "multiple":
    sel_X = X.iloc[ROW_INDICES]; sel_y = y.iloc[ROW_INDICES]
elif MODE == "random":
    rng   = np.random.default_rng(RANDOM_SEED)
    idxs  = rng.choice(len(X), size=RANDOM_N, replace=False)
    sel_X = X.iloc[idxs]; sel_y = y.iloc[idxs]

# ── Predict ───────────────────────────────────────────────
result = predict_house_price(model, preprocessor, sel_X)
result.insert(0, "actual_price_usd", np.exp(sel_y.values).astype(int))
result["error_%"]    = ((result["predicted_price_usd"] - result["actual_price_usd"])
                        / result["actual_price_usd"] * 100).round(1)
result["within_10%"] = result["error_%"].abs() <= 10

# ── Print ─────────────────────────────────────────────────
print("="*58)
print("   🏠  HOUSE PRICE PREDICTION RESULTS")
print("="*58)
for i, (idx, row) in enumerate(result.iterrows(), 1):
    tick = "PASS" if row["within_10%"] else "MISS"
    house = sel_X.loc[idx]
    print(f"\n  House {i}  (Row {idx})")
    print(f"  TotalSF: {int(house['TotalSF']):,}  |  Quality: {int(house['OverallQual'])}/10  "
          f"|  GrLivArea: {int(house['GrLivArea']):,}")
    print(f"  GarageCars: {int(house['GarageCars'])}  |  TotalBath: {house['TotalBath']}  "
          f"|  GarageArea: {int(house['GarageArea'])}")
    print(f"  -----------------------------------------------")
    print(f"  Actual    : ${int(row['actual_price_usd']):>10,}")
    print(f"  Predicted : ${int(row['predicted_price_usd']):>10,}  ({row['error_%']:+.1f}%)  [{tick}]")
    print(f"  Range     : ${int(row['lower_bound_usd']):,} – ${int(row['upper_bound_usd']):,}")

if len(result) > 1:
    acc = result["within_10%"].sum()
    mae = result["error_%"].abs().mean()
    print(f"\n{'='*58}")
    print(f"  Within ±10%  : {acc}/{len(result)}")
    print(f"  Mean Error   : {mae:.1f}%")
print("="*58 + "\n")
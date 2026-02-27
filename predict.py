"""
predict.py
──────────
Predict house prices using REAL rows from train.csv.

HOW TO USE:
    Option A — Predict a specific row by index:
        Set MODE = "single" and change ROW_INDEX to any number (0 to 1457)

    Option B — Predict multiple rows at once:
        Set MODE = "multiple" and add index numbers to ROW_INDICES list

    Option C — Predict a random sample:
        Set MODE = "random" and change RANDOM_N to however many you want

Then run:
    python predict.py
"""

import numpy as np
import pandas as pd

from data_loader   import load_data
from preprocessing import Preprocessor
from models        import get_mlr_model
from training      import split_data, fit_model
from evaluation    import predict_house_price


# ══════════════════════════════════════════════════════════
#  ✏️  CONFIGURE YOUR PREDICTION HERE
# ══════════════════════════════════════════════════════════

MODE = "single"       # Options: "single" | "multiple" | "random"

ROW_INDEX  = 0        # Used when MODE = "single"   (0 to 1457)

ROW_INDICES = [0, 5, 42, 100, 500]   # Used when MODE = "multiple"

RANDOM_N   = 5        # Used when MODE = "random"   (how many houses)
RANDOM_SEED = 42      # Change for different random picks


# ══════════════════════════════════════════════════════════
#  Train model on the same dataset
# ══════════════════════════════════════════════════════════
print("\n" + "="*54)
print("   HOUSE PRICE PREDICTOR -- Using train.csv")
print("="*54)
print("\n[Step 1] Loading data and training model...")

X, y          = load_data("data/train.csv")
X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
preprocessor  = Preprocessor()
X_train_sc    = preprocessor.fit_transform(X_train)
model         = get_mlr_model()
model         = fit_model(model, X_train_sc, y_train)

print("[Step 1] Done!\n")


# ══════════════════════════════════════════════════════════
#  Select rows from the dataset based on MODE
# ══════════════════════════════════════════════════════════
print(f"[Step 2] Selecting houses (MODE = '{MODE}')...")

if MODE == "single":
    selected_X = X.iloc[[ROW_INDEX]]
    selected_y = y.iloc[[ROW_INDEX]]

elif MODE == "multiple":
    selected_X = X.iloc[ROW_INDICES]
    selected_y = y.iloc[ROW_INDICES]

elif MODE == "random":
    rng        = np.random.default_rng(RANDOM_SEED)
    idxs       = rng.choice(len(X), size=RANDOM_N, replace=False)
    selected_X = X.iloc[idxs]
    selected_y = y.iloc[idxs]

else:
    raise ValueError(f"Unknown MODE '{MODE}'. Choose: single | multiple | random")

print(f"[Step 2] Selected {len(selected_X)} house(s)\n")


# ══════════════════════════════════════════════════════════
#  Predict
# ══════════════════════════════════════════════════════════
print("[Step 3] Predicting prices...\n")

result = predict_house_price(model, preprocessor, selected_X)

# Add actual price for comparison
result.insert(0, "actual_price_usd", np.exp(selected_y.values).astype(int))

# Add error columns
result["error_usd"]  = result["predicted_price_usd"] - result["actual_price_usd"]
result["error_pct"]  = (result["error_usd"] / result["actual_price_usd"] * 100).round(1)
result["within_10%"] = result["error_pct"].abs() <= 10


# ══════════════════════════════════════════════════════════
#  Print results
# ══════════════════════════════════════════════════════════
print("="*60)
print("   HOUSE PRICE PREDICTION RESULTS")
print("="*60)

for pos, (df_idx, row) in enumerate(result.iterrows()):
    actual    = row["actual_price_usd"]
    predicted = row["predicted_price_usd"]
    lower     = row["lower_bound_usd"]
    upper     = row["upper_bound_usd"]
    err       = row["error_usd"]
    err_pct   = row["error_pct"]
    tick      = "PASS" if row["within_10%"] else "MISS"

    # Pull key readable features from the original row
    house = selected_X.loc[df_idx]
    qual  = int(house["OverallQual"])
    area  = int(house["GrLivArea"])
    yr    = int(house["YearBuilt"])
    beds  = int(house["BedroomAbvGr"])
    baths = house["TotalBath"]
    cars  = int(house["GarageCars"])

    print(f"\n  House (Row {df_idx})")
    print(f"  Quality: {qual}/10  |  Living Area: {area:,} sqft  |  Built: {yr}")
    print(f"  Bedrooms: {beds}  |  Bathrooms: {baths}  |  Garage: {cars} cars")
    print(f"  ---------------------------------------------------")
    print(f"  Actual Price    :  ${actual:>10,}")
    print(f"  Predicted Price :  ${predicted:>10,}")
    print(f"  Price Range     :  ${lower:,} -- ${upper:,}")
    print(f"  Error           :  ${err:>+10,}  ({err_pct:+.1f}%)  [{tick}]")

# Summary for multiple predictions
if len(result) > 1:
    acc = result["within_10%"].sum()
    mae = result["error_usd"].abs().mean()
    print(f"\n{'='*60}")
    print(f"  SUMMARY for {len(result)} houses:")
    print(f"    Within +-10% error : {acc}/{len(result)} houses")
    print(f"    Mean Abs Error     : ${mae:,.0f}")
    print("="*60)

print()
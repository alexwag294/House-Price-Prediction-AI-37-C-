# preprocessing.py
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

NUMERIC_FEATURES = [
    "GrLivArea", "BedroomAbvGr", "FullBath",
    "GarageCars", "GarageArea",
    "OverallQual", "LotArea", "YearBuilt"
]

CATEGORICAL_FEATURES = []

def create_preprocessor():
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, NUMERIC_FEATURES)
    ])

    return preprocessor
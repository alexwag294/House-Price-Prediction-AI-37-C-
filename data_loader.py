
import pandas as pd


def load_data(filepath: str = "data/train.csv") -> tuple[pd.DataFrame, pd.Series]:
    """
    Load the house price dataset.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.

    Returns
    -------
    X : pd.DataFrame  — feature matrix
    y : pd.Series     — log-transformed SalePrice (target)
    """
    df = pd.read_csv(filepath)

    if "SalePrice" not in df.columns:
        raise ValueError("CSV must contain a 'SalePrice' column.")

    y = df["SalePrice"]           # already log-transformed in this dataset
    X = df.drop(columns=["SalePrice"])

    print(f"[DataLoader] Loaded  →  {X.shape[0]} rows  |  {X.shape[1]} features")
    return X, y
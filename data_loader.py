import pandas as pd

def load_dataset(path):
    df = pd.read_csv(path)
    print("\n===== DATASET INFORMATION =====")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("\nFirst 5 rows:\n", df.head())
    return df
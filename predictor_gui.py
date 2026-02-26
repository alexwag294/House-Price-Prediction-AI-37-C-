# predictor_gui.py
import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

NUMERIC_FEATURES = [
    "GrLivArea", "BedroomAbvGr", "FullBath",
    "GarageCars", "GarageArea",
    "OverallQual", "LotArea", "YearBuilt"
]

class HousePriceGUI:

    def __init__(self, root):
        self.root = root
        self.root.title("🏠 House Price Predictor")
        self.root.geometry("500x620")
        self.root.configure(bg="#1e1e2f")
        self.model = None

        self.build_ui()
        self.train_model()

    def train_model(self):
        try:
            df = pd.read_csv("data/train.csv")
            X = df[NUMERIC_FEATURES]
            y = df["SalePrice"]

            numeric_transformer = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            preprocessor = ColumnTransformer([
                ("num", numeric_transformer, NUMERIC_FEATURES)
            ])

            self.model = Pipeline([
                ("preprocessor", preprocessor),
                ("regressor", RandomForestRegressor(
                    n_estimators=200, random_state=42))
            ])

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            self.model.fit(X_train, y_train)
            r2 = self.model.score(X_test, y_test)

            self.status_label.config(
                text=f"✅ Model Ready  |  R² = {r2:.3f}",
                fg="#00ffcc"
            )
        except Exception as e:
            messagebox.showerror("Error", f"Could not train model:\n{e}")

    def build_ui(self):
        # Title
        tk.Label(
            self.root,
            text="🏠 House Price Predictor",
            font=("Arial", 18, "bold"),
            bg="#1e1e2f", fg="white"
        ).pack(pady=15)

        # Status
        self.status_label = tk.Label(
            self.root,
            text="⏳ Training model...",
            font=("Arial", 10),
            bg="#1e1e2f", fg="yellow"
        )
        self.status_label.pack()

        # Form frame
        form = tk.Frame(self.root, bg="#2a2a3f", padx=20, pady=20)
        form.pack(padx=30, pady=15, fill="both")

        fields = [
            ("Living Area (sq ft)",     "GrLivArea",    "e.g. 1500"),
            ("Bedrooms",                "BedroomAbvGr", "e.g. 3"),
            ("Bathrooms",               "FullBath",     "e.g. 2"),
            ("Garage Cars",             "GarageCars",   "e.g. 2"),
            ("Garage Area (sq ft)",     "GarageArea",   "e.g. 400"),
            ("Overall Quality (1-10)",  "OverallQual",  "e.g. 7"),
            ("Lot Area (sq ft)",        "LotArea",      "e.g. 8000"),
            ("Year Built",              "YearBuilt",    "e.g. 2005"),
        ]

        self.entries = {}

        for i, (label, key, placeholder) in enumerate(fields):
            tk.Label(
                form, text=label,
                bg="#2a2a3f", fg="white",
                font=("Arial", 10), anchor="w"
            ).grid(row=i, column=0, sticky="w", pady=5)

            entry = tk.Entry(form, font=("Arial", 11), width=18,
                             fg="grey", bg="#3a3a5f",
                             insertbackground="white",
                             relief="flat")
            entry.insert(0, placeholder)
            entry.bind("<FocusIn>",  lambda e, en=entry, ph=placeholder: self.clear_placeholder(e, en, ph))
            entry.bind("<FocusOut>", lambda e, en=entry, ph=placeholder: self.restore_placeholder(e, en, ph))
            entry.grid(row=i, column=1, padx=15, pady=5)
            self.entries[key] = entry

        # Predict button
        tk.Button(
            self.root,
            text="🔮 Predict Price",
            font=("Arial", 13, "bold"),
            bg="#ff9800", fg="white",
            relief="flat", padx=10, pady=8,
            command=self.predict
        ).pack(pady=10)

        # Result label
        self.result_label = tk.Label(
            self.root,
            text="💰 Predicted price will appear here",
            font=("Arial", 13, "bold"),
            bg="#1e1e2f", fg="#00ffcc",
            wraplength=400
        )
        self.result_label.pack(pady=10)

    def clear_placeholder(self, event, entry, placeholder):
        if entry.get() == placeholder:
            entry.delete(0, tk.END)
            entry.config(fg="white")

    def restore_placeholder(self, event, entry, placeholder):
        if entry.get() == "":
            entry.insert(0, placeholder)
            entry.config(fg="grey")

    def predict(self):
        if self.model is None:
            messagebox.showwarning("Wait", "Model is still training!")
            return
        try:
            data = {key: [float(self.entries[key].get())]
                    for key in NUMERIC_FEATURES}
            input_df = pd.DataFrame(data)
            price = self.model.predict(input_df)[0]
            self.result_label.config(
                text=f"💰 Estimated Price: ${price:,.0f}"
            )
        except ValueError:
            messagebox.showerror(
                "Input Error",
                "Please fill in all fields with valid numbers!"
            )


if __name__ == "__main__":
    root = tk.Tk()
    app = HousePriceGUI(root)
    root.mainloop()
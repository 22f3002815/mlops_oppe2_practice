# train_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

# adjust path as needed in your jupyter instance
df = pd.read_csv("data/v0/transactions_2022.csv")

# Drop rows with NaN values in the target column
print(f"Original dataset size: {len(df)}")
df = df.dropna(subset=['Class'])
print(f"After dropping NaN in Class: {len(df)}")

# Or drop all rows with any NaN values
# df = df.dropna()

# feature list
features = ["Time"] + [f"V{i}" for i in range(1,29)] + ["Amount"]
X = df[features]
y = df["Class"]

# Verify no NaN values remain
print(f"NaN in X: {X.isna().sum().sum()}")
print(f"NaN in y: {y.isna().sum()}")

# small sample to speed up in notebook — remove for full training
X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

os.makedirs("app/model", exist_ok=True)
joblib.dump(model, "app/model/model.pkl")
print("Saved model to app/model/model.pkl")

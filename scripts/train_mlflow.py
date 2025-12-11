"""
scripts/train_mlflow.py

Usage example:
  python scripts/train_mlflow.py \
    --data data/poisoned/poisoned_2_percent.csv \
    --poisoning_level 2 \
    --experiment-name "poisoning_experiment" \
    --run-name "poison-2pct" \
    --seed 42
"""

import argparse
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
import joblib
import mlflow
import mlflow.sklearn

FEATURES = ["Time"] + [f"V{i}" for i in range(1,29)] + ["Amount"]

def load_data(path):
    df = pd.read_csv(path)
    # Drop rows where target is NaN
    before = len(df)
    df = df.dropna(subset=["Class"])
    after = len(df)
    print(f"Dropped {before - after} rows with NaN in Class")

    X = df[FEATURES]
    y = df["Class"]
    return X, y

def train_and_log(data_path, poisoning_level, experiment_name, run_name, seed=42):
    X, y = load_data(data_path)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    # Model: RandomForest with class_weight balanced
    model = RandomForestClassifier(n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=seed)

    # MLflow experiment setup
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        # log input dataset as param (path)
        mlflow.log_param("dataset", data_path)
        mlflow.log_param("poisoning_level", poisoning_level)
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_val", len(X_val))

        # train
        model.fit(X_train, y_train)

        # predict val
        preds = model.predict(X_val)
        f1 = f1_score(y_val, preds)
        prec = precision_score(y_val, preds)
        rec = recall_score(y_val, preds)

        # log metrics
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)

        # save model artifact and log it
        os.makedirs("artifacts", exist_ok=True)
        model_path = os.path.join("artifacts", f"rf_poison_{poisoning_level}pct.joblib")
        joblib.dump(model, model_path)
        mlflow.sklearn.log_model(model, artifact_path="model")
        mlflow.log_artifact(model_path, artifact_path="raw_model_file")

        print(f"Run logged: f1={f1:.4f} prec={prec:.4f} rec={rec:.4f}")

    return f1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to csv dataset")
    parser.add_argument("--poisoning_level", type=int, required=True, help="Integer percent (2,8,20)")
    parser.add_argument("--experiment-name", default="poisoning_experiment")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_name = args.run_name or f"poison-{args.poisoning_level}pct"
    train_and_log(args.data, args.poisoning_level, args.experiment_name, run_name, seed=args.seed)

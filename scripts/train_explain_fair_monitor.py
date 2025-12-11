#!/usr/bin/env python3
# scripts/train_explain_fair_monitor.py
"""
End-to-end training + explainability + fairness + drift monitoring.

Usage:
 python scripts/train_explain_fair_monitor.py \
   --v0 data/v0/transactions_2022_with_location.csv \
   --v1 data/v1/transactions_2023.csv \
   --experiment "explain_fair_monitor" \
   --run_name "xgb_with_location" \
   --seed 42
"""

import argparse
import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.xgboost
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from fairlearn.metrics import demographic_parity_difference
import joblib
import warnings

warnings.filterwarnings("ignore")

# Features list
FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


def clean_df_labels_and_features(df, features, label_col="Class"):
    """
    - Report NA/inf counts for label and features
    - Drop rows where label is NaN or infinite
    - Replace non-finite feature values (NaN/inf) with 0.0
    - Return cleaned df (with same index where possible) and a report dict
    """
    report = {}

    # Ensure df is a DataFrame copy to avoid modifying original
    df = df.copy()

    # label issues
    if label_col in df.columns:
        label_ser = pd.to_numeric(df[label_col], errors="coerce")
        n_label_na = int(label_ser.isna().sum())
        n_label_inf = int(np.isinf(label_ser.fillna(0)).sum())
    else:
        n_label_na = 0
        n_label_inf = 0

    report["label_na"] = n_label_na
    report["label_inf"] = n_label_inf

    # feature issues
    feat_na = {}
    feat_inf = {}
    for f in features:
        if f in df.columns:
            s = pd.to_numeric(df[f], errors="coerce")
            feat_na[f] = int(s.isna().sum())
            feat_inf[f] = int(np.isinf(s.fillna(0)).sum())
        else:
            feat_na[f] = None
            feat_inf[f] = None

    report["feature_na_counts"] = feat_na
    report["feature_inf_counts"] = feat_inf

    # Drop rows with missing or non-finite labels
    if label_col in df.columns:
        # Mask of finite numeric labels
        label_numeric = pd.to_numeric(df[label_col], errors="coerce")
        is_finite_label = label_numeric.notna() & (~np.isinf(label_numeric))
        before = len(df)
        df_clean = df.loc[is_finite_label].copy()
        after = len(df_clean)
        report["dropped_rows_for_label"] = before - after
    else:
        df_clean = df.copy()
        report["dropped_rows_for_label"] = 0

    # Replace non-finite features with 0.0
    for f in features:
        if f in df_clean.columns:
            df_clean[f] = pd.to_numeric(df_clean[f], errors="coerce")
            df_clean[f] = df_clean[f].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return df_clean, report


def prepare_Xy(df, use_location=True):
    df2 = df.copy()
    df2, report = clean_df_labels_and_features(df2, FEATURES, label_col="Class")

    print("Data quality report (summary):")
    print(f"  dropped_rows_for_label: {report.get('dropped_rows_for_label')}")
    feats_with_na = {k: v for k, v in report["feature_na_counts"].items() if v and v > 0}
    if feats_with_na:
        print("  feature NA counts (sample):", list(feats_with_na.items())[:5])

    # Build X with numeric features only
    X = df2[FEATURES].copy()
    for f in FEATURES:
        X[f] = pd.to_numeric(X[f], errors="coerce")
        X[f] = X[f].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if use_location and "location" in df2.columns:
        loc_dummies = pd.get_dummies(df2["location"], prefix="loc")
        loc_dummies = loc_dummies.reindex(sorted(loc_dummies.columns), axis=1)
        X = pd.concat([X, loc_dummies], axis=1)

    # y (safe cast after dropping bad rows)
    y = pd.to_numeric(df2["Class"], errors="coerce").fillna(0).astype(int)
    assert len(X) == len(y), "X and y length mismatch after cleaning"
    return X, y, df2

def compute_shap_and_save(model, X_train, seed=42):
    """
    Compute SHAP values for an XGBoost tree model using TreeExplainer
    and save a beeswarm summary plot.
    """
    shap_png = "artifacts/shap_summary.png"
    os.makedirs(os.path.dirname(shap_png), exist_ok=True)

    shap_sample = X_train.sample(n=min(200, len(X_train)), random_state=seed)
    X_shap = shap_sample.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)

    try:
        # TreeExplainer first (fast)
        if hasattr(model, "get_booster"):
            explainer = shap.TreeExplainer(model.get_booster())
        else:
            explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_shap.values)
    except Exception:
        # Very small KernelExplainer fallback (slow but robust)
        f = lambda data: model.predict_proba(data)[:, 1]
        background = X_shap.sample(n=min(50, len(X_shap)), random_state=seed).values
        explainer = shap.KernelExplainer(f, background)
        shap_values = explainer.shap_values(X_shap.values, nsamples=100)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_shap.values, feature_names=X_shap.columns, show=False)
    plt.tight_layout()
    plt.savefig(shap_png, bbox_inches="tight", dpi=150)
    plt.clf()
    return shap_png

    # except Exception as e_callable:
    #     # If callable explainer fails, try TreeExplainer on booster or model as last resort
    #     try:
    #         # TreeExplainer on booster if available
    #         if hasattr(model, "get_booster"):
    #             explainer = shap.TreeExplainer(model.get_booster())
    #         else:
    #             explainer = shap.TreeExplainer(model)
    #         try:
    #             shap_vals = explainer.shap_values(shap_sample)
    #             plt.figure(figsize=(10, 6))
    #             # older shap versions expect numpy arrays
    #             try:
    #                 shap.summary_plot(shap_vals, shap_sample, show=False)
    #             except Exception:
    #                 shap.summary_plot(np.array(shap_vals), shap_sample, show=False)
    #             plt.tight_layout()
    #             plt.savefig(shap_png, bbox_inches="tight", dpi=150)
    #             plt.clf()
    #             return shap_png
    #         except Exception as e_tree_vals:
    #             # final fallback failed
    #             raise RuntimeError(f"Callable explainer failed ({e_callable}); TreeExplainer fallback failed ({e_tree_vals})")
    #     except Exception as e_tree:
    #         # both approaches failed
    #         raise RuntimeError(f"SHAP explainers failed: callable error: {e_callable}; tree error: {e_tree}")


def main(args):
    os.makedirs("artifacts", exist_ok=True)
    # Load v0 (with location)
    df_v0 = pd.read_csv(args.v0)
    print("v0 shape:", df_v0.shape)

    # prepare X,y and cleaned df (preserves indices)
    X_v0, y_v0, df_v0_clean = prepare_Xy(df_v0, use_location=True)

    # split keeping index alignment (train_test_split preserves indices)
    X_train, X_val, y_train, y_val = train_test_split(
        X_v0, y_v0, test_size=0.2, stratify=y_v0, random_state=args.seed
    )

    # XGBoost params
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "use_label_encoder": False,
        "verbosity": 0,
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
    }
    model = xgb.XGBClassifier(**params)

    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_param("model", "xgboost")
        mlflow.log_param("seed", args.seed)
        mlflow.log_param("features", ",".join(X_train.columns.tolist()))
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_val", len(X_val))

        # train
        model.fit(X_train, y_train)

        # evaluate on v0 val
        preds_val = model.predict(X_val)
        probs_val = model.predict_proba(X_val)[:, 1]
        f1_val = f1_score(y_val, preds_val)
        prec_val = precision_score(y_val, preds_val, zero_division=0)
        rec_val = recall_score(y_val, preds_val, zero_division=0)
        mlflow.log_metric("f1_v0_val", float(f1_val))
        mlflow.log_metric("precision_v0_val", float(prec_val))
        mlflow.log_metric("recall_v0_val", float(rec_val))

        # SHAP explainability
        try:
            shap_path = compute_shap_and_save(model, X_train, seed=args.seed)
            mlflow.log_artifact(shap_path, artifact_path="explainability")
            print(f"Saved SHAP summary plot to {shap_path}")
        except Exception as ex_shap:
            print("WARNING: SHAP explainer creation failed:", ex_shap)
            with open("artifacts/shap_error.txt", "w") as f:
                f.write("SHAP explainer failed:\n")
                f.write(str(ex_shap) + "\n")
            mlflow.log_artifact("artifacts/shap_error.txt", artifact_path="explainability")


        # Fairness: demographic parity difference on validation set
        try:
            df_val = df_v0_clean.loc[X_val.index]
        except Exception:
            df_val = df_v0_clean.iloc[X_val.index]

        if "location" in df_val.columns:
            sensitive = df_val["location"].values
        else:
            sensitive = np.array(["Unknown"] * len(df_val))

        dp_diff = demographic_parity_difference(y_val.values, preds_val, sensitive_features=sensitive)
        mlflow.log_metric("demographic_parity_difference_v0_val", float(dp_diff))

        # Save fairness summary
        with open("artifacts/fairness.txt", "w") as f:
            f.write(f"demographic_parity_difference_v0_val: {dp_diff}\n")
        mlflow.log_artifact("artifacts/fairness.txt", artifact_path="fairness")

        # Save model artifact
        mlflow.xgboost.log_model(model, artifact_path="model")
        joblib.dump(model, "artifacts/xgb_model.joblib")
        mlflow.log_artifact("artifacts/xgb_model.joblib", artifact_path="model_raw")

        # Now run on v1 (may not have location). Align columns with X_train
        df_v1 = pd.read_csv(args.v1)
        print("v1 shape:", df_v1.shape)

        # build X_v1_base from v1 features; ensure numeric & cleaned
        X_v1_base = df_v1[FEATURES].copy()
        for f in FEATURES:
            X_v1_base[f] = pd.to_numeric(X_v1_base[f], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # Add loc_ columns present in X_train (so columns match)
        loc_cols = [c for c in X_train.columns if c.startswith("loc_")]
        for c in loc_cols:
            X_v1_base[c] = 0  # neutral / unknown location

        # Reindex columns to exactly match X_train column order
        X_v1 = X_v1_base.reindex(columns=X_train.columns, fill_value=0)

        # ensure lengths align with y_v1
        y_v1 = pd.to_numeric(df_v1["Class"], errors="coerce").fillna(0).astype(int)

        preds_v1 = model.predict(X_v1)
        probs_v1 = model.predict_proba(X_v1)[:, 1]
        f1_v1 = f1_score(y_v1, preds_v1, zero_division=0)
        prec_v1 = precision_score(y_v1, preds_v1, zero_division=0)
        rec_v1 = recall_score(y_v1, preds_v1, zero_division=0)
        mlflow.log_metric("f1_v1_full", float(f1_v1))
        mlflow.log_metric("precision_v1_full", float(prec_v1))
        mlflow.log_metric("recall_v1_full", float(rec_v1))

        # save drift comparison plot (bar plot of f1_v0_val vs f1_v1_full)
        plt.figure(figsize=(6, 4))
        sns.barplot(x=["v0_val", "v1_full"], y=[f1_val, f1_v1])
        plt.ylim(0, 1)
        plt.title("F1 comparison: v0_val vs v1_full")
        plt.ylabel("F1 score")
        drift_png = "artifacts/drift_comparison.png"
        plt.savefig(drift_png, bbox_inches="tight", dpi=150)
        plt.clf()
        mlflow.log_artifact(drift_png, artifact_path="monitoring")

        print("Logged metrics: f1_v0_val=%.4f f1_v1_full=%.4f dp_diff=%.4f" % (f1_val, f1_v1, dp_diff))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--v0", required=True, help="v0 dataset with location")
    parser.add_argument("--v1", required=True, help="v1 dataset (transactions_2023.csv)")
    parser.add_argument("--experiment", default="explain_fair_monitor")
    parser.add_argument("--run_name", default="xgb_explain_fair_monitor")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)

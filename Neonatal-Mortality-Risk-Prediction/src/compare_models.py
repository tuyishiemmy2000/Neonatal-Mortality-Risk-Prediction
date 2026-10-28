"""
Compare Logistic Regression (baseline) with the saved Random Forest models.
Run after src/train_model.py:
    python src/compare_models.py --data data/merged_table.csv

Thresholds target 60% recall on training out-of-fold predictions. They are
screening cutoffs for research evaluation, not clinical thresholds.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss, precision_score, recall_score,
    roc_auc_score, f1_score,
)
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from build_features import build_features, TARGET, PREGNANCY_FEATURES, DELIVERY_FEATURES

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "merged_table.csv"
OUTPUTS = ROOT / "outputs"
MODELS = OUTPUTS / "models"
TARGET_RECALL = 0.60
RANDOM_STATE = 42


def make_logistic(features):
    numeric = [f for f in features if f != "District"]
    pre = ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]), numeric),
        ("district", OneHotEncoder(handle_unknown="ignore"), ["District"]),
    ])
    return Pipeline([
        ("preprocessor", pre),
        ("logistic_regression", LogisticRegression(max_iter=3000, solver="lbfgs")),
    ])


def select_threshold(y, p):
    from sklearn.metrics import precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(y, p)
    eligible = np.where(recall[:-1] >= TARGET_RECALL)[0]
    return float(thresholds[eligible[-1]]) if len(eligible) else float(np.min(p))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    args = parser.parse_args()

    raw = pd.read_csv(args.data, low_memory=False)
    df = build_features(raw)
    y = df[TARGET].astype(int)
    groups = df["Case ID"].astype(str)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(df, y, groups))
    train, test = df.iloc[train_idx], df.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    g_train = groups.iloc[train_idx]

    thresholds_path = OUTPUTS / "model_thresholds.json"
    rf_thresholds = json.loads(thresholds_path.read_text()) if thresholds_path.exists() else {}
    results = []
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    for stage, features in [("pregnancy", PREGNANCY_FEATURES), ("delivery", DELIVERY_FEATURES)]:
        X_train, X_test = train[features], test[features]
        lr = make_logistic(features)
        oof = cross_val_predict(
            lr, X_train, y_train, cv=cv, groups=g_train,
            method="predict_proba", n_jobs=1
        )[:, 1]
        lr_threshold = select_threshold(y_train, oof)
        lr.fit(X_train, y_train)
        p_lr = lr.predict_proba(X_test)[:, 1]

        rf_path = MODELS / f"{stage}_model.joblib"
        if not rf_path.exists():
            raise FileNotFoundError(f"Missing {rf_path}; run src/train_model.py first.")
        rf = joblib.load(rf_path)
        p_rf = rf.predict_proba(X_test)[:, 1]
        for name, p, threshold in [
            ("Logistic Regression baseline", p_lr, lr_threshold),
            ("Calibrated Random Forest", p_rf, float(rf_thresholds.get(stage, 0.5))),
        ]:
            pred = (p >= threshold).astype(int)
            results.append({
                "stage": stage,
                "model": name,
                "threshold": threshold,
                "roc_auc": roc_auc_score(y_test, p) if y_test.nunique() > 1 else np.nan,
                "average_precision": average_precision_score(y_test, p),
                "brier_score": brier_score_loss(y_test, p),
                "recall": recall_score(y_test, pred, zero_division=0),
                "precision": precision_score(y_test, pred, zero_division=0),
                "f1": f1_score(y_test, pred, zero_division=0),
                "flagged_pct": float(pred.mean()),
                "test_records": len(y_test),
                "test_deaths": int(y_test.sum()),
            })

    result = pd.DataFrame(results)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUTS / "model_comparison.csv", index=False)
    print(result.round(4).to_string(index=False))
    print(f"\nSaved: {OUTPUTS / 'model_comparison.csv'}")
    print(
        "\nCompare models using ROC-AUC, average precision, Brier score, recall, precision and "
        "flagged share together. Do not choose a model on ROC-AUC alone."
    )


if __name__ == "__main__":
    main()

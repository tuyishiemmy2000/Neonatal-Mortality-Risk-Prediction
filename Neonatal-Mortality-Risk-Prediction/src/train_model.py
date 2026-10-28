"""
Train separate pregnancy-stage and delivery-stage Random Forest models.

Run from project root:
    python src/train_model.py --data data/merged_table.csv

The mother/Case ID is kept within a single fold to reduce sibling leakage.
Thresholds target 60% recall on training out-of-fold predictions; they are
research screening cutoffs, not clinically validated referral thresholds.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss, precision_score, recall_score,
    roc_auc_score, f1_score, confusion_matrix,
)
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from build_features import (
    build_features, PREGNANCY_FEATURES, DELIVERY_FEATURES, TARGET,
    CalibratedRiskModel,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "merged_table.csv"
OUTPUTS = ROOT / "outputs"
MODELS = OUTPUTS / "models"
TARGET_RECALL = 0.60
RANDOM_STATE = 42


def make_base_model(features):
    numeric_features = [f for f in features if f != "District"]
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median", keep_empty_features=True), numeric_features),
            ("district", OneHotEncoder(handle_unknown="ignore"), ["District"]),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    rf = RandomForestClassifier(
        n_estimators=40,
        min_samples_leaf=8,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    return Pipeline([("preprocessor", preprocessor), ("random_forest", rf)])


def choose_threshold(y_true, probabilities, target_recall=TARGET_RECALL):
    """Highest threshold that achieves target recall on training OOF predictions."""
    from sklearn.metrics import precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    valid = np.where(recall[:-1] >= target_recall)[0]
    if len(valid) == 0:
        warnings.warn(
            f"No threshold achieved recall {target_recall:.2f}; using the lowest "
            "out-of-fold probability as a high-sensitivity fallback."
        )
        return float(np.min(probabilities))
    return float(thresholds[valid[-1]])


def wilson_interval(events, total, z=1.96):
    if total <= 0:
        return np.nan, np.nan
    p = events / total
    denom = 1 + z*z/total
    center = (p + z*z/(2*total)) / denom
    half = z * np.sqrt((p*(1-p)/total) + z*z/(4*total*total)) / denom
    return max(0.0, center-half), min(1.0, center+half)


def district_summary(df, probabilities, stage):
    work = df[["District", TARGET]].copy()
    work["risk"] = np.asarray(probabilities)
    national_mean = float(work["risk"].mean())
    grouped = work.groupby("District", dropna=False).agg(
        births=(TARGET, "size"),
        observed_deaths=(TARGET, "sum"),
        mean_risk=("risk", "mean"),
        expected_deaths=("risk", "sum"),
    ).reset_index()
    grouped["observed_deaths"] = grouped["observed_deaths"].astype(int)
    grouped["national_mean_risk"] = national_mean
    # Difference is in percentage points, not a percent change.
    # The national mean is one stage-specific reference shared by every district.
    grouped["diff"] = (grouped["mean_risk"] - national_mean) * 100.0
    grouped["observed_rate_per_1000"] = (
        grouped["observed_deaths"] / grouped["births"] * 1000.0
    )
    ci = [
        wilson_interval(int(row.observed_deaths), int(row.births))
        for row in grouped.itertuples()
    ]
    grouped["observed_rate_ci_low_per_1000"] = [lo * 1000 for lo, hi in ci]
    grouped["observed_rate_ci_high_per_1000"] = [hi * 1000 for lo, hi in ci]
    # Relative ranking bands: bottom third = Low, middle third = Medium,
    # top third = High. These labels are relative comparisons, not clinical cutoffs.
    grouped["level"] = pd.qcut(
        grouped["mean_risk"].rank(method="first"),
        q=3,
        labels=["Low", "Medium", "High"],
    ).astype(str)
    grouped["uncertainty_note"] = np.where(
        (grouped["births"] < 100) | (grouped["observed_deaths"] < 5),
        "Interpret cautiously: small number of records and/or observed deaths",
        "Descriptive estimate; still subject to model and sampling uncertainty",
    )
    grouped["stage"] = stage
    grouped = grouped.sort_values("mean_risk", ascending=False).reset_index(drop=True)
    return grouped


def calibration_bins(y_true, probabilities, stage, bins=5):
    frame = pd.DataFrame({
        "observed": np.asarray(y_true).astype(int),
        "predicted": np.asarray(probabilities, dtype=float),
    })
    frame["bin"] = pd.qcut(
        frame["predicted"].rank(method="first"),
        q=bins,
        labels=False,
        duplicates="drop",
    ) + 1
    out = frame.groupby("bin", as_index=False).agg(
        records=("observed", "size"),
        observed_deaths=("observed", "sum"),
        mean_predicted_risk=("predicted", "mean"),
        observed_rate=("observed", "mean"),
    )
    out["stage"] = stage
    return out


def fit_stage(df, features, stage, y_train, y_test, X_train, X_test, groups_train, groups_all):
    X_all = df[features].copy()
    X_train_stage = X_train[features].copy()
    X_test_stage = X_test[features].copy()
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    # Group-aware OOF scores used to fit a simple sigmoid probability mapping.
    base_for_oof = make_base_model(features)
    oof_raw_train = cross_val_predict(
        base_for_oof, X_train_stage, y_train,
        cv=cv, groups=groups_train, method="predict_proba", n_jobs=1
    )[:, 1]
    calibrator = LogisticRegression(max_iter=1000)
    calibrator.fit(oof_raw_train.reshape(-1, 1), y_train)
    oof_cal_train = calibrator.predict_proba(oof_raw_train.reshape(-1, 1))[:, 1]
    threshold = choose_threshold(y_train, oof_cal_train)

    # Fit final model on the complete training partition.
    base_model = make_base_model(features)
    base_model.fit(X_train_stage, y_train)
    model = CalibratedRiskModel(base_model, calibrator)

    p_test = model.predict_proba(X_test_stage)[:, 1]
    y_pred = (p_test >= threshold).astype(int)
    metrics = {
        "stage": stage,
        "records": int(len(df)),
        "observed_neonatal_deaths": int(df[TARGET].sum()),
        "roc_auc": float(roc_auc_score(y_test, p_test)) if y_test.nunique() > 1 else np.nan,
        "pr_auc": float(average_precision_score(y_test, p_test)),
        "brier_score": float(brier_score_loss(y_test, p_test)),
        "mean_predicted_risk_test": float(np.mean(p_test)),
        "observed_rate_test": float(np.mean(y_test)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "threshold": float(threshold),
        "flagged_pct": float(np.mean(y_pred)),
        "test_records": int(len(y_test)),
        "test_deaths": int(y_test.sum()),
        "true_positive": int(((y_test == 1) & (y_pred == 1)).sum()),
        "false_positive": int(((y_test == 0) & (y_pred == 1)).sum()),
        "false_negative": int(((y_test == 1) & (y_pred == 0)).sum()),
        "true_negative": int(((y_test == 0) & (y_pred == 0)).sum()),
    }

    calibration_df = calibration_bins(y_test, p_test, stage)

    # District summaries combine out-of-fold predictions for training records
    # with independent held-out predictions for test records. This avoids using
    # in-sample predictions for the training portion without requiring a second
    # full-cohort cross-validation run.
    oof_cal_all = pd.Series(index=df.index, dtype=float)
    oof_cal_all.loc[X_train.index] = oof_cal_train
    oof_cal_all.loc[X_test.index] = p_test
    district = district_summary(df, oof_cal_all.loc[df.index].to_numpy(), stage)

    # Export transformed feature importance, including one-hot district indicators.
    preprocessor = base_model.named_steps["preprocessor"]
    names = preprocessor.get_feature_names_out()
    importance = base_model.named_steps["random_forest"].feature_importances_
    importance_df = pd.DataFrame({"feature": names, "importance": importance})
    importance_df = importance_df.sort_values("importance", ascending=False)

    return model, threshold, metrics, district, importance_df, oof_cal_all, calibration_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="Path to merged DHS CSV")
    args = parser.parse_args()

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(args.data, low_memory=False)
    audit_path = OUTPUTS / "data_quality_conflicts.csv"
    df = build_features(raw, audit_path=str(audit_path))

    if df[TARGET].nunique() != 2:
        raise ValueError("The analytic dataset must contain both outcome classes.")
    if df[TARGET].sum() < 30:
        warnings.warn("Fewer than 30 neonatal deaths remain; estimates may be unstable.")

    print(f"Analytic births: {len(df):,}")
    print(f"Neonatal deaths: {int(df[TARGET].sum()):,}")
    print(f"Outcome rate in analytic sample: {df[TARGET].mean():.2%}")
    audit_df = pd.read_csv(audit_path)
    print(f"Rows with conflicting duplicate-key values logged: {len(audit_df):,}")
    if not audit_df.empty:
        audit_df.groupby("column", as_index=False).size().rename(
            columns={"size": "conflicting_birth_keys"}
        ).sort_values("conflicting_birth_keys", ascending=False).to_csv(
            OUTPUTS / "data_quality_conflict_summary.csv", index=False
        )

    groups = df["Case ID"].astype(str)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(df, df[TARGET], groups))
    train_df, test_df = df.iloc[train_idx].copy(), df.iloc[test_idx].copy()
    y_train = train_df[TARGET].astype(int)
    y_test = test_df[TARGET].astype(int)
    groups_train = train_df["Case ID"].astype(str)
    print(f"Grouped train/test split: {len(train_df):,}/{len(test_df):,} records")
    print(f"Test neonatal deaths: {int(y_test.sum())}")

    all_metrics = []
    all_calibration = []
    thresholds = {}
    for stage, features in [
        ("pregnancy", PREGNANCY_FEATURES),
        ("delivery", DELIVERY_FEATURES),
    ]:
        model, threshold, metrics, district, importance, oof_risk, calibration_df = fit_stage(
            df=df,
            features=features,
            stage=stage,
            y_train=y_train,
            y_test=y_test,
            X_train=train_df,
            X_test=test_df,
            groups_train=groups_train,
            groups_all=groups,
        )
        joblib.dump(model, MODELS / f"{stage}_model.joblib")
        thresholds[stage] = threshold
        all_metrics.append(metrics)
        all_calibration.append(calibration_df)
        district.to_csv(OUTPUTS / f"{stage}_district_risk.csv", index=False)
        importance.to_csv(OUTPUTS / f"{stage}_feature_importance.csv", index=False)
        print(f"\n{stage.upper()} model")
        print(f"  threshold (training OOF, target recall {TARGET_RECALL:.0%}): {threshold:.4f}")
        print(f"  test ROC-AUC: {metrics['roc_auc']:.3f}")
        print(f"  test average precision: {metrics['pr_auc']:.3f}")
        print(f"  test Brier score: {metrics['brier_score']:.4f}")
        print(f"  test recall / precision: {metrics['recall']:.3f} / {metrics['precision']:.3f}")
        print(f"  test flagged: {metrics['flagged_pct']:.1%}")

    with open(OUTPUTS / "model_thresholds.json", "w", encoding="utf-8") as f:
        json.dump(thresholds, f, indent=2)

    summary = pd.DataFrame(all_metrics)
    # Add the cohort-wide out-of-fold mean risk for display; stage means differ by model.
    for stage in ["pregnancy", "delivery"]:
        district = pd.read_csv(OUTPUTS / f"{stage}_district_risk.csv")
        summary.loc[summary["stage"] == stage, "oof_mean_predicted_risk"] = (
            district["mean_risk"] * district["births"]
        ).sum() / district["births"].sum()
    summary.to_csv(OUTPUTS / "national_risk_summary.csv", index=False)
    pd.concat(all_calibration, ignore_index=True).to_csv(
        OUTPUTS / "calibration_summary.csv", index=False
    )

    # Backward-compatible combined file for the existing app's expected path.
    delivery = pd.read_csv(OUTPUTS / "delivery_district_risk.csv")
    delivery.to_csv(OUTPUTS / "district_risk.csv", index=False)
    print(f"\nSaved models and summaries under: {OUTPUTS}")


if __name__ == "__main__":
    main()

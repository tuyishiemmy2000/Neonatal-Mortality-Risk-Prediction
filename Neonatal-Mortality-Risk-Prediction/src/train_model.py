"""Train the Random Forest, evaluate it, and save district-level results.

Run from the project root:  python src/train_model.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (average_precision_score, classification_report,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline

sys.path.append(str(Path(__file__).parent))
from build_features import FEATURES, TARGET, build_features

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed_data_cleaned.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

TARGET_RECALL = 0.60      # change to 0.50 or 0.70 to compare
SEED = 42

# 1. data
df = build_features(pd.read_csv(DATA))
X, y = df[FEATURES], df[TARGET].astype(int)
print("Rows:", len(df), "| neonatal deaths:", int(y.sum()))

# 2. split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)

# 3. model
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
    ("random_forest", RandomForestClassifier(
        n_estimators=300, min_samples_leaf=5,
        class_weight="balanced", random_state=SEED)),
])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# 4. choose the cutoff from the TRAINING data only (out-of-fold predictions)
p_oof = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
prec, rec, thr = precision_recall_curve(y_train, p_oof)
i = np.where(rec[:-1] >= TARGET_RECALL)[0][-1]
cutoff = thr[i]
print(f"Chosen cutoff for recall >= {TARGET_RECALL}: {cutoff:.3f}")

# 5. train and test once
model.fit(X_train, y_train)
p_test = model.predict_proba(X_test)[:, 1]
y_pred = (p_test >= cutoff).astype(int)

print("\nROC-AUC:", round(roc_auc_score(y_test, p_test), 3))
print("PR-AUC:", round(average_precision_score(y_test, p_test), 3))
print("Recall:", round(recall_score(y_test, y_pred), 3))
print("Precision:", round(precision_score(y_test, y_pred, zero_division=0), 3))
print("F1:", round(f1_score(y_test, y_pred), 3))
print("Share of babies flagged:", round(y_pred.mean(), 3))
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))
print(classification_report(y_test, y_pred, labels=[0, 1],
      target_names=["Survived", "Neonatal death"], zero_division=0))

# 6. feature importance
rf = model.named_steps["random_forest"]
names = model.named_steps["imputer"].get_feature_names_out(FEATURES)
imp = pd.Series(rf.feature_importances_, index=names).sort_values(ascending=False)
imp.to_csv(OUT / "feature_importance.csv", header=["importance"])

# 7. district results (out-of-fold risk for every child)
df["risk"] = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
df["flagged"] = (df["risk"] >= cutoff).astype(int)
d = df.groupby("District").agg(
    births=("risk", "size"),
    observed_deaths=(TARGET, "sum"),
    mean_risk=("risk", "mean"),
    flagged_pct=("flagged", "mean")).reset_index()
d["deaths_per_1000"] = (d["observed_deaths"] / d["births"] * 1000).round(1)
d["flagged_pct"] = (d["flagged_pct"] * 100).round(1)
d["diff"] = d["mean_risk"] - df["risk"].mean()
d["level"] = pd.qcut(d["mean_risk"], 3, labels=["Low", "Medium", "High"])
d = d.sort_values("diff", ascending=False)
d.to_csv(OUT / "district_risk.csv", index=False)

# 8. chart
colors = {"High": "#d62728", "Medium": "#ff9f1c", "Low": "#2ca02c"}
fig, ax = plt.subplots(figsize=(8, 9))
ax.barh(d["District"], d["diff"], color=d["level"].map(colors))
ax.invert_yaxis()
ax.axvline(0, color="grey", lw=0.8)
ax.set_xlabel("Predicted neonatal mortality risk vs national average")
ax.set_title("Neonatal mortality risk by district (Rwanda DHS 2025)")
ax.legend(handles=[Patch(color=c, label=l) for l, c in colors.items()],
          title="Risk", loc="lower right")
plt.tight_layout()
plt.savefig(OUT / "district_risk.png", dpi=150)
print("\nSaved results to", OUT)

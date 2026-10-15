import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (recall_score, precision_score, f1_score, confusion_matrix,
                             classification_report, roc_auc_score,
                             average_precision_score, precision_recall_curve)

df = build_features(pd.read_csv(r"C:\Users\user\Pictures\My project\processed_data_cleaned.csv"))
X = df[FEATURES].copy()
y = df[TARGET].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
    ("random_forest", RandomForestClassifier(
        n_estimators=300, min_samples_leaf=5,
        class_weight="balanced", random_state=42))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
p_oof = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
prec, rec, thr = precision_recall_curve(y_train, p_oof)

TARGET_RECALL = 0.60
i = np.where(rec[:-1] >= TARGET_RECALL)[0][-1]
cutoff = thr[i]
print("Chosen cutoff:", round(cutoff, 3))

model.fit(X_train, y_train)
p_test = model.predict_proba(X_test)[:, 1]
y_pred = (p_test >= cutoff).astype(int)

print("ROC-AUC:", round(roc_auc_score(y_test, p_test), 3))
print("PR-AUC:", round(average_precision_score(y_test, p_test), 3))
print("Recall:", round(recall_score(y_test, y_pred), 3))
print("Precision:", round(precision_score(y_test, y_pred, zero_division=0), 3))
print("F1:", round(f1_score(y_test, y_pred), 3))
print("Share of babies flagged:", round(y_pred.mean(), 3))
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))
print(classification_report(y_test, y_pred, labels=[0, 1],
      target_names=["Survived", "Neonatal death"], zero_division=0))

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

logit = make_pipeline(
    StandardScaler(),
    LogisticRegression(class_weight="balanced", max_iter=1000)
)

rf = RandomForestClassifier(
    n_estimators=300,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42
)

logit.fit(X_train, y_train)
rf.fit(X_train, y_train)


# MODEL EVALUATION AND COMPARISON

import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    recall_score,
    precision_score,
    f1_score
)

thresholds = {
    "Logistic Regression": 0.50,
    "Random Forest": 0.0844
}

results = []

for name, model in [
    ("Logistic Regression", logit),
    ("Random Forest", rf)
]:
    probabilities = model.predict_proba(X_test)[:, 1]

    cutoff = thresholds[name]
    predictions = (probabilities >= cutoff).astype(int)

    results.append({
        "Model": name,
        "Threshold": cutoff,
        "ROC-AUC": roc_auc_score(y_test, probabilities),
        "PR-AUC": average_precision_score(y_test, probabilities),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "F1-score": f1_score(
            y_test, predictions, zero_division=0
        )
    })

    print("\\n" + "=" * 45)
    print(name)
    print(f"Threshold: {cutoff:.4f}")
    print("=" * 45)

    print("\\nConfusion matrix:")
    print(confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ))

    print("\\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Survived", "Neonatal death"],
        zero_division=0
    ))

    print(
        f"Share of babies flagged: {predictions.mean():.1%}"
    )

print("\\nFINAL MODEL COMPARISON")
comparison = pd.DataFrame(results)
print(comparison.round(3).to_string(index=False))


# FEATURE IMPORTANCE AND MODEL INTERPRETATION

import pandas as pd

rf_importance = pd.Series(
    rf.feature_importances_,
    index=FEATURES,
    name="Random Forest Importance"
).sort_values(ascending=False)

print("RANDOM FOREST FEATURE IMPORTANCE")
print(rf_importance)

logit_coefficients = pd.Series(
    logit.named_steps["logisticregression"].coef_[0],
    index=FEATURES,
    name="Logistic Regression Coefficient"
).sort_values(ascending=False)

print("\nLOGISTIC REGRESSION COEFFICIENTS")
print(logit_coefficients)

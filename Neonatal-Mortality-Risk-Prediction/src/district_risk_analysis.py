df["risk"] = rf.predict_proba(X)[:, 1]
df.groupby("District")["risk"].mean().sort_values(ascending=False)

from sklearn.model_selection import StratifiedKFold, cross_val_predict

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
df["risk"] = cross_val_predict(rf, X, y, cv=cv, method="predict_proba")[:, 1]

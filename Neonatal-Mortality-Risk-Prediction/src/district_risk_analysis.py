df["risk"] = rf.predict_proba(X)[:, 1]
df.groupby("District")["risk"].mean().sort_values(ascending=False)

from sklearn.model_selection import StratifiedKFold, cross_val_predict

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
df["risk"] = cross_val_predict(rf, X, y, cv=cv, method="predict_proba")[:, 1]

national_average = df["risk"].mean()

d = df.groupby("District")["risk"].mean().reset_index()

d["diff"] = (d["risk"] - national_average) * 100

d["level"] = d["diff"].apply(
    lambda x: "High" if x > 0.5
    else "Low" if x < -0.5
    else "Medium"
)

d = d.sort_values("diff", ascending=False)

print(f"National average predicted risk: {national_average * 100:.2f}%")
print(d[["District", "risk", "diff", "level"]].to_string(index=False))

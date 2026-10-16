df["risk"] = rf.predict_proba(X)[:, 1]
df.groupby("District")["risk"].mean().sort_values(ascending=False)

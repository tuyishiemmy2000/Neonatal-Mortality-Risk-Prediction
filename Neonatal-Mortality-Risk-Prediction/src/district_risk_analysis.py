
def analyze_district_risk(df, rf, X):
    """Calculate average predicted neonatal mortality risk by district."""

    df = df.copy()

    # Predict probability of neonatal death
    df["risk"] = rf.predict_proba(X)[:, 1]

    # Rank districts from highest to lowest average predicted risk
    district_risk = (
        df.groupby("District")["risk"]
        .mean()
        .sort_values(ascending=False)
    )

    return district_risk

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

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

colors = {
    "High": "red",
    "Medium": "orange",
    "Low": "green"
}

fig, ax = plt.subplots(figsize=(10, 9))

# Draw the bars
bars = ax.barh(
    d["District"],
    d["diff"],
    color=d["level"].map(colors)
)

# Add percentage-point labels to each bar
for bar, value in zip(bars, d["diff"]):
    if value >= 0:
        ax.text(
            value + 0.05,
            bar.get_y() + bar.get_height() / 2,
            f"+{value:.2f}%",
            va="center",
            fontsize=8
        )
    else:
        ax.text(
            value - 0.05,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}%",
            va="center",
            ha="right",
            fontsize=8
        )

# Put highest-risk district at the top
ax.invert_yaxis()

# National average reference line
ax.axvline(0, color="grey", lw=0.8)

ax.set_xlabel(
    "Difference in Predicted Neonatal Mortality Risk from the National Average (percentage points)"
)
ax.set_title(
    "District-Level Comparison of Predicted Neonatal Mortality Risk "
    "with the National Average — Rwanda DHS 2025"
)

# Legend
handles = [
    Patch(color=c, label=lvl)
    for lvl, c in colors.items()
]
ax.legend(handles=handles, title="Risk", loc="lower right")

plt.tight_layout()
plt.show()

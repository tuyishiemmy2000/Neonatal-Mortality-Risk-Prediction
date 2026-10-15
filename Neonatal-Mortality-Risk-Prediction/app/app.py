"""Optional Streamlit dashboard. Run: streamlit run app/app.py
Needs outputs/district_risk.csv (created by src/train_model.py)."""
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

d = pd.read_csv(Path(__file__).resolve().parents[1] / "outputs" / "district_risk.csv")

st.title("Neonatal mortality risk by district")
levels = st.sidebar.multiselect("Risk level", ["High", "Medium", "Low"],
                                default=["High", "Medium", "Low"])
d = d[d["level"].isin(levels)]

chart = alt.Chart(d).mark_bar().encode(
    x=alt.X("diff:Q", title="Predicted risk vs national average"),
    y=alt.Y("District:N", sort="-x", title=None),
    color=alt.Color("level:N", title="risk",
        scale=alt.Scale(domain=["High", "Medium", "Low"],
                        range=["#d62728", "#ff9f1c", "#2ca02c"])),
    tooltip=["District", "births", "observed_deaths", "deaths_per_1000",
             alt.Tooltip("mean_risk:Q", format=".3f"), "flagged_pct"])
st.altair_chart(chart, use_container_width=True)
st.dataframe(d.drop(columns=["diff"]))

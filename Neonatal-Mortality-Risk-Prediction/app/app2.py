
from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt


# ============================================================
# 1. PAGE CONFIGURATION AND FILE PATHS
# ============================================================

st.set_page_config(
    page_title="Rwanda Neonatal Risk Dashboard",
    page_icon="🏥",
    layout="wide",
)

# This is the OUTER folder containing your existing outputs folder.
ROOT = Path(__file__).resolve().parents[1]
    r"C:\Users\user\Downloads\Neonatal-Mortality-Risk-Prediction-main (3)\Neonatal-Mortality-Risk-Prediction-main"
)

OUTPUTS = ROOT / "outputs"
MODELS = OUTPUTS / "models"

PREGNANCY_MODEL_PATH = MODELS / "pregnancy_model.joblib"
DELIVERY_MODEL_PATH = MODELS / "delivery_model.joblib"
THRESHOLDS_PATH = OUTPUTS / "model_thresholds.json"
NATIONAL_PATH = OUTPUTS / "national_risk_summary.csv"
PREGNANCY_DISTRICT_PATH = OUTPUTS / "pregnancy_district_risk.csv"
DELIVERY_DISTRICT_PATH = OUTPUTS / "delivery_district_risk.csv"
DISTRICT_PATH = OUTPUTS / "district_risk.csv"


# ============================================================
# 2. MODEL FEATURES
# These must match the features used during training.
# ============================================================

PREGNANCY_FEATURES = [
    "mother_age",
    "birth_order",
    "anc_visits",
    "first_anc_month",
    "minutes_to_facility",
    "walks_to_facility",
    "is_urban",
    "prev_caesarean",
    "smokes",
    "wealth",
    "mother_edu",
    "district_code",
]

DELIVERY_FEATURES = [
    "mother_age",
    "birth_order",
    "anc_visits",
    "first_anc_month",
    "birth_weight",
    "minutes_to_facility",
    "is_male",
    "is_urban",
    "caesarean",
    "prev_caesarean",
    "smokes",
    "facility_delivery",
    "walks_to_facility",
    "wealth",
    "mother_edu",
    "district_code",
]


# ============================================================
# 3. LOAD SAVED FILES
# ============================================================

@st.cache_resource
def load_model(path_string):
    return joblib.load(path_string)


@st.cache_data
def load_csv(path_string):
    return pd.read_csv(path_string)


@st.cache_data
def load_thresholds(path_string):
    with open(path_string, "r", encoding="utf-8") as f:
        return json.load(f)


st.title("Rwanda Neonatal Mortality Risk Dashboard")
st.caption(
    "Exploratory analysis of neonatal mortality and model-estimated risk"
)

st.warning(
    "Research prototype only. These models have not been established as "
    "safe for clinical decision-making. A prediction is not a diagnosis "
    "and must not replace clinical assessment or referral protocols."
)

required_files = [
    PREGNANCY_MODEL_PATH,
    DELIVERY_MODEL_PATH,
    THRESHOLDS_PATH,
    NATIONAL_PATH,
    PREGNANCY_DISTRICT_PATH,
    DELIVERY_DISTRICT_PATH,
]

missing_files = [p for p in required_files if not p.exists()]

if missing_files:
    st.error("Some required files could not be found:")
    for path in missing_files:
        st.write(f"- `{path}`")
    st.info(
        "Check that the saved files are in the outer outputs folder "
        "and that both model files are inside outputs/models."
    )
    st.stop()

try:
    pregnancy_model = load_model(str(PREGNANCY_MODEL_PATH))
    delivery_model = load_model(str(DELIVERY_MODEL_PATH))
    thresholds = load_thresholds(str(THRESHOLDS_PATH))
    national = load_csv(str(NATIONAL_PATH))
    pregnancy_districts = load_csv(str(PREGNANCY_DISTRICT_PATH))
    delivery_districts = load_csv(str(DELIVERY_DISTRICT_PATH))
except Exception as exc:
    st.error(f"Could not load one of the saved files: {exc}")
    st.stop()


# Validate the expected feature names when the model exposes them.
def check_model_features(model, expected, stage):
    if hasattr(model, "feature_names_in_"):
        actual = list(model.feature_names_in_)
        if actual != expected:
            st.error(
                f"The {stage} model's stored feature names or order do "
                "not match this dashboard's feature configuration."
            )
            st.write("Model features:", actual)
            st.write("Dashboard features:", expected)
            st.stop()


check_model_features(
    pregnancy_model, PREGNANCY_FEATURES, "pregnancy"
)
check_model_features(
    delivery_model, DELIVERY_FEATURES, "delivery"
)

pregnancy_threshold = float(thresholds["pregnancy"])
delivery_threshold = float(thresholds["delivery"])


# ============================================================
# 4. SIDEBAR NAVIGATION
# ============================================================

page = st.sidebar.radio(
    "Navigate dashboard",
    [
        "National overview",
        "Pregnancy prediction",
        "Delivery prediction",
        "District analysis",
        "Model information",
    ],
)

st.sidebar.divider()
st.sidebar.caption("Rwanda neonatal mortality research prototype")


# ============================================================
# 5. NATIONAL OVERVIEW
# ============================================================

if page == "National overview":

    st.header("National overview")

    required_metrics = {
        "stage",
        "records",
        "observed_neonatal_deaths",
        "roc_auc",
        "pr_auc",
        "recall",
        "precision",
        "threshold",
    }

    if not required_metrics.issubset(national.columns):
        st.error(
            "The national summary CSV is missing expected columns. "
            "Check national_risk_summary.csv."
        )
        st.write("Columns found:", list(national.columns))
        st.stop()

    # The same births are analysed at both stages, so do not add
    # the pregnancy and delivery counts together.
    pregnancy_row = national[
        national["stage"].astype(str).str.lower() == "pregnancy"
    ]
    delivery_row = national[
        national["stage"].astype(str).str.lower() == "delivery"
    ]

    if pregnancy_row.empty or delivery_row.empty:
        st.error(
            "The national summary must have one pregnancy row and "
            "one delivery row."
        )
        st.stop()

    p = pregnancy_row.iloc[0]
    d = delivery_row.iloc[0]

    records = int(p["records"])
    deaths = int(p["observed_neonatal_deaths"])
    descriptive_rate = deaths / records * 1000 if records else 0

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Analysed birth records", f"{records:,}")
    c2.metric("Observed neonatal deaths", f"{deaths:,}")
    c3.metric(
        "Descriptive deaths per 1,000 records",
        f"{descriptive_rate:.1f}",
    )
    c4.metric(
        "Pregnancy model ROC-AUC",
        f"{float(p['roc_auc']):.3f}",
    )

    st.caption(
        "The descriptive rate is calculated from this analytic dataset. "
        "It is not necessarily the official national neonatal mortality rate."
    )

    st.subheader("Model performance comparison")

    comparison = national.copy()
    comparison["stage"] = comparison["stage"].str.title()

    display_cols = [
        "stage",
        "records",
        "observed_neonatal_deaths",
        "roc_auc",
        "pr_auc",
        "recall",
        "precision",
        "threshold",
    ]

    if "flagged_pct" in comparison.columns:
        display_cols.append("flagged_pct")

    st.dataframe(
        comparison[display_cols].round(3),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Average model-estimated risk")

    risk_data = pd.DataFrame({
        "Stage": ["Pregnancy", "Delivery"],
        "Mean predicted risk (%)": [
            float(p.get("oof_mean_predicted_risk", np.nan)) * 100,
            float(d.get("oof_mean_predicted_risk", np.nan)) * 100,
        ],
    })

    if risk_data["Mean predicted risk (%)"].notna().any():
        chart = alt.Chart(risk_data).mark_bar().encode(
            x=alt.X("Stage:N", title=None),
            y=alt.Y(
                "Mean predicted risk (%):Q",
                title="Mean predicted risk (%)",
            ),
            tooltip=["Stage", "Mean predicted risk (%)"],
        )
        st.altair_chart(chart, use_container_width=True)

    st.info(
        "ROC-AUC measures ranking ability. Precision-recall metrics are "
        "particularly important when neonatal deaths are rare. These "
        "summaries should not be interpreted as proof of clinical readiness."
    )


# ============================================================
# 6. PREGNANCY-STAGE INDIVIDUAL PREDICTION
# ============================================================

elif page == "Pregnancy prediction":

    st.header("Pregnancy-stage risk assessment")

    st.write(
        "Enter information available during pregnancy. Use the same "
        "numeric coding and units used to prepare the training dataset."
    )

    with st.form("pregnancy_form"):

        col1, col2 = st.columns(2)

        with col1:
            mother_age = st.number_input(
                "Mother's age (years)", 10, 60, 25
            )
            birth_order = st.number_input(
                "Birth order", 1, 20, 1
            )
            anc_visits = st.number_input(
                "ANC visits recorded", 0, 20, 4
            )
            first_anc_month = st.number_input(
                "Month of first ANC visit", 0, 10, 4
            )
            minutes_to_facility = st.number_input(
                "Travel time to facility (minutes)", 0, 1000, 30
            )
            walks_to_facility = st.selectbox(
                "walks_to_facility numeric code", [0, 1]
            )

        with col2:
            is_urban = st.selectbox("is_urban numeric code", [0, 1])
            prev_caesarean = st.selectbox(
                "prev_caesarean numeric code", [0, 1]
            )
            smokes = st.selectbox("smokes numeric code", [0, 1])
            wealth = st.selectbox(
                "Wealth quintile code", [1, 2, 3, 4, 5]
            )
            mother_edu = st.selectbox(
                "Education code", [0, 1, 2, 3]
            )
            district_code = st.number_input(
                "District code used in training",
                min_value=0,
                max_value=10000,
                value=1,
            )

        st.caption(
            "For coded fields, 0 and 1 are raw model input values, not "
            "guaranteed labels for No/Yes. Confirm their meanings in "
            "build_features.py. Enter the original training district code."
        )

        submitted = st.form_submit_button(
            "Estimate pregnancy-stage risk",
            type="primary",
        )

    if submitted:
        row = pd.DataFrame([{
            "mother_age": mother_age,
            "birth_order": birth_order,
            "anc_visits": anc_visits,
            "first_anc_month": first_anc_month,
            "minutes_to_facility": minutes_to_facility,
            "walks_to_facility": walks_to_facility,
            "is_urban": is_urban,
            "prev_caesarean": prev_caesarean,
            "smokes": smokes,
            "wealth": wealth,
            "mother_edu": mother_edu,
            "district_code": district_code,
        }], columns=PREGNANCY_FEATURES)

        probability = float(
            pregnancy_model.predict_proba(row)[0, 1]
        )

        flagged = probability >= pregnancy_threshold

        st.subheader("Prediction result")
        a, b = st.columns(2)
        a.metric("Estimated probability", f"{probability:.1%}")
        b.metric("Model threshold", f"{pregnancy_threshold:.1%}")

        if flagged:
            st.warning(
                "The estimate is at or above the model's selected "
                "flagging threshold."
            )
        else:
            st.info(
                "The estimate is below the model's selected "
                "flagging threshold. This does not guarantee a safe outcome."
            )

        st.caption(
            "This is a model estimate, not a validated clinical risk "
            "category. Do not use it alone to make care or referral decisions."
        )


# ============================================================
# 7. DELIVERY-STAGE INDIVIDUAL PREDICTION
# ============================================================

elif page == "Delivery prediction":

    st.header("Delivery-stage risk assessment")

    st.write(
        "Enter maternal information plus delivery and newborn information. "
        "Use the same numeric codes and units as the training data."
    )

    with st.form("delivery_form"):

        col1, col2 = st.columns(2)

        with col1:
            mother_age = st.number_input(
                "Mother's age (years)", 10, 60, 25, key="d_age"
            )
            birth_order = st.number_input(
                "Birth order", 1, 20, 1, key="d_order"
            )
            anc_visits = st.number_input(
                "ANC visits recorded", 0, 20, 4, key="d_anc"
            )
            first_anc_month = st.number_input(
                "Month of first ANC visit", 0, 10, 4, key="d_firstanc"
            )
            birth_weight = st.number_input(
                "Birth weight (confirm unit used in training)",
                min_value=0.0,
                max_value=10000.0,
                value=3000.0,
                step=50.0,
            )
            minutes_to_facility = st.number_input(
                "Travel time to facility (minutes)",
                0, 1000, 30, key="d_travel"
            )
            is_male = st.selectbox(
                "is_male numeric code", [0, 1]
            )
            is_urban = st.selectbox(
                "is_urban numeric code", [0, 1], key="d_urban"
            )

        with col2:
            caesarean = st.selectbox(
                "caesarean numeric code", [0, 1]
            )
            prev_caesarean = st.selectbox(
                "prev_caesarean numeric code", [0, 1], key="d_prev"
            )
            smokes = st.selectbox(
                "smokes numeric code", [0, 1], key="d_smokes"
            )
            facility_delivery = st.selectbox(
                "facility_delivery numeric code", [0, 1]
            )
            walks_to_facility = st.selectbox(
                "walks_to_facility numeric code", [0, 1], key="d_walks"
            )
            wealth = st.selectbox(
                "Wealth quintile code", [1, 2, 3, 4, 5], key="d_wealth"
            )
            mother_edu = st.selectbox(
                "Education code", [0, 1, 2, 3], key="d_edu"
            )
            district_code = st.number_input(
                "District code used in training",
                min_value=0,
                max_value=10000,
                value=1,
                key="d_district",
            )

        st.caption(
            "Confirm binary codes, education, wealth, birth-weight units, "
            "and district-code mapping against build_features.py."
        )

        submitted = st.form_submit_button(
            "Estimate delivery-stage risk",
            type="primary",
        )

    if submitted:
        row = pd.DataFrame([{
            "mother_age": mother_age,
            "birth_order": birth_order,
            "anc_visits": anc_visits,
            "first_anc_month": first_anc_month,
            "birth_weight": birth_weight,
            "minutes_to_facility": minutes_to_facility,
            "is_male": is_male,
            "is_urban": is_urban,
            "caesarean": caesarean,
            "prev_caesarean": prev_caesarean,
            "smokes": smokes,
            "facility_delivery": facility_delivery,
            "walks_to_facility": walks_to_facility,
            "wealth": wealth,
            "mother_edu": mother_edu,
            "district_code": district_code,
        }], columns=DELIVERY_FEATURES)

        probability = float(
            delivery_model.predict_proba(row)[0, 1]
        )

        flagged = probability >= delivery_threshold

        st.subheader("Prediction result")
        a, b = st.columns(2)
        a.metric("Estimated probability", f"{probability:.1%}")
        b.metric("Model threshold", f"{delivery_threshold:.1%}")

        if flagged:
            st.warning(
                "The estimate is at or above the model's selected "
                "flagging threshold."
            )
        else:
            st.info(
                "The estimate is below the model's selected flagging "
                "threshold. This does not guarantee a safe outcome."
            )

        st.caption(
            "This estimate is not a diagnosis and should not replace "
            "clinical assessment, newborn examination, or established "
            "referral protocols."
        )


# ============================================================
# 8. DISTRICT ANALYSIS
# ============================================================

elif page == "District analysis":

    st.header("District-level risk analysis")

    stage_choice = st.radio(
        "Choose prediction stage",
        ["Pregnancy", "Delivery"],
        horizontal=True,
    )

    if stage_choice == "Pregnancy":
        district = pregnancy_districts.copy()
    else:
        district = delivery_districts.copy()

    required_columns = {
        "District",
        "mean_risk",
        "diff",
        "level",
        "births",
        "observed_deaths",
    }

    missing = required_columns - set(district.columns)

    if missing:
        st.error(
            "District file is missing columns: "
            + ", ".join(sorted(missing))
        )
        st.write("Columns found:", list(district.columns))
        st.stop()

    levels = st.multiselect(
        "Filter relative risk group",
        ["High", "Medium", "Low"],
        default=["High", "Medium", "Low"],
    )

    filtered = district[district["level"].isin(levels)].copy()

    if filtered.empty:
        st.info("No districts match the selected filters.")
    else:
        c1, c2, c3 = st.columns(3)

        c1.metric("Districts displayed", len(filtered))
        c2.metric(
            "Highest mean predicted risk",
            f"{filtered['mean_risk'].max():.1%}",
        )
        c3.metric(
            "Highest observed deaths in displayed data",
            f"{int(filtered['observed_deaths'].max()):,}",
        )

        chart = alt.Chart(filtered).mark_bar().encode(
            x=alt.X(
                "diff:Q",
                title="Difference from national mean predicted risk (percentage points)",
            ),
            y=alt.Y(
                "District:N",
                sort="-x",
                title=None,
            ),
            color=alt.Color(
                "level:N",
                scale=alt.Scale(
                    domain=["High", "Medium", "Low"],
                    range=["#d62728", "#ffbf00", "#2ca02c"],
                ),
                title="Relative group",
            ),
            tooltip=[
                "District",
                alt.Tooltip("mean_risk:Q", format=".2%"),
                alt.Tooltip("diff:Q", format=".2f"),
                "level",
                "births",
                "observed_deaths",
            ],
        )

        st.altair_chart(chart, use_container_width=True)

        st.subheader("District data")
        display_columns = [
            "District",
            "births",
            "observed_deaths",
            "mean_risk",
            "diff",
            "level",
        ]

        if "flagged_pct" in filtered.columns:
            display_columns.append("flagged_pct")

        st.dataframe(
            filtered[display_columns].round(3),
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download displayed district data",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name=f"{stage_choice.lower()}_district_risk.csv",
            mime="text/csv",
        )

    st.caption(
        "Mean predicted risk is the average model-estimated probability. "
        "The High/Medium/Low groups compare districts with the national "
        "average using the analysis script's chosen cutoff; they are not "
        "official district mortality classifications."
    )


# ============================================================
# 9. MODEL INFORMATION
# ============================================================

elif page == "Model information":

    st.header("Model information and limitations")

    st.subheader("Selected thresholds")

    c1, c2 = st.columns(2)
    c1.metric("Pregnancy threshold", f"{pregnancy_threshold:.2%}")
    c2.metric("Delivery threshold", f"{delivery_threshold:.2%}")

    st.write(
        "The threshold is the probability cutoff selected by the training "
        "analysis to target a specified recall. It is not a clinically "
        "approved definition of high risk."
    )

    st.subheader("Available performance results")
    st.dataframe(national, use_container_width=True, hide_index=True)

    st.subheader("Feature importance")

    for stage in ["pregnancy", "delivery"]:
        importance_path = OUTPUTS / f"{stage}_feature_importance.csv"

        if importance_path.exists():
            st.write(f"**{stage.title()} model**")
            importance = load_csv(str(importance_path))

            if {"feature", "importance"}.issubset(importance.columns):
                importance_chart = alt.Chart(importance).mark_bar().encode(
                    x=alt.X("importance:Q", title="Feature importance"),
                    y=alt.Y("feature:N", sort="-x", title=None),
                    tooltip=["feature", "importance"],
                )
                st.altair_chart(
                    importance_chart,
                    use_container_width=True,
                )
            else:
                st.dataframe(importance, use_container_width=True)

        else:
            st.info(
                f"No feature-importance file found for the {stage} model."
            )

    st.subheader("Interpretation and safety")

    st.markdown(
        """
        - The dataset is imbalanced: neonatal deaths are uncommon compared
          with surviving births.
        - Precision, recall, PR-AUC, calibration, and external validation
          matter in addition to ROC-AUC.
        - Out-of-fold results used for threshold selection are not a fully
          independent final evaluation.
        - The model's training codes and measurement units must match the
          dashboard's input values.
        - Do not enter identifiable patient information into this prototype.
        - The application does not connect to OpenMRS or a hospital database.
        """
    )

    st.write("**Loaded model files**")
    st.code(
        f"Pregnancy model: {PREGNANCY_MODEL_PATH}\n"
        f"Delivery model: {DELIVERY_MODEL_PATH}\n"
        f"Thresholds: {THRESHOLDS_PATH}"
    )

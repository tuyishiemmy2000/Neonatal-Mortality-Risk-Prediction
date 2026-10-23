
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

## Locate the project folder containing your outputs folder.
ROOT = Path(__file__).resolve().parents[1]

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

st.header("Machine Learning Method")

st.info(
    "This dashboard uses the Random Forest Classifier "
    "for neonatal mortality risk prediction."
)


# ============================================================
# SIDEBAR NAVIGATION
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
    key="main_dashboard_navigation",
)


st.sidebar.divider()
st.sidebar.caption("Rwanda neonatal mortality research prototype")

# ============================================================
# DISPLAY ONLY THE RELEVANT MACHINE LEARNING SECTION
# ============================================================

if page == "Pregnancy prediction":
    st.header("Pregnancy Assessment")
    st.write("**Algorithm:** Random Forest")
    st.write(
        "Uses information available during pregnancy to estimate "
        "neonatal mortality risk before delivery."
    )

elif page == "Delivery prediction":
    st.header("Delivery Assessment")
    st.write("**Algorithm:** Random Forest")
    st.write(
        "Uses maternal, pregnancy, and delivery-stage information "
        "to estimate neonatal mortality risk when the baby is born."
    )

elif page == "District analysis":
    st.header("District Risk Analysis")
    st.write("**Algorithm:** Random Forest")
    st.write(
        "Compares average model-estimated risk across districts "
        "using Random Forest predictions."
    )

elif page == "Model information":
    st.header("Model Information")
    st.write("**Algorithm:** Random Forest Classifier")
    st.write(
        "Explains model performance, selected thresholds, "
        "feature importance, and limitations."
    )

st.divider()


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

except Exception as e:
    st.error(f"Could not load required project files: {e}")
    st.stop()

# District mapping goes AFTER the try/except block
district_names = sorted(
    set(pregnancy_districts["District"].dropna().astype(str))
    | set(delivery_districts["District"].dropna().astype(str))
)

district_to_code = {
    name: code for code, name in enumerate(district_names)
}
    # District names for user-friendly dropdowns.
# cat.codes assigns codes in alphabetical category order.
district_names = sorted(
    set(pregnancy_districts["District"].dropna().astype(str))
    | set(delivery_districts["District"].dropna().astype(str))
)

district_to_code = {
    name: code for code, name in enumerate(district_names)
}



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
# 
# ============================================================
# 6. PREGNANCY-STAGE INDIVIDUAL PREDICTION
# ============================================================

elif page == "Pregnancy prediction":

    st.header("Pregnancy-stage risk assessment")
    st.write(
        "Enter information available during pregnancy. "
        "Choose descriptive answers instead of numeric codes."
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
                "Month of first ANC visit", 1, 10, 4
            )
            minutes_to_facility = st.number_input(
                "Travel time to facility (minutes)",
                0, 1000, 30
            )

            walks_label = st.selectbox(
                "Does the mother walk to the health facility?",
                ["No", "Yes"]
            )
            walks_to_facility = {"No": 0, "Yes": 1}[walks_label]

        with col2:
            residence_label = st.selectbox(
                "Place of residence",
                ["Rural", "Urban"]
            )
            is_urban = {"Rural": 0, "Urban": 1}[residence_label]

            prev_label = st.selectbox(
                "Previous caesarean section",
                ["No", "Yes"]
            )
            prev_caesarean = {"No": 0, "Yes": 1}[prev_label]

            smoke_label = st.selectbox(
                "Does the mother smoke cigarettes?",
                ["No", "Yes"]
            )
            smokes = {"No": 0, "Yes": 1}[smoke_label]

            wealth_label = st.selectbox(
                "Household wealth quintile",
                ["Poorest", "Poorer", "Middle", "Richer", "Richest"]
            )
            wealth = {
                "Poorest": 1,
                "Poorer": 2,
                "Middle": 3,
                "Richer": 4,
                "Richest": 5,
            }[wealth_label]

            education_label = st.selectbox(
                "Mother's highest education level",
                ["No education", "Primary", "Secondary", "Higher"]
            )
            mother_edu = {
                "No education": 0,
                "Primary": 1,
                "Secondary": 2,
                "Higher": 3,
            }[education_label]

            selected_district = st.selectbox(
                "District",
                district_names
            )
            district_code = district_to_code[selected_district]

        st.caption(
            "The descriptive choices are converted to numeric model "
            "inputs. District codes must match the training mapping."
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
                "The estimate is below the model's selected threshold. "
                "This does not guarantee a safe outcome."
            )

        st.caption(
            "Research estimate only. It is not a diagnosis or a "
            "validated clinical risk category. Do not use it alone "
            "to make care or referral decisions."
        )

# ============================================================

# ============================================================
# 7. DELIVERY-STAGE INDIVIDUAL PREDICTION
# ============================================================

elif page == "Delivery prediction":

    st.header("Delivery-stage risk assessment")
    st.write(
        "Enter maternal information and information available "
        "at delivery or after the newborn's birth."
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
                "Month of first ANC visit", 1, 10, 4,
                key="d_firstanc"
            )
            birth_weight = st.number_input(
                "Birth weight (confirm training-data unit)",
                min_value=0.0,
                max_value=10000.0,
                value=3000.0,
                step=50.0,
            )
            minutes_to_facility = st.number_input(
                "Travel time to facility (minutes)",
                0, 1000, 30, key="d_travel"
            )

            sex_label = st.selectbox(
                "Sex of newborn",
                ["Female", "Male"]
            )
            is_male = {"Female": 0, "Male": 1}[sex_label]

            residence_label = st.selectbox(
                "Place of residence",
                ["Rural", "Urban"], key="d_residence"
            )
            is_urban = {"Rural": 0, "Urban": 1}[residence_label]

            walks_label = st.selectbox(
                "Does the mother walk to the health facility?",
                ["No", "Yes"], key="d_walks_label"
            )
            walks_to_facility = {"No": 0, "Yes": 1}[walks_label]

        with col2:
            caesarean_label = st.selectbox(
                "Was this delivery by caesarean section?",
                ["No", "Yes"]
            )
            caesarean = {"No": 0, "Yes": 1}[caesarean_label]

            prev_label = st.selectbox(
                "Previous caesarean section",
                ["No", "Yes"], key="d_prev_label"
            )
            prev_caesarean = {"No": 0, "Yes": 1}[prev_label]

            smoke_label = st.selectbox(
                "Does the mother smoke cigarettes?",
                ["No", "Yes"], key="d_smoke_label"
            )
            smokes = {"No": 0, "Yes": 1}[smoke_label]

            delivery_place_label = st.selectbox(
                "Place of delivery",
                [
                    "Health facility",
                    "Respondent's home or other home",
                ]
            )
            facility_delivery = {
                "Health facility": 1,
                "Respondent's home or other home": 0,
            }[delivery_place_label]

            wealth_label = st.selectbox(
                "Household wealth quintile",
                ["Poorest", "Poorer", "Middle", "Richer", "Richest"],
                key="d_wealth_label"
            )
            wealth = {
                "Poorest": 1,
                "Poorer": 2,
                "Middle": 3,
                "Richer": 4,
                "Richest": 5,
            }[wealth_label]

            education_label = st.selectbox(
                "Mother's highest education level",
                ["No education", "Primary", "Secondary", "Higher"],
                key="d_education_label"
            )
            mother_edu = {
                "No education": 0,
                "Primary": 1,
                "Secondary": 2,
                "Higher": 3,
            }[education_label]

            selected_district = st.selectbox(
                "District",
                district_names,
                key="d_district_name"
            )
            district_code = district_to_code[selected_district]

        st.caption(
            "Descriptive choices are converted to numeric model inputs. "
            "Confirm birth-weight units and district-code mapping against "
            "the training pipeline."
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
                "The estimate is below the model's selected threshold. "
                "This does not guarantee a safe outcome."
            )

        st.caption(
            "Research estimate only. It is not a diagnosis and must "
            "not replace clinical assessment, newborn examination, "
            "or established referral protocols."
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

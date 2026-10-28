


from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

from altair.datasets import data
from altair.datasets import data
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from sklearn.inspection import permutation_importance
import base64

import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import streamlit as st
import base64
from pathlib import Path

# 1. Configure the dashboard
import streamlit as st
import base64
from pathlib import Path

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# Keep this before other Streamlit commands
# --------------------------------------------------
st.set_page_config(
    page_title="Neonatal Mortality Risk Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# 2. LOAD BACKGROUND IMAGE
# The image must be in the same folder as app2.py
# --------------------------------------------------
BACKGROUND_PATH = Path(__file__).resolve().parent / "background.jpg.png"

if BACKGROUND_PATH.is_file():

    encoded_image = base64.b64encode(
        BACKGROUND_PATH.read_bytes()
    ).decode("utf-8")

    # --------------------------------------------------
    # 3. BACKGROUND AND READABILITY STYLES
    # --------------------------------------------------
    st.markdown(
        f"""
        <style>

        /* Background image with a soft white overlay */
        .stApp {{
            background-image:
                linear-gradient(
                    rgba(255, 255, 255, 0.90),
                    rgba(255, 255, 255, 0.94)
                ),
                url("data:image/png;base64,{encoded_image}");

            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}

        /* Improve readability of headings and text */
        h1, h2, h3, p, label {{
            color: #183536;
        }}

        /* Give bordered dashboard containers a clean surface */
        div[data-testid="stVerticalBlockBorderWrapper"] {{
            background-color: rgba(255, 255, 255, 0.92);
            border-radius: 12px;
        }}

        /* Make metric cards easier to read */
        div[data-testid="stMetric"] {{
            background-color: rgba(255, 255, 255, 0.96);
            padding: 14px;
            border-radius: 12px;
            border: 1px solid #E1E8E7;
        }}

        </style>
        """,
        unsafe_allow_html=True,
    )

else:
    st.warning(f"Background image not found: {BACKGROUND_PATH}")


# --------------------------------------------------
# 4. YOUR EXISTING DASHBOARD CODE STARTS BELOW
# Keep your current functions, navigation, charts,
# prediction forms, and other pages below this line.
# --------------------------------------------------


warnings.filterwarnings("ignore", category=FutureWarning)
# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
OUTPUTS = ROOT / "outputs"
MODEL_DIR = OUTPUTS / "models"
DATA_PATH = ROOT / "data" / "merged_table.csv"
PREGNANCY_MODEL_PATH = MODEL_DIR / "pregnancy_model.joblib"
DELIVERY_MODEL_PATH = MODEL_DIR / "delivery_model.joblib"
THRESHOLDS_PATH = OUTPUTS / "model_thresholds.json"
PREGNANCY_DISTRICT_PATH = OUTPUTS / "pregnancy_district_risk.csv"
DELIVERY_DISTRICT_PATH = OUTPUTS / "delivery_district_risk.csv"


if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    from build_features import PREGNANCY_FEATURES, DELIVERY_FEATURES
except ImportError as exc:
    raise ImportError(
        "Could not import PREGNANCY_FEATURES and DELIVERY_FEATURES "
        "from src/build_features.py. Check that the file and constants exist."
    ) from exc

PREGNANCY_MODEL_PATH = MODEL_DIR / "pregnancy_model.joblib"
DELIVERY_MODEL_PATH = MODEL_DIR / "delivery_model.joblib"
THRESHOLD_PATH = OUTPUTS / "model_thresholds.json"

NATIONAL_PATH = OUTPUTS / "national_risk_summary.csv"
PREGNANCY_DISTRICT_PATH = OUTPUTS / "pregnancy_district_risk.csv"
DELIVERY_DISTRICT_PATH = OUTPUTS / "delivery_district_risk.csv"
COMPARISON_PATH = OUTPUTS / "model_comparison.csv"

st.set_page_config(
    page_title="Rwanda Neonatal Mortality Risk Dashboard",
    
    page_icon=r"C:\Users\user\Downloads\Neonatal Mortality Project Report.png",

    layout="wide",
)


# ============================================================
# 2. GENERAL HELPERS
# ============================================================

def first_existing_path(*paths: Path) -> Path | None:
    """Return the first path that exists."""
    for path in paths:
        if path.exists():
            return path
    return None


@st.cache_data
def load_csv(path_string: str) -> pd.DataFrame:
    """Load a CSV file and cache the result."""
    return pd.read_csv(path_string)


@st.cache_resource
def load_model(path_string: str):
    """Load an existing trained model."""
    return joblib.load(path_string)



def load_thresholds(path_string: str) -> dict:
    """Read the current threshold values from the JSON file."""
    with open(path_string, "r", encoding="utf-8") as file:
        return json.load(file)



def read_optional_csv(path):
    path = Path(path)  # Convert a string to a Path object

    if not path.exists():
        return None

    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        st.warning(f"Could not read {path.name}: {e}")
        return None
    """Return an empty DataFrame when an optional file is unavailable."""
    if not path.exists():
        return pd.DataFrame()

    try:
        return load_csv(str(path))
    except Exception as exc:
        st.warning(f"Could not read {path.name}: {exc}")
        return pd.DataFrame()


def get_threshold(stage: str) -> float:
    """
    Read the screening threshold saved during model development.

    The names below accommodate common threshold JSON formats.
    Confirm that the keys match your training script.
    """
    if not THRESHOLD_PATH.exists():
        raise FileNotFoundError(
            f"Threshold file not found: {THRESHOLD_PATH}"
        )

    thresholds = load_thresholds(str(THRESHOLD_PATH))

    aliases = {
        "pregnancy": [
            "pregnancy",
            "before_birth",
            "pregnancy_threshold",
            "before_birth_threshold",
        ],
        "delivery": [
            "delivery",
            "after_delivery",
            "delivery_threshold",
            "after_delivery_threshold",
        ],
    }

    for key in aliases[stage]:
        if key in thresholds:
            value = thresholds[key]

            if isinstance(value, dict):
                for nested_key in ("threshold", "cutoff", "value"):
                    if nested_key in value:
                        value = value[nested_key]
                        break

            try:
                value = float(value)
                if 0 <= value <= 1:
                    return value
            except (TypeError, ValueError):
                continue

    raise ValueError(
        f"Could not find a valid {stage} threshold in "
        f"{THRESHOLD_PATH.name}. Check the JSON keys."
    )



def check_model_features(model, expected_features: list[str]) -> None:
    """Validate that dashboard features match the saved model."""
    model_features = getattr(model, "feature_names_in_", None)

    if model_features is None:
        raise ValueError(
            "The saved model does not expose feature_names_in_. "
            "Please check the model training code."
        )

    model_features = list(model_features)
    expected_features = list(expected_features)

    missing_from_model = [
        feature for feature in expected_features
        if feature not in model_features
    ]

    unexpected_in_model = [
        feature for feature in model_features
        if feature not in expected_features
    ]

    if missing_from_model or unexpected_in_model:
        raise ValueError(
            "Model and dashboard feature lists do not match. "
            f"Missing from model: {missing_from_model}. "
            f"Unexpected in model: {unexpected_in_model}."
        )



def get_district_names() -> list[str]:
    """Get district names from the available merged dataset."""
    if not DATA_PATH.exists():
        return []

    try:
        data = load_csv(str(DATA_PATH))
        if "District" in data.columns:
            names = data["District"].dropna().astype(str).unique()
            return sorted(names.tolist())
    except Exception:
        pass

    return []


def make_district_code_map() -> dict[str, int]:
    """
    Build a sorted district-to-code mapping.

    IMPORTANT: This must match the encoding used when the models were
    trained. If build_features.py uses a different mapping, replace this
    function with that exact saved mapping.
    """
    districts = get_district_names()
    return {name: index for index, name in enumerate(districts)}


def show_missing_files() -> None:
    """Report required files without crashing the entire application."""
    required = [
        PREGNANCY_MODEL_PATH,
        DELIVERY_MODEL_PATH,
        THRESHOLD_PATH,
    ]

    missing = [str(path) for path in required if not path.exists()]

    if missing:
        st.error("Some required project files are missing:")
        for path in missing:
            st.write(f"- `{path}`")

        st.info(
            "Check that the trained models and model_thresholds.json "
            "are saved in the outputs folder."
        )


# ============================================================
# 3. MODEL AND THRESHOLD LOADING
# ============================================================


@st.cache_resource
def get_model(stage: str):
    """Load the saved model for the requested prediction stage."""
    if stage in ("pregnancy", "before_birth", "before_delivery"):
        model_path = PREGNANCY_MODEL_PATH
    elif stage in ("delivery", "after_delivery"):
        model_path = DELIVERY_MODEL_PATH
    else:
        raise ValueError(f"Unknown prediction stage: {stage}")

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file was not found: {model_path}"
        )

    return joblib.load(model_path)

    if stage == "pregnancy":
        path = PREGNANCY_MODEL_PATH
    else:
        path = DELIVERY_MODEL_PATH

    if not path.exists():
        raise FileNotFoundError(f"Saved model not found: {path}")

    return load_model(str(path))



def get_stage_features(stage: str) -> list[str]:
    """Return the feature list that matches the saved model."""
    if stage in ("pregnancy", "before_birth", "before_delivery"):
        return list(PREGNANCY_FEATURES)

    if stage in ("delivery", "after_delivery"):
        return list(DELIVERY_FEATURES)

    raise ValueError(f"Unknown prediction stage: {stage}")



# ============================================================
# 4. INPUT FORM
# ============================================================

def collect_prediction_inputs(stage: str) -> dict:
    """
    Collect inputs using the feature names from build_features.py.

    Values must use the same units and encodings used during training.
    """
    features = get_stage_features(stage)
    values = {}

    district_names = get_district_names()

    st.subheader("Maternal and pregnancy information")

    for feature in features:

        # Maternal age
        if feature == "mother_age":
            values[feature] = st.number_input(
                "Mother's age (years)",
                min_value=10,
                max_value=60,
                value=25,
                step=1,
            )

        # Birth order
        elif feature == "birth_order":
            values[feature] = st.number_input(
                "Birth order",
                min_value=1,
                max_value=20,
                value=1,
                step=1,
            )

        # Antenatal care
        elif feature == "anc_visits":
            values[feature] = st.number_input(
                "Number of ANC visits",
                min_value=0,
                max_value=20,
                value=4,
                step=1,
            )

        elif feature == "first_anc_month":
            values[feature] = st.number_input(
                "Month of first ANC visit",
                min_value=1,
                max_value=9,
                value=3,
                step=1,
            )

        # Access to healthcare
        elif feature == "minutes_to_facility":
            values[feature] = st.number_input(
                "Travel time to health facility (minutes)",
                min_value=0,
                max_value=600,
                value=30,
                step=5,
            )

        elif feature == "walks_to_facility":
            answer = st.selectbox(
                "Does the mother walk to the health facility?",
                ["No", "Yes"],
                key=f"{stage}_{feature}",
            )
            values[feature] = 1 if answer == "Yes" else 0

        elif feature == "is_urban":
            answer = st.selectbox(
                "Place of residence",
                ["Rural", "Urban"],
                key=f"{stage}_{feature}",
            )
            values[feature] = 1 if answer == "Urban" else 0
        
        elif feature == "District":
            if district_names:
                values[feature] = st.selectbox(
                    "District",
                    district_names,
                    key=f"{stage}_district",
                )
            else:
                st.error(
                    "District names could not be loaded. "
                    "Check the configured data file."
                )
                values[feature] = None

        # Previous health and behavioural factors
        elif feature == "prev_caesarean":
            answer = st.selectbox(
                "Previous caesarean section",
                ["No", "Yes"],
                key=f"{stage}_{feature}",
            )
            values[feature] = 1 if answer == "Yes" else 0

        elif feature == "smokes":
            answer = st.selectbox(
                "Does the mother smoke cigarettes?",
                ["No", "Yes"],
                key=f"{stage}_{feature}",
            )
            values[feature] = 1 if answer == "Yes" else 0

        # Wealth and education encodings
        elif feature == "wealth":
            wealth_options = {
                "Poorest": 1,
                "Poorer": 2,
                "Middle": 3,
                "Richer": 4,
                "Richest": 5,
            }
            wealth_label = st.selectbox(
                "Household wealth quintile",
                list(wealth_options.keys()),
                index=0,
                key=f"{stage}_{feature}",
            )
            values[feature] = wealth_options[wealth_label]

        elif feature == "mother_edu":
            education_options = {
                "No education": 0,
                "Primary": 1,
                "Secondary": 2,
                "Higher": 3,
            }
            education_label = st.selectbox(
                "Mother's highest education level",
                list(education_options.keys()),
                index=0,
                key=f"{stage}_{feature}",
            )
            values[feature] = education_options[education_label]

        # Delivery-only features
        elif feature == "birth_weight":
            st.subheader("Birth and delivery information")
            weight = st.number_input(
                "Birth weight (grams)",
                min_value=300,
                max_value=7000,
                value=3000,
                step=50,
                key=f"{stage}_{feature}",
            )
            values[feature] = weight

        elif feature == "is_male":
            answer = st.selectbox(
                "Sex of child",
                ["Female", "Male"],
                key=f"{stage}_{feature}",
            )
            values[feature] = 1 if answer == "Male" else 0

        elif feature == "caesarean":
            answer = st.selectbox(
                "Was delivery by caesarean section?",
                ["No", "Yes"],
                key=f"{stage}_{feature}",
            )
            values[feature] = 1 if answer == "Yes" else 0

        elif feature == "facility_delivery":
            answer = st.selectbox(
                "Place of delivery",
                ["Health facility", "Home or other"],
                key=f"{stage}_{feature}",
            )
            values[feature] = (
                1 if answer == "Health facility" else 0
            )

        # Catch unexpected features instead of silently omitting them.
        else:
            st.warning(
                f"The feature `{feature}` does not have a dedicated "
                "input control yet. A numeric input is being used."
            )
            values[feature] = st.number_input(
                f"{feature.replace('_', ' ').title()}",
                value=0.0,
                key=f"{stage}_{feature}",
            )

    return values


# ============================================================
# 5. PREDICTION AND CLINICAL REVIEW
# ============================================================

def show_prediction_result(
    stage: str,
    values: dict,
) -> None:
    """Run the saved model and display a cautious screening result."""
    try:
        model = get_model(stage)
        features = get_stage_features(stage)
        threshold = get_threshold(stage)

        

        check_model_features(model, features)


        input_frame = pd.DataFrame(
            [[values[feature] for feature in features]],
            columns=features,
        )

        if not hasattr(model, "predict_proba"):
            st.error(
                "The saved model does not support predict_proba()."
            )
            return

        probability = float(model.predict_proba(input_frame)[0, 1])

        if not np.isfinite(probability):
            st.error("The model returned an invalid probability.")
            return

        flagged = probability >= threshold

        st.divider()
        st.subheader("Estimated screening result")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Estimated probability",
            f"{probability * 100:.2f}%",
        )
        col2.metric(
            "Saved screening threshold",
            f"{threshold * 100:.2f}%",
        )
        col3.metric(
            "Model screening flag",
            "Flagged" if flagged else "Not flagged",
        )

        if flagged:
            st.warning(
                "This profile meets the model's screening threshold. Woman needs further clinical Attention "
            )
        else:
            st.info(
                "This profile is below the model's screening threshold. "
                "This does not establish that the pregnancy or newborn "
                "is clinically low risk. Continue routine assessment "
                "and follow approved protocols."
            )

        st.markdown("**Information submitted to the model**")
        st.dataframe(
            pd.DataFrame(
                {
                    "Feature": list(values.keys()),
                    "Entered value": list(values.values()),
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        result = {
            "assessment_stage": stage,
            "estimated_probability": probability,
            "threshold": threshold,
            "screening_flag": bool(flagged),
            **values,
        }

        st.download_button(
            "Download assessment result",
            data=pd.DataFrame([result]).to_csv(index=False),
            file_name=f"{stage}_screening_result.csv",
            mime="text/csv",
        )

        

    except Exception as exc:
        st.error(f"Unable to generate the prediction: {exc}")
        st.info(
            "Check that the model, threshold file, feature names, "
            "and feature encodings match your training pipeline."
        )



def pregnancy_prediction_page() -> None:
    st.title("Pregnancy-stage prediction")

    st.write(
        "Estimate model-predicted neonatal mortality probability using "
        "information intended to be available before delivery."
    )

    with st.form("pregnancy_prediction_form"):
        values = collect_prediction_inputs("pregnancy")

        submitted = st.form_submit_button(
            "Estimate pregnancy-stage probability",
            type="primary",
        )

    if submitted:
        show_prediction_result("pregnancy", values)

    st.subheader("Where to go next")

    st.write(
        "Use the National overview to review summary statistics, "
        "or visit District analysis to compare average predicted "
        "neonatal mortality risk across districts."
    )



def delivery_prediction_page() -> None:
    st.title("Delivery-stage prediction")
    st.write(
        "Estimate model-predicted neonatal mortality probability using "
        "pregnancy information plus the delivery and birth features "
        "included in the trained model."
    )

    st.warning(
        "This assessment uses delivery-stage information and is not "
        "a substitute for newborn examination, emergency assessment, "
        "or approved clinical protocols."
    )

    with st.form("delivery_prediction_form"):
        values = collect_prediction_inputs("delivery")
        submitted = st.form_submit_button(
            "Estimate delivery-stage probability",
            type="primary",
        )

    if submitted:
        show_prediction_result("delivery", values)


# ============================================================
# 6. NATIONAL OVERVIEW
# ============================================================


def national_overview_page() -> None:
    st.title("Rwanda Neonatal Mortality Risk Dashboard")

    st.write(
        "National overview of the study dataset and model-generated "
        "risk estimates."
    )

    st.info(
        "This is a research prototype. Descriptive statistics from the "
        "study dataset are not necessarily national population estimates."
    )

    # Locate and load the study dataset
    data_path = first_existing_path(
        DATA_PATH,
        ROOT / "merged_table.csv",
    )

    data = (
        read_optional_csv(data_path)
        if data_path
        else pd.DataFrame()
    )

    summary = read_optional_csv(NATIONAL_PATH)
    comparison = read_optional_csv(COMPARISON_PATH)

    # Study dataset statistics
    if data is not None and not data.empty:
        st.subheader("Study dataset")

        c1, c2 = st.columns(2)

        c1.metric(
            "Records in loaded dataset",
            f"{len(data):,}",
        )

        c2.metric(
            "Districts represented",
            (
                f"{data['District'].nunique():,}"
                if "District" in data.columns
                else "Not available"
            ),
        )
    else:
        st.warning(
            "The merged dataset was not found at the expected project "
            "path. Dataset-level descriptive statistics are unavailable."
        )

    # Saved national risk summary
    if summary is not None and not summary.empty:
        st.subheader("Saved national risk summary")
        st.dataframe(summary, use_container_width=True)
    else:
        st.info(
            "The saved national risk summary is not available. "
            "Check the configured national summary CSV path."
        )

    # Model comparison results
    if comparison is not None and not comparison.empty:
        st.subheader("Model comparison")
        st.dataframe(comparison, use_container_width=True)
    else:
        st.info(
            "Model comparison results are not available. "
            "Check whether the model comparison CSV exists in the outputs folder."
        )

    # Dashboard interpretation
    st.subheader("How to interpret this dashboard")

    st.markdown(
        """
        - **Descriptive statistics** summarize the available study data.
        - **Estimated probability** is generated by a trained model.
        - **Screening threshold** is a cutoff selected during model
          development; it is not a clinical guideline.
        - **District average predicted risk** is a model-based summary
          and is not the same as observed neonatal mortality.
        """
    )



# ============================================================
# 7. DISTRICT ANALYSIS
# ============================================================

def find_column(
    frame: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    """Find the first matching column, ignoring case and spaces."""
    normalized = {
        str(column).strip().lower().replace(" ", "_"): column
        for column in frame.columns
    }

    for candidate in candidates:
        key = candidate.strip().lower().replace(" ", "_")
        if key in normalized:
            return normalized[key]

    return None






import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------
# Look and feel
# --------------------------------------------------------------------
COLORS = {"High": "#C62828", "Medium": "#E5A000", "Low": "#238636"}
CAT_ORDER = ["High", "Medium", "Low"]
STAGE_COLORS = {"Pregnancy": "#378ADD", "Delivery": "#D85A30"}

RISK_CANDIDATES = [
    "mean_predicted_risk", "average_predicted_risk",
    "predicted_risk", "mean_risk", "average_risk",
]
DISTRICT_CANDIDATES = ["District", "district_name", "district", "District Name"]
BIRTH_CANDIDATES = [
    "observed_births", "number_of_births", "total_births",
    "births", "n_births", "sample_size", "n",
]
DEATH_CANDIDATES = [
    "observed_deaths", "observed_neonatal_deaths",
    "neonatal_deaths", "deaths",
]

CSS = """
<style>
.hero {
    background: #0F3D3E; border-radius: 14px;
    padding: 26px 30px; margin-bottom: 14px;
}
.hero h1 { color: #FFFFFF; font-size: 1.9rem; margin: 0; padding: 0; }
.hero p  { color: #A9D6CF; margin: 6px 0 0 0; font-size: 0.98rem; }
.kpi {
    background: #FFFFFF; border: 1px solid #DADFD9; border-radius: 12px;
    padding: 14px 18px;
}
.kpi .label { color: #5F6B66; font-size: 0.82rem; }
.kpi .value { color: #13302F; font-size: 1.7rem; font-weight: 700; line-height: 1.3; }
.kpi .value.small { font-size: 1.15rem; }
.story {
    background: #EAF3F1; border-left: 5px solid #0F3D3E;
    border-radius: 6px; padding: 12px 16px; margin: 14px 0;
    color: #13302F; font-size: 0.98rem;
}
</style>
"""


def _kpi(label: str, value: str, small: bool = False) -> str:
    cls = "value small" if small else "value"
    return (
        f'<div class="kpi"><div class="label">{label}</div>'
        f'<div class="{cls}">{value}</div></div>'
    )


# --------------------------------------------------------------------
# Data loading and preparation
# --------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_stage(path: str) -> pd.DataFrame:
    """Read one saved district summary and return a clean table with
    columns: District, Risk (in %), Births, Deaths (the last two may be NaN)."""
    raw = read_optional_csv(path)
    if raw is None or raw.empty:
        return pd.DataFrame()

    district_col = find_column(raw, DISTRICT_CANDIDATES)
    risk_col = find_column(raw, RISK_CANDIDATES)
    if district_col is None or risk_col is None:
        return pd.DataFrame()

    births_col = find_column(raw, BIRTH_CANDIDATES)
    deaths_col = find_column(raw, DEATH_CANDIDATES)

    out = pd.DataFrame({
        "District": raw[district_col].astype(str).str.strip(),
        "Risk": pd.to_numeric(raw[risk_col], errors="coerce"),
    })
    out["Births"] = (
        pd.to_numeric(raw[births_col], errors="coerce") if births_col else np.nan
    )
    out["Deaths"] = (
        pd.to_numeric(raw[deaths_col], errors="coerce") if deaths_col else np.nan
    )
    out = out.dropna(subset=["District", "Risk"])
    if out.empty:
        return out

    # Saved values may be proportions (0.025) or percentages (2.5).
    if out["Risk"].max() <= 1:
        out["Risk"] = out["Risk"] * 100

    # One row per district.
    out = out.groupby("District", as_index=False).agg(
        Risk=("Risk", "mean"),
        Births=("Births", lambda s: s.sum(min_count=1)),
        Deaths=("Deaths", lambda s: s.sum(min_count=1)),
    )
    return out


def add_categories(df: pd.DataFrame, k: float) -> pd.DataFrame:
    """High / Medium / Low relative to the district mean, using k standard
    deviations as the band. These are relative research categories."""
    df = df.copy()
    mean = df["Risk"].mean()
    band = k * df["Risk"].std(ddof=0)
    df["Category"] = np.select(
        [df["Risk"] > mean + band, df["Risk"] < mean - band],
        ["High", "Low"],
        default="Medium",
    )
    return df


# --------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------
def tile_map(df: pd.DataFrame, ncols: int = 6) -> alt.Chart:
    """Each district is a coloured tile, ordered from highest to lowest risk."""
    d = df.sort_values("Risk", ascending=False).reset_index(drop=True)
    d["col"] = d.index % ncols
    d["row"] = d.index // ncols
    d["label"] = d["Risk"].map(lambda x: f"{x:.2f}%")
    n_rows = int(d["row"].max()) + 1

    color = alt.Color(
        "Category:N",
        scale=alt.Scale(domain=CAT_ORDER, range=[COLORS[c] for c in CAT_ORDER]),
        legend=alt.Legend(title="Relative risk", orient="bottom"),
    )
    text_color = alt.condition(
        alt.datum.Category == "Medium", alt.value("#2B2B2B"), alt.value("white")
    )
    base = alt.Chart(d).encode(
        x=alt.X("col:O", axis=None),
        y=alt.Y("row:O", axis=None, sort="ascending"),
    )
    rects = base.mark_rect(cornerRadius=10, stroke="white", strokeWidth=5).encode(
        color=color,
        tooltip=[
            alt.Tooltip("District:N"),
            alt.Tooltip("Risk:Q", title="Predicted risk (%)", format=".2f"),
            alt.Tooltip("Category:N", title="Relative category"),
        ],
    )
    names = base.mark_text(dy=-10, fontSize=12).encode(
        text="District:N", color=text_color
    )
    values = base.mark_text(dy=10, fontSize=16, fontWeight="bold").encode(
        text="label:N", color=text_color
    )
    return (
        (rects + names + values)
        .properties(height=max(120, n_rows * 78))
        .configure_view(strokeWidth=0)
    )


def ranked_bars(df: pd.DataFrame) -> alt.Chart:
    d = df.sort_values("Risk", ascending=False)
    base = alt.Chart(d).encode(
        y=alt.Y("District:N", sort="-x", title=None),
        x=alt.X(
            "Risk:Q",
            title="Average predicted neonatal mortality risk (%)",
            scale=alt.Scale(zero=True),
            axis=alt.Axis(format=".1f"),
        ),
    )
    bars = base.mark_bar(cornerRadiusEnd=6, size=16).encode(
        color=alt.Color(
            "Category:N",
            scale=alt.Scale(domain=CAT_ORDER, range=[COLORS[c] for c in CAT_ORDER]),
            legend=None,
        ),
        tooltip=[
            alt.Tooltip("District:N"),
            alt.Tooltip("Risk:Q", title="Predicted risk (%)", format=".2f"),
            alt.Tooltip("Category:N", title="Relative category"),
        ],
    )
    labels = base.mark_text(align="left", dx=5, fontSize=11).encode(
        text=alt.Text("Risk:Q", format=".2f")
    )
    return (bars + labels).properties(height=max(300, len(d) * 28))


def dumbbell(preg: pd.DataFrame, deliv: pd.DataFrame) -> alt.Chart | None:
    wide = preg[["District", "Risk"]].merge(
        deliv[["District", "Risk"]],
        on="District", suffixes=("_preg", "_deliv"),
    ).rename(columns={"Risk_preg": "Pregnancy", "Risk_deliv": "Delivery"})
    if wide.empty:
        return None

    order = wide.sort_values("Delivery", ascending=False)["District"].tolist()
    long = wide.melt(
        id_vars="District", value_vars=["Pregnancy", "Delivery"],
        var_name="Stage", value_name="Risk",
    )
    y = alt.Y("District:N", sort=order, title=None)
    lo = min(wide["Pregnancy"].min(), wide["Delivery"].min())
    hi = max(wide["Pregnancy"].max(), wide["Delivery"].max())
    pad = (hi - lo) * 0.1 or 0.1
    xscale = alt.Scale(domain=[lo - pad, hi + pad])

    lines = alt.Chart(wide).mark_rule(color="#B9C2BD", strokeWidth=4).encode(
        y=y,
        x=alt.X("Pregnancy:Q", scale=xscale,
                title="Predicted neonatal mortality risk (%)"),
        x2="Delivery:Q",
    )
    dots = alt.Chart(long).mark_circle(size=130, opacity=1).encode(
        y=y,
        x=alt.X("Risk:Q", scale=xscale),
        color=alt.Color(
            "Stage:N",
            scale=alt.Scale(
                domain=list(STAGE_COLORS), range=list(STAGE_COLORS.values())
            ),
            legend=alt.Legend(title=None, orient="top"),
        ),
        tooltip=[
            alt.Tooltip("District:N"),
            alt.Tooltip("Stage:N"),
            alt.Tooltip("Risk:Q", title="Predicted risk (%)", format=".2f"),
        ],
    )
    return (lines + dots).properties(height=max(300, len(wide) * 28))


# --------------------------------------------------------------------
# Page
# --------------------------------------------------------------------
def _stage_selector() -> str:
    options = ["Pregnancy stage", "Delivery stage"]
    if hasattr(st, "segmented_control"):  # Streamlit 1.40+
        choice = st.segmented_control(
            "Assessment stage", options, default=options[0],
            key="district_stage", label_visibility="collapsed",
        )
        return choice or options[0]
    return st.radio(
        "Assessment stage", options, horizontal=True,
        key="district_stage", label_visibility="collapsed",
    )




def get_district_neonatal_deaths(data_path):
    """Count births and observed neonatal deaths by district."""
    births = pd.read_csv(data_path)

    required = {
        "District",
        "Child is Alive",
        "Age at death category",
        "current_age_of_children",
    }
    missing = required - set(births.columns)

    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing)}"
        )

    # Clean district names and count all birth records by district.
    births["District"] = (
        births["District"].astype("string").str.strip()
    )

    total_births = (
        births.dropna(subset=["District"])
        .groupby("District")
        .size()
        .rename("Births")
        .reset_index()
    )

    alive = (
        births["Child is Alive"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    age_death = (
        births["Age at death category"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    current_age = pd.to_numeric(
        births["current_age_of_children"],
        errors="coerce",
    )

    # Deaths from Day 0 through Day 27.
    neonatal = age_death.str.contains(
        r"day\s*(?:[0-9]|1[0-9]|2[0-7])\b",
        regex=True,
        na=False,
    )

    # Deaths after the neonatal period are not neonatal deaths.
    later_death = (
        alive.isin(["no", "0"])
        & age_death.notna()
        & age_death.ne("")
        & ~neonatal
    )

    # Living children are eligible controls at age >= 1 month.
    living_eligible = (
        alive.isin(["yes", "1"])
        & (current_age >= 1)
    )

    births["neonatal_death"] = np.nan
    births.loc[neonatal, "neonatal_death"] = 1
    births.loc[later_death, "neonatal_death"] = 0
    births.loc[living_eligible, "neonatal_death"] = 0

    eligible = births.dropna(
        subset=["District", "neonatal_death"]
    ).copy()

    counts = (
        eligible.groupby("District", as_index=False)
        .agg(
            Observed_neonatal_deaths=("neonatal_death", "sum"),
            Eligible_records=("neonatal_death", "count"),
        )
    )

    counts["Observed_neonatal_deaths"] = (
        counts["Observed_neonatal_deaths"].astype(int)
    )

    # Attach total birth records, including records excluded
    # from the eligible outcome analysis.
    counts = total_births.merge(
        counts,
        on="District",
        how="left",
    )

    counts["Observed_neonatal_deaths"] = (
        counts["Observed_neonatal_deaths"]
        .fillna(0)
        .astype(int)
    )

    counts["Eligible_records"] = (
        counts["Eligible_records"]
        .fillna(0)
        .astype(int)
    )

    return counts

DETAIL_CSS = """
<style>
.detail {
    background: #FFFFFF; border: 1px solid #DADFD9; border-radius: 14px;
    padding: 18px 20px;
}
.detail h3 { margin: 0 0 8px 0; color: #0F3D3E; }
.detail .badge {
    display: inline-block; color: #FFFFFF; border-radius: 14px;
    padding: 3px 14px; font-size: 0.85rem; font-weight: 700;
}
.detail .row { display: flex; justify-content: space-between;
    padding: 9px 0; border-bottom: 1px solid #EEF1EE; }
.detail .row:last-child { border-bottom: none; }
.detail .k { color: #5F6B66; font-size: 0.9rem; }
.detail .v { color: #13302F; font-size: 1.15rem; font-weight: 700; }
.hint {
    background: #EAF3F1; border-left: 5px solid #0F3D3E; border-radius: 6px;
    padding: 14px 16px; color: #13302F;
}
</style>
"""
def show_district_details(row: pd.Series, national_mean: float) -> None:
    color = COLORS[row["Category"]]
    births = int(row["Births"])
    deaths = int(row["Observed_neonatal_deaths"])
    diff = row["Risk"] - national_mean

    st.markdown(DETAIL_CSS, unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="detail">
          <h3>{row['District']}
            <span class="badge" style="background:{color}; margin-left:10px;">
              {row['Category']}
            </span>
          </h3>
          <div class="row"><span class="k">Number of births</span>
               <span class="v">{births:,}</span></div>
          <div class="row"><span class="k">Observed neonatal deaths</span>
               <span class="v">{deaths:,}</span></div>
          <div class="row"><span class="k">Average predicted risk</span>
               <span class="v">{row['Risk']:.2f}%</span></div>
          <div class="row"><span class="k">Difference from mean</span>
               <span class="v">{diff:+.2f} pp</span></div>
          <div class="row"><span class="k">Category</span>
               <span class="v" style="color:{color}">{row['Category']}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def diverging_bars(df: pd.DataFrame, national_mean: float) -> alt.Chart:
    """Bars show each district's difference from the mean (pp). Clickable."""
    d = df.copy()
    d["Difference"] = d["Risk"] - national_mean

    click = alt.selection_point(name="district_click", fields=["District"])

    return (
        alt.Chart(d)
        .mark_bar(cursor="pointer", cornerRadiusEnd=3)
        .encode(
            y=alt.Y("District:N", sort="-x", title=None,
                    axis=alt.Axis(labelFontSize=12)),
            x=alt.X("Difference:Q",
                    title="Difference from national mean predicted risk "
                          "(percentage points)"),
            color=alt.Color(
                "Category:N", title="Relative group",
                scale=alt.Scale(domain=CAT_ORDER,
                                range=[COLORS[c] for c in CAT_ORDER]),
                legend=alt.Legend(orient="top-right"),
            ),
            opacity=alt.condition(click, alt.value(1.0), alt.value(0.45)),
            tooltip=[
                alt.Tooltip("District:N"),
                alt.Tooltip("Risk:Q", title="Predicted risk (%)", format=".2f"),
                alt.Tooltip("Difference:Q", title="Difference (pp)", format="+.2f"),
                alt.Tooltip("Category:N", title="Relative group"),
            ],
        )
        .add_params(click)
        .properties(height=max(450, len(d) * 22))
    )

    if data is None or data.empty:
        st.warning("No district risk data is available.")
        return

    chart_data = data.copy()

    required = {"District", "Risk"}
    if not required.issubset(chart_data.columns):
        st.error("District data must contain 'District' and 'Risk' columns.")
        return

    chart_data["Risk"] = pd.to_numeric(
        chart_data["Risk"], errors="coerce"
    )
    chart_data = chart_data.dropna(subset=["Risk"])
    chart_data = chart_data.sort_values("Risk", ascending=False)

    if chart_data.empty:
        st.warning("No valid predicted risk values are available.")
        return

    st.subheader(f"{stage} — District risk ranking")

    chart = (
        alt.Chart(chart_data)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            y=alt.Y(
                "District:N",
                sort="-x",
                title=None
            ),
            x=alt.X(
                "Risk:Q",
                title="Average predicted neonatal mortality risk",
                axis=alt.Axis(format="%")
            ),
            tooltip=[
                alt.Tooltip("District:N", title="District"),
                alt.Tooltip("Risk:Q", title="Predicted risk", format=".2%")
            ]
        )
        .properties(height=max(300, min(900, len(chart_data) * 28)))
    )

    st.altair_chart(chart, width="stretch")

    st.dataframe(
        chart_data[["District", "Risk"]].rename(
            columns={"Risk": "Average predicted risk"}
        ),
        hide_index=True,
        width="stretch"
    )


def district_analysis_page() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(
        '<div class="hero"><h1>Where is newborn risk highest?</h1>'
        "<p>Model-predicted neonatal mortality risk across Rwanda's districts</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    stage_label = _stage_selector()
    stage = "pregnancy" if stage_label == "Pregnancy stage" else "delivery"
    path = PREGNANCY_DISTRICT_PATH if stage == "pregnancy" else DELIVERY_DISTRICT_PATH

    with st.expander("Category settings"):
        k = st.slider(
            "Band width in standard deviations",
            min_value=0.1, max_value=1.5, value=0.5, step=0.1,
            help="High = above mean + band, Low = below mean - band, "
                 "Medium = within the band.",
            key=f"district_k_{stage}",
        )

    data = load_stage(str(path))
    if data.empty:
        st.warning(
            f"No usable district summary at {path}. It needs a district "
            "column and a mean predicted risk column."
        )
        return

    # ---- Attach observed births and deaths (no Births_x / Births_y) ----
    death_counts = get_district_neonatal_deaths(str(DATA_PATH))
    death_counts["District"] = death_counts["District"].astype(str).str.strip()
    data["District"] = data["District"].astype(str).str.strip()

    data = data.drop(columns=["Births", "Deaths"], errors="ignore").merge(
        death_counts[["District", "Births", "Observed_neonatal_deaths"]],
        on="District", how="left",
    )
    data["Births"] = data["Births"].fillna(0).astype(int)
    data["Observed_neonatal_deaths"] = (
        data["Observed_neonatal_deaths"].fillna(0).astype(int)
    )

    data = add_categories(data, k)
    mean = data["Risk"].mean()
    n = len(data)
    counts = data["Category"].value_counts()
    top = data.sort_values("Risk", ascending=False).iloc[0]

    # ---- KPI cards ----
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(_kpi("Districts analysed", f"{n:,}"), unsafe_allow_html=True)
    c2.markdown(_kpi("Mean predicted risk", f"{mean:.2f}%"), unsafe_allow_html=True)
    c3.markdown(_kpi("High-risk districts", str(int(counts.get("High", 0)))),
                unsafe_allow_html=True)
    c4.markdown(
        _kpi("Highest district", f"{top['District']}  {top['Risk']:.2f}%", small=True),
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="story"><b>{top["District"]}</b> has the highest predicted '
        f'risk at {top["Risk"]:.2f}%, {top["Risk"] - mean:.2f} percentage points '
        f"above the district mean. {int(counts.get('High', 0))} of {n} districts "
        f"are in the High group, {int(counts.get('Low', 0))} in the Low group.</div>",
        unsafe_allow_html=True,
    )

    tab_overview, tab_compare, tab_data = st.tabs(
        ["Overview", "Compare stages", "Data"]
    )

    # ---- Tab 1: bar chart + click details ----
    with tab_overview:
        st.subheader("Difference from the national mean predicted risk")

        event = st.altair_chart(
            diverging_bars(data, mean),
            use_container_width=True,
            on_select="rerun",
            key=f"diverging_{stage}",
        )

        # Read which bar was clicked
        selected = None
        if event and event.selection:
            picked = event.selection.get("district_click", [])
            if picked:
                selected = picked[0]["District"]

        if selected:
            row = data[data["District"] == selected].iloc[0]
            show_district_details(row, mean)
        else:
            st.markdown(DETAIL_CSS, unsafe_allow_html=True)
            st.markdown(
                '<div class="hint">Click a bar to see the number of births, '
                "neonatal deaths, average predicted risk and category for "
                "that district.</div>",
                unsafe_allow_html=True,
            )

        with st.expander("Tile map of all districts"):
            st.altair_chart(tile_map(data), use_container_width=True)

    # ---- Tab 2: pregnancy vs delivery ----
    with tab_compare:
        st.subheader("Pregnancy vs delivery stage")
        preg = load_stage(str(PREGNANCY_DISTRICT_PATH))
        deliv = load_stage(str(DELIVERY_DISTRICT_PATH))
        if preg.empty or deliv.empty:
            st.info("Both stage summaries are needed for this comparison.")
        else:
            chart = dumbbell(preg, deliv)
            if chart is None:
                st.info("No districts appear in both summaries.")
            else:
                st.altair_chart(chart, use_container_width=True)

    # ---- Tab 3: data table ----
    with tab_data:
        f1, f2 = st.columns([1, 2])
        levels = f1.multiselect("Relative risk categories", CAT_ORDER,
                                default=CAT_ORDER, key=f"district_levels_{stage}")
        search = f2.text_input("Search by district name",
                               key=f"district_search_{stage}")

        view = data[data["Category"].isin(levels)]
        if search.strip():
            view = view[view["District"].str.contains(
                search.strip(), case=False, na=False, regex=False)]
        view = view.sort_values("Risk", ascending=False)

        if view.empty:
            st.info("No districts match your search and selected categories.")
        else:
            table = view.rename(columns={
                "Risk": "Average predicted risk (%)",
                "Category": "Relative risk category",
                "Births": "Number of births",
                "Observed_neonatal_deaths": "Number of neonatal deaths",
            })
            table["Average predicted risk (%)"] = table["Average predicted risk (%)"].round(2)
            st.dataframe(table, use_container_width=True, hide_index=True)
            st.download_button(
                "Download district results (CSV)",
                data=table.to_csv(index=False).encode("utf-8"),
                file_name=f"{stage}_district_risk_analysis.csv",
                mime="text/csv",
                key=f"download_district_{stage}",
            )

    st.caption(
        "Predicted risk is a model estimate, not the observed neonatal "
        "mortality rate. Categories are relative and exploratory, not "
        "clinical thresholds or official district classifications."
    )

def national_overview_page() -> None:
    st.title("Rwanda Neonatal Mortality Risk Dashboard")

    st.write(
        "Explore the study dataset, child survival status, "
        "and saved model evaluation results."
    )

    st.info(
        "Research prototype only. Results describe the available study "
        "data and model estimates; they are not necessarily national "
        "population estimates or clinical classifications."
    )

    # --------------------------------------------------------
    # 1. LOAD DATA
    # --------------------------------------------------------
    data_path = first_existing_path(
        DATA_PATH,
        ROOT / "merged_table.csv",
    )

    data = (
        read_optional_csv(data_path)
        if data_path
        else pd.DataFrame()
    )

    summary = read_optional_csv(NATIONAL_PATH)
    comparison = read_optional_csv(COMPARISON_PATH)

    # --------------------------------------------------------
    # 2. KEY METRICS
    # --------------------------------------------------------
    st.subheader("Study at a glance")

    if data is not None and not data.empty:
        total_records = len(data)

        total_districts = (
            data["District"].nunique()
            if "District" in data.columns
            else None
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Records in dataset",
            f"{total_records:,}",
        )

        c2.metric(
            "Districts represented",
            f"{total_districts:,}"
            if total_districts is not None
            else "Unavailable",
        )

        c3.metric(
            "Dataset variables",
            f"{data.shape[1]:,}",
        )

        # ----------------------------------------------------
        # 3. PIE CHART: CHILD SURVIVAL STATUS
        # ----------------------------------------------------
        st.subheader("Child survival status")

        status_column = next(
            (
                col for col in data.columns
                if col.strip().lower() == "child is alive"
            ),
            None,
        )

        if status_column is not None:
            status_data = (
                data[status_column]
                .dropna()
                .astype(str)
                .str.strip()
            )

            status_counts = (
                status_data
                .value_counts()
                .rename_axis("Child status")
                .reset_index(name="Records")
            )

            if not status_counts.empty:
                chart = (
                    alt.Chart(status_counts)
                    .mark_arc(
                        innerRadius=65,
                        outerRadius=115,
                    )
                    .encode(
                        theta=alt.Theta(
                            "Records:Q",
                            stack=True,
                        ),
                        color=alt.Color(
                            "Child status:N",
                            title="Recorded status",
                        ),
                        tooltip=[
                            alt.Tooltip(
                                "Child status:N",
                                title="Child status",
                            ),
                            alt.Tooltip(
                                "Records:Q",
                                title="Records",
                                format=",",
                            ),
                        ],
                    )
                    .properties(
                        title="Distribution of recorded child status",
                        height=350,
                    )
                )

                left, right = st.columns([3, 2])

                with left:
                    st.altair_chart(
                        chart,
                        use_container_width=True,
                    )

                with right:
                    st.markdown("**Status breakdown**")
                    st.dataframe(
                        status_counts,
                        hide_index=True,
                        use_container_width=True,
                    )

                    st.caption(
                        "This chart shows the recorded alive/deceased "
                        "status of children. Deceased children are not "
                        "necessarily neonatal deaths; age at death must "
                        "be checked separately."
                    )
        else:
            st.info(
                "The pie chart is unavailable because the dataset "
                "does not contain a 'Child is Alive' column."
            )

    else:
        st.warning(
            "The merged dataset was not found. Check DATA_PATH and "
            "the location of merged_table.csv."
        )

    # --------------------------------------------------------
    # 4. SAVED NATIONAL RISK SUMMARY
    # --------------------------------------------------------
    st.divider()
    st.subheader("National model-risk summary")

    if summary is not None and not summary.empty:
        st.dataframe(
            summary,
            hide_index=True,
            use_container_width=True,
        )

        st.caption(
            "Model-predicted risk is an estimate and is not the same "
            "as the observed neonatal mortality rate."
        )
    else:
        st.info(
            "The saved national risk summary is not available. "
            "Check the configured NATIONAL_PATH."
        )

    # --------------------------------------------------------
    # 5. MODEL COMPARISON
    # --------------------------------------------------------
    st.divider()
    st.subheader("Model performance comparison")

    if comparison is not None and not comparison.empty:
        st.dataframe(
            comparison,
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info(
            "Model comparison results are not available. "
            "Check the configured COMPARISON_PATH."
        )

    # --------------------------------------------------------
    # 6. INTERPRETATION
    # --------------------------------------------------------
    st.divider()
    st.subheader("How to interpret this dashboard")

    st.markdown(
        """
        - **Records:** observations in the loaded study dataset.
        - **Child survival status:** recorded alive/deceased status,
          not a count of neonatal deaths alone.
        - **Predicted risk:** an estimate produced by a trained model.
        - **Screening threshold:** a model-development cutoff, not
          an approved clinical guideline.
        - **District average predicted risk:** a model-based estimate,
          not the observed district neonatal mortality rate.
        """
    )




def pregnancy_prediction_page() -> None:
    st.title("Pregnancy-stage prediction")

    st.write(
        "Estimate model-predicted neonatal mortality probability using "
        "information intended to be available before delivery."
    )

    

    with st.form("pregnancy_prediction_form"):
        values = collect_prediction_inputs("pregnancy")
        submitted = st.form_submit_button(
            "Estimate pregnancy-stage probability",
            type="primary",
        )

    if submitted:
        show_prediction_result("pregnancy", values)


def delivery_prediction_page() -> None:
    st.title("Delivery-stage prediction")

    st.write(
        "Estimate model-predicted neonatal mortality probability using "
        "pregnancy information plus delivery and birth information."
    )

    

    with st.form("delivery_prediction_form"):
        values = collect_prediction_inputs("delivery")
        submitted = st.form_submit_button(
            "Estimate delivery-stage probability",
            type="primary",
        )

    if submitted:
        show_prediction_result("delivery", values)


INTERVENTIONS_BY_RISK = {
    "High": [
        "Strengthen emergency obstetric and newborn referral readiness",
        "Review transport barriers and delays to health facilities",
        "Assess newborn resuscitation and essential newborn care readiness",
        "Review care capacity for low-birth-weight and preterm newborns",
        "Conduct neonatal death and service-gap reviews",
        "Target community outreach to vulnerable mothers and newborns",
        "Antenatal care follow-up",
        "Facility delivery and referral",
        "Postnatal visits",
        "Data quality and review",
    ],
    "Medium": [
        "Identify gaps in antenatal attendance and early booking",
        "Improve access to delivery facilities and referral transport",
        "Strengthen follow-up for mothers with identified risk factors",
        "Review early postnatal care and newborn danger-sign education",
        "Monitor facility-level service gaps",
        "Community health worker outreach",
        "Staff training",
        "Data quality and review",
    ],
    "Low": [
        "Maintain routine antenatal, delivery and postnatal services",
        "Continue essential newborn care and danger-sign education",
        "Maintain emergency referral pathways and preparedness",
        "Monitor neonatal outcomes and data completeness",
        "Check whether small samples or missing data affect risk estimates",
        "Community health worker outreach",
        "Staff training",
        "Data quality and review",
    ],
}


def public_health_action_planner_page() -> None:
    st.markdown(CSS, unsafe_allow_html=True)

    st.markdown(
        '<div class="hero"><h1>Public-health action planner</h1>'
        '<p>Turn district risk estimates into a prioritised action plan</p></div>',
        unsafe_allow_html=True,
    )

    # 1. Select the assessment stage
    stage_label = st.radio(
        "Assessment stage",
        ["Pregnancy stage", "Delivery stage"],
        horizontal=True,
        key="planner_stage",
    )

    path = (
        PREGNANCY_DISTRICT_PATH
        if stage_label == "Pregnancy stage"
        else DELIVERY_DISTRICT_PATH
    )

    # 2. Load district data BEFORE using the variable data
    data = load_stage(str(path))

    if data is None or data.empty:
        st.warning(
            "No usable district summary found. "
            "Check the district summary CSV files and paths."
        )
        return

    # 3. Create risk categories
    k = st.slider(
        "Category band (standard deviations)",
        min_value=0.1,
        max_value=1.5,
        value=0.5,
        step=0.1,
        key="planner_k",
    )

    data = add_categories(data, k)
    data = data.sort_values("Risk", ascending=False)

    priority = data[data["Category"] == "High"]

    # 4. Summary metrics
    c1, c2, c3 = st.columns(3)

    c1.markdown(
        _kpi("Priority districts", str(len(priority))),
        unsafe_allow_html=True,
    )

    c2.markdown(
        _kpi("Mean predicted risk", f"{data['Risk'].mean():.2f}%"),
        unsafe_allow_html=True,
    )

    c3.markdown(
        _kpi(
            "Highest district",
            f"{data.iloc[0]['District']} "
            f"{data.iloc[0]['Risk']:.2f}%",
            small=True,
        ),
        unsafe_allow_html=True,
    )

    # 5. Choose districts
    st.subheader("1. Choose districts to plan for")

    chosen = st.multiselect(
        "Districts",
        options=data["District"].tolist(),
        default=priority["District"].tolist(),
        key="planner_districts",
    )

    if not chosen:
        st.info("Select at least one district to build a plan.")
        return

    # 6. Build the action plan
    st.subheader("2.  SUGGESTED ACTION PLAN FOR EACH DISTRICT")

    

    plan = data.loc[
        data["District"].isin(chosen),
        ["District", "Risk", "Category"],
    ].copy()

    plan["Risk"] = plan["Risk"].round(2)

    def default_intervention(category):
        options = INTERVENTIONS_BY_RISK.get(category, [])
        return (
            options[0]
            if options
            else "Data quality and review"
        )

    plan["Action area"] = plan["Category"].apply(default_intervention)
    plan["Responsible team"] = ""
    plan["Timeline"] = ""
    plan["Status"] = "Not started"

    # 7. Editable table
    all_interventions = list(dict.fromkeys(
        action
        for actions in INTERVENTIONS_BY_RISK.values()
        for action in actions
    ))

    edited = st.data_editor(
        plan,
        hide_index=True,
        use_container_width=True,
        key="planner_table",
        disabled=["District", "Risk", "Category"],
        column_config={
            "Action area": st.column_config.SelectboxColumn(
                "Suggested intervention",
                options=all_interventions,
                required=True,
            ),
            "Responsible team": st.column_config.TextColumn(
                "Responsible team"
            ),
            "Timeline": st.column_config.TextColumn(
                "Timeline"
            ),
            "Status": st.column_config.SelectboxColumn(
                "Status",
                options=["Not started", "In progress", "Done"],
            ),
        },
    )

    # 8. Download the action plan
    st.download_button(
        "Download action plan (CSV)",
        data=edited.to_csv(index=False).encode("utf-8"),
        file_name="district_action_plan.csv",
        mime="text/csv",
        key="planner_download",
    )

    st.caption(
        "Risk values are model estimates, not observed mortality rates. "
        "Priority groups are relative and exploratory."
    )


from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------
# PATHS: change these to where your notebook saves the files
# --------------------------------------------------------------------

from pathlib import Path

# Project root: the folder containing app/ and outputs/
ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = ROOT / "outputs"

# Use the model comparison results already generated by your project
METRICS_PATH = OUTPUTS_DIR / "model_comparison.csv"

# These reports are optional until you generate them
IMPORTANCE_PATH = OUTPUTS_DIR / "feature_importance.csv"
GROUP_RISK_PATH = OUTPUTS_DIR / "risk_by_group.csv"

REPO_NAME = "Neonatal-Mortality-Risk-Prediction"
DATA_SOURCE = "Rwanda Demographic and Health Survey (DHS) 2025"


# Optional: filled by main() so the overview can link to the other pages.
if "PAGE_REGISTRY" not in globals():
    PAGE_REGISTRY: dict = {}

# Readable names for DHS variable codes. Check each one against your DHS
# recode manual and edit freely. Unknown names are shown as they are.
FEATURE_LABELS = {
    "v012": "Mother's age",
    "v106": "Mother's education",
    "v025": "Urban or rural residence",
    "v024": "Region",
    "v190": "Household wealth",
    "b4": "Sex of child",
    "b0": "Twin or single birth",
    "bord": "Birth order",
    "b11": "Gap since previous birth",
    "m13": "Timing of first antenatal visit",
    "m14": "Number of antenatal visits",
    "m15": "Place of delivery",
    "m17": "Caesarean delivery",
    "m19": "Birth weight",
}


# --------------------------------------------------------------------
# District summary paths (used for the KPI cards and district snapshot)
# --------------------------------------------------------------------
if "PREGNANCY_DISTRICT_PATH" not in globals():
    PREGNANCY_DISTRICT_PATH = "outputs/pregnancy_district_summary.csv"

# --------------------------------------------------------------------
# Small helpers (skipped automatically if your app already defines them)
# --------------------------------------------------------------------
if "read_optional_csv" not in globals():
    def read_optional_csv(path) -> pd.DataFrame:
        p = Path(str(path))
        if not p.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(p)
        except Exception:
            return pd.DataFrame()

if "find_column" not in globals():
    def find_column(df: pd.DataFrame, candidates):
        lookup = {str(c).strip().lower(): c for c in df.columns}
        for name in candidates:
            hit = lookup.get(str(name).strip().lower())
            if hit is not None:
                return hit
        return None


def _ov_kpi(label: str, value: str, small: bool = False) -> str:
    cls = "value small" if small else "value"
    return (
        f'<div class="ov-kpi"><div class="label">{label}</div>'
        f'<div class="{cls}">{value}</div></div>'
    )


@st.cache_data(show_spinner=False)
def _ov_load_district(path: str) -> pd.DataFrame:
    """Read a saved district summary into: District, Risk (%), Births, Deaths."""
    raw = read_optional_csv(path)
    if raw is None or raw.empty:
        return pd.DataFrame()
    dcol = find_column(raw, ["District", "district_name", "district", "District Name"])
    rcol = find_column(raw, ["mean_predicted_risk", "average_predicted_risk",
                             "predicted_risk", "mean_risk", "average_risk"])
    if dcol is None or rcol is None:
        return pd.DataFrame()
    bcol = find_column(raw, ["observed_births", "number_of_births", "total_births",
                             "births", "n_births", "sample_size", "n"])
    xcol = find_column(raw, ["observed_deaths", "observed_neonatal_deaths",
                             "neonatal_deaths", "deaths"])
    out = pd.DataFrame({
        "District": raw[dcol].astype(str).str.strip(),
        "Risk": pd.to_numeric(raw[rcol], errors="coerce"),
        "Births": pd.to_numeric(raw[bcol], errors="coerce") if bcol else np.nan,
        "Deaths": pd.to_numeric(raw[xcol], errors="coerce") if xcol else np.nan,
    }).dropna(subset=["District", "Risk"])
    if out.empty:
        return out
    if out["Risk"].max() <= 1:          # proportions -> percentages
        out["Risk"] = out["Risk"] * 100
    return out.groupby("District", as_index=False).agg(
        Risk=("Risk", "mean"),
        Births=("Births", lambda s: s.sum(min_count=1)),
        Deaths=("Deaths", lambda s: s.sum(min_count=1)),
    )

OV_CSS = """
<style>
.ov-hero { background: #0F3D3E; border-radius: 14px; padding: 26px 30px; margin-bottom: 14px; }
.ov-hero h1 { color: #FFFFFF; font-size: 1.9rem; margin: 0; padding: 0; }
.ov-hero p  { color: #A9D6CF; margin: 6px 0 0 0; font-size: 0.98rem; }
.ov-kpi { background: #FFFFFF; border: 1px solid #DADFD9; border-radius: 12px; padding: 14px 18px; }
.ov-kpi .label { color: #5F6B66; font-size: 0.82rem; }
.ov-kpi .value { color: #13302F; font-size: 1.7rem; font-weight: 700; line-height: 1.3; }
.ov-kpi .value.small { font-size: 1.15rem; }
.flow { display: flex; gap: 10px; align-items: stretch; flex-wrap: wrap; }
.step {
    flex: 1; min-width: 150px; background: #FFFFFF;
    border: 1px solid #DADFD9; border-radius: 12px; padding: 14px 14px;
}
.step .num {
    display: inline-block; width: 24px; height: 24px; border-radius: 50%;
    background: #0F3D3E; color: #FFFFFF; text-align: center;
    line-height: 24px; font-size: 0.8rem; font-weight: 700; margin-bottom: 6px;
}
.step .t { font-weight: 700; color: #13302F; font-size: 0.95rem; }
.step .d { color: #5F6B66; font-size: 0.82rem; margin-top: 3px; }
.arrow { align-self: center; color: #0F3D3E; font-size: 1.4rem; }
.stage {
    background: #FFFFFF; border: 1px solid #DADFD9; border-radius: 12px;
    padding: 16px 18px; height: 100%;
}
.stage h4 { margin: 0 0 6px 0; color: #0F3D3E; }
.stage p  { color: #3E4A46; font-size: 0.92rem; margin: 4px 0; }
.stage .tag {
    display: inline-block; background: #EAF3F1; color: #0F3D3E;
    border-radius: 12px; padding: 2px 10px; margin: 3px 4px 0 0;
    font-size: 0.8rem;
}
.footer { color: #7A857F; font-size: 0.82rem; text-align: center; margin-top: 18px; }
</style>
"""


# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _load_csv(path: str) -> pd.DataFrame:
    df = read_optional_csv(path)
    return pd.DataFrame() if df is None else df


def _col(df: pd.DataFrame, name: str):
    """Case-insensitive column lookup."""
    for c in df.columns:
        if str(c).strip().lower() == name.lower():
            return c
    return None


def _missing(path: str, expected: str) -> None:
    st.info(
        f"Not available yet: `{path}` was not found. Save it from your "
        f"notebook with columns: {expected}."
    )


def _go_button(label: str, key: str, icon: str) -> None:
    page = PAGE_REGISTRY.get(key)
    if page is not None:
        st.page_link(page, label=label, icon=icon, use_container_width=True)


# --------------------------------------------------------------------
# Page
# --------------------------------------------------------------------
def national_overview_page() -> None:
    st.markdown(OV_CSS, unsafe_allow_html=True)

    # ---- 1. Purpose banner -----------------------------------------
    st.markdown(
        '<div class="ov-hero"><h1>Neonatal Mortality Risk in Rwanda</h1>'
        "<p>Estimating each newborn's risk of dying in the first 28 days, "
        "so district health teams can act earlier. A research dashboard "
        "built on " + DATA_SOURCE + ".</p></div>",
        unsafe_allow_html=True,
    )

    # ---- 2. National KPI cards -------------------------------------
    preg = _ov_load_district(str(PREGNANCY_DISTRICT_PATH))
    metrics = _load_csv(METRICS_PATH)

    n_districts = len(preg) if not preg.empty else None
    births = deaths = rate = mean_risk = None
    if not preg.empty:
        if preg["Births"].notna().any():
            births = preg["Births"].sum()
            mean_risk = float(np.average(preg["Risk"], weights=preg["Births"].fillna(0)))
        else:
            mean_risk = float(preg["Risk"].mean())
        if preg["Deaths"].notna().any():
            deaths = preg["Deaths"].sum()
        if births and deaths is not None:
            rate = deaths / births * 1000

    best_auc = None
    if not metrics.empty and _col(metrics, "AUC") is not None:
        auc = pd.to_numeric(metrics[_col(metrics, "AUC")], errors="coerce")
        best_auc = auc.max() if auc.notna().any() else None

    def fmt(v, f):
        return "n/a" if v is None or pd.isna(v) else f.format(v)

    k = st.columns(5)
    k[0].markdown(_ov_kpi("Births analysed", fmt(births, "{:,.0f}")), unsafe_allow_html=True)
    k[1].markdown(_ov_kpi("Observed neonatal deaths", fmt(deaths, "{:,.0f}")), unsafe_allow_html=True)
    k[2].markdown(_ov_kpi("Deaths per 1,000 births", fmt(rate, "{:.1f}")), unsafe_allow_html=True)
    k[3].markdown(_ov_kpi("Mean predicted risk", fmt(mean_risk, "{:.2f}%")), unsafe_allow_html=True)
    k[4].markdown(_ov_kpi("Best model AUC", fmt(best_auc, "{:.2f}")), unsafe_allow_html=True)
    st.caption(
        "Observed counts come from the survey data. Predicted risk is a model "
        "estimate."
    )

    # ---- 3. How it works -------------------------------------------
    st.subheader("How the project works")
    st.markdown(
        '<div class="flow">'
        '<div class="step"><div class="num">1</div><div class="t">Survey data</div>'
        f'<div class="d">{DATA_SOURCE} birth records</div></div>'
        '<div class="arrow">&#10140;</div>'
        '<div class="step"><div class="num">2</div><div class="t">Features</div>'
        '<div class="d">Mother, child, household and care factors are cleaned and prepared</div></div>'
        '<div class="arrow">&#10140;</div>'
        '<div class="step"><div class="num">3</div><div class="t">Two risk models</div>'
        '<div class="d">Logistic Regression and Random Forest, at two stages</div></div>'
        '<div class="arrow">&#10140;</div>'
        '<div class="step"><div class="num">4</div><div class="t">District risk</div>'
        '<div class="d">Predicted risk is summarised for each district</div></div>'
        '<div class="arrow">&#10140;</div>'
        '<div class="step"><div class="num">5</div><div class="t">Action planner</div>'
        '<div class="d">Priority districts become a practical action plan</div></div>'
        "</div>",
        unsafe_allow_html=True,
    )

    # ---- 4. The two stages -----------------------------------------
    st.subheader("Two moments when risk can be assessed")
    s1, s2 = st.columns(2)
    s1.markdown(
        '<div class="stage"><h4>Pregnancy stage</h4>'
        "<p>Uses what is known <b>before the baby is born</b>, so action can "
        "start during pregnancy.</p>"
        '<span class="tag">Mother\'s age</span><span class="tag">Education</span>'
        '<span class="tag">Residence</span><span class="tag">Wealth</span>'
        '<span class="tag">Birth order</span><span class="tag">Gap since last birth</span>'
        '<span class="tag">Antenatal care</span></div>',
        unsafe_allow_html=True,
    )
    s2.markdown(
        '<div class="stage"><h4>Delivery stage</h4>'
        "<p>Adds information that is known <b>at delivery</b>, giving a more "
        "complete picture for the newborn.</p>"
        '<span class="tag">Everything from pregnancy</span>'
        '<span class="tag">Place of delivery</span>'
        '<span class="tag">Delivery type</span>'
        '<span class="tag">Birth details</span></div>',
        unsafe_allow_html=True,
    )

    # ---- 5-7. Tabs --------------------------------------------------
    st.subheader("What the models show")
    tab_perf, tab_feat, tab_group = st.tabs(
        ["Model performance", "Key risk factors", "Risk by group"]
    )

    with tab_perf:
        if metrics.empty:
            _missing(METRICS_PATH, "Stage, Model, AUC (and optionally Recall, Precision)")
        else:
            st.dataframe(metrics, use_container_width=True, hide_index=True)
            stage_c, model_c, auc_c = (
                _col(metrics, "Stage"), _col(metrics, "Model"), _col(metrics, "AUC")
            )
            if stage_c and model_c and auc_c:
                chart = (
                    alt.Chart(metrics)
                    .mark_bar(cornerRadiusEnd=5, color="#0F3D3E")
                    .encode(
                        x=alt.X(f"{model_c}:N", title=None),
                        y=alt.Y(f"{auc_c}:Q", title="AUC",
                                scale=alt.Scale(domain=[0.5, 1.0])),
                        column=alt.Column(f"{stage_c}:N", title=None),
                        tooltip=[stage_c, model_c, alt.Tooltip(f"{auc_c}:Q", format=".3f")],
                    )
                    .properties(height=240, width=180)
                )
                st.altair_chart(chart)
            

    with tab_feat:
        imp = _load_csv(IMPORTANCE_PATH)
        if imp.empty:
            _missing(IMPORTANCE_PATH, "Stage, Feature, Importance")
        else:
            sc, fc, ic = _col(imp, "Stage"), _col(imp, "Feature"), _col(imp, "Importance")
            if not (fc and ic):
                st.error("The importance file needs Feature and Importance columns.")
            else:
                view = imp.copy()
                if sc:
                    stages = view[sc].dropna().unique().tolist()
                    pick = st.radio("Stage", stages, horizontal=True, key="overview_imp_stage")
                    view = view[view[sc] == pick]
                view = view.copy()
                view["Factor"] = view[fc].map(lambda f: FEATURE_LABELS.get(str(f), str(f)))
                view = view.sort_values(ic, ascending=False).head(10)
                chart = (
                    alt.Chart(view)
                    .mark_bar(cornerRadiusEnd=5, color="#0F3D3E", size=18)
                    .encode(
                        y=alt.Y("Factor:N", sort="-x", title=None),
                        x=alt.X(f"{ic}:Q", title="Importance in the Random Forest"),
                        tooltip=["Factor:N", alt.Tooltip(f"{ic}:Q", format=".3f")],
                    )
                    .properties(height=max(260, len(view) * 30))
                )
                st.altair_chart(chart, use_container_width=True)
                st.caption(
                    "Importance shows how much the model relied on each factor. "
                    "It does not prove that a factor causes neonatal death."
                )

    with tab_group:
        grp = _load_csv(GROUP_RISK_PATH)
        if grp.empty:
            _missing(GROUP_RISK_PATH, "Grouping, Group, Risk")
        else:
            gc, lc, rc = _col(grp, "Grouping"), _col(grp, "Group"), _col(grp, "Risk")
            if not (gc and lc and rc):
                st.error("The group file needs Grouping, Group and Risk columns.")
            else:
                choice = st.selectbox(
                    "Compare risk by", grp[gc].dropna().unique().tolist(),
                    key="overview_group",
                )
                view = grp[grp[gc] == choice]
                chart = (
                    alt.Chart(view)
                    .mark_bar(cornerRadiusEnd=5, color="#238636", size=22)
                    .encode(
                        y=alt.Y(f"{lc}:N", sort="-x", title=None),
                        x=alt.X(f"{rc}:Q", title="Average predicted risk (%)",
                                scale=alt.Scale(zero=True)),
                        tooltip=[lc, alt.Tooltip(f"{rc}:Q", format=".2f")],
                    )
                    .properties(height=max(200, len(view) * 38))
                )
                st.altair_chart(chart, use_container_width=True)

    # ---- 8. District snapshot --------------------------------------
    st.subheader("District snapshot")
    if preg.empty:
        st.info("District summary not available yet.")
    else:
        ranked = preg.sort_values("Risk", ascending=False)
        a, b = st.columns(2)
        for col, title, part in (
            (a, "Five highest predicted risk", ranked.head(5)),
            (b, "Five lowest predicted risk", ranked.tail(5).iloc[::-1]),
        ):
            with col:
                st.markdown(f"**{title}** (pregnancy stage)")
                show = part[["District", "Risk"]].rename(
                    columns={"Risk": "Predicted risk (%)"}
                )
                show["Predicted risk (%)"] = show["Predicted risk (%)"].round(2)
                st.dataframe(show, hide_index=True, use_container_width=True)

    # ---- 9. Data and limitations -----------------------------------
    with st.expander("Data and limitations"):
        st.markdown(
            f"- **Data:** {DATA_SOURCE}, birth records merged with household "
            "and mother-level variables.\n"
            "- **Outcome:** death within the first 28 days of life.\n"
            "- **Predicted risk** is a model estimate. It is not the observed "
            "neonatal mortality rate.\n"
            "- **Survey data** has sampling error, and district estimates "
            "based on few births can be unstable.\n"
            "- **Risk categories** (High, Medium, Low) are relative to the "
            "district mean. They are not clinical thresholds.\n"
            
        )

    # ---- 10. Where to go next --------------------------------------
    
    n1, n2, n3, n4 = st.columns(4)
    with n1:
        _go_button("Pregnancy prediction", "pregnancy", ":material/pregnant_woman:")
    with n2:
        _go_button("Delivery prediction", "delivery", ":material/child_care:")
    with n3:
        _go_button("District risk analysis", "district", ":material/map:")
    with n4:
        _go_button("Action planner", "planner", ":material/assignment:")

    # ---- 11. Footer -------------------------------------------------
    st.markdown(
        f'<div class="footer">Neonatal Mortality Risk Prediction | '
        f"GitHub: {REPO_NAME} | Data: {DATA_SOURCE}</div>",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------
# To preview this page on its own, uncomment the lines below and run:
#     streamlit run national_overview_page.py
# --------------------------------------------------------------------
# st.set_page_config(page_title="National overview", layout="wide")
# national_overview_page()

# ============================================================
# MODEL INFORMATION PAGE
# ============================================================

def model_information_page() -> None:
   
    # 3. Load saved comparison results
    # --------------------------------------------------------
    st.header("1. Model performance")

    comparison_path = ROOT / "outputs" / "model_comparison.csv"

    if not comparison_path.exists():
        st.info(
            "The model comparison CSV is not available yet. "
            "Run src/compare_models.py in your terminal, then "
            "refresh the dashboard."
        )
    else:
        try:
            results = pd.read_csv(comparison_path)

            required = {
                "stage", "model", "roc_auc",
                "average_precision", "precision",
                "recall", "f1", "flagged_pct", "threshold"
            }

            missing = required - set(results.columns)

            if missing:
                st.error(
                    "The comparison file is missing these columns: "
                    + ", ".join(sorted(missing))
                )
            else:
                stage_choice = st.selectbox(
                    "Assessment stage",
                    ["Both stages", "Pregnancy", "Delivery"],
                    key="model_info_stage",
                )

                view = results.copy()

                view["stage"] = view["stage"].astype(str)
                view["model"] = view["model"].astype(str)

                if stage_choice != "Both stages":
                    stage_key = (
                        "pregnancy"
                        if stage_choice == "Pregnancy"
                        else "delivery"
                    )

                    stage_aliases = {
                        "pregnancy": {
                            "pregnancy", "before_birth",
                        },
                        "delivery": {
                            "delivery", "after_delivery",
                        },
                    }

                    view = view[
                        view["stage"].str.lower().isin(
                            stage_aliases[stage_key]
                        )
                    ]

                if view.empty:
                    st.warning(
                        "No saved results were found for this stage."
                    )
                else:
                    display = view[
                        [
                            "stage", "model", "roc_auc",
                            "average_precision", "precision",
                            "recall", "f1", "flagged_pct",
                            "threshold",
                        ]
                    ].copy()

                    display = display.rename(columns={
                        "stage": "Stage",
                        "model": "Model",
                        "roc_auc": "ROC-AUC",
                        "average_precision": "PR-AUC",
                        "precision": "Precision",
                        "recall": "Recall",
                        "f1": "F1-score",
                        "flagged_pct": "Flagged (%)",
                        "threshold": "Threshold",
                    })

                    for column in [
                        "Precision", "Recall",
                        "F1-score", "Flagged (%)",
                    ]:
                        display[column] = (
                            pd.to_numeric(
                                display[column], errors="coerce"
                            ) * 100
                        )

                    st.dataframe(
                        display.style.format({
                            "ROC-AUC": "{:.3f}",
                            "PR-AUC": "{:.3f}",
                            "Precision": "{:.2f}%",
                            "Recall": "{:.2f}%",
                            "F1-score": "{:.2f}%",
                            "Flagged (%)": "{:.2f}%",
                            "Threshold": "{:.4f}",
                        }),
                        use_container_width=True,
                        hide_index=True,
                    )

                    st.caption(
                        "Precision, recall, F1-score and flagged "
                        "percentage are displayed as percentages. "
                        "ROC-AUC and PR-AUC remain scores between "
                        "0 and 1. The saved threshold determines "
                        "which predictions are flagged."
                    )

                    # Visual comparison of ROC-AUC
                    chart_data = view.copy()
                    chart_data["roc_auc"] = pd.to_numeric(
                        chart_data["roc_auc"], errors="coerce"
                    )
                    chart_data = chart_data.dropna(
                        subset=["roc_auc"]
                    )

                    if not chart_data.empty:
                        st.subheader("ROC-AUC comparison")

                        chart = (
                            alt.Chart(chart_data)
                            .mark_bar()
                            .encode(
                                x=alt.X(
                                    "model:N",
                                    title="Model",
                                ),
                                y=alt.Y(
                                    "roc_auc:Q",
                                    title="ROC-AUC",
                                    scale=alt.Scale(domain=[0, 1]),
                                ),
                                color=alt.Color(
                                    "model:N",
                                    title="Model",
                                ),
                                column=alt.Column(
                                    "stage:N",
                                    title="Assessment stage",
                                ),
                                tooltip=[
                                    "stage:N",
                                    "model:N",
                                    alt.Tooltip(
                                        "roc_auc:Q",
                                        title="ROC-AUC",
                                        format=".3f",
                                    ),
                                ],
                            )
                            .properties(height=300)
                        )

                        st.altair_chart(
                            chart,
                            use_container_width=True,
                        )

        except Exception as exc:
            st.error(
                f"Could not load model comparison results: {exc}"
            )

    # --------------------------------------------------------
    # 4. Explain performance metrics
    # --------------------------------------------------------
    st.header("4. What the performance metrics mean")

    with st.expander("ROC-AUC"):
        st.write(
            "Measures how well a model ranks children who experienced "
            "the outcome above those who did not. A value of 0.5 "
            "indicates chance-level discrimination; 1.0 indicates "
            "perfect discrimination."
        )

    with st.expander("PR-AUC / Average Precision"):
        st.write(
            "Summarises precision and recall across prediction "
            "thresholds. It is especially informative when neonatal "
            "deaths are rare. Interpret it relative to the outcome "
            "prevalence in the evaluation sample."
        )

    with st.expander("Precision"):
        st.write(
            "Among records flagged by the model, the proportion "
            "that actually experienced the defined outcome."
        )

    with st.expander("Recall"):
        st.write(
            "Among records that actually experienced the outcome, "
            "the proportion flagged by the model."
        )

    with st.expander("F1-score"):
        st.write(
            "The harmonic mean of precision and recall. It balances "
            "these two metrics but does not measure every aspect "
            "of model performance."
        )

    with st.expander("Screening threshold"):
        st.write(
            "The probability cutoff used to flag a record. Lowering "
            "the cutoff often increases recall and the number of "
            "records flagged, while precision may decrease."
        )

    # --------------------------------------------------------
    

    st.caption(
        "Project: Predicting neonatal mortality risk in Rwanda from "
        "antenatal and delivery information. for Research prototype."
    )

def main() -> None:
    st.sidebar.title("Dashboard navigation")

    page = st.sidebar.radio(
    "Choose a page",
    [
        "National overview",
        "Pregnancy prediction",
        "Delivery prediction",
        "District analysis",
        "Public-health action planner",
        "Model information"
        
    ],
)

    
    
    if page == "National overview":
        national_overview_page()

    elif page == "Pregnancy prediction":
        pregnancy_prediction_page()

    elif page == "Delivery prediction":
        delivery_prediction_page()

    elif page == "District analysis":
        district_analysis_page()

    elif page == "Public-health action planner":
        public_health_action_planner_page()

    elif page == "Model information":
        model_information_page()



if __name__ == "__main__":
    main()

"""
Feature preparation for the Rwanda neonatal mortality risk research prototype.

Important:
- One analytic row is created per mother/birth key.
- Conflicting duplicated values are set to missing rather than arbitrarily choosing a row.
- Living infants younger than one completed month are excluded from the outcome because
  they have not yet completed the 28-day neonatal outcome window.
- Missing predictors are imputed inside the training pipeline, not here.
"""
from __future__ import annotations

import re
import numpy as np
import pandas as pd

TARGET = "neonatal_death"
KEYS = ["Case ID", "Birth Index"]

PREGNANCY_FEATURES = [
    "mother_age", "birth_order", "anc_visits", "first_anc_month",
    "minutes_to_facility", "walks_to_facility", "is_urban",
    "prev_caesarean", "smokes", "wealth", "mother_edu", "District",
]
DELIVERY_FEATURES = PREGNANCY_FEATURES[:-1] + [
    "birth_weight", "is_male", "caesarean", "facility_delivery", "District"
]
# Backward-compatible name: full delivery-stage feature list.
FEATURES = DELIVERY_FEATURES

WEALTH = {"Poorest": 1, "Poorer": 2, "Middle": 3, "Richer": 4, "Richest": 5}
EDU = {"No education": 0, "Primary": 1, "Secondary": 2, "Higher": 3}

RAW_COLUMNS_USED = [
    "Case ID", "Birth Index", "Child is Alive", "Age at death category",
    "current_age_of_children", "District", "Mother's Age", "ANC Visits",
    "First ANC Timing", "Birth Weight", "Minutes to the nearest health facility",
    "Transport to Health Facility", "Place of Residence",
    "Previous Caesarean Section", "Smokes Cigarette", "Wealth Status",
    "Highest Educational Level_y", "Sex of Child", "Caesarean Delivery",
    "Place of Delivery",
]


def _one_consistent_value(series: pd.Series):
    """Return the sole nonmissing value; conflicting values become NaN."""
    vals = series.dropna().drop_duplicates()
    if len(vals) == 1:
        return vals.iloc[0]
    return np.nan


def consolidate_birth_records(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Consolidate duplicate mother/birth rows and return a conflict audit table."""
    missing = [c for c in RAW_COLUMNS_USED if c not in raw.columns]
    if missing:
        raise ValueError("Input data is missing required columns: " + ", ".join(missing))

    df = raw.copy()
    # Normalise whitespace so visually identical DHS IDs are grouped together.
    df["Case ID"] = (
        df["Case ID"].astype("string").str.replace(r"\s+", " ", regex=True).str.strip()
    )
    df["Birth Index"] = pd.to_numeric(df["Birth Index"], errors="coerce")

    # Rows with no reliable birth key cannot be safely linked to a child.
    df = df.dropna(subset=KEYS).copy()

    value_cols = [c for c in df.columns if c not in KEYS]
    audit_rows = []
    consolidated = []
    for (case_id, birth_index), group in df.groupby(KEYS, dropna=False, sort=False):
        row = {"Case ID": case_id, "Birth Index": birth_index}
        for col in value_cols:
            nonmissing = group[col].dropna().drop_duplicates()
            if len(nonmissing) > 1:
                audit_rows.append({
                    "Case ID": case_id,
                    "Birth Index": birth_index,
                    "column": col,
                    "conflicting_nonmissing_values": " | ".join(map(str, nonmissing.tolist())),
                    "action": "set to missing before imputation",
                })
                row[col] = np.nan
            elif len(nonmissing) == 1:
                row[col] = nonmissing.iloc[0]
            else:
                row[col] = np.nan
        consolidated.append(row)

    return pd.DataFrame(consolidated), pd.DataFrame(
        audit_rows,
        columns=["Case ID", "Birth Index", "column", "conflicting_nonmissing_values", "action"],
    )


def died_in_neonatal_period(value):
    """Return 1 for death at age 0–27 days, 0 for later death, NaN if unknown."""
    if pd.isna(value):
        return np.nan
    s = str(value).strip().lower()
    match = re.fullmatch(r"(\d+)\s*days?", s)
    if match:
        days = int(match.group(1))
        return 1.0 if 0 <= days < 28 else 0.0
    if s in {"day 0", "0 day", "0 days"}:
        return 1.0
    if re.fullmatch(r"\d+\s*months?", s) or re.fullmatch(r"\d+\s*years?", s):
        return 0.0
    return np.nan


def build_features(raw: pd.DataFrame, audit_path: str | None = None) -> pd.DataFrame:
    """
    Return one row per child with a valid binary outcome and engineered predictors.
    Birth weight is kept in grams to match the uploaded data and dashboard inputs.
    """
    df, audit = consolidate_birth_records(raw)
    if audit_path:
        audit.to_csv(audit_path, index=False)

    alive = df["Child is Alive"].astype("string").str.strip().str.lower().eq("yes")
    current_age_months = pd.to_numeric(df["current_age_of_children"], errors="coerce")
    age_at_death = df["Age at death category"].map(died_in_neonatal_period)

    # A living infant younger than 1 completed month has not completed the neonatal
    # outcome window. Do not label that infant as a negative outcome.
    df[TARGET] = np.where(
        alive & current_age_months.ge(1), 0.0,
        np.where(alive, np.nan, age_at_death)
    )

    # DHS missing / don't know / not weighed codes.
    df["Birth Weight"] = pd.to_numeric(df["Birth Weight"], errors="coerce").replace(
        [9996, 9998, 9999], np.nan
    )
    df["ANC Visits"] = pd.to_numeric(df["ANC Visits"], errors="coerce").replace(
        [98, 99], np.nan
    )
    df["First ANC Timing"] = pd.to_numeric(df["First ANC Timing"], errors="coerce").replace(
        [98, 99], np.nan
    )

    df["mother_age"] = pd.to_numeric(df["Mother's Age"], errors="coerce")
    df["birth_order"] = pd.to_numeric(df["Birth Index"], errors="coerce")
    df["anc_visits"] = df["ANC Visits"]
    df["first_anc_month"] = df["First ANC Timing"]
    df["birth_weight"] = df["Birth Weight"]
    df["minutes_to_facility"] = pd.to_numeric(
        df["Minutes to the nearest health facility"], errors="coerce"
    )
    df["is_male"] = df["Sex of Child"].astype("string").str.strip().str.lower().eq("male").fillna(False).astype(int)
    df["is_urban"] = df["Place of Residence"].astype("string").str.strip().str.lower().eq("urban").fillna(False).astype(int)
    df["caesarean"] = df["Caesarean Delivery"].astype("string").str.strip().str.lower().eq("yes").fillna(False).astype(int)
    df["prev_caesarean"] = df["Previous Caesarean Section"].astype("string").str.strip().str.lower().eq("yes").fillna(False).astype(int)
    df["smokes"] = df["Smokes Cigarette"].astype("string").str.strip().str.lower().eq("yes").fillna(False).astype(int)
    place = df["Place of Delivery"].astype("string").str.strip()
    df["facility_delivery"] = (
        (~place.isin(["Respondent's home", "Other home"]) & place.notna())
    ).astype(int)
    df["walks_to_facility"] = (
        df["Transport to Health Facility"].astype("string").str.strip().str.lower().eq("walking")
    ).astype(int)
    df["wealth"] = df["Wealth Status"].map(WEALTH)
    df["mother_edu"] = df["Highest Educational Level_y"].map(EDU)
    df["District"] = df["District"].astype("string").str.strip()

    # Do not impute here: medians are learned from training folds in the model pipeline.
    df = df.dropna(subset=[TARGET, "District"]).copy()
    df[TARGET] = df[TARGET].astype(int)
    df = df.reset_index(drop=True)
    return df


class CalibratedRiskModel:
    """Pickle-friendly wrapper: Random Forest score mapped to calibrated probability."""
    def __init__(self, base_model, calibrator):
        self.base_model = base_model
        self.calibrator = calibrator
        self.feature_names_in_ = getattr(base_model, "feature_names_in_", None)
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        raw_probability = self.base_model.predict_proba(X)[:, 1]
        calibrated = self.calibrator.predict_proba(
            np.asarray(raw_probability).reshape(-1, 1)
        )[:, 1]
        calibrated = np.clip(calibrated, 0.0, 1.0)
        return np.column_stack([1.0 - calibrated, calibrated])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

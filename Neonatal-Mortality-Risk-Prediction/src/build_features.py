import numpy as np
import pandas as pd
import re

FEATURES = [
    "mother_age", "birth_order", "anc_visits", "first_anc_month",
    "birth_weight", "minutes_to_facility", "is_male", "is_urban",
    "caesarean", "prev_caesarean", "smokes", "facility_delivery",
    "walks_to_facility", "wealth", "mother_edu", "district_code"
]

TARGET = "neonatal_death"

WEALTH = {
    "Poorest": 1, "Poorer": 2, "Middle": 3,
    "Richer": 4, "Richest": 5
}

EDU = {
    "No education": 0, "Primary": 1,
    "Secondary": 2, "Higher": 3
}


def died_in_neonatal_period(s):
    if pd.isna(s):
        return np.nan

    s = str(s).strip().lower()

    if s in ["day 0", "0 days", "0 day"]:
        return 1.0

    match = re.fullmatch(r"(\d+)\s*days?", s)

    if match:
        days = int(match.group(1))
        return 1.0 if 0 <= days < 28 else 0.0

    if re.fullmatch(r"\d+\s*months?", s):
        return 0.0

    if re.fullmatch(r"\d+\s*years?", s):
        return 0.0

    return np.nan


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df.drop_duplicates(
        subset=["Case ID", "Birth Index"]
    ).reset_index(drop=True)

    died = df["Age at death category"].map(
        died_in_neonatal_period
    )

    df[TARGET] = np.where(
        df["Child is Alive"] == "Yes", 0.0, died
    )

    df["Birth Weight"] = df["Birth Weight"].replace(9998, np.nan)
    df["ANC Visits"] = df["ANC Visits"].replace(98, np.nan)
    df["First ANC Timing"] = df["First ANC Timing"].replace(98, np.nan)

    df["mother_age"] = df["Mother's Age"]
    df["birth_order"] = df["Birth Index"]
    df["anc_visits"] = df["ANC Visits"]
    df["first_anc_month"] = df["First ANC Timing"]
    df["birth_weight"] = df["Birth Weight"]
    df["minutes_to_facility"] = df[
        "Minutes to the nearest health facility"
    ]

    df["is_male"] = (df["Sex of Child"] == "Male").astype(int)
    df["is_urban"] = (df["Place of Residence"] == "Urban").astype(int)
    df["caesarean"] = (df["Caesarean Delivery"] == "Yes").astype(int)
    df["prev_caesarean"] = (
        df["Previous Caesarean Section"] == "Yes"
    ).astype(int)
    df["smokes"] = (df["Smokes Cigarette"] == "Yes").astype(int)

    df["facility_delivery"] = (
        ~df["Place of Delivery"].isin(
            ["Respondent's home", "Other home"]
        )
    ).astype(int)

    df["walks_to_facility"] = (
        df["Transport to Health Facility"] == "Walking"
    ).astype(int)

    df["wealth"] = df["Wealth Status"].map(WEALTH)
    df["mother_edu"] = df["Highest Educational Level_y"].map(EDU)
    df["district_code"] = df["District"].astype("category").cat.codes

    df[FEATURES] = df[FEATURES].fillna(df[FEATURES].median())

    return df.dropna(subset=[TARGET]).reset_index(drop=True)

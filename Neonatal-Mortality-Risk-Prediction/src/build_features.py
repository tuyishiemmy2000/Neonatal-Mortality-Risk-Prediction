"""Build the modelling table for neonatal mortality risk prediction (Rwanda DHS 2025)."""
import numpy as np
import pandas as pd

FEATURES = ["mother_age", "birth_order", "anc_visits", "first_anc_month",
            "birth_weight", "minutes_to_facility", "is_male", "is_urban",
            "caesarean", "prev_caesarean", "smokes", "facility_delivery",
            "walks_to_facility", "wealth", "mother_edu", "district_code"]
TARGET = "neonatal_death"

WEALTH = {"Poorest": 1, "Poorer": 2, "Middle": 3, "Richer": 4, "Richest": 5}
EDU = {"No education": 0, "Primary": 1, "Secondary": 2, "Higher": 3}


def died_in_neonatal_period(s) -> float:
    """'Day 0', '3 days', '28 days' -> 1; '1 month', '2 years' -> 0; missing -> NaN."""
    if pd.isna(s):
        return np.nan
    s = str(s).lower()
    if s.startswith("day 0"):
        return 1.0
    n, unit = s.split()
    return 1.0 if unit.startswith("day") and int(n) <= 28 else 0.0


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # one row per child (the merge repeated each birth several times)
    df = df.drop_duplicates(subset=["Case ID", "Birth Index"]).reset_index(drop=True)

    # target: alive -> 0 ; dead -> 1 only if died within 28 days
    died = df["Age at death category"].map(died_in_neonatal_period)
    df[TARGET] = np.where(df["Child is Alive"] == "Yes", 0.0, died)

    # clean DHS "don't know" codes
    df["Birth Weight"] = df["Birth Weight"].replace(9998, np.nan)
    df["ANC Visits"] = df["ANC Visits"].replace(98, np.nan)
    df["First ANC Timing"] = df["First ANC Timing"].replace(98, np.nan)

    # features
    df["mother_age"] = df["Mother's Age"]
    df["birth_order"] = df["Birth Index"]
    df["anc_visits"] = df["ANC Visits"]
    df["first_anc_month"] = df["First ANC Timing"]
    df["birth_weight"] = df["Birth Weight"]
    df["minutes_to_facility"] = df["Minutes to the nearest health facility"]
    df["is_male"] = (df["Sex of Child"] == "Male").astype(int)
    df["is_urban"] = (df["Place of Residence"] == "Urban").astype(int)
    df["caesarean"] = (df["Caesarean Delivery"] == "Yes").astype(int)
    df["prev_caesarean"] = (df["Previous Caesarean Section"] == "Yes").astype(int)
    df["smokes"] = (df["Smokes Cigarette"] == "Yes").astype(int)
    df["facility_delivery"] = (~df["Place of Delivery"].isin(
        ["Respondent's home", "Other home"])).astype(int)
    df["walks_to_facility"] = (df["Transport to Health Facility"] == "Walking").astype(int)
    df["wealth"] = df["Wealth Status"].map(WEALTH)
    df["mother_edu"] = df["Highest Educational Level_y"].map(EDU)
    df["district_code"] = df["District"].astype("category").cat.codes

    # missing values are filled later, inside the model pipeline (SimpleImputer),
    # so the medians are learned from the training data only

    # drop children whose neonatal outcome is unknown
    return df.dropna(subset=[TARGET]).reset_index(drop=True)


if __name__ == "__main__":
    data = build_features(pd.read_csv("data/processed_data_cleaned.csv"))
    print(data.shape)
    print(data[TARGET].value_counts())

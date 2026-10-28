"""
Clinical interpretation helpers for the Rwanda neonatal risk research prototype.

These helpers provide transparent input checks and model-performance summaries.
They do not diagnose conditions, estimate causal effects, or replace local clinical guidance.
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    precision_score, recall_score,
)

FEATURE_META = {
    "mother_age": {"label": "Mother's age (years)", "stage": "before", "type": "context"},
    "birth_order": {"label": "Birth order", "stage": "before", "type": "context"},
    "anc_visits": {"label": "ANC visits recorded", "stage": "before", "type": "care_history"},
    "first_anc_month": {"label": "Month of first ANC visit", "stage": "before", "type": "care_history"},
    "minutes_to_facility": {"label": "Travel time to facility (minutes)", "stage": "before", "type": "access"},
    "walks_to_facility": {"label": "Walking is the transport to the facility", "stage": "before", "type": "access"},
    "is_urban": {"label": "Urban residence", "stage": "before", "type": "context"},
    "prev_caesarean": {"label": "Previous caesarean section", "stage": "before", "type": "obstetric_history"},
    "smokes": {"label": "Cigarette smoking reported", "stage": "before", "type": "modifiable_exposure"},
    "wealth": {"label": "Household wealth quintile code", "stage": "before", "type": "context"},
    "mother_edu": {"label": "Maternal education code", "stage": "before", "type": "context"},
    "District": {"label": "District", "stage": "both", "type": "context"},
    "birth_weight": {"label": "Birth weight (grams)", "stage": "after", "type": "newborn_measure"},
    "is_male": {"label": "Male newborn", "stage": "after", "type": "newborn_characteristic"},
    "caesarean": {"label": "Caesarean delivery", "stage": "after", "type": "delivery_history"},
    "facility_delivery": {"label": "Delivery in a health facility", "stage": "after", "type": "delivery_context"},
}


def features_for_stage(stage: str, pregnancy_features, delivery_features):
    """Return the configured feature names for a valid assessment stage."""
    stage = stage.strip().lower()
    if stage in {"pregnancy", "before", "before_birth"}:
        return list(pregnancy_features)
    if stage in {"delivery", "after", "after_delivery"}:
        return list(delivery_features)
    raise ValueError("stage must be pregnancy/before or delivery/after")


def clinical_review_prompts(stage: str, values: dict) -> list[str]:
    """
    Return transparent checks based on entered values.
    These are prompts for review, not the model's feature attributions or care orders.
    """
    stage = stage.strip().lower()
    notes = []
    age = values.get("mother_age")
    if age is not None and np.isfinite(float(age)) and (float(age) < 18 or float(age) >= 35):
        notes.append(
            "Maternal age is below 18 or 35 and older: consider this alongside the full maternal assessment and current local guidance; age alone does not determine care."
        )
    if values.get("prev_caesarean") == 1:
        notes.append(
            "Previous caesarean section reported: review the obstetric history and delivery plan using local protocols."
        )
    if values.get("smokes") == 1:
        notes.append(
            "Cigarette smoking reported: consider appropriate cessation support under local guidance."
        )
    if values.get("anc_visits", 0) < 4:
        notes.append(
            "Fewer than four ANC visits are recorded in this input: check the history and compare with the current national ANC schedule. This is a review prompt, not a validated risk cutoff."
        )
    if values.get("first_anc_month", 0) > 3:
        notes.append(
            "The first ANC visit is recorded after the first trimester: verify timing and follow-up needs against local guidance."
        )
    if values.get("minutes_to_facility", 0) >= 60:
        notes.append(
            "Travel time is recorded as at least 60 minutes: consider access, transport and referral-planning barriers. This is an operational prompt, not a clinical threshold."
        )
    if stage in {"delivery", "after", "after_delivery"}:
        weight = values.get("birth_weight")
        if weight is not None and np.isfinite(float(weight)) and float(weight) < 2500:
            notes.append(
                "Birth weight is below 2,500 g, a conventional low-birth-weight marker: follow the approved newborn assessment and referral protocol."
            )
        if values.get("facility_delivery") == 0:
            notes.append(
                "Home or non-facility delivery reported: confirm that the newborn receives appropriate assessment and postnatal follow-up under local guidance."
            )
    if not notes:
        notes.append(
            "No additional rule-based review prompt was triggered by these entered values. Continue routine clinical assessment and approved care protocols."
        )
    return notes


def evaluate_probabilities(y_true, probabilities, threshold: float) -> dict:
    """Calculate useful screening metrics; expects an independent evaluation sample."""
    y = np.asarray(y_true).astype(int)
    p = np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)
    pred = (p >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan,
        "average_precision": float(average_precision_score(y, p)),
        "brier_score": float(brier_score_loss(y, p)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "flagged_share": float(pred.mean()),
        "mean_predicted_risk": float(p.mean()),
        "observed_rate": float(y.mean()),
    }

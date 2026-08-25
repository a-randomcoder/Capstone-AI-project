"""
Common patient input schema for the Maternal Digital Twin prototype.

All modules receive the same patient_data dict and extract only the fields they need.
This is a research/student decision-support schema — not a clinical record standard.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

DEMOGRAPHICS = ["age", "ethnicity"]
PREGNANCY = [
    "gestational_age",
    "booking_gestational_age",
    "pre_pregnancy_bmi",
    "obs_score_l",
    "lmp_known",
    "usg",
]
VITALS = [
    "systolic_bp",
    "diastolic_bp",
    "pulse",
    "body_temp",
]
GLUCOSE = [
    "blood_sugar",
    "early_rbs_mgdl",
    "early_ppbs_mgdl",
    "early_hba1c_percent",
    "early_ogtt_performed",
    "early_ogtt_fasting_mgdl",
    "early_ogtt_1h_mgdl",
    "early_ogtt_2h_mgdl",
]
HEMATOLOGY = [
    "trbc",
    "mcv",
    "mch",
    "mchc",
    "rdw",
    "reticulocyte_count",
    "reticulocyte_pct",
    "serum_iron",
    "tibc",
    "transferrin_saturation",
    "total_bilirubin",
    "direct_bilirubin",
    "indirect_bilirubin",
    "urea",
    "creatinine",
]
CLINICAL_SIGNS = [
    "pallor",
    "edema",
    "icterus",
    "dietary_habits",
    "history_iron_supplementation",
    "history_blood_transfusion",
    "family_history_hemoglobinopathy",
    "history_allergy",
]
RISK_FLAGS = [
    "family_history_dm",
    "previous_gdm",
    "pcos",
    "previous_macrosomia",
]
THYROID = [
    "tsh_baseline",
    "ft3_baseline",
    "ft4_baseline",
    "tpo_baseline",
    "household_income",
    "parity",
    "family_history_diabetes",
    "smoking_exposure",
    "alcohol_consumption",
    "folic_acid_supplementation",
    "vd_supplementation",
]

ALL_OPTIONAL_KEYS = (
    DEMOGRAPHICS
    + PREGNANCY
    + VITALS
    + GLUCOSE
    + HEMATOLOGY
    + CLINICAL_SIGNS
    + RISK_FLAGS
    + THYROID
)


def empty_patient() -> Dict[str, Any]:
    """Return a patient_data dict with all known keys set to None."""
    return {k: None for k in ALL_OPTIONAL_KEYS}


def sample_patient() -> Dict[str, Any]:
    """
    Illustrative sample for demos / tests.
    Values are plausible synthetic ranges — not a real patient record.
    """
    return {
        "age": 28,
        "ethnicity": "Asian",
        "gestational_age": 24.0,
        "booking_gestational_age": 10.0,
        "pre_pregnancy_bmi": 26.5,
        "obs_score_l": 1,
        "lmp_known": "Yes",
        "usg": "Yes",
        "systolic_bp": 135,
        "diastolic_bp": 88,
        "pulse": 86,
        "body_temp": 98.2,
        "blood_sugar": 12.0,
        "early_rbs_mgdl": 118.0,
        "early_ppbs_mgdl": 145.0,
        "early_hba1c_percent": 5.6,
        "early_ogtt_performed": 0,
        "early_ogtt_fasting_mgdl": None,
        "early_ogtt_1h_mgdl": None,
        "early_ogtt_2h_mgdl": None,
        "trbc": 3.9,
        "mcv": 78.0,
        "mch": 25.0,
        "mchc": 31.5,
        "rdw": 15.2,
        "reticulocyte_count": 0.05,
        "reticulocyte_pct": 1.2,
        "serum_iron": 55.0,
        "tibc": 420.0,
        "transferrin_saturation": 13.1,
        "total_bilirubin": 0.5,
        "direct_bilirubin": 0.15,
        "indirect_bilirubin": 0.35,
        "urea": 18.0,
        "creatinine": 0.7,
        "pallor": "Yes",
        "edema": "No",
        "icterus": "No",
        "dietary_habits": "Vegetarian",
        "history_iron_supplementation": "No",
        "history_blood_transfusion": "No",
        "family_history_hemoglobinopathy": "No",
        "history_allergy": "No",
        "family_history_dm": 0,
        "previous_gdm": 0,
        "pcos": 0,
        "previous_macrosomia": 0,
        # Thyroid baseline (first trimester)
        "tsh_baseline": 2.8,
        "ft3_baseline": 4.6,
        "ft4_baseline": 14.5,
        "tpo_baseline": 12.0,
        "household_income": 2,
        "parity": 1,
        "family_history_diabetes": 0,
        "smoking_exposure": 0,
        "alcohol_consumption": 0,
        "folic_acid_supplementation": 2,
        "vd_supplementation": 1,
    }


def standardize_output(
    condition: str,
    prediction: str,
    probability: Optional[float] = None,
    class_probabilities: Optional[Dict[str, float]] = None,
    important_factors: Optional[List[Dict[str, Any]]] = None,
    notes: str = "",
    status: str = "ok",
) -> Dict[str, Any]:
    """Uniform return shape for every predict_* function."""
    return {
        "condition": condition,
        "prediction": prediction,
        "probability": probability,
        "class_probabilities": class_probabilities or {},
        "important_factors": important_factors or [],
        "notes": notes,
        "status": status,
    }

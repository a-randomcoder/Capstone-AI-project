"""
Anemia module - prediction interface.

Uses Track B Logistic Regression artifacts regenerated from anemia_module.ipynb
(RANDOM_STATE=42). Hb, PCV, and who_pred are excluded due to target leakage.

This is a student research prototype, not a clinical diagnostic system.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

MODULE_DIR = Path(__file__).resolve().parents[2] / "models" / "anemia"

# Common-schema key → training column name (Track B)
FEATURE_MAP = {
    "obs_score_l": "Obs Score L",
    "lmp_known": "LMP known",
    "usg": "USG",
    "gestational_age": "POG (in weeks)",
    "history_blood_transfusion": "History of blood transfusion during pregnancy",
    "family_history_hemoglobinopathy": "Family history of hemoglobinopathy",
    "dietary_habits": "Dietary habits",
    "history_allergy": "History of any type of allergy",
    "history_iron_supplementation": "History of iron supplementation",
    "pulse": "Pulse (bpm)",
    "icterus": "Icterus",
    "pallor": "Pallor",
    "edema": "Edema",
    "trbc": "TRBC (in 10^6 /microL)",
    "mcv": "MCV (in fL)",
    "mch": "MCH (in pg)",
    "mchc": "MCHC (in gm/dL)",
    "rdw": "RDW (%)",
    "reticulocyte_count": "Reticulocyte count (in 10^6 /microL)",
    "reticulocyte_pct": "Reticulocyte%",
    "serum_iron": "Serum iron (in microg/dL)",
    "tibc": "TIBC (in microg/dL)",
    "transferrin_saturation": "Transferrin saturation (%)",
    "total_bilirubin": "Total bilirubin (mg/dL)",
    "direct_bilirubin": "Direct (mg/dL)",
    "indirect_bilirubin": "Indirect (mg/dL)",
    "urea": "Urea (mg/dL)",
    "creatinine": "Creatinine (mg/dL)",
    "systolic_bp": "SBP",
    "diastolic_bp": "DBP",
}

# Also accept training column names directly
REVERSE_ALIASES = {v: v for v in FEATURE_MAP.values()}

YES_NO = {"Yes": 1, "No": 0, "yes": 1, "no": 0, 1: 1, 0: 0, True: 1, False: 0}
DIET = {
    "Non-Vegetarian": 1,
    "Vegetarian": 0,
    "non-vegetarian": 1,
    "vegetarian": 0,
    1: 1,
    0: 0,
}

_model = None
_preprocessor = None
_label_encoder = None
_metadata = None
_feature_importance = None


def _load():
    global _model, _preprocessor, _label_encoder, _metadata, _feature_importance
    if _model is not None:
        return
    _model = joblib.load(MODULE_DIR / "anemia_best_model.joblib")
    _preprocessor = joblib.load(MODULE_DIR / "anemia_preprocessor.joblib")
    _label_encoder = joblib.load(MODULE_DIR / "anemia_label_encoder.joblib")
    with open(MODULE_DIR / "anemia_model_metadata.json") as f:
        _metadata = json.load(f)
    try:
        _feature_importance = pd.read_csv(
            MODULE_DIR / "anemia_feature_importance.csv", index_col=0
        )
    except Exception:
        _feature_importance = None


def _resolve_value(patient_data: dict, train_col: str) -> Any:
    """Look up a Track B column from common schema keys or raw training names."""
    # Direct training name
    if train_col in patient_data and patient_data[train_col] is not None:
        return patient_data[train_col]
    # Common schema key
    for common, mapped in FEATURE_MAP.items():
        if mapped == train_col and common in patient_data and patient_data[common] is not None:
            return patient_data[common]
    return None


def _encode(col: str, val: Any) -> Any:
    if col == "Dietary habits":
        if isinstance(val, str) and val in DIET:
            return DIET[val]
        return val
    yes_no_cols = {
        "LMP known",
        "USG",
        "History of blood transfusion during pregnancy",
        "Family history of hemoglobinopathy",
        "History of any type of allergy",
        "History of iron supplementation",
        "Icterus",
        "Pallor",
        "Edema",
    }
    if col in yes_no_cols:
        if val in YES_NO:
            return YES_NO[val]
        return val
    return val


def predict_anemia(patient_data: dict, top_k_factors: int = 5) -> dict:
    """
    Predict maternal anemia severity (Track B - Hb/PCV excluded).

    Returns standardized dict:
      condition, prediction, probability, class_probabilities,
      important_factors, notes, status
    """
    _load()
    track_B = _metadata["raw_feature_list_track_B"]
    missing = []
    row = {}
    for col in track_B:
        val = _resolve_value(patient_data, col)
        if val is None:
            missing.append(col)
        else:
            row[col] = _encode(col, val)

    if missing:
        return {
            "condition": "Anemia",
            "prediction": "Insufficient input",
            "probability": None,
            "class_probabilities": {},
            "important_factors": [],
            "notes": (
                f"Missing required Track B fields: {missing[:8]}"
                + ("..." if len(missing) > 8 else "")
                + ". Hb/PCV are intentionally not used (leakage)."
            ),
            "status": "missing_features",
        }

    X_input = pd.DataFrame([row], columns=track_B)
    X_proc = _preprocessor.transform(X_input)

    pred_idx = int(_model.predict(X_proc)[0])
    proba = _model.predict_proba(X_proc)[0]
    classes = [str(c) for c in _label_encoder.classes_]
    pred_name = classes[pred_idx]
    class_probs = {cls: round(float(p), 4) for cls, p in zip(classes, proba)}
    probability = float(proba[pred_idx])

    # Important factors from global |coef| ranking (Logistic Regression)
    important = []
    if _feature_importance is not None:
        # index may be feature names
        fi = _feature_importance.copy()
        if fi.shape[1] >= 1:
            col = fi.columns[0]
            ranked = fi[col].abs().sort_values(ascending=False)
            for feat, imp in ranked.head(top_k_factors).items():
                important.append(
                    {
                        "feature": str(feat),
                        "shap_contribution": round(float(imp), 4),
                        "source": "global_|coef|_importance",
                    }
                )

    return {
        "condition": "Anemia",
        "prediction": pred_name,
        "probability": round(probability, 4),
        "class_probabilities": class_probs,
        "important_factors": important,
        "notes": (
            "Track B Logistic Regression. Hb/PCV/who_pred excluded due to leakage "
            "(WHO Hb thresholds recover ~91.4% of labels). Severe class is rare (~9 samples). "
            "Research prototype - not a diagnosis."
        ),
        "status": "ok",
    }


if __name__ == "__main__":
    from src.integration.schemas import sample_patient

    # Need full Track B fields - map sample into training names for a quick smoke test
    p = sample_patient()
    print(json.dumps(predict_anemia(p), indent=2))

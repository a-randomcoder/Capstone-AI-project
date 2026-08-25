"""
GDM (Gestational Diabetes Mellitus) module - prediction interface.

Uses the existing Random Forest pipeline (gdm_best_model.joblib) from the
GDM notebook. Dataset is SYNTHETIC (~65% GDM prevalence). Not clinically validated.

Research prototype - not a clinical diagnostic system.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = ROOT / "models" / "gdm"

GDM_FEATURES = [
    "age",
    "pre_pregnancy_bmi",
    "ethnicity",
    "family_history_dm",
    "previous_gdm",
    "pcos",
    "previous_macrosomia",
    "booking_gestational_age",
    "early_rbs_mgdl",
    "early_ppbs_mgdl",
    "early_hba1c_percent",
    "early_ogtt_performed",
    "early_ogtt_fasting_mgdl",
    "early_ogtt_1h_mgdl",
    "early_ogtt_2h_mgdl",
]

BINARY_KEYS = {
    "family_history_dm",
    "previous_gdm",
    "pcos",
    "previous_macrosomia",
    "early_ogtt_performed",
}

_model = None


def _load():
    global _model
    if _model is not None:
        return
    candidates = [
        MODULE_DIR / "gdm_best_model.joblib",
        ROOT / "Capstone-AI-Project" / "Notebook" / "models" / "gdm_best_model.joblib",
        ROOT / "models" / "gdm_best_model.joblib",
    ]
    for c in candidates:
        if c.exists() and c.stat().st_size > 0:
            _model = joblib.load(c)
            return
    raise FileNotFoundError(f"gdm_best_model.joblib not found in {candidates}")


def _get(patient_data: dict, key: str):
    if key in patient_data and patient_data[key] is not None:
        return patient_data[key]
    aliases = {
        "booking_gestational_age": ["gestational_age_booking", "gestational_age"],
        "pre_pregnancy_bmi": ["bmi", "bmi_prepregnancy"],
    }
    for a in aliases.get(key, []):
        if a in patient_data and patient_data[a] is not None:
            return patient_data[a]
    return None


def _to_binary(val) -> Any:
    if val is None:
        return None
    if isinstance(val, (int, float, np.integer, np.floating)):
        return int(val)
    s = str(val).strip().lower()
    if s in ("yes", "y", "true", "1"):
        return 1
    if s in ("no", "n", "false", "0"):
        return 0
    return val


def predict_gdm(patient_data: dict, top_k_factors: int = 5) -> dict:
    """Estimate GDM probability using the saved Random Forest pipeline."""
    _load()

    row: Dict[str, Any] = {}
    critical_missing = []
    for col in GDM_FEATURES:
        val = _get(patient_data, col)
        if col in BINARY_KEYS:
            val = _to_binary(val)
        row[col] = val

    for req in ["age", "pre_pregnancy_bmi", "ethnicity"]:
        if row.get(req) is None:
            critical_missing.append(req)

    if critical_missing:
        return {
            "condition": "GDM",
            "prediction": "Insufficient input",
            "probability": None,
            "class_probabilities": {},
            "important_factors": [],
            "notes": (
                f"Missing critical fields: {critical_missing}. "
                "Model trained on SYNTHETIC data (~65% GDM prevalence) - not clinically validated."
            ),
            "status": "missing_features",
        }

    X = pd.DataFrame([row], columns=GDM_FEATURES)

    pred = int(_model.predict(X)[0])
    proba = _model.predict_proba(X)[0]
    label = "GDM likely (synthetic model)" if pred == 1 else "GDM unlikely (synthetic model)"
    class_probs = {
        "No GDM": round(float(proba[0]), 4),
        "GDM": round(float(proba[1]), 4),
    }
    probability = round(float(proba[pred]), 4)

    important: List[Dict[str, Any]] = []
    try:
        clf = _model.named_steps["classifier"]
        pre = _model.named_steps["preprocessor"]
        try:
            feat_names = list(pre.get_feature_names_out())
        except Exception:
            feat_names = [f"f{i}" for i in range(clf.n_features_in_)]
        if hasattr(clf, "feature_importances_"):
            imp = pd.Series(clf.feature_importances_, index=feat_names).sort_values(
                ascending=False
            )
            for feat, val in imp.head(top_k_factors).items():
                clean = str(feat).replace("num__", "").replace("cat__", "")
                important.append(
                    {
                        "feature": clean,
                        "shap_contribution": round(float(val), 4),
                        "source": "rf_feature_importance",
                    }
                )
    except Exception:
        pass

    return {
        "condition": "GDM",
        "prediction": label,
        "probability": probability,
        "class_probabilities": class_probs,
        "important_factors": important,
        "notes": (
            "Random Forest pipeline trained on SYNTHETIC data (n~10,000, ~65% GDM prevalence). "
            "Not clinically validated. Not a diagnostic model. OGTT fields are median-imputed when missing."
        ),
        "status": "ok",
    }


if __name__ == "__main__":
    import json
    from src.integration.schemas import sample_patient

    print(json.dumps(predict_gdm(sample_patient()), indent=2))

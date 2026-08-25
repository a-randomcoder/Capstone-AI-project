"""
Thyroid dysfunction module - prediction interface.

Uses Random Forest pipeline (thyroid_final_model.pkl) regenerated from
thyroid_model_complete.ipynb with identical methodology (RANDOM_STATE=42,
tuned RF: n_estimators=400, max_depth=8, min_samples_leaf=10).

Target: thyroid_dysfunction_later among women NOT SCH-consistent at baseline
(TSH>4.0 & FT4 12-22 at occ2 or occ3). Prediction time = end of first trimester.

Research prototype - not a clinical diagnostic system.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = ROOT / "models" / "thyroid"

FEATURES = [
    "age", "bmi", "tsh_baseline", "ft3_baseline", "ft4_baseline", "tpo_baseline",
    "household_income", "parity", "family_history_diabetes", "smoking_exposure",
    "alcohol_consumption", "folic_acid_supplementation", "vd_supplementation",
    "log_tsh_baseline",
]

ALIASES = {
    "age": ["age", "Age"],
    "bmi": ["bmi", "BMI", "pre_pregnancy_bmi"],
    "tsh_baseline": ["tsh_baseline", "TSH", "tsh"],
    "ft3_baseline": ["ft3_baseline", "FT3", "ft3"],
    "ft4_baseline": ["ft4_baseline", "FT4", "ft4"],
    "tpo_baseline": ["tpo_baseline", "TPO", "tpo", "TPO_clean"],
    "household_income": ["household_income"],
    "parity": ["parity"],
    "family_history_diabetes": ["family_history_diabetes", "family_history_dm"],
    "smoking_exposure": ["smoking_exposure"],
    "alcohol_consumption": ["alcohol_consumption"],
    "folic_acid_supplementation": ["folic_acid_supplementation"],
    "vd_supplementation": ["vd_supplementation"],
}

_model = None
_meta = None


def _load():
    global _model, _meta
    if _model is not None:
        return
    candidates = [
        MODULE_DIR / "thyroid_final_model.pkl",
        ROOT / "models" / "thyroid_final_model.pkl",
        ROOT / "thyroid_final_model.pkl",
    ]
    path = next((c for c in candidates if c.exists() and c.stat().st_size > 0), None)
    if path is None:
        raise FileNotFoundError(f"thyroid_final_model.pkl not found in {candidates}")
    _model = joblib.load(path)
    meta_path = MODULE_DIR / "thyroid_feature_metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            _meta = json.load(f)


def _get(patient_data: dict, feature: str):
    for key in ALIASES.get(feature, [feature]):
        if key in patient_data and patient_data[key] is not None:
            return patient_data[key]
    return None


def predict_thyroid(patient_data: dict, top_k_factors: int = 5) -> Dict[str, Any]:
    """Estimate risk of later thyroid dysfunction from first-trimester baseline features."""
    try:
        _load()
    except FileNotFoundError as e:
        return {
            "condition": "Thyroid",
            "prediction": "Model missing",
            "probability": None,
            "class_probabilities": {},
            "important_factors": [],
            "notes": str(e),
            "status": "missing_model",
        }

    row: Dict[str, Any] = {}
    missing = []
    for feat in FEATURES:
        if feat == "log_tsh_baseline":
            continue
        val = _get(patient_data, feat)
        if val is None and feat in ("age", "tsh_baseline", "ft4_baseline"):
            missing.append(feat)
        row[feat] = val

    if missing:
        return {
            "condition": "Thyroid",
            "prediction": "Insufficient input",
            "probability": None,
            "class_probabilities": {},
            "important_factors": [],
            "notes": f"Missing critical fields: {missing}.",
            "status": "missing_features",
        }

    tsh = row.get("tsh_baseline")
    row["log_tsh_baseline"] = float(np.log1p(float(tsh))) if tsh is not None else None

    X = pd.DataFrame([row], columns=FEATURES)
    pred = int(_model.predict(X)[0])
    proba = _model.predict_proba(X)[0]
    label = (
        "later thyroid dysfunction risk elevated"
        if pred == 1
        else "later thyroid dysfunction risk not elevated"
    )
    class_probs = {
        "no later dysfunction": round(float(proba[0]), 4),
        "later dysfunction": round(float(proba[1]), 4),
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
            imp = pd.Series(clf.feature_importances_, index=feat_names).sort_values(ascending=False)
            for feat, val in imp.head(top_k_factors).items():
                clean = str(feat).replace("num__", "").replace("cat__", "")
                important.append({
                    "feature": clean,
                    "shap_contribution": round(float(val), 4),
                    "source": "rf_feature_importance",
                })
    except Exception:
        pass

    notes = (
        "Random Forest (tuned). Target = later SCH-consistent status among baseline-negative women "
        "(TSH>4.0 mIU/L and FT4 12-22 pmol/L at occ2 or occ3). Single-center retrospective data. "
        "Not a diagnosis. ROC-AUC~0.66 on held-out test set."
    )
    if _meta:
        notes += f" Prevalence in study pop ~ {_meta.get('target', {}).get('prevalence_pct', '?')}%."

    return {
        "condition": "Thyroid",
        "prediction": label,
        "probability": probability,
        "class_probabilities": class_probs,
        "important_factors": important,
        "notes": notes,
        "status": "ok",
    }


if __name__ == "__main__":
    sample = {
        "age": 30, "bmi": 22.5, "tsh_baseline": 2.8, "ft3_baseline": 4.6,
        "ft4_baseline": 14.5, "tpo_baseline": 12.0, "household_income": 2,
        "parity": 1, "family_history_diabetes": 0, "smoking_exposure": 0,
        "alcohol_consumption": 0, "folic_acid_supplementation": 2, "vd_supplementation": 1,
    }
    print(json.dumps(predict_thyroid(sample), indent=2))

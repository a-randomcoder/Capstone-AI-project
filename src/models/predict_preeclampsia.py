"""
Preeclampsia module - prediction interface.

Uses the existing XGBoost model + preprocessing artifacts.
Target is a RiskLevel-derived PROXY (high risk), NOT confirmed clinical preeclampsia.

Research prototype - not a clinical diagnostic system.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = ROOT / "models" / "preeclampsia"

_model = None
_prep = None


def _resolve(name: str) -> Path:
    candidates = [
        MODULE_DIR / name,
        ROOT / name,  # existing flat repo layout
    ]
    for c in candidates:
        if c.exists() and c.stat().st_size > 0:
            return c
    raise FileNotFoundError(f"Could not find {name} in {candidates}")


def _load():
    global _model, _prep
    if _model is not None:
        return
    _model = joblib.load(_resolve("preeclampsia_model.pkl"))
    _prep = joblib.load(_resolve("preeclampsia_preprocessing.pkl"))


def _get(patient_data: dict, *keys, default=None):
    for k in keys:
        if k in patient_data and patient_data[k] is not None:
            return patient_data[k]
    return default


def predict_preeclampsia(patient_data: dict, top_k_factors: int = 5) -> dict:
    """
    Estimate hypertensive / preeclampsia-relevant risk proxy.

    Required inputs (aliases accepted):
      age, systolic_bp (SystolicBP), diastolic_bp (DiastolicBP),
      blood_sugar (BS), body_temp (BodyTemp), pulse/heart_rate (HeartRate)
    """
    _load()
    age = _get(patient_data, "age", "Age")
    sbp = _get(patient_data, "systolic_bp", "SystolicBP", "SBP")
    dbp = _get(patient_data, "diastolic_bp", "DiastolicBP", "DBP")
    bs = _get(patient_data, "blood_sugar", "BS", "blood_glucose")
    temp = _get(patient_data, "body_temp", "BodyTemp")
    hr = _get(patient_data, "pulse", "heart_rate", "HeartRate")

    missing = [
        name
        for name, val in [
            ("age", age),
            ("systolic_bp", sbp),
            ("diastolic_bp", dbp),
            ("blood_sugar", bs),
            ("body_temp", temp),
            ("pulse/heart_rate", hr),
        ]
        if val is None
    ]
    if missing:
        return {
            "condition": "Preeclampsia",
            "prediction": "Insufficient input",
            "probability": None,
            "class_probabilities": {},
            "important_factors": [],
            "notes": f"Missing required fields: {missing}. Target is a RiskLevel proxy, not clinical preeclampsia.",
            "status": "missing_features",
        }

    sbp = float(sbp)
    dbp = float(dbp)
    mean_arterial_pressure = dbp + (sbp - dbp) / 3.0
    pulse_pressure = sbp - dbp
    bp_risk_flag = 1 if (sbp >= 140 or dbp >= 90) else 0

    feature_cols = _prep["feature_cols"]
    row = {
        "Age": float(age),
        "SystolicBP": sbp,
        "DiastolicBP": dbp,
        "BS": float(bs),
        "BodyTemp": float(temp),
        "HeartRate": float(hr),
        "mean_arterial_pressure": mean_arterial_pressure,
        "pulse_pressure": pulse_pressure,
        "bp_risk_flag": bp_risk_flag,
    }
    X = pd.DataFrame([row])[feature_cols]

    # Model was trained on unscaled features for XGBoost (scaler saved but XGB used raw X)
    # Per training script: xgb.fit(X_train, y_train) without scaler; scaler was for LR only.
    # The preprocessing pickle stores scaler + feature_cols; XGB does not need scaling.
    pred = int(_model.predict(X)[0])
    proba = _model.predict_proba(X)[0]
    # Convention from training: target 1 = high-risk proxy
    label = "high risk (proxy)" if pred == 1 else "low/mid risk (proxy)"
    class_probs = {
        "low/mid risk (proxy)": round(float(proba[0]), 4),
        "high risk (proxy)": round(float(proba[1]), 4),
    }
    probability = round(float(proba[pred]), 4)

    important = []
    try:
        import shap

        explainer = shap.TreeExplainer(_model)
        sv = explainer.shap_values(X)
        if isinstance(sv, list):
            sample = sv[1][0] if pred == 1 else sv[0][0]
        else:
            sample = sv[0]
        contrib = pd.Series(sample, index=feature_cols).sort_values(key=abs, ascending=False)
        for feat, val in contrib.head(top_k_factors).items():
            important.append(
                {
                    "feature": str(feat),
                    "shap_contribution": round(float(val), 4),
                    "source": "shap",
                }
            )
    except Exception:
        # Fallback: model feature importances
        if hasattr(_model, "feature_importances_"):
            imp = pd.Series(_model.feature_importances_, index=feature_cols).sort_values(
                ascending=False
            )
            for feat, val in imp.head(top_k_factors).items():
                important.append(
                    {
                        "feature": str(feat),
                        "shap_contribution": round(float(val), 4),
                        "source": "feature_importance",
                    }
                )

    return {
        "condition": "Preeclampsia",
        "prediction": label,
        "probability": probability,
        "class_probabilities": class_probs,
        "important_factors": important,
        "notes": (
            "Target is a RiskLevel-derived PROXY from the UCI Maternal Health Risk dataset, "
            "NOT confirmed clinical preeclampsia. Single-snapshot features only; no longitudinal BP trend. "
            "Research prototype - not a diagnosis."
        ),
        "status": "ok",
    }


if __name__ == "__main__":
    import json
    from src.integration.schemas import sample_patient

    print(json.dumps(predict_preeclampsia(sample_patient()), indent=2))

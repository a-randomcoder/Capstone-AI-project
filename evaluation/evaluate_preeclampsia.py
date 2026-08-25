#!/usr/bin/env python3
"""
Evaluate the frozen Preeclampsia XGBoost model on external data.

Target is a RiskLevel-derived PROXY (high risk vs low/mid), NOT confirmed
clinical preeclampsia. Does NOT retrain.

Default target column: RiskLevel (values containing 'high' -> positive class).
Or supply a binary 0/1 column via --target.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation._common import (  # noqa: E402
    compute_classification_metrics,
    load_table,
    print_metrics,
    require_columns,
    save_evaluation_bundle,
)

OFFICIAL = (
    "This model uses a RiskLevel-derived PROXY label from the UCI Maternal Health Risk dataset, "
    "not confirmed clinical preeclampsia. Official training metrics live in preeclampsia_results.md "
    "in the repo root. This script reports metrics on YOUR uploaded file only."
)

BASE_FEATURES = ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]


def _resolve_artifacts():
    candidates_model = [
        ROOT / "models" / "preeclampsia" / "preeclampsia_model.pkl",
        ROOT / "preeclampsia_model.pkl",
    ]
    candidates_prep = [
        ROOT / "models" / "preeclampsia" / "preeclampsia_preprocessing.pkl",
        ROOT / "preeclampsia_preprocessing.pkl",
    ]
    model_path = next(p for p in candidates_model if p.exists())
    prep_path = next(p for p in candidates_prep if p.exists())
    return joblib.load(model_path), joblib.load(prep_path)


def map_target(series: pd.Series) -> np.ndarray:
    """Map RiskLevel-like strings or 0/1 to binary proxy (1 = high risk)."""
    out = []
    for v in series:
        if pd.isna(v):
            out.append(np.nan)
            continue
        if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
            out.append(int(v))
            continue
        s = str(v).strip().lower()
        if s in {"1", "true", "yes", "high", "high risk", "highrisk"}:
            out.append(1)
        elif "high" in s:
            out.append(1)
        else:
            out.append(0)
    return np.array(out, dtype=float)


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    sbp = df["SystolicBP"].astype(float)
    dbp = df["DiastolicBP"].astype(float)
    out = pd.DataFrame(
        {
            "Age": df["Age"].astype(float),
            "SystolicBP": sbp,
            "DiastolicBP": dbp,
            "BS": df["BS"].astype(float),
            "BodyTemp": df["BodyTemp"].astype(float),
            "HeartRate": df["HeartRate"].astype(float),
            "mean_arterial_pressure": dbp + (sbp - dbp) / 3.0,
            "pulse_pressure": sbp - dbp,
            "bp_risk_flag": ((sbp >= 140) | (dbp >= 90)).astype(int),
        }
    )
    return out


def run(data_path: Path, target_col: str = "RiskLevel") -> Path:
    model, prep = _resolve_artifacts()
    feature_cols = list(prep["feature_cols"])

    df = load_table(data_path)
    require_columns(df, BASE_FEATURES + [target_col], "Preeclampsia evaluation")

    work = df[BASE_FEATURES + [target_col]].copy().dropna()
    y = map_target(work[target_col])
    mask = ~np.isnan(y)
    work = work.loc[mask].reset_index(drop=True)
    y_true = y[mask].astype(int)

    X = engineer(work)[feature_cols]
    # XGBoost was fit on unscaled features (scaler stored but unused at predict time)
    y_pred = model.predict(X).astype(int)
    y_proba = model.predict_proba(X)

    labels = [0, 1]
    label_names = ["low/mid risk (proxy)", "high risk (proxy)"]
    metrics = compute_classification_metrics(y_true, y_pred, y_proba=y_proba, labels=labels)

    pred_table = work.copy()
    pred_table["actual"] = [label_names[i] for i in y_true]
    pred_table["predicted"] = [label_names[i] for i in y_pred]
    pred_table["actual_code"] = y_true
    pred_table["predicted_code"] = y_pred

    out = save_evaluation_bundle(
        condition="preeclampsia",
        metrics=metrics,
        y_true=np.array([label_names[i] for i in y_true]),
        y_pred=np.array([label_names[i] for i in y_pred]),
        labels=label_names,
        pred_table=pred_table,
        source_note=f"External test file: {data_path} (proxy label from '{target_col}')",
        official_note=OFFICIAL,
        y_proba=y_proba,
        proba_columns=["proba_low_mid_proxy", "proba_high_proxy"],
    )
    print_metrics(metrics)
    print(f"\nResults saved under: {out}")
    return out


def main():
    p = argparse.ArgumentParser(description="Evaluate frozen Preeclampsia proxy model")
    p.add_argument("--data", required=True, help="Path to CSV/Excel test file")
    p.add_argument("--target", default="RiskLevel", help="Target column (RiskLevel or 0/1)")
    args = p.parse_args()
    run(Path(args.data), target_col=args.target)


if __name__ == "__main__":
    main()

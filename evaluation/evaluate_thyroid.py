#!/usr/bin/env python3
"""
Evaluate the frozen Thyroid Random Forest pipeline on external data.

Target: later SCH-consistent dysfunction among baseline-negative women.
Does NOT retrain. Single-center retrospective methodology preserved.
"""

from __future__ import annotations

import argparse
import json
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

MODULE_DIR = ROOT / "models" / "thyroid"
DEFAULT_TARGET = "thyroid_dysfunction_later"

BASE_FEATURES = [
    "age",
    "bmi",
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

OFFICIAL = (
    "Official held-out metrics from regenerator/notebook (RANDOM_STATE=42): "
    "ROC-AUC≈0.6588, PR-AUC≈0.2033, F1≈0.2759. "
    "Target = later SCH-consistent status among baseline-negative women. "
    "This script reports metrics on YOUR uploaded file only."
)


def load_model():
    candidates = [
        MODULE_DIR / "thyroid_final_model.pkl",
        ROOT / "thyroid_final_model.pkl",
    ]
    path = next(p for p in candidates if p.exists() and p.stat().st_size > 0)
    model = joblib.load(path)
    meta_path = MODULE_DIR / "thyroid_feature_metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return model, meta


def map_target(series: pd.Series) -> np.ndarray:
    out = []
    for v in series:
        if pd.isna(v):
            out.append(np.nan)
            continue
        if isinstance(v, (int, float, np.integer, np.floating)):
            out.append(int(v))
            continue
        s = str(v).strip().lower()
        if s in {"1", "yes", "true", "positive", "later dysfunction"}:
            out.append(1)
        else:
            out.append(0)
    return np.array(out, dtype=float)


def run(data_path: Path, target_col: str = DEFAULT_TARGET) -> Path:
    model, meta = load_model()
    df = load_table(data_path)
    require_columns(df, BASE_FEATURES + [target_col], "Thyroid evaluation")

    work = df[BASE_FEATURES + [target_col]].copy()
    y = map_target(work[target_col])
    mask = ~np.isnan(y)
    work = work.loc[mask].reset_index(drop=True)
    y_true = y[mask].astype(int)

    X = work[BASE_FEATURES].copy()
    X["log_tsh_baseline"] = np.log1p(X["tsh_baseline"].astype(float))
    feature_order = list(meta.get("all_features_in_order") or (BASE_FEATURES + ["log_tsh_baseline"]))
    X = X[feature_order]

    y_pred = model.predict(X).astype(int)
    y_proba = model.predict_proba(X)

    labels = [0, 1]
    label_names = ["no later dysfunction", "later dysfunction"]
    metrics = compute_classification_metrics(y_true, y_pred, y_proba=y_proba, labels=labels)

    pred_table = work.copy()
    pred_table["actual"] = [label_names[i] for i in y_true]
    pred_table["predicted"] = [label_names[i] for i in y_pred]
    pred_table["actual_code"] = y_true
    pred_table["predicted_code"] = y_pred

    out = save_evaluation_bundle(
        condition="thyroid",
        metrics=metrics,
        y_true=np.array([label_names[i] for i in y_true]),
        y_pred=np.array([label_names[i] for i in y_pred]),
        labels=label_names,
        pred_table=pred_table,
        source_note=f"External test file: {data_path}",
        official_note=OFFICIAL,
        y_proba=y_proba,
        proba_columns=["proba_no_later", "proba_later"],
    )
    print_metrics(metrics)
    print(f"\nResults saved under: {out}")
    return out


def main():
    p = argparse.ArgumentParser(description="Evaluate frozen Thyroid model")
    p.add_argument("--data", required=True, help="Path to CSV/Excel test file")
    p.add_argument("--target", default=DEFAULT_TARGET, help="Target column name")
    args = p.parse_args()
    run(Path(args.data), target_col=args.target)


if __name__ == "__main__":
    main()

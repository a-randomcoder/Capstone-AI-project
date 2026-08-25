#!/usr/bin/env python3
"""
Evaluate the frozen GDM Random Forest pipeline on external data.

Model was trained on SYNTHETIC data (~65% GDM prevalence). Not clinically validated.
Does NOT retrain.
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
DEFAULT_TARGET = "gdm_outcome"

OFFICIAL = (
    "GDM model was trained on SYNTHETIC data (n≈10k, ~65% prevalence). "
    "It is not clinically validated. Official notebook metrics are in Capstone-AI-Project/Notebook/. "
    "This script reports metrics on YOUR uploaded file only."
)


def load_model():
    candidates = [
        ROOT / "models" / "gdm" / "gdm_best_model.joblib",
        ROOT / "Capstone-AI-Project" / "Notebook" / "models" / "gdm_best_model.joblib",
    ]
    path = next(p for p in candidates if p.exists() and p.stat().st_size > 0)
    return joblib.load(path)


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
        if s in {"1", "yes", "true", "gdm", "positive"}:
            out.append(1)
        else:
            out.append(0)
    return np.array(out, dtype=float)


def run(data_path: Path, target_col: str = DEFAULT_TARGET) -> Path:
    model = load_model()
    df = load_table(data_path)
    require_columns(df, GDM_FEATURES + [target_col], "GDM evaluation")

    work = df[GDM_FEATURES + [target_col]].copy()
    y = map_target(work[target_col])
    # allow missing feature values (pipeline imputes); require target
    mask = ~np.isnan(y)
    work = work.loc[mask].reset_index(drop=True)
    y_true = y[mask].astype(int)
    X = work[GDM_FEATURES]

    y_pred = model.predict(X).astype(int)
    y_proba = model.predict_proba(X)

    labels = [0, 1]
    label_names = ["No GDM", "GDM"]
    metrics = compute_classification_metrics(y_true, y_pred, y_proba=y_proba, labels=labels)

    pred_table = work.copy()
    pred_table["actual"] = [label_names[i] for i in y_true]
    pred_table["predicted"] = [label_names[i] for i in y_pred]
    pred_table["actual_code"] = y_true
    pred_table["predicted_code"] = y_pred

    out = save_evaluation_bundle(
        condition="gdm",
        metrics=metrics,
        y_true=np.array([label_names[i] for i in y_true]),
        y_pred=np.array([label_names[i] for i in y_pred]),
        labels=label_names,
        pred_table=pred_table,
        source_note=f"External test file: {data_path}",
        official_note=OFFICIAL,
        y_proba=y_proba,
        proba_columns=["proba_no_gdm", "proba_gdm"],
    )
    print_metrics(metrics)
    print(f"\nResults saved under: {out}")
    return out


def main():
    p = argparse.ArgumentParser(description="Evaluate frozen GDM synthetic model")
    p.add_argument("--data", required=True, help="Path to CSV/Excel test file")
    p.add_argument("--target", default=DEFAULT_TARGET, help="Target column name")
    args = p.parse_args()
    run(Path(args.data), target_col=args.target)


if __name__ == "__main__":
    main()

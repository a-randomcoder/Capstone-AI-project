#!/usr/bin/env python3
"""
Evaluate the frozen Anemia Track B Logistic Regression model on external data.

Does NOT retrain. Uses models/anemia/*.joblib + anemia_model_metadata.json.

Target column (default): "Severity of anemia (on the basis of Hb)"
  values: Normal | Mild | Moderate | Severe

Hb / PCV / who_pred are intentionally excluded (leakage).
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

MODULE_DIR = ROOT / "models" / "anemia"
DEFAULT_TARGET = "Severity of anemia (on the basis of Hb)"
YES_NO = {"Yes": 1, "No": 0, "yes": 1, "no": 0, 1: 1, 0: 0, True: 1, False: 0}
DIET = {
    "Non-Vegetarian": 1,
    "Vegetarian": 0,
    "non-vegetarian": 1,
    "vegetarian": 0,
    1: 1,
    0: 0,
}
BINARY_COLS = {
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

OFFICIAL = (
    "Official held-out metrics (from training notebook / metadata, RANDOM_STATE=42): "
    "accuracy≈0.785, balanced_accuracy≈0.696, f1_macro≈0.675, f1_weighted≈0.788. "
    "Severe class is rare (~9 samples). This script reports metrics on YOUR uploaded file only."
)


def load_artifacts():
    model = joblib.load(MODULE_DIR / "anemia_best_model.joblib")
    preprocessor = joblib.load(MODULE_DIR / "anemia_preprocessor.joblib")
    label_encoder = joblib.load(MODULE_DIR / "anemia_label_encoder.joblib")
    meta = json.loads((MODULE_DIR / "anemia_model_metadata.json").read_text())
    return model, preprocessor, label_encoder, meta


def encode_row(col: str, val):
    if col == "Dietary habits":
        if isinstance(val, str) and val in DIET:
            return DIET[val]
        return val
    if col in BINARY_COLS:
        if val in YES_NO:
            return YES_NO[val]
        return val
    return val


def prepare_features(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        rows.append({c: encode_row(c, r[c]) for c in feature_cols})
    return pd.DataFrame(rows, columns=feature_cols)


def run(data_path: Path, target_col: str = DEFAULT_TARGET) -> Path:
    model, preprocessor, label_encoder, meta = load_artifacts()
    feature_cols = meta["raw_feature_list_track_B"]
    classes = list(meta.get("class_order") or list(label_encoder.classes_))

    df = load_table(data_path)
    require_columns(df, feature_cols + [target_col], "Anemia evaluation")

    # Drop rows with missing target or features
    work = df[feature_cols + [target_col]].copy().dropna()
    if work.empty:
        raise ValueError("No complete rows after dropping NA in features/target.")

    X_raw = prepare_features(work, feature_cols)
    X = preprocessor.transform(X_raw)
    y_true_labels = work[target_col].astype(str).values
    # map unknown labels
    known = set(str(c) for c in label_encoder.classes_)
    mask = np.array([lab in known for lab in y_true_labels])
    if not mask.all():
        dropped = int((~mask).sum())
        print(f"Warning: dropping {dropped} rows with target labels outside model classes {sorted(known)}")
        X = X[mask]
        y_true_labels = y_true_labels[mask]
        work = work.loc[mask].reset_index(drop=True)

    y_true = label_encoder.transform(y_true_labels)
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    pred_labels = label_encoder.inverse_transform(y_pred)

    metrics = compute_classification_metrics(
        y_true, y_pred, y_proba=y_proba, labels=list(range(len(label_encoder.classes_)))
    )
    # also human-readable label metrics via string
    metrics_str = compute_classification_metrics(
        y_true_labels, pred_labels, y_proba=None, labels=list(label_encoder.classes_)
    )
    # keep numeric roc_auc from encoded version
    metrics_str["roc_auc"] = metrics.get("roc_auc")

    pred_table = work.reset_index(drop=True).copy()
    pred_table["actual"] = y_true_labels
    pred_table["predicted"] = pred_labels

    proba_cols = [f"proba_{c}" for c in label_encoder.classes_]
    out = save_evaluation_bundle(
        condition="anemia",
        metrics=metrics_str,
        y_true=y_true_labels,
        y_pred=pred_labels,
        labels=list(label_encoder.classes_),
        pred_table=pred_table,
        source_note=f"External test file: {data_path}",
        official_note=OFFICIAL,
        y_proba=y_proba,
        proba_columns=proba_cols,
    )
    print_metrics(metrics_str)
    print(f"\nResults saved under: {out}")
    return out


def main():
    p = argparse.ArgumentParser(description="Evaluate frozen Anemia Track B model on external data")
    p.add_argument("--data", required=True, help="Path to CSV/Excel test file")
    p.add_argument("--target", default=DEFAULT_TARGET, help="Target column name")
    args = p.parse_args()
    run(Path(args.data), target_col=args.target)


if __name__ == "__main__":
    main()

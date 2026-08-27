#!/usr/bin/env python3
"""
TEMPORARY AUDIT EXPERIMENT — Preeclampsia proxy models, Stratified 5-Fold CV

Compares Logistic Regression vs Random Forest using the SAME dataset, target
definition, feature engineering, and preprocessing choices as the original
preeclampsia training pipeline (run_model.py / preeclampsia_results.md).

This script does NOT:
  - modify production models or artifacts
  - tune hyperparameters
  - change methodology
  - use a separate held-out test set for model selection/tuning

Target is a RiskLevel-derived PROXY (high risk vs low/mid), NOT confirmed
clinical preeclampsia.

Reproducible with random_state=42.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
N_SPLITS = 5

FEATURE_COLS = [
    "Age",
    "SystolicBP",
    "DiastolicBP",
    "BS",
    "BodyTemp",
    "HeartRate",
    "mean_arterial_pressure",
    "pulse_pressure",
    "bp_risk_flag",
]


def resolve_data_path() -> Path:
    """Locate maternal_health_risk.csv relative to common project layouts."""
    here = Path(__file__).resolve()
    root = here.parents[1]
    candidates = [
        root / "data" / "raw" / "maternal_health_risk.csv",
        root / "raw" / "maternal_health_risk.csv",
        root / "maternal_health_risk.csv",
        here.parents[2] / "data" / "raw" / "maternal_health_risk.csv",
    ]
    # also search a few sibling artifact trees
    for p in Path("/home/workdir/artifacts").rglob("maternal_health_risk.csv"):
        candidates.append(p)
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not find maternal_health_risk.csv. "
        "Place it under data/raw/ or pass via working tree."
    )


def load_and_prepare(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Exact methodology from original pipeline:
      - target = (RiskLevel == 'high risk').astype(int)  # PROXY label
      - engineered: MAP, pulse_pressure, bp_risk_flag (SBP>=140 or DBP>=90)
      - feature_cols as in run_model.py
    """
    df = pd.read_csv(path)
    required = ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate", "RiskLevel"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")

    work = df[required].copy().dropna()
    # PROXY target — not confirmed clinical preeclampsia
    y = (work["RiskLevel"].astype(str).str.strip().str.lower() == "high risk").astype(int).values

    sbp = work["SystolicBP"].astype(float)
    dbp = work["DiastolicBP"].astype(float)
    X = pd.DataFrame(
        {
            "Age": work["Age"].astype(float).values,
            "SystolicBP": sbp.values,
            "DiastolicBP": dbp.values,
            "BS": work["BS"].astype(float).values,
            "BodyTemp": work["BodyTemp"].astype(float).values,
            "HeartRate": work["HeartRate"].astype(float).values,
            "mean_arterial_pressure": (dbp + (sbp - dbp) / 3.0).values,
            "pulse_pressure": (sbp - dbp).values,
            "bp_risk_flag": ((sbp >= 140) | (dbp >= 90)).astype(int).values,
        }
    )[FEATURE_COLS]

    return X, y


def fold_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }


def run_cv(X: pd.DataFrame, y: np.ndarray) -> dict[str, list[dict]]:
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    results = {"Logistic Regression": [], "Random Forest": []}

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        # --- Logistic Regression: StandardScaler fit on train fold only ---
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        lr = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
        lr.fit(X_train_s, y_train)
        lr_pred = lr.predict(X_test_s)
        lr_prob = lr.predict_proba(X_test_s)[:, 1]
        results["Logistic Regression"].append(fold_metrics(y_test, lr_pred, lr_prob))

        # --- Random Forest: unscaled features (same as original run_model.py) ---
        rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        rf_prob = rf.predict_proba(X_test)[:, 1]
        results["Random Forest"].append(fold_metrics(y_test, rf_pred, rf_prob))

        print(f"Fold {fold}/{N_SPLITS} complete.")

    return results


def summarize(fold_list: list[dict]) -> dict[str, str]:
    keys = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc"]
    out = {}
    for k in keys:
        vals = np.array([d[k] for d in fold_list], dtype=float)
        out[k] = f"{vals.mean():.4f} ± {vals.std(ddof=1):.4f}"
    return out


def mean_std(fold_list: list[dict], key: str) -> tuple[float, float]:
    vals = np.array([d[key] for d in fold_list], dtype=float)
    return float(vals.mean()), float(vals.std(ddof=1))


def main() -> None:
    data_path = resolve_data_path()
    print("=" * 72)
    print("TEMPORARY AUDIT: Preeclampsia proxy — Stratified 5-Fold CV")
    print("=" * 72)
    print(f"Dataset: {data_path}")
    print("Target: RiskLevel == 'high risk' (PROXY — not clinical preeclampsia)")
    print(f"Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")
    print(f"CV: StratifiedKFold n_splits={N_SPLITS}, shuffle=True, random_state={RANDOM_STATE}")
    print("Models: LogisticRegression (scaled, class_weight=balanced),")
    print("        RandomForest (unscaled, n_estimators=300, max_depth=8, class_weight=balanced)")
    print("No hyperparameter tuning. No held-out test used for selection.")
    print()

    X, y = load_and_prepare(data_path)
    print(f"Rows after dropna: {len(X)}")
    print(f"Class balance: high-risk={int(y.sum())} ({y.mean():.3f}), "
          f"low/mid={int((1 - y).sum())} ({1 - y.mean():.3f})")
    print()

    results = run_cv(X, y)

    print()
    print("=" * 72)
    print("RESULTS (mean ± std across 5 folds)")
    print("=" * 72)

    summaries = {}
    for name, folds in results.items():
        summaries[name] = summarize(folds)
        print(f"\n{name}")
        print("-" * 40)
        for metric, val in summaries[name].items():
            print(f"  {metric:20s}  {val}")

    # Rank by mean F1 then mean ROC-AUC; also consider stability (lower std better)
    ranking = []
    for name, folds in results.items():
        f1_m, f1_s = mean_std(folds, "f1")
        auc_m, auc_s = mean_std(folds, "roc_auc")
        bal_m, bal_s = mean_std(folds, "balanced_accuracy")
        ranking.append((name, f1_m, f1_s, auc_m, auc_s, bal_m, bal_s))

    ranking.sort(key=lambda t: (t[1], t[3]), reverse=True)
    best = ranking[0]
    other = ranking[1]

    print()
    print("=" * 72)
    print("CONCLUSION")
    print("=" * 72)
    print(
        f"Strongest overall (by mean F1, then ROC-AUC): {best[0]} "
        f"(F1 {best[1]:.4f} ± {best[2]:.4f}, ROC-AUC {best[3]:.4f} ± {best[4]:.4f})."
    )
    # Stability: lower std of F1 and ROC-AUC
    best_stability = best[2] + best[4]
    other_stability = other[2] + other[4]
    if best_stability <= other_stability:
        print(
            f"Most stable across folds (lowest F1+AUC std): {best[0]} "
            f"(F1 std {best[2]:.4f}, AUC std {best[4]:.4f})."
        )
        print(f"\n=> {best[0]} has the strongest and most stable performance across folds.")
    else:
        print(
            f"Most stable across folds (lowest F1+AUC std): {other[0]} "
            f"(F1 std {other[2]:.4f}, AUC std {other[4]:.4f})."
        )
        if best[0] == other[0]:
            print(f"\n=> {best[0]} has the strongest and most stable performance across folds.")
        else:
            print(
                f"\n=> {best[0]} is strongest on mean metrics; "
                f"{other[0]} is more stable (lower fold-to-fold variance)."
            )
            # Still pick one primary statement based on combined view
            # Prefer higher mean F1 if close stability; else note both
            print(
                f"For this audit, primary recommendation by mean performance: {best[0]}."
            )

    print()
    print("Reminder: target is a RiskLevel proxy, not confirmed clinical preeclampsia.")
    print("This is a temporary audit experiment only — no production artifacts were modified.")


if __name__ == "__main__":
    main()

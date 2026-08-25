"""Shared helpers for offline evaluation. Does not retrain models."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = EVAL_ROOT / "results"


def load_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Test data not found: {path}")
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        sep = "\t" if suffix == ".tsv" else ","
        return pd.read_csv(path, sep=sep)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type: {suffix}. Use CSV or Excel.")


def require_columns(df: pd.DataFrame, required: Sequence[str], context: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{context}: missing required columns ({len(missing)}):\n"
            + "\n".join(f"  - {c}" for c in missing)
        )


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    labels: Optional[Sequence[Any]] = None,
    average_for_binary_auc: str = "binary",
) -> Dict[str, Any]:
    """Compute standard classification metrics. ROC-AUC when probabilities allow."""
    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "precision_weighted": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
    }

    roc_auc = None
    if y_proba is not None:
        try:
            n_classes = y_proba.shape[1] if y_proba.ndim == 2 else 1
            if n_classes == 2:
                pos = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
                roc_auc = float(roc_auc_score(y_true, pos))
            elif n_classes > 2:
                roc_auc = float(
                    roc_auc_score(
                        y_true,
                        y_proba,
                        multi_class="ovr",
                        average="macro",
                        labels=labels,
                    )
                )
        except Exception as exc:  # noqa: BLE001
            roc_auc = None
            metrics["roc_auc_error"] = str(exc)
    if roc_auc is not None and isinstance(roc_auc, float) and (roc_auc != roc_auc):
        roc_auc = None
    metrics["roc_auc"] = roc_auc
    return metrics


def per_class_table(
    y_true: np.ndarray, y_pred: np.ndarray, labels: Optional[Sequence[Any]] = None
) -> pd.DataFrame:
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    rows = []
    for key, vals in report.items():
        if key in ("accuracy",):
            continue
        if not isinstance(vals, dict):
            continue
        rows.append(
            {
                "label": key,
                "precision": vals.get("precision"),
                "recall": vals.get("recall"),
                "f1-score": vals.get("f1-score"),
                "support": vals.get("support"),
            }
        )
    return pd.DataFrame(rows)


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


def save_evaluation_bundle(
    condition: str,
    metrics: Dict[str, Any],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Sequence[Any],
    pred_table: pd.DataFrame,
    source_note: str,
    official_note: str,
    y_proba: Optional[np.ndarray] = None,
    proba_columns: Optional[List[str]] = None,
) -> Path:
    out = RESULTS_ROOT / condition
    out.mkdir(parents=True, exist_ok=True)

    payload = {
        "condition": condition,
        "evaluation_type": "external_test_data",
        "source_note": source_note,
        "official_held_out_note": official_note,
        "n_rows": int(len(y_true)),
        "metrics": metrics,
        "labels": [str(x) for x in labels],
    }
    (out / "metrics.json").write_text(json.dumps(_json_safe(payload), indent=2))

    cm = confusion_matrix(y_true, y_pred, labels=list(labels))
    cm_df = pd.DataFrame(
        cm,
        index=[f"actual_{x}" for x in labels],
        columns=[f"pred_{x}" for x in labels],
    )
    cm_df.to_csv(out / "confusion_matrix.csv")

    report_txt = classification_report(
        y_true, y_pred, labels=list(labels), zero_division=0
    )
    (out / "classification_report.txt").write_text(report_txt)

    per_class_table(y_true, y_pred, labels=list(labels)).to_csv(
        out / "per_class_metrics.csv", index=False
    )

    table = pred_table.copy()
    if y_proba is not None and proba_columns:
        for i, col in enumerate(proba_columns):
            table[col] = y_proba[:, i]
    table.to_csv(out / "predictions.csv", index=False)

    wrong = table[table["actual"].astype(str) != table["predicted"].astype(str)]
    wrong.to_csv(out / "incorrect_predictions.csv", index=False)

    lines = [
        f"# {condition} — external test evaluation",
        "",
        f"- Rows evaluated: **{len(y_true)}**",
        f"- Incorrect: **{len(wrong)}**",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "|--------|------:|",
    ]
    for k, v in metrics.items():
        if isinstance(v, float) and v == v:
            lines.append(f"| {k} | {v:.4f} |")
        else:
            lines.append(f"| {k} | {v} |")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            source_note,
            "",
            official_note,
            "",
        ]
    )
    (out / "summary.md").write_text("\n".join(lines))
    return out


def print_metrics(metrics: Dict[str, Any]) -> None:
    print("\nMetrics")
    print("-" * 40)
    for k, v in metrics.items():
        if isinstance(v, float) and v == v:
            print(f"  {k:24s} {v:.4f}")
        else:
            print(f"  {k:24s} {v}")

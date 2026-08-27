#!/usr/bin/env python3
"""
TEMPORARY AUDIT: Thyroid model — Stratified 5-Fold CV.

This experiment does NOT modify production models or artifacts.
It reproduces the existing thyroid preprocessing/model pipeline and
evaluates it using stratified 5-fold cross-validation.
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)


RANDOM_STATE = 42

ROOT = Path(__file__).resolve().parents[1]

src = next(
    p for p in [
        ROOT / "origin_data.xlsx",
        ROOT / "data" / "raw" / "origin_data.xlsx",
    ]
    if p.exists()
)

df = pd.read_excel(src)


def sch(tsh, ft4):
    return (tsh > 4.0) & (ft4 >= 12) & (ft4 <= 22)


# Same target construction as reproduce_thyroid_heldout.py
s1 = sch(df["TSH"], df["FT4"])
s2 = sch(df["TSH2"], df["FT42"])
s3 = sch(df["TSH3"], df["FT43"])

base = ~s1

target = ((s2 | s3) & base).astype(int)

m = df.loc[base].copy()
m["target"] = target.loc[base].values


def parse(v):
    if pd.isna(v):
        return np.nan

    if isinstance(v, (int, float)):
        return float(v)

    s = str(v).strip().replace("＞", ">").replace("＜", "<")

    if s.lower() in {"未测", "？", "?", "na", "n/a", ""}:
        return np.nan

    z = re.search(r"[-+]?[0-9]*\.?[0-9]+", s)

    return float(z.group()) if z else np.nan


m["TPO_clean"] = m["TPO"].apply(parse)


RAW = {
    "Age": "age",
    "BMI": "bmi",
    "TSH": "tsh_baseline",
    "FT3": "ft3_baseline",
    "FT4": "ft4_baseline",
    "TPO_clean": "tpo_baseline",
    "Annual household income, 1= less than 50,000, 2= 50,000 to 100,000 = more than 100,000": "household_income",
    "parity": "parity",
    "Family history of diabetes": "family_history_diabetes",
    "Smoking or secondhand smoke exposure for more than 15 minutes per week": "smoking_exposure",
    "Alcohol consumption": "alcohol_consumption",
    "Folic acid supplementation": "folic_acid_supplementation",
    "VD supplementation is not 0 before pregnancy 1 early pregnancy 2": "vd_supplementation",
}


m = m.rename(columns=RAW)

features = list(RAW.values()) + ["log_tsh_baseline"]

m["log_tsh_baseline"] = np.log1p(m["tsh_baseline"])

X = m[features]
y = m["target"]


num = [
    "age",
    "bmi",
    "tsh_baseline",
    "ft3_baseline",
    "ft4_baseline",
    "tpo_baseline",
    "log_tsh_baseline",
]

cat = [
    "household_income",
    "parity",
    "family_history_diabetes",
    "smoking_exposure",
    "alcohol_consumption",
    "folic_acid_supplementation",
    "vd_supplementation",
]


def make_pipeline():

    pre = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num,
            ),
            (
                "cat",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(strategy="most_frequent"),
                        ),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore"),
                        ),
                    ]
                ),
                cat,
            ),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=8,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return Pipeline(
        [
            ("preprocessor", pre),
            ("classifier", model),
        ]
    )


print("=" * 72)
print("TEMPORARY AUDIT: Thyroid — Stratified 5-Fold CV")
print("=" * 72)

print(f"Dataset: {src}")
print("Target: later thyroid dysfunction risk")
print("Target definition: ((TSH2/FT42 OR TSH3/FT43) AND NOT baseline SCH)")
print(f"Rows after baseline exclusion: {len(m)}")
print(f"Positive prevalence: {y.mean():.4f}")
print()
print("Model: Random Forest")
print("Preprocessing: median imputation + StandardScaler + OneHotEncoder")
print("Parameters:")
print("  n_estimators=400")
print("  max_depth=8")
print("  min_samples_leaf=10")
print("  class_weight=balanced")
print("  random_state=42")
print()
print("CV: StratifiedKFold(n_splits=5, shuffle=True, random_state=42)")
print("No hyperparameter tuning.")
print("No production artifacts modified.")
print()


cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE,
)


metrics = {
    "accuracy": [],
    "balanced_accuracy": [],
    "precision": [],
    "recall": [],
    "f1": [],
    "roc_auc": [],
    "pr_auc": [],
}


for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    pipe = make_pipeline()

    pipe.fit(X_train, y_train)

    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]

    values = {
        "accuracy": accuracy_score(y_test, pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, pred),
        "precision": precision_score(
            y_test,
            pred,
            zero_division=0,
        ),
        "recall": recall_score(
            y_test,
            pred,
            zero_division=0,
        ),
        "f1": f1_score(
            y_test,
            pred,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(y_test, proba),
        "pr_auc": average_precision_score(y_test, proba),
    }

    for key, value in values.items():
        metrics[key].append(value)

    print(
        f"Fold {fold}/5 complete "
        f"(F1={values['f1']:.4f}, "
        f"ROC-AUC={values['roc_auc']:.4f})"
    )


print()
print("=" * 72)
print("RESULTS (mean ± std across 5 folds)")
print("=" * 72)

for key, values in metrics.items():

    values = np.array(values)

    print(
        f"{key:<20} "
        f"{values.mean():.4f} ± {values.std():.4f}"
    )


f1_mean = np.mean(metrics["f1"])
auc_mean = np.mean(metrics["roc_auc"])

f1_std = np.std(metrics["f1"])
auc_std = np.std(metrics["roc_auc"])


print()
print("=" * 72)
print("CONCLUSION")
print("=" * 72)

print(
    f"Mean F1:      {f1_mean:.4f} ± {f1_std:.4f}"
)

print(
    f"Mean ROC-AUC: {auc_mean:.4f} ± {auc_std:.4f}"
)

print()

print(
    "This is a temporary audit experiment only. "
    "No production model or artifact was modified."
)

print(
    "Reminder: this predicts later thyroid dysfunction risk "
    "from baseline/first-trimester information; it is not a diagnosis."
)

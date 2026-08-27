#!/usr/bin/env python3
"""Reproduce the preeclampsia training comparison without changing production artifacts.

This is an audit/reproduction script. It trains LR, RF and XGBoost on the same
UCI-style maternal-risk data and reports held-out metrics. It does NOT overwrite
models/ artifacts or evaluation results.
"""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = [ROOT / "data" / "raw" / "maternal_health_risk.csv", ROOT / "maternal_health_risk.csv"]
src = next((p for p in CANDIDATES if p.exists()), None)
if src is None:
    raise FileNotFoundError("Could not find maternal_health_risk.csv in data/raw or repository root.")

df = pd.read_csv(src)
df["target"] = (df["RiskLevel"] == "high risk").astype(int)
df["mean_arterial_pressure"] = df["DiastolicBP"] + (df["SystolicBP"] - df["DiastolicBP"]) / 3
df["pulse_pressure"] = df["SystolicBP"] - df["DiastolicBP"]
df["bp_risk_flag"] = ((df["SystolicBP"] >= 140) | (df["DiastolicBP"] >= 90)).astype(int)

features = ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate", "mean_arterial_pressure", "pulse_pressure", "bp_risk_flag"]
X, y = df[features], df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Logistic Regression": (LogisticRegression(max_iter=1000, class_weight="balanced"), X_train_scaled, X_test_scaled),
    "Random Forest": (RandomForestClassifier(n_estimators=300, max_depth=8, class_weight="balanced", random_state=42), X_train, X_test),
    "XGBoost": (XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, eval_metric="logloss", random_state=42), X_train, X_test),
}

print(f"Dataset: {src}")
print(f"Shape: {df.shape}")
print(f"Positive proxy prevalence: {y.mean():.3f}")
print(f"Train/test: {len(X_train)}/{len(X_test)}")
print()
print("Model comparison on the same stratified 20% held-out split")
print("=" * 90)
print(f"{'Model':<22} {'Accuracy':>10} {'BalAcc':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'ROC-AUC':>10}")

for name, (model, xtr, xte) in models.items():
    model.fit(xtr, y_train)
    pred = model.predict(xte)
    proba = model.predict_proba(xte)[:, 1]
    vals = [
        accuracy_score(y_test, pred),
        balanced_accuracy_score(y_test, pred),
        precision_score(y_test, pred, zero_division=0),
        recall_score(y_test, pred, zero_division=0),
        f1_score(y_test, pred, zero_division=0),
        roc_auc_score(y_test, proba),
    ]
    print(f"{name:<22} " + " ".join(f"{v:10.4f}" for v in vals))

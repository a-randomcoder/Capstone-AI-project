#!/usr/bin/env python3
"""Reproduce the Anemia Track B held-out experiment without saving artifacts."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parents[1]
CSV = next(p for p in [ROOT / "data/raw/CBC_Datasets.csv", ROOT / "CBC_Datasets.csv"] if p.exists())
TARGET = "Severity of anemia (on the basis of Hb)"
CLASS_ORDER = ["Normal", "Mild", "Moderate", "Severe"]
BINARY_COLS = ["LMP known", "USG", "History of blood transfusion during pregnancy", "Family history of hemoglobinopathy", "History of any type of allergy", "History of iron supplementation", "Icterus", "Pallor", "Edema"]
LEAK = ["Hb (in gm/dL)", "PCV (%)", "who_pred"]

def who_rule(hb):
    if hb < 7: return "Severe"
    if hb < 10: return "Moderate"
    if hb < 11: return "Mild"
    return "Normal"

df = pd.read_csv(CSV).drop_duplicates().reset_index(drop=True)
df["who_pred"] = df["Hb (in gm/dL)"].apply(who_rule)
for c in BINARY_COLS:
    df[c] = df[c].map({"Yes": 1, "No": 0})
df["Dietary habits"] = df["Dietary habits"].map({"Non-Vegetarian": 1, "Vegetarian": 0})
le = LabelEncoder().fit(CLASS_ORDER)
df["target_enc"] = le.transform(df[TARGET])
features = [c for c in df.columns if c not in [TARGET, "target_enc"] + LEAK]
X, y = df[features], df["target_enc"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
binary_like = set(BINARY_COLS + ["Dietary habits"])
num = [c for c in X.select_dtypes(include=[np.number]).columns if c not in binary_like]
pre = ColumnTransformer([("scale", StandardScaler(), num)], remainder="passthrough")
Xtr, Xte = pre.fit_transform(X_train), pre.transform(X_test)
sw = compute_sample_weight("balanced", y_train)
models = {
    "Logistic Regression": LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=2, class_weight="balanced", random_state=RANDOM_STATE),
    "XGBoost": xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, eval_metric="mlogloss", random_state=RANDOM_STATE),
}
for name, model in models.items():
    if name == "XGBoost": model.fit(Xtr, y_train, sample_weight=sw)
    else: model.fit(Xtr, y_train)
    pred = model.predict(Xte)
    print(name)
    print(f"  accuracy={accuracy_score(y_test,pred):.4f}")
    print(f"  balanced_accuracy={balanced_accuracy_score(y_test,pred):.4f}")
    print(f"  precision_macro={precision_score(y_test,pred,average='macro',zero_division=0):.4f}")
    print(f"  recall_macro={recall_score(y_test,pred,average='macro',zero_division=0):.4f}")
    print(f"  f1_macro={f1_score(y_test,pred,average='macro',zero_division=0):.4f}")
    print(f"  f1_weighted={f1_score(y_test,pred,average='weighted',zero_division=0):.4f}")

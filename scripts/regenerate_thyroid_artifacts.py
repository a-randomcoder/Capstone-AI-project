#!/usr/bin/env python3
"""Regenerate thyroid RF pipeline exactly as thyroid_model_complete.ipynb (RANDOM_STATE=42)."""
import warnings
warnings.filterwarnings("ignore")
import re
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import joblib

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "models" / "thyroid"
OUT.mkdir(parents=True, exist_ok=True)

candidates = [ROOT / "origin_data.xlsx", ROOT / "data" / "raw" / "origin_data.xlsx"]
src = next(p for p in candidates if p.exists())
df = pd.read_excel(src)

def sch_flag(tsh, ft4):
    return (tsh > 4.0) & (ft4 >= 12) & (ft4 <= 22)

sch1 = sch_flag(df["TSH"], df["FT4"])
sch2 = sch_flag(df["TSH2"], df["FT42"])
sch3 = sch_flag(df["TSH3"], df["FT43"])
baseline_neg = ~sch1
target = ((sch2 | sch3) & baseline_neg).astype(int)
model_df = df.loc[baseline_neg].copy()
model_df["thyroid_dysfunction_later"] = target.loc[baseline_neg].values

def parse_lab(v):
    if pd.isna(v): return np.nan
    if isinstance(v, (int, float)): return float(v)
    s = str(v).strip()
    if s.lower() in {"未测", "？", "?", "na", "n/a", ""}: return np.nan
    s = s.replace("＞", ">").replace("＜", "<")
    m = re.search(r"[-+]?[0-9]*\.?[0-9]+", s)
    return float(m.group()) if m else np.nan

model_df["TPO_clean"] = model_df["TPO"].apply(parse_lab)
RAW = {
    "Age": "age", "BMI": "bmi", "TSH": "tsh_baseline", "FT3": "ft3_baseline",
    "FT4": "ft4_baseline", "TPO_clean": "tpo_baseline",
    "Annual household income, 1= less than 50,000, 2= 50,000 to 100,000 = more than 100,000": "household_income",
    "parity": "parity", "Family history of diabetes": "family_history_diabetes",
    "Smoking or secondhand smoke exposure for more than 15 minutes per week": "smoking_exposure",
    "Alcohol consumption": "alcohol_consumption",
    "Folic acid supplementation": "folic_acid_supplementation",
    "VD supplementation is not 0 before pregnancy 1 early pregnancy 2": "vd_supplementation",
}
model_df = model_df.rename(columns=RAW)
FEATURES = list(RAW.values())
model_df["log_tsh_baseline"] = np.log1p(model_df["tsh_baseline"])
FEATURES.append("log_tsh_baseline")
X, y = model_df[FEATURES], model_df["thyroid_dysfunction_later"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
num = ["age", "bmi", "tsh_baseline", "ft3_baseline", "ft4_baseline", "tpo_baseline", "log_tsh_baseline"]
cat = ["household_income", "parity", "family_history_diabetes", "smoking_exposure",
       "alcohol_consumption", "folic_acid_supplementation", "vd_supplementation"]
pre = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat),
])
pipe = Pipeline([
    ("preprocessor", pre),
    ("classifier", RandomForestClassifier(
        n_estimators=400, max_depth=8, min_samples_leaf=10,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)),
])
pipe.fit(X_train, y_train)
proba = pipe.predict_proba(X_test)[:, 1]
pred = pipe.predict(X_test)
print("ROC-AUC", round(roc_auc_score(y_test, proba), 4))
joblib.dump(pipe, OUT / "thyroid_final_model.pkl")
json.dump({
    "target": {"name": "thyroid_dysfunction_later", "prevalence_pct": round(float(y.mean()*100), 2)},
    "all_features_in_order": FEATURES,
    "best_model": "Random Forest",
    "random_state": RANDOM_STATE,
    "test_metrics": {
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "pr_auc": round(float(average_precision_score(y_test, proba)), 4),
        "f1": round(float(f1_score(y_test, pred)), 4),
    },
}, open(OUT / "thyroid_feature_metadata.json", "w"), indent=2)
print("Saved to", OUT)

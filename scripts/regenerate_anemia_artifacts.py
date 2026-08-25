#!/usr/bin/env python3
"""Regenerate anemia Track B artifacts (same as anemia_module.ipynb, RANDOM_STATE=42)."""
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
import xgboost as xgb
import joblib

RANDOM_STATE = 42
CLASS_ORDER = ["Normal", "Mild", "Moderate", "Severe"]
TARGET = "Severity of anemia (on the basis of Hb)"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "models" / "anemia"
OUT.mkdir(parents=True, exist_ok=True)

csv_path = next(p for p in [ROOT / "data" / "raw" / "CBC_Datasets.csv", ROOT / "CBC_Datasets.csv"] if p.exists())
df = pd.read_csv(csv_path).drop_duplicates().reset_index(drop=True)

def who_rule(hb):
    if hb < 7: return "Severe"
    if hb < 10: return "Moderate"
    if hb < 11: return "Mild"
    return "Normal"
df["who_pred"] = df["Hb (in gm/dL)"].apply(who_rule)

binary_cols = ["LMP known", "USG", "History of blood transfusion during pregnancy",
               "Family history of hemoglobinopathy", "History of any type of allergy",
               "History of iron supplementation", "Icterus", "Pallor", "Edema"]
df_model = df.copy()
for col in binary_cols:
    df_model[col] = df_model[col].map({"Yes": 1, "No": 0})
df_model["Dietary habits"] = df_model["Dietary habits"].map({"Non-Vegetarian": 1, "Vegetarian": 0})
label_encoder = LabelEncoder()
label_encoder.fit(CLASS_ORDER)
df_model["target_enc"] = label_encoder.transform(df_model[TARGET])

leak_cols = ["Hb (in gm/dL)", "PCV (%)", "who_pred"]
track_B_features = [c for c in df_model.columns if c not in [TARGET, "target_enc"] + leak_cols]
X_B, y = df_model[track_B_features], df_model["target_enc"]
X_B_train, X_B_test, y_train, y_test = train_test_split(X_B, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

binary_like = set(binary_cols + ["Dietary habits"])
numeric_to_scale = [c for c in X_B.select_dtypes(include=[np.number]).columns if c not in binary_like]
preprocessor_B = ColumnTransformer([("scale", StandardScaler(), numeric_to_scale)], remainder="passthrough")
X_tr = preprocessor_B.fit_transform(X_B_train)
X_te = preprocessor_B.transform(X_B_test)
processed = numeric_to_scale + [c for c in X_B_train.columns if c not in numeric_to_scale]
sw = compute_sample_weight("balanced", y_train)

models, results = {}, {}
def eval_(name, pred):
    results[name] = dict(accuracy=accuracy_score(y_test, pred), balanced_accuracy=balanced_accuracy_score(y_test, pred),
        f1_macro=f1_score(y_test, pred, average="macro"), f1_weighted=f1_score(y_test, pred, average="weighted"))

lr = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE)
lr.fit(X_tr, y_train); models["Logistic Regression"] = lr; eval_("Logistic Regression", lr.predict(X_te))
rf = RandomForestClassifier(n_estimators=400, min_samples_leaf=2, class_weight="balanced", random_state=RANDOM_STATE)
rf.fit(X_tr, y_train); models["Random Forest"] = rf; eval_("Random Forest", rf.predict(X_te))
xg = xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
    eval_metric="mlogloss", random_state=RANDOM_STATE)
xg.fit(X_tr, y_train, sample_weight=sw); models["XGBoost"] = xg; eval_("XGBoost", xg.predict(X_te))

cmp = pd.DataFrame(results).T.round(3)
best_name = cmp["f1_macro"].idxmax()
best = models[best_name]
print("Best:", best_name, cmp.loc[best_name].to_dict())
imp = pd.Series(np.abs(best.coef_).mean(axis=0) if hasattr(best, "coef_") else best.feature_importances_, index=processed).sort_values(ascending=False)

joblib.dump(best, OUT / "anemia_best_model.joblib")
joblib.dump(preprocessor_B, OUT / "anemia_preprocessor.joblib")
joblib.dump(label_encoder, OUT / "anemia_label_encoder.joblib")
json.dump({
    "best_model_name": best_name,
    "raw_feature_list_track_B": track_B_features,
    "numeric_features_scaled": numeric_to_scale,
    "passthrough_features": [c for c in track_B_features if c not in numeric_to_scale],
    "class_order": list(map(str, label_encoder.classes_)),
    "excluded_due_to_leakage": leak_cols,
    "test_metrics": cmp.loc[best_name].to_dict(),
}, open(OUT / "anemia_model_metadata.json", "w"), indent=2)
imp.to_csv(OUT / "anemia_feature_importance.csv", header=["importance"])
print("Saved to", OUT)

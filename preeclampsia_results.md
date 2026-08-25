# Preeclampsia Risk Model — Results

## Dataset
- Source: UCI Maternal Health Risk Dataset (https://archive.ics.uci.edu/dataset/863/pregnant%2Bhealth%2Brisk)
- N = 1,014 records
- Features: Age, SystolicBP, DiastolicBP, BS (blood sugar), BodyTemp, HeartRate
- Target: **Proxy label.** This dataset does not contain a direct preeclampsia diagnosis. We treat "high risk" (from the original `RiskLevel` column, which is strongly influenced by elevated blood pressure) as a proxy for hypertensive/preeclampsia-relevant risk, since preeclampsia is fundamentally a hypertensive pregnancy disorder. This is a simplification appropriate for a student prototype, not a clinical diagnosis label.

## Engineered features (BP-trend-related contribution)
- `mean_arterial_pressure` = DiastolicBP + (SystolicBP − DiastolicBP) / 3
- `pulse_pressure` = SystolicBP − DiastolicBP
- `bp_risk_flag` = 1 if SystolicBP ≥ 140 or DiastolicBP ≥ 90, else 0

Note: this dataset provides a single snapshot per patient rather than repeated measurements over gestational weeks, so true longitudinal BP trend could not be computed here. These derived features are the realistic MVP; true longitudinal trend tracking is documented as future work for when repeated-measurement data becomes available (see Limitations).

## Input schema (for integration with digital twin — Person 4)
```json
{
  "age": float,
  "systolic_bp": float,
  "diastolic_bp": float,
  "blood_sugar": float,
  "body_temp": float,
  "heart_rate": float
}
```
Derived internally by the preprocessing pipeline: mean_arterial_pressure, pulse_pressure, bp_risk_flag.

## Models compared

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.667 | 0.852 | 0.748 | 0.935 |
| Random Forest | 0.875 | 0.907 | 0.891 | 0.966 |
| **XGBoost (chosen)** | **0.907** | **0.907** | **0.907** | **0.970** |

## Chosen model and why
XGBoost was selected as the final model. It achieved the highest ROC-AUC (0.970) and F1 score (0.907) among the three baselines, with balanced precision and recall — important in a risk-flagging context where both missed cases (false negatives) and false alarms (false positives) carry real costs. Random Forest was a close second; Logistic Regression, while a useful interpretable baseline, lagged notably behind the tree-based models.

For context: published clinical preeclampsia ML studies using comparable clinical/BP features typically report ROC-AUC in the 0.74–0.85 range. Our result is higher, which should be read cautiously — it likely reflects the proxy-label setup and a relatively clean, small, single-source dataset rather than a claim that this prototype outperforms clinical-grade systems.

## Explainability
Feature importance was assessed using SHAP (TreeExplainer) on the XGBoost model. Mean absolute SHAP values:

| Feature | Mean |SHAP| |
|---|---|
| BS (blood sugar) | 2.111 |
| BodyTemp | 1.045 |
| SystolicBP | 1.028 |
| Age | 0.619 |
| mean_arterial_pressure | 0.378 |
| pulse_pressure | 0.241 |
| HeartRate | 0.229 |
| DiastolicBP | 0.125 |
| bp_risk_flag | 0.000 |

**Key observation:** Blood sugar is the single strongest driver of this model's predictions — notably stronger than any BP-related feature. This is an important, honest finding about the proxy-label setup: the underlying `RiskLevel` category in this dataset captures general high-risk pregnancy signal (which includes glucose-related risk), not a hypertension-specific signal. This should be stated clearly when presenting results — the model is best described as a "general maternal hypertensive/metabolic risk" model built on this dataset, rather than a pure preeclampsia-specific model. Note also that the engineered `bp_risk_flag` contributed nothing to the model (SHAP = 0) — XGBoost extracted equivalent information directly from the raw BP values, making the binary flag redundant for this particular model, though it may still be useful for the human-readable insight layer (e.g., "BP flagged as elevated: yes/no") even if not for prediction itself.

## Known limitations
- **Proxy label**: target is derived from a general "high risk" pregnancy category, not a confirmed preeclampsia diagnosis. Reported metrics should be interpreted as "hypertensive pregnancy risk," not preeclampsia detection specifically.
- **No true longitudinal BP trend**: dataset provides one snapshot per patient; trend features are derived (MAP, pulse pressure) rather than computed from repeated measurements over time.
- **Single dataset, no external validation**: results are from an internal train/test split (80/20) on ~1,014 records from Bangladeshi healthcare sources; generalizability to other populations is unverified.
- **Not a diagnostic tool**: model outputs are risk indicators intended to support discussion with a healthcare provider, not a diagnosis or treatment recommendation.
- **Class imbalance handling**: `class_weight='balanced'` was used for Logistic Regression and Random Forest to address target imbalance; this should be documented if metrics are compared against unweighted baselines elsewhere in the project.

## Files produced
- `preeclampsia_model.pkl` — trained XGBoost classifier
- `preeclampsia_preprocessing.pkl` — dict containing `scaler` (StandardScaler, fit on training data), `feature_cols` (ordered list), `model_type`
- `run_model.py` — full training script (from load → feature engineering → train → evaluate → save)

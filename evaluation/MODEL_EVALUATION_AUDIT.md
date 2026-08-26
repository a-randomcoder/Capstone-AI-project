# Model Evaluation Audit

## Purpose

This document defines the evaluation work for the four Maternal Digital Twin modules. It keeps **official held-out evaluation**, **external validation**, and **single-patient/manual testing** separate.

The goal is to obtain defensible performance numbers for the models and to investigate the requested 86–95% classroom target without tuning models against the final test data.

A test set must remain unseen during model development; changing the model after inspecting test results turns that test set into part of the development loop and makes the final metric optimistic. Cross-validation can be used on the training portion when model comparison or tuning is needed. See scikit-learn's model-evaluation guidance.

## Current repository audit

| Module | Current evidence in repository | Status |
|---|---|---|
| Anemia | `models/anemia/anemia_model_metadata.json`: accuracy ≈ 0.785, balanced accuracy ≈ 0.696, macro F1 ≈ 0.675 | Official reference exists |
| Preeclampsia proxy | Training method in the recovered `run_model.py`: stratified 80/20 split, random_state=42; LR/RF/XGBoost compared | Reproducible training logic recovered, exact printed metrics still need a local run |
| GDM | `notebooks/gdm_prediction_module.ipynb`; dataset is explicitly synthetic | Official reference must be read from/executed from the notebook; do not describe it as real-world validation |
| Thyroid | `models/thyroid/thyroid_feature_metadata.json`: ROC-AUC ≈ 0.6588 | Official reference exists |

## Preeclampsia training protocol recovered

The recovered training code uses:

1. `RiskLevel == "high risk"` as a **proxy** target.
2. Nine engineered features:
   - Age
   - SystolicBP
   - DiastolicBP
   - BS
   - BodyTemp
   - HeartRate
   - mean arterial pressure
   - pulse pressure
   - blood-pressure risk flag
3. `train_test_split(..., test_size=0.2, random_state=42, stratify=y)`.
4. Logistic Regression on standardized training data.
5. Random Forest on raw engineered features.
6. XGBoost on raw engineered features.
7. Precision, recall, F1 and ROC-AUC comparison.
8. XGBoost selected as the saved production artifact in that training script.

The saved XGBoost artifact is therefore not to be re-fit merely to improve an evaluation number. The evaluation should first reproduce the existing experiment.

## Evaluation order

### Phase 1 — reproduce existing held-out results

Run each module using its existing training methodology and record the metrics produced by the training workflow. Do not change hyperparameters yet.

Required metrics where applicable:

- accuracy
- balanced accuracy
- precision
- recall
- macro F1
- weighted F1
- ROC-AUC
- PR-AUC where useful for imbalanced binary targets

### Phase 2 — compare models

Where the training workflow already compares multiple models, retain the comparison. For preeclampsia this is LR vs RF vs XGBoost.

Do **not** choose a model solely because its accuracy is closest to 86–95%. The selection metric should remain the metric justified by the module's training design, with class imbalance considered.

### Phase 3 — investigate low scores

If a model is below the expected classroom range, inspect:

- target/class balance
- target leakage
- feature leakage
- preprocessing leakage
- train/test contamination
- class-specific precision and recall
- confusion matrix
- ROC-AUC / PR-AUC
- overfitting gap between train and held-out performance
- whether the target itself is a proxy or synthetic outcome

Only after this audit should hyperparameter tuning be considered.

### Phase 4 — external validation

External datasets are evaluated separately with the **frozen production artifact**. They must not be used to tune the production model.

The existing external preeclampsia validation already demonstrated why this distinction matters: the Tanzania cohort produced poor transfer performance under documented feature/domain shift. That result is evidence about external transfer, not evidence that the implementation is broken.

## Target table for the final report

| Module | Selected model | Accuracy | Balanced accuracy | Macro F1 | ROC-AUC | Evaluation type |
|---|---|---:|---:|---:|---:|---|
| Anemia | existing selected model | 0.785 reference | 0.696 reference | 0.675 reference | pending/if available | official held-out |
| Preeclampsia proxy | XGBoost | pending local reproduction | pending | pending | pending | official held-out |
| GDM | existing selected model | pending | pending | pending | pending | official held-out |
| Thyroid | Random Forest | pending/metadata | pending | pending | 0.6588 reference | official held-out |

## Important interpretation rule

The requested 86–95% range is a **project/classroom expectation**, not a scientific guarantee. A defensible report must show the measured result even when it is below that range. If a result is unexpectedly high, leakage and target construction must also be checked rather than assuming that the high score is automatically good.

## Existing external evaluation interface

The repository already provides `evaluation/evaluate_all.py`, which accepts separate external datasets for anemia, preeclampsia, GDM and thyroid and evaluates the frozen artifacts without retraining. It writes metrics, confusion matrices, classification reports, predictions and incorrect predictions under `evaluation/results/<condition>/`.

This audit deliberately does **not** overwrite model artifacts or fabricate missing metrics.

## Local reproduction commands

From the repository root:

```bash
# Preeclampsia: run the recovered training script and capture the LR/RF/XGB comparison.
python path/to/recovered/run_model.py

# Inspect existing official references.
cat models/anemia/anemia_model_metadata.json
cat models/thyroid/thyroid_feature_metadata.json

# External frozen-model evaluation (only when an appropriate external file exists).
python evaluation/evaluate_preeclampsia.py --data evaluation/test_data/preeclampsia_sample.csv
```

The exact GDM command should be taken from the notebook after confirming its training/evaluation cells and path assumptions; the notebook currently contains Windows-specific data-path text, so it should not be silently treated as a portable command.

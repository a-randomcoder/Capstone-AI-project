# Model Evaluation Audit

## Purpose

This document defines the evaluation work for the four Maternal Digital Twin modules. It keeps **official held-out evaluation**, **external validation**, and **single-patient/manual testing** separate.

The goal is to obtain defensible performance numbers for the models and to investigate the requested 86–95% classroom target without tuning models against the final test data.

A test set must remain unseen during model development; changing the model after inspecting test results turns that test set into part of the development loop and makes the final metric optimistic. Cross-validation can be used on the training portion when model comparison or tuning is needed.

## Current repository audit

| Module | Current evidence in repository | Status |
|---|---|---|
| Anemia | `models/anemia/anemia_model_metadata.json`: accuracy ≈ 0.785, balanced accuracy ≈ 0.696, macro F1 ≈ 0.675 | Official reference exists |
| Preeclampsia proxy | Recovered `run_model.py`: stratified 80/20 split, random_state=42; LR/RF/XGBoost compared | Reproduction script added |
| GDM | `notebooks/gdm_prediction_module.ipynb`; dataset is explicitly synthetic; notebook imports LR/RF and optional XGBoost | Reproduction script added for the clearly recoverable LR/RF experiment |
| Thyroid | `models/thyroid/thyroid_feature_metadata.json`: ROC-AUC ≈ 0.6588 | Reproduction script added |

## Reproducible held-out audits added

The branch contains four non-destructive scripts:

- `evaluation/reproduce_preeclampsia_heldout.py`
- `evaluation/reproduce_anemia_heldout.py`
- `evaluation/reproduce_gdm_heldout.py`
- `evaluation/reproduce_thyroid_heldout.py`

These scripts are intentionally **audit/reproduction scripts**. They do not overwrite production model artifacts, do not tune against test results, and do not change the frontend.

### Anemia

The reproduction follows the repository's Track B generator: `train_test_split(..., test_size=0.2, random_state=42, stratify=y)`, excludes Hb/PCV/who_pred leakage features, applies the same preprocessing, and compares Logistic Regression, Random Forest and XGBoost. The source script explicitly defines these steps. 

### Preeclampsia proxy

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

The saved XGBoost artifact is therefore not to be re-fit merely to improve an evaluation number. The reproduction script evaluates the three model families on the same held-out split defined by the recovered training methodology.

### GDM

The notebook states that the dataset is entirely synthetic and that the module is a student decision-support prototype rather than clinical validation. It imports Logistic Regression, Random Forest and optional XGBoost, uses `RANDOM_STATE = 42`, and evaluates model performance. The repository's frozen external evaluator uses the saved GDM pipeline and explicitly treats its results as external-file metrics only.

Because the notebook is stored as a large executed `.ipynb` and its exact XGBoost training cell is not safely recoverable from the repository API in this audit, the added reproduction script covers the clearly recoverable Logistic Regression and Random Forest experiment. It does not invent XGBoost hyperparameters.

### Thyroid

The repository's regeneration script explicitly uses `train_test_split(..., test_size=0.2, random_state=42, stratify=y)`, the baseline-negative cohort, the engineered `log_tsh_baseline` feature, the documented numerical/categorical preprocessing, and a Random Forest classifier. The added reproduction script mirrors that methodology and reports accuracy, balanced accuracy, precision, recall, F1, ROC-AUC and PR-AUC.

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

Where the training workflow already compares multiple models, retain the comparison. For preeclampsia this is LR vs RF vs XGBoost; for anemia it is LR vs RF vs XGBoost; for GDM the safely recoverable comparison is LR vs RF; thyroid's documented production workflow is Random Forest.

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
python evaluation/reproduce_preeclampsia_heldout.py
python evaluation/reproduce_anemia_heldout.py
python evaluation/reproduce_gdm_heldout.py
python evaluation/reproduce_thyroid_heldout.py
```

These commands reproduce the held-out experiments on the training datasets using the documented split methodology. They print the metrics but do not save or replace production model files.

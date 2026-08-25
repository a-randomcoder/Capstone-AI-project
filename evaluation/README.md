# Internal model evaluation

Offline evaluation and single-patient testing for the four frozen maternal-health modules.

**This folder is for internal / research testing only.**  
It is separate from the Streamlit dashboard (`frontend/app.py` is not modified).

---

## Evaluation types (keep them distinct)

| Type | What it is | How to run |
|------|------------|------------|
| **Official held-out** | Metrics already reported when each model was trained (notebooks / metadata) | Documented below; do **not** treat external-file metrics as a replacement |
| **External test data** | A CSV/Excel **you supply**, scored with the **frozen** production artifacts | `evaluate_*.py` / `evaluate_all.py` |
| **Manual single-patient** | Interactively or via JSON enter one patient; uses production `predict_*` wrappers | `manual_test.py` |

Evaluation scripts **never retrain** and **never write** into `models/`.

---

## Prerequisites

From the repository root (after dependencies and model artifacts are available):

```bash
pip install -r requirements.txt
# if anemia/thyroid joblibs are missing:
bash scripts/bootstrap_integration.sh
```

---

## Official held-out metrics (reference only)

These come from training notebooks / regenerators (`RANDOM_STATE=42`). They are **not** recomputed by the external-file scripts unless you point those scripts at the same held-out split.

| Module | Official reference | Notes |
|--------|-------------------|--------|
| **Anemia** | `models/anemia/anemia_model_metadata.json` → accuracy≈0.785, balanced_acc≈0.696, f1_macro≈0.675 | Track B; **Hb/PCV/who_pred excluded** (leakage). Severe class rare. |
| **Preeclampsia** | `preeclampsia_results.md` | Target is a **RiskLevel proxy**, not confirmed clinical preeclampsia. |
| **GDM** | GDM notebook under `Capstone-AI-Project/Notebook/` | Trained on **synthetic** data (~65% prevalence); not clinically validated. |
| **Thyroid** | `models/thyroid/thyroid_feature_metadata.json` → ROC-AUC≈0.6588 | Later SCH-consistent risk among baseline-negative women; single-center. |

---

## External test data evaluation

### Expected columns

**Anemia** (`evaluate_anemia.py`)

- Features: Track B training names (see metadata `raw_feature_list_track_B`), e.g. `Obs Score L`, `LMP known`, `POG (in weeks)`, `TRBC (in 10^6 /microL)`, …, `SBP`, `DBP`
- Target (default): `Severity of anemia (on the basis of Hb)` ∈ {Normal, Mild, Moderate, Severe}
- Do **not** rely on Hb/PCV as model inputs (excluded)

**Preeclampsia** (`evaluate_preeclampsia.py`)

- Features: `Age`, `SystolicBP`, `DiastolicBP`, `BS`, `BodyTemp`, `HeartRate`
- Target (default): `RiskLevel` (string with “high” → positive proxy class) or a 0/1 column via `--target`

**GDM** (`evaluate_gdm.py`)

- Features: `age`, `pre_pregnancy_bmi`, `ethnicity`, `family_history_dm`, `previous_gdm`, `pcos`, `previous_macrosomia`, `booking_gestational_age`, `early_rbs_mgdl`, `early_ppbs_mgdl`, `early_hba1c_percent`, `early_ogtt_performed`, `early_ogtt_fasting_mgdl`, `early_ogtt_1h_mgdl`, `early_ogtt_2h_mgdl`
- Target (default): `gdm_outcome` (0/1)

**Thyroid** (`evaluate_thyroid.py`)

- Features: `age`, `bmi`, `tsh_baseline`, `ft3_baseline`, `ft4_baseline`, `tpo_baseline`, `household_income`, `parity`, `family_history_diabetes`, `smoking_exposure`, `alcohol_consumption`, `folic_acid_supplementation`, `vd_supplementation`  
  (`log_tsh_baseline` is computed inside the script)
- Target (default): `thyroid_dysfunction_later` (0/1)

### Commands

```bash
# Single module
python evaluation/evaluate_anemia.py --data path/to/anemia_test.csv
python evaluation/evaluate_preeclampsia.py --data path/to/pe_test.csv
python evaluation/evaluate_gdm.py --data path/to/gdm_test.csv
python evaluation/evaluate_thyroid.py --data path/to/thyroid_test.csv

# Optional custom target column name
python evaluation/evaluate_gdm.py --data gdm_test.csv --target gdm_outcome

# All modules you have files for
python evaluation/evaluate_all.py \
  --anemia-data evaluation/test_data/anemia_sample.csv \
  --preeclampsia-data evaluation/test_data/preeclampsia_sample.csv \
  --gdm-data evaluation/test_data/gdm_sample.csv \
  --thyroid-data evaluation/test_data/thyroid_sample.csv
```

### Outputs (`evaluation/results/<condition>/`)

| File | Content |
|------|---------|
| `metrics.json` | accuracy, balanced accuracy, precision/recall (macro & weighted), macro/weighted F1, ROC-AUC when available |
| `confusion_matrix.csv` | Confusion matrix |
| `classification_report.txt` | sklearn classification report |
| `per_class_metrics.csv` | Per-class precision / recall / F1 / support |
| `predictions.csv` | Actual vs predicted (+ probabilities when available) |
| `incorrect_predictions.csv` | Rows the model got wrong |
| `summary.md` | Short human-readable summary |

---

## Manual single-patient test

```bash
# Interactive prompts (defaults from sample_patient)
python evaluation/manual_test.py

# Non-interactive sample
python evaluation/manual_test.py --sample

# One module only
python evaluation/manual_test.py --sample --module thyroid

# From a JSON patient file
python evaluation/manual_test.py --json path/to/patient.json
```

Writes `evaluation/results/manual_last_run.json`.

---

## Metrics (what they mean)

| Metric | Meaning |
|--------|---------|
| **Accuracy** | Fraction of rows correctly classified |
| **Balanced accuracy** | Average of recall per class (better under imbalance) |
| **Precision (macro)** | Unweighted mean precision across classes |
| **Recall (macro)** | Unweighted mean recall across classes |
| **Macro F1** | Unweighted mean F1 across classes |
| **Weighted F1** | F1 averaged by class support |
| **ROC-AUC** | Ranking quality from predicted probabilities (binary or OVR multiclass when supported) |

External-file metrics depend entirely on the file you upload (label quality, prevalence, domain shift). They are **not** a substitute for the official held-out numbers above.

---

## Limitations (preserved)

- **Anemia** — Hb / PCV / who_pred excluded due to target leakage.
- **Preeclampsia** — RiskLevel **proxy** label, not confirmed clinical preeclampsia.
- **GDM** — **synthetic** training data; not clinically validated.
- **Thyroid** — later SCH-consistent risk; single-center; modest discrimination (ROC-AUC≈0.66).
- Research prototype only — **not a diagnostic system**.

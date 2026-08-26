# Maternal Digital Twin

**Student AI/ML capstone** — maternal-health decision-support prototype that combines four independent condition modules into one transparent maternal health profile.

> **This is NOT a medical diagnostic system.**  
> It does not diagnose disease, prescribe treatment, or replace clinical judgment.  
> No combined clinical “overall pregnancy risk score” is computed.

---

## Architecture

```
Patient information
        ↓
Common patient schema (src/integration/schemas.py)
        ↓
 ┌──────────┬──────────────┬─────────┬─────────┐
 │  Anemia  │ Preeclampsia │   GDM   │ Thyroid │
 │  (REAL)  │    (REAL)    │ (REAL*) │  (REAL) │
 └──────────┴──────────────┴─────────┴─────────┘
        ↓
Condition predictions + contributing factors
        ↓
Maternal Health Profile (generate_maternal_profile)
        ↓
Streamlit dashboard (frontend/app.py)
```

\* GDM uses a model trained on **synthetic** data.

---

## Module status

| Module | Status | Model | Critical caveats |
|--------|--------|-------|------------------|
| **Anemia** | Complete | Logistic Regression (Track B) | Hb / PCV / who_pred **excluded** (leakage; WHO thresholds recover ~91.4% of labels). Macro F1 ≈ 0.675. Severe class ~9 samples. |
| **Preeclampsia** | Complete | XGBoost | Target is a **RiskLevel-derived proxy**, not confirmed clinical preeclampsia. |
| **GDM** | Complete (synthetic) | Random Forest pipeline | **Synthetic** dataset (n≈10k, ~65% prevalence). Not clinically validated. |
| **Thyroid** | Complete | Random Forest | Later SCH-consistent risk among baseline-negative women; single-center; ROC-AUC ≈ 0.66. |

---

## Repository structure

```
Capstone-AI-project/
├── README.md
├── SETUP.md
├── requirements.txt
├── run_demo.py
├── .gitignore
├── .gitattributes
│
├── data/raw/                 # authoritative raw datasets
│   ├── CBC_Datasets.csv
│   ├── maternal_health_risk.csv
│   ├── gdm_synthetic_data.csv
│   └── origin_data.xlsx
│
├── models/                   # one folder per condition
│   ├── anemia/
│   ├── preeclampsia/
│   ├── gdm/
│   └── thyroid/
│
├── src/
│   ├── models/               # predict_anemia / preeclampsia / gdm / thyroid
│   └── integration/          # schemas, aggregator, prediction_service
│
├── notebooks/                # original training notebooks (methodology preserved)
│   ├── anemia_module.ipynb
│   ├── gdm_prediction_module.ipynb
│   └── thyroid_module.ipynb
│
├── scripts/                  # regenerate anemia/thyroid; bootstrap
├── frontend/app.py           # Streamlit dashboard
├── evaluation/               # offline evaluation only (not in the app)
└── outputs/
```

**Note:** `bash scripts/bootstrap_integration.sh` migrates any leftover legacy paths into this layout and regenerates anemia + thyroid artifacts if missing.

---

## Quick start

```bash
pip install -r requirements.txt
bash scripts/bootstrap_integration.sh   # migrate + build anemia/thyroid if missing
python run_demo.py
streamlit run frontend/app.py
```

See [SETUP.md](SETUP.md) for details.

---

## Evaluation

Internal offline evaluation lives under `evaluation/` (separate from the dashboard).  
See [evaluation/README.md](evaluation/README.md).

---

## Limitations

- **Anemia** — Hb / PCV / who_pred excluded due to target leakage.
- **Preeclampsia** — RiskLevel **proxy** label, not confirmed clinical preeclampsia.
- **GDM** — **synthetic** training data; not clinically validated.
- **Thyroid** — later SCH-consistent risk; single-center retrospective data; modest discrimination.
- Research / student decision-support prototype only — **not a diagnostic system**.

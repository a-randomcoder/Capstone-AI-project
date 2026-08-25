# Maternal Digital Twin

**Student AI/ML capstone** — maternal-health decision-support prototype that combines four condition modules into one transparent maternal health profile.

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
 │  (REAL)  │    (REAL)    │ (REAL*) │ Pending │
 └──────────┴──────────────┴─────────┴─────────┘
        ↓
Condition predictions + contributing factors
        ↓
Maternal Health Profile (generate_maternal_profile)
        ↓
Streamlit dashboard
```

\* GDM uses a model trained on **synthetic** data.

---

## Module status

| Module | Status | Model | Critical caveats |
|--------|--------|-------|------------------|
| **Anemia** | Complete | Logistic Regression (Track B) | Hb / PCV / who_pred **excluded** (leakage; WHO thresholds recover ~91.4% of labels). Macro F1 ≈ 0.675. Severe class ~9 samples. |
| **Preeclampsia** | Complete | XGBoost | Target is a **RiskLevel-derived proxy**, not confirmed clinical preeclampsia. |
| **GDM** | Complete (synthetic) | Random Forest pipeline | **Synthetic** dataset (n≈10k, ~65% prevalence). Not clinically validated. |
| **Thyroid** | Placeholder | — | `predict_thyroid()` returns Pending until artifacts arrive. |

---

## Repository structure

```
maternal_digital_twin/
├── data/raw/
│   ├── CBC_Datasets.csv              # anemia
│   ├── maternal_health_risk.csv      # preeclampsia (UCI)
│   └── gdm_synthetic_data.csv        # GDM (synthetic)
├── models/
│   ├── anemia/
│   │   ├── anemia_best_model.joblib
│   │   ├── anemia_preprocessor.joblib
│   │   ├── anemia_label_encoder.joblib
│   │   ├── anemia_model_metadata.json
│   │   └── anemia_feature_importance.csv
│   ├── preeclampsia/
│   │   ├── preeclampsia_model.pkl
│   │   ├── preeclampsia_preprocessing.pkl
│   │   └── preeclampsia_results.md
│   ├── gdm/
│   │   └── gdm_best_model.joblib
│   └── thyroid/                      # empty — waiting for teammate
├── src/
│   ├── models/
│   │   ├── predict_anemia.py
│   │   ├── predict_preeclampsia.py
│   │   ├── predict_gdm.py
│   │   └── predict_thyroid.py        # placeholder
│   └── integration/
│       ├── schemas.py
│       ├── aggregator.py             # generate_maternal_profile()
│       └── prediction_service.py
├── notebooks/
│   ├── anemia_module.ipynb
│   └── gdm_prediction_module.ipynb
├── frontend/
│   └── app.py                        # Streamlit dashboard
├── outputs/
├── requirements.txt
└── README.md
```

---

## Setup

```bash
cd Capstone-AI-project   # or your clone path
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Run the prediction pipeline (CLI)

From the project root:

```bash
python run_demo.py
# or
python -m src.integration.aggregator
```

Or in Python:

```python
from src.integration.schemas import sample_patient
from src.integration.aggregator import generate_maternal_profile

profile = generate_maternal_profile(sample_patient())
print(profile["modules"]["Anemia"]["prediction"])
print(profile["summary"])
```

---

## Run the Streamlit dashboard

```bash
streamlit run frontend/app.py
```

1. Optionally click **Load sample patient** in the sidebar.  
2. Adjust fields as needed.  
3. Click **Analyze Maternal Health**.  
4. View four condition cards + maternal health profile summary.

---

## Adding the thyroid module later

1. Place artifacts under `models/thyroid/`.  
2. Replace the body of `src/models/predict_thyroid.py` so it returns the same keys as the other modules:
   - `condition`, `prediction`, `probability`, `class_probabilities`, `important_factors`, `notes`, `status`
3. No changes required to the aggregator or dashboard.

---

## Datasets (summary)

- **Anemia:** CBC Benchmark (473 pregnant women → 464 after removing 9 duplicates). Target = severity on the basis of Hb.  
- **Preeclampsia:** UCI Maternal Health Risk (n=1,014). Proxy label from `RiskLevel`.  
- **GDM:** Synthetic (n=10,000). Elevated prevalence; for prototype use only.  
- **Thyroid:** Not yet available.

---

## Limitations (must preserve)

1. **Anemia:** Target is near-deterministic from Hb; Hb/PCV were deliberately excluded from the predictive model. Severe class is tiny.  
2. **Preeclampsia:** Label is a **proxy**, not a clinical diagnosis of preeclampsia. Single-time-point features only.  
3. **GDM:** Synthetic data and elevated prevalence; metrics are not transferable to real clinical populations.  
4. **System:** Research/student decision-support prototype. No external clinical validation. No treatment recommendations. No overall risk percentage.

---

## License / academic use

Capstone coursework. Do not deploy as a clinical product.

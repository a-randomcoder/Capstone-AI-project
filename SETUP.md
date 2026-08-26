# Setup — Maternal Digital Twin

## Quick start

```bash
git clone https://github.com/a-randomcoder/Capstone-AI-project.git
cd Capstone-AI-project
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
bash scripts/bootstrap_integration.sh   # migrates legacy paths + regenerates anemia/thyroid if missing
python run_demo.py
streamlit run frontend/app.py
```

## Models

| Module | Artifact location | How obtained |
|--------|-------------------|--------------|
| Anemia | `models/anemia/*.joblib` | `python scripts/regenerate_anemia_artifacts.py` (from `data/raw/CBC_Datasets.csv` or legacy root) |
| Preeclampsia | `models/preeclampsia/*.pkl` | Migrated by bootstrap from repo root if needed |
| GDM | `models/gdm/gdm_best_model.joblib` | Migrated by bootstrap from `Capstone-AI-Project/Notebook/models/` if needed |
| Thyroid | `models/thyroid/thyroid_final_model.pkl` | `python scripts/regenerate_thyroid_artifacts.py` (from `data/raw/origin_data.xlsx` or legacy root) |

## Data

Authoritative raw datasets under `data/raw/` (bootstrap moves legacy copies there):

- `CBC_Datasets.csv` — anemia
- `maternal_health_risk.csv` — preeclampsia proxy
- `gdm_synthetic_data.csv` — GDM (synthetic)
- `origin_data.xlsx` — thyroid

## Evaluation (offline only)

See `evaluation/README.md`. Not part of the Streamlit app.

## Limitations

No overall clinical risk score. Anemia excludes Hb/PCV (leakage). Preeclampsia = RiskLevel proxy. GDM = synthetic. Thyroid = single-center SCH-later risk. Research prototype — not a diagnostic system.

# Setup — Maternal Digital Twin

## Bootstrap (recommended)

```bash
git clone https://github.com/a-randomcoder/Capstone-AI-project.git
cd Capstone-AI-project
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/bootstrap_integration.sh
python run_demo.py
streamlit run frontend/app.py
```

## Models

| Module | Artifact | Source |
|--------|----------|--------|
| Anemia | `models/anemia/*.joblib` | `python scripts/regenerate_anemia_artifacts.py` |
| Preeclampsia | `preeclampsia_model.pkl` | Already in repo root |
| GDM | `Capstone-AI-Project/Notebook/models/gdm_best_model.joblib` | Already in repo |
| Thyroid | `models/thyroid/thyroid_final_model.pkl` | `python scripts/regenerate_thyroid_artifacts.py` (needs `origin_data.xlsx`) |

If `src/models/predict_anemia.py` is missing after clone:

```bash
python scripts/assemble_part1.py
python scripts/assemble_part2.py
```

## Limitations

No overall clinical risk score. Anemia excludes Hb/PCV (leakage). Preeclampsia = RiskLevel proxy. GDM = synthetic. Thyroid = single-center SCH-later risk. Not a diagnostic system.

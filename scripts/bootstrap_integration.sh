#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "[0] Ensure target directories..."
mkdir -p data/raw models/anemia models/preeclampsia models/gdm models/thyroid notebooks

echo "[0b] Migrate legacy data paths if needed..."
[ -f CBC_Datasets.csv ] && [ ! -f data/raw/CBC_Datasets.csv ] && mv CBC_Datasets.csv data/raw/CBC_Datasets.csv || true
[ -f origin_data.xlsx ] && [ ! -f data/raw/origin_data.xlsx ] && mv origin_data.xlsx data/raw/origin_data.xlsx || true
[ -f raw/maternal_health_risk.csv ] && [ ! -f data/raw/maternal_health_risk.csv ] && mkdir -p data/raw && mv raw/maternal_health_risk.csv data/raw/ || true
[ -f Capstone-AI-Project/data/gdm_synthetic_data.csv ] && [ ! -f data/raw/gdm_synthetic_data.csv ] && mv Capstone-AI-Project/data/gdm_synthetic_data.csv data/raw/ || true

echo "[0c] Migrate legacy model paths if needed..."
[ -f Capstone-AI-Project/Notebook/models/gdm_best_model.joblib ] && [ ! -f models/gdm/gdm_best_model.joblib ] && mv Capstone-AI-Project/Notebook/models/gdm_best_model.joblib models/gdm/ || true
[ -f preeclampsia_model.pkl ] && [ ! -f models/preeclampsia/preeclampsia_model.pkl ] && mv preeclampsia_model.pkl models/preeclampsia/ || true
[ -f preeclampsia_preprocessing.pkl ] && [ ! -f models/preeclampsia/preeclampsia_preprocessing.pkl ] && mv preeclampsia_preprocessing.pkl models/preeclampsia/ || true
[ -f preeclampsia_results.md ] && [ ! -f models/preeclampsia/preeclampsia_results.md ] && mv preeclampsia_results.md models/preeclampsia/ || true

echo "[0d] Migrate notebooks if needed..."
[ -f anemia_module.ipynb ] && [ ! -f notebooks/anemia_module.ipynb ] && mv anemia_module.ipynb notebooks/ || true
[ -f Capstone-AI-Project/Notebook/gdm_prediction_module.ipynb ] && [ ! -f notebooks/gdm_prediction_module.ipynb ] && mv Capstone-AI-Project/Notebook/gdm_prediction_module.ipynb notebooks/ || true
[ -f thyroid_model_complete.ipynb ] && [ ! -f notebooks/thyroid_module.ipynb ] && mv thyroid_model_complete.ipynb notebooks/thyroid_module.ipynb || true

echo "[1] Anemia joblibs..."
python scripts/regenerate_anemia_artifacts.py
echo "[2] Thyroid model..."
python scripts/regenerate_thyroid_artifacts.py
echo "Bootstrap complete."
echo "Run: python run_demo.py"
echo "Or:  streamlit run frontend/app.py"

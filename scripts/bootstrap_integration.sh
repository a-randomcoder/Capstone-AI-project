#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
echo "[1] Decode remaining sources..."
python scripts/decode_sources.py
echo "[2] Anemia joblibs..."
python scripts/regenerate_anemia_artifacts.py
echo "[3] Thyroid model..."
python scripts/regenerate_thyroid_artifacts.py
echo "[4] Link GDM + preeclampsia..."
mkdir -p models/gdm models/preeclampsia
[ -f Capstone-AI-Project/Notebook/models/gdm_best_model.joblib ] && cp -n Capstone-AI-Project/Notebook/models/gdm_best_model.joblib models/gdm/ || true
[ -f preeclampsia_model.pkl ] && cp -n preeclampsia_model.pkl models/preeclampsia/ || true
[ -f preeclampsia_preprocessing.pkl ] && cp -n preeclampsia_preprocessing.pkl models/preeclampsia/ || true
echo "Bootstrap complete."
echo "Run: python run_demo.py"
echo "Or:  streamlit run frontend/app.py"

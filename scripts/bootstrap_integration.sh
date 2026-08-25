#!/usr/bin/env bash
# Bootstrap remaining integration sources + model artifacts after clone.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "[1/4] Installing remaining Python sources (if needed)..."
if [ ! -f src/models/predict_anemia.py ]; then
  python scripts/install_part1.py
fi
if [ ! -f frontend/app.py ]; then
  python scripts/install_part2.py
fi

echo "[2/4] Regenerating anemia artifacts (exact Track B, F1=0.675)..."
python scripts/regenerate_anemia_artifacts.py

echo "[3/4] Regenerating thyroid artifacts (exact RF tuned, ROC-AUC=0.6588)..."
python scripts/regenerate_thyroid_artifacts.py

echo "[4/4] Ensuring GDM/preeclampsia paths..."
mkdir -p models/gdm models/preeclampsia
if [ ! -f models/gdm/gdm_best_model.joblib ] && [ -f Capstone-AI-Project/Notebook/models/gdm_best_model.joblib ]; then
  cp Capstone-AI-Project/Notebook/models/gdm_best_model.joblib models/gdm/
fi
if [ ! -f models/preeclampsia/preeclampsia_model.pkl ] && [ -f preeclampsia_model.pkl ]; then
  cp preeclampsia_model.pkl models/preeclampsia/
  cp preeclampsia_preprocessing.pkl models/preeclampsia/
fi

echo "Done. Run: python run_demo.py   OR   streamlit run frontend/app.py"

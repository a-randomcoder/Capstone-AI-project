# Finish setup after clone

## 1. Clone / pull

```bash
git clone https://github.com/a-randomcoder/Capstone-AI-project.git
cd Capstone-AI-project
git pull origin main
```

## 2. Install deps

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Regenerate anemia joblibs (exact Track B pipeline)

```bash
python scripts/regenerate_anemia_artifacts.py
```

This recreates `models/anemia/*.joblib` with Macro F1 0.675 (same as notebook).

## 4. GDM model path

Already present at:
`Capstone-AI-Project/Notebook/models/gdm_best_model.joblib`

## 5. Preeclampsia models

Already at repo root:
- `preeclampsia_model.pkl`
- `preeclampsia_preprocessing.pkl`

## 6. Run

```bash
python run_demo.py
streamlit run frontend/app.py
```

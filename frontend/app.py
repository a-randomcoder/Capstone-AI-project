"""
Maternal Digital Twin - Streamlit dashboard

Student research / decision-support prototype.
NOT a medical diagnostic system.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.integration.prediction_service import analyze_patient
from src.integration.schemas import sample_patient

st.set_page_config(
    page_title="Maternal Digital Twin",
    page_icon="🤰",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .condition-card {
        border: 1px solid #e0e0e0; border-radius: 12px; padding: 1.1rem 1.3rem;
        margin-bottom: 0.8rem; background: #fafafa;
    }
    .pred-value { font-size: 1.3rem; font-weight: 700; color: #1a1a2e; }
    .disclaimer {
        background: #fff8e6; border-left: 4px solid #f0ad4e;
        padding: 0.8rem 1rem; margin: 0.8rem 0 1.2rem 0; font-size: 0.9rem;
    }
    .profile-box {
        background: #f0f7ff; border-radius: 10px; padding: 1.1rem;
        border: 1px solid #cce0ff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🤰 Maternal Digital Twin")
st.caption("Student AI/ML decision-support prototype · Not a diagnostic device")

st.markdown(
    """
    <div class="disclaimer">
    <strong>Important:</strong> This system does <em>not</em> diagnose, prescribe treatment,
    or replace clinical judgment. Outputs are independent module estimates for research /
    educational use only. No combined clinical risk score is produced.
    <br/>Anemia excludes Hb/PCV (leakage). Preeclampsia uses a RiskLevel <em>proxy</em> label.
    GDM model is trained on <em>synthetic</em> data. Thyroid uses baseline RF (SCH target).
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar inputs
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Patient information")
    st.caption("Enter available values. Leave blank if unknown.")

    if st.button("Load sample patient"):
        st.session_state["use_sample"] = True

    sample = sample_patient() if st.session_state.get("use_sample") else {}

    st.subheader("Demographics")
    age = st.number_input("Age (years)", min_value=0.0, max_value=60.0, value=float(sample.get("age") or 28), step=1.0)
    ethnicity = st.selectbox(
        "Ethnicity",
        ["Asian", "Caucasian", "African", "Hispanic", "Other"],
        index=0 if not sample else ["Asian", "Caucasian", "African", "Hispanic", "Other"].index(sample.get("ethnicity", "Asian"))
        if sample.get("ethnicity") in ["Asian", "Caucasian", "African", "Hispanic", "Other"] else 0,
    )

    st.subheader("Pregnancy")
    gestational_age = st.number_input("Gestational age / POG (weeks)", min_value=0.0, max_value=45.0, value=float(sample.get("gestational_age") or 24.0), step=0.5)
    booking_ga = st.number_input("Booking gestational age (weeks)", min_value=0.0, max_value=45.0, value=float(sample.get("booking_gestational_age") or 10.0), step=0.5)
    bmi = st.number_input("Pre-pregnancy BMI", min_value=10.0, max_value=60.0, value=float(sample.get("pre_pregnancy_bmi") or 26.5), step=0.1)
    obs_score = st.number_input("Obs Score L", min_value=0, max_value=10, value=int(sample.get("obs_score_l") or 1))
    lmp_known = st.selectbox("LMP known", ["Yes", "No"], index=0)
    usg = st.selectbox("USG", ["Yes", "No"], index=0)

    st.subheader("Vitals")
    sbp = st.number_input("Systolic BP", min_value=60, max_value=250, value=int(sample.get("systolic_bp") or 135))
    dbp = st.number_input("Diastolic BP", min_value=30, max_value=150, value=int(sample.get("diastolic_bp") or 88))
    pulse = st.number_input("Pulse / heart rate", min_value=40, max_value=200, value=int(sample.get("pulse") or 86))
    body_temp = st.number_input("Body temperature (°F)", min_value=90.0, max_value=110.0, value=float(sample.get("body_temp") or 98.2), step=0.1)

    st.subheader("Glucose / GDM risk")
    blood_sugar = st.number_input("Blood sugar (BS, preeclampsia unit)", min_value=0.0, max_value=50.0, value=float(sample.get("blood_sugar") or 12.0), step=0.1)
    early_rbs = st.number_input("Early RBS (mg/dL)", min_value=0.0, max_value=400.0, value=float(sample.get("early_rbs_mgdl") or 118.0), step=1.0)
    early_ppbs = st.number_input("Early PPBS (mg/dL)", min_value=0.0, max_value=400.0, value=float(sample.get("early_ppbs_mgdl") or 145.0), step=1.0)
    early_hba1c = st.number_input("Early HbA1c (%)", min_value=3.0, max_value=15.0, value=float(sample.get("early_hba1c_percent") or 5.6), step=0.1)
    early_ogtt_performed = st.selectbox("Early OGTT performed", [0, 1], index=0)
    family_history_dm = st.selectbox("Family history of DM", [0, 1], index=0)
    previous_gdm = st.selectbox("Previous GDM", [0, 1], index=0)
    pcos = st.selectbox("PCOS", [0, 1], index=0)
    previous_macrosomia = st.selectbox("Previous macrosomia", [0, 1], index=0)

    st.subheader("Hematology (no Hb/PCV - excluded)")
    trbc = st.number_input("TRBC (10^6/µL)", min_value=0.0, max_value=8.0, value=float(sample.get("trbc") or 3.9), step=0.01)
    mcv = st.number_input("MCV (fL)", min_value=40.0, max_value=140.0, value=float(sample.get("mcv") or 78.0), step=0.1)
    mch = st.number_input("MCH (pg)", min_value=10.0, max_value=40.0, value=float(sample.get("mch") or 25.0), step=0.1)
    mchc = st.number_input("MCHC (g/dL)", min_value=20.0, max_value=40.0, value=float(sample.get("mchc") or 31.5), step=0.1)
    rdw = st.number_input("RDW (%)", min_value=5.0, max_value=30.0, value=float(sample.get("rdw") or 15.2), step=0.1)
    ret_count = st.number_input("Reticulocyte count", min_value=0.0, max_value=5.0, value=float(sample.get("reticulocyte_count") or 0.05), step=0.01)
    ret_pct = st.number_input("Reticulocyte %", min_value=0.0, max_value=20.0, value=float(sample.get("reticulocyte_pct") or 1.2), step=0.1)
    serum_iron = st.number_input("Serum iron (µg/dL)", min_value=0.0, max_value=300.0, value=float(sample.get("serum_iron") or 55.0), step=1.0)
    tibc = st.number_input("TIBC (µg/dL)", min_value=0.0, max_value=600.0, value=float(sample.get("tibc") or 420.0), step=1.0)
    ts = st.number_input("Transferrin saturation (%)", min_value=0.0, max_value=100.0, value=float(sample.get("transferrin_saturation") or 13.1), step=0.1)
    tbil = st.number_input("Total bilirubin", min_value=0.0, max_value=10.0, value=float(sample.get("total_bilirubin") or 0.5), step=0.01)
    dbil = st.number_input("Direct bilirubin", min_value=0.0, max_value=10.0, value=float(sample.get("direct_bilirubin") or 0.15), step=0.01)
    ibil = st.number_input("Indirect bilirubin", min_value=0.0, max_value=10.0, value=float(sample.get("indirect_bilirubin") or 0.35), step=0.01)
    urea = st.number_input("Urea (mg/dL)", min_value=0.0, max_value=100.0, value=float(sample.get("urea") or 18.0), step=0.1)
    creat = st.number_input("Creatinine (mg/dL)", min_value=0.0, max_value=10.0, value=float(sample.get("creatinine") or 0.7), step=0.01)

    st.subheader("Clinical signs")
    pallor = st.selectbox("Pallor", ["Yes", "No"], index=0 if sample.get("pallor") == "Yes" else 1)
    edema = st.selectbox("Edema", ["Yes", "No"], index=1)
    icterus = st.selectbox("Icterus", ["Yes", "No"], index=1)
    diet = st.selectbox("Dietary habits", ["Vegetarian", "Non-Vegetarian"], index=0)
    iron_supp = st.selectbox("History of iron supplementation", ["Yes", "No"], index=1)
    transfusion = st.selectbox("History of blood transfusion", ["Yes", "No"], index=1)
    fam_hb = st.selectbox("Family history of hemoglobinopathy", ["Yes", "No"], index=1)
    allergy = st.selectbox("History of allergy", ["Yes", "No"], index=1)

    run = st.button("Analyze Maternal Health", type="primary")

# ---------------------------------------------------------------------------
# Build patient_data
# ---------------------------------------------------------------------------
patient_data = {
    "age": age,
    "ethnicity": ethnicity,
    "gestational_age": gestational_age,
    "booking_gestational_age": booking_ga,
    "pre_pregnancy_bmi": bmi,
    "obs_score_l": obs_score,
    "lmp_known": lmp_known,
    "usg": usg,
    "systolic_bp": sbp,
    "diastolic_bp": dbp,
    "pulse": pulse,
    "body_temp": body_temp,
    "blood_sugar": blood_sugar,
    "early_rbs_mgdl": early_rbs,
    "early_ppbs_mgdl": early_ppbs,
    "early_hba1c_percent": early_hba1c,
    "early_ogtt_performed": early_ogtt_performed,
    "early_ogtt_fasting_mgdl": None,
    "early_ogtt_1h_mgdl": None,
    "early_ogtt_2h_mgdl": None,
    "family_history_dm": family_history_dm,
    "previous_gdm": previous_gdm,
    "pcos": pcos,
    "previous_macrosomia": previous_macrosomia,
    "trbc": trbc,
    "mcv": mcv,
    "mch": mch,
    "mchc": mchc,
    "rdw": rdw,
    "reticulocyte_count": ret_count,
    "reticulocyte_pct": ret_pct,
    "serum_iron": serum_iron,
    "tibc": tibc,
    "transferrin_saturation": ts,
    "total_bilirubin": tbil,
    "direct_bilirubin": dbil,
    "indirect_bilirubin": ibil,
    "urea": urea,
    "creatinine": creat,
    "pallor": pallor,
    "edema": edema,
    "icterus": icterus,
    "dietary_habits": diet,
    "history_iron_supplementation": iron_supp,
    "history_blood_transfusion": transfusion,
    "family_history_hemoglobinopathy": fam_hb,
    "history_allergy": allergy,
}


def render_card(result: dict):
    condition = result.get("condition", "")
    prediction = result.get("prediction", "-")
    status = result.get("status", "")
    proba = result.get("probability")
    class_probs = result.get("class_probabilities") or {}
    factors = result.get("important_factors") or []
    notes = result.get("notes", "")

    color = "#888"
    if status == "pending":
        color = "#b8860b"
    elif status == "missing_features":
        color = "#c0392b"
    elif status == "ok":
        color = "#1a7f37"

    st.markdown(f'<div class="condition-card">', unsafe_allow_html=True)
    st.markdown(f"**{condition}**  ·  `{status}`")
    st.markdown(f'<div class="pred-value" style="color:{color}">{prediction}</div>', unsafe_allow_html=True)
    if proba is not None:
        st.write(f"Predicted-class probability: **{proba:.3f}**")
    if class_probs:
        st.write("Class probabilities:")
        st.json(class_probs)
    if factors:
        st.write("Important contributing factors:")
        for f in factors:
            src = f.get("source", "")
            st.write(f"- `{f.get('feature')}`: {f.get('shap_contribution')} ({src})")
    if notes:
        st.caption(notes)
    st.markdown("</div>", unsafe_allow_html=True)


if run:
    with st.spinner("Running modules…"):
        profile = analyze_patient(patient_data)

    st.subheader("Condition modules")
    cols = st.columns(2)
    modules = profile["modules"]
    order = ["Anemia", "Preeclampsia", "GDM", "Thyroid"]
    for i, name in enumerate(order):
        with cols[i % 2]:
            render_card(modules[name])

    st.subheader("Maternal Health Profile")
    st.markdown('<div class="profile-box">', unsafe_allow_html=True)
    summary = profile["summary"]
    st.markdown("**Flagged (module-level):**")
    if summary["flagged_conditions"]:
        for item in summary["flagged_conditions"]:
            st.write(f"- {item}")
    else:
        st.write("- None flagged by current module outputs")
    st.markdown("**Within typical / proxy-low / informational:**")
    for item in summary["within_typical_range_or_proxy_low"]:
        st.write(f"- {item}")
    st.markdown("**Pending / incomplete:**")
    for item in summary["pending_or_incomplete"]:
        st.write(f"- {item}")
    st.info(summary["note"])
    st.markdown("</div>", unsafe_allow_html=True)
    st.warning(profile["disclaimer"])
else:
    st.info("Fill patient fields in the sidebar (or load the sample), then click **Analyze Maternal Health**.")

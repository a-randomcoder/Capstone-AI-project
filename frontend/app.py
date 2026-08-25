"""
Maternal Digital Twin - Streamlit dashboard

Student research / decision-support prototype.
NOT a medical diagnostic system.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.integration.prediction_service import analyze_patient
from src.integration.schemas import sample_patient

# ---------------------------------------------------------------------------
# Page config & light styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Maternal Health Profile",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.4rem; padding-bottom: 2rem; }
    div[data-testid="stMetricValue"] { font-size: 1.15rem; }
    .mh-card {
        border: 1px solid #e6e6e6;
        border-radius: 10px;
        padding: 1rem 1.15rem 0.85rem 1.15rem;
        background: #ffffff;
        margin-bottom: 0.6rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
        height: 100%;
    }
    .mh-card-title {
        font-size: 0.85rem;
        letter-spacing: 0.02em;
        text-transform: uppercase;
        color: #5a5a5a;
        margin-bottom: 0.25rem;
        font-weight: 600;
    }
    .mh-prediction {
        font-size: 1.2rem;
        font-weight: 650;
        color: #1a1a1a;
        line-height: 1.3;
        margin-bottom: 0.35rem;
    }
    .mh-status-ok { color: #1b6b3a; }
    .mh-status-warn { color: #8a6d1d; }
    .mh-status-miss { color: #8b2e2e; }
    .mh-subtle { color: #666; font-size: 0.88rem; }
    .mh-limit-box {
        background: #fafafa;
        border: 1px solid #e8e8e8;
        border-left: 3px solid #9a9a9a;
        padding: 0.75rem 1rem;
        font-size: 0.9rem;
        color: #444;
        margin: 0.5rem 0 1rem 0;
    }
    .mh-section-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #777;
        font-weight: 600;
        margin-bottom: 0.35rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Human-readable card titles (display only; backend keys unchanged)
CARD_TITLES = {
    "Anemia": "Anemia",
    "Preeclampsia": "Preeclampsia-related proxy risk",
    "GDM": "Gestational diabetes (GDM)",
    "Thyroid": "Thyroid dysfunction",
}

MODULE_ORDER = ["Anemia", "Preeclampsia", "GDM", "Thyroid"]


def _pct(p: Optional[float]) -> str:
    if p is None:
        return "—"
    return f"{100.0 * float(p):.1f}%"


def _status_label(status: str) -> str:
    mapping = {
        "ok": "Complete",
        "pending": "Pending",
        "missing_features": "Incomplete input",
        "missing_model": "Model unavailable",
    }
    return mapping.get(status, status or "—")


def _status_class(status: str) -> str:
    if status == "ok":
        return "mh-status-ok"
    if status in ("pending", "missing_model"):
        return "mh-status-warn"
    if status == "missing_features":
        return "mh-status-miss"
    return ""


def render_condition_card(key: str, result: Dict[str, Any]) -> None:
    """Render one condition module as a polished card."""
    title = CARD_TITLES.get(key, result.get("condition", key))
    prediction = result.get("prediction") or "—"
    status = result.get("status") or ""
    proba = result.get("probability")
    class_probs: Dict[str, float] = result.get("class_probabilities") or {}
    factors: List[Dict[str, Any]] = result.get("important_factors") or []
    notes = result.get("notes") or ""

    st.markdown('<div class="mh-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="mh-card-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="mh-prediction">{prediction}</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.metric(
            label="Predicted-class probability",
            value=_pct(proba) if status == "ok" else "—",
        )
    with c2:
        st.markdown(
            f'<p class="mh-subtle">Status: '
            f'<span class="{_status_class(status)}">{_status_label(status)}</span></p>',
            unsafe_allow_html=True,
        )

    if class_probs and status == "ok":
        st.caption("Class probabilities")
        # Values as percentages for a readable bar chart
        chart_df = pd.DataFrame(
            {
                "Class": list(class_probs.keys()),
                "Probability (%)": [round(100.0 * float(v), 2) for v in class_probs.values()],
            }
        ).set_index("Class")
        st.bar_chart(chart_df, height=160, use_container_width=True)

    if factors and status == "ok":
        st.caption("Top contributing features")
        rows = []
        for f in factors[:5]:
            rows.append(
                {
                    "Feature": str(f.get("feature", "")),
                    "Contribution": f.get("shap_contribution"),
                }
            )
        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            use_container_width=True,
            height=min(38 + 35 * len(rows), 220),
        )

    if notes:
        st.caption(notes)

    with st.expander("View technical details"):
        st.json(
            {
                "condition": result.get("condition"),
                "prediction": result.get("prediction"),
                "probability": result.get("probability"),
                "class_probabilities": class_probs,
                "important_factors": factors,
                "status": status,
                "notes": notes,
            }
        )

    st.markdown("</div>", unsafe_allow_html=True)


def render_profile_summary(summary: Dict[str, Any]) -> None:
    st.subheader("Maternal profile summary")
    st.caption(
        "Independent module outcomes only. No combined clinical risk score is computed."
    )

    flagged = summary.get("flagged_conditions") or []
    typical = summary.get("within_typical_range_or_proxy_low") or []
    pending = summary.get("pending_or_incomplete") or []

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown('<div class="mh-section-label">Flagged modules</div>', unsafe_allow_html=True)
        if flagged:
            for item in flagged:
                st.markdown(f"- {item}")
        else:
            st.markdown("*None flagged by current module outputs*")
    with col_b:
        st.markdown(
            '<div class="mh-section-label">Within typical / proxy-low</div>',
            unsafe_allow_html=True,
        )
        if typical:
            for item in typical:
                st.markdown(f"- {item}")
        else:
            st.markdown("*None*")
    with col_c:
        st.markdown(
            '<div class="mh-section-label">Pending / incomplete</div>',
            unsafe_allow_html=True,
        )
        if pending:
            for item in pending:
                st.markdown(f"- {item}")
        else:
            st.markdown("*None*")

    if summary.get("note"):
        st.info(summary["note"])


def render_probability_comparison(modules: Dict[str, Any]) -> None:
    """Horizontal comparison of predicted-class probability across completed modules."""
    rows = []
    for key in MODULE_ORDER:
        r = modules.get(key) or {}
        if r.get("status") == "ok" and r.get("probability") is not None:
            rows.append(
                {
                    "Module": CARD_TITLES.get(key, key),
                    "Predicted-class probability (%)": round(
                        100.0 * float(r["probability"]), 2
                    ),
                }
            )
    if not rows:
        return

    st.subheader("Module probability comparison")
    st.caption(
        "Predicted-class probability for each completed module "
        "(not an overall maternal risk score)."
    )
    df = pd.DataFrame(rows).set_index("Module")
    st.bar_chart(df, height=220, use_container_width=True)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Maternal Health Profile")
st.caption(
    "Student AI/ML decision-support prototype · Independent condition modules · Not a diagnostic device"
)

st.markdown(
    """
    <div class="mh-limit-box">
    <strong>Limitations (read before interpreting results)</strong><br/>
    This system does <em>not</em> diagnose disease, prescribe treatment, or replace clinical judgment.
    No combined clinical “overall pregnancy risk” percentage is produced.<br/>
    • <strong>Anemia</strong> — Hb / PCV / who_pred excluded (target leakage).<br/>
    • <strong>Preeclampsia</strong> — RiskLevel-derived <em>proxy</em> label, not confirmed clinical preeclampsia.<br/>
    • <strong>GDM</strong> — model trained on <em>synthetic</em> data; not clinically validated.<br/>
    • <strong>Thyroid</strong> — later SCH-consistent risk from first-trimester labs; single-center data; modest discrimination.
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar — patient inputs (functionality preserved)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Patient information")
    st.caption("Enter available values. Leave fields at defaults if unknown.")

    if st.button("Load sample patient", use_container_width=True):
        st.session_state["use_sample"] = True

    sample = sample_patient() if st.session_state.get("use_sample") else {}

    st.subheader("Demographics")
    age = st.number_input(
        "Age (years)",
        min_value=0.0,
        max_value=60.0,
        value=float(sample.get("age") or 28),
        step=1.0,
    )
    ethnicity_opts = ["Asian", "Caucasian", "African", "Hispanic", "Other"]
    eth_default = sample.get("ethnicity", "Asian")
    eth_index = ethnicity_opts.index(eth_default) if eth_default in ethnicity_opts else 0
    ethnicity = st.selectbox("Ethnicity", ethnicity_opts, index=eth_index)

    st.subheader("Pregnancy")
    gestational_age = st.number_input(
        "Gestational age / POG (weeks)",
        min_value=0.0,
        max_value=45.0,
        value=float(sample.get("gestational_age") or 24.0),
        step=0.5,
    )
    booking_ga = st.number_input(
        "Booking gestational age (weeks)",
        min_value=0.0,
        max_value=45.0,
        value=float(sample.get("booking_gestational_age") or 10.0),
        step=0.5,
    )
    bmi = st.number_input(
        "Pre-pregnancy BMI",
        min_value=10.0,
        max_value=60.0,
        value=float(sample.get("pre_pregnancy_bmi") or 26.5),
        step=0.1,
    )
    obs_score = st.number_input(
        "Obs Score L",
        min_value=0,
        max_value=10,
        value=int(sample.get("obs_score_l") or 1),
    )
    lmp_known = st.selectbox("LMP known", ["Yes", "No"], index=0)
    usg = st.selectbox("USG", ["Yes", "No"], index=0)

    st.subheader("Vitals")
    sbp = st.number_input(
        "Systolic BP",
        min_value=60,
        max_value=250,
        value=int(sample.get("systolic_bp") or 135),
    )
    dbp = st.number_input(
        "Diastolic BP",
        min_value=30,
        max_value=150,
        value=int(sample.get("diastolic_bp") or 88),
    )
    pulse = st.number_input(
        "Pulse / heart rate",
        min_value=40,
        max_value=200,
        value=int(sample.get("pulse") or 86),
    )
    body_temp = st.number_input(
        "Body temperature (°F)",
        min_value=90.0,
        max_value=110.0,
        value=float(sample.get("body_temp") or 98.2),
        step=0.1,
    )

    st.subheader("Glucose / GDM risk")
    blood_sugar = st.number_input(
        "Blood sugar (BS, preeclampsia unit)",
        min_value=0.0,
        max_value=50.0,
        value=float(sample.get("blood_sugar") or 12.0),
        step=0.1,
    )
    early_rbs = st.number_input(
        "Early RBS (mg/dL)",
        min_value=0.0,
        max_value=400.0,
        value=float(sample.get("early_rbs_mgdl") or 118.0),
        step=1.0,
    )
    early_ppbs = st.number_input(
        "Early PPBS (mg/dL)",
        min_value=0.0,
        max_value=400.0,
        value=float(sample.get("early_ppbs_mgdl") or 145.0),
        step=1.0,
    )
    early_hba1c = st.number_input(
        "Early HbA1c (%)",
        min_value=3.0,
        max_value=15.0,
        value=float(sample.get("early_hba1c_percent") or 5.6),
        step=0.1,
    )
    early_ogtt_performed = st.selectbox("Early OGTT performed", [0, 1], index=0)
    family_history_dm = st.selectbox("Family history of DM", [0, 1], index=0)
    previous_gdm = st.selectbox("Previous GDM", [0, 1], index=0)
    pcos = st.selectbox("PCOS", [0, 1], index=0)
    previous_macrosomia = st.selectbox("Previous macrosomia", [0, 1], index=0)

    st.subheader("Hematology (Hb/PCV excluded)")
    trbc = st.number_input(
        "TRBC (10^6/µL)",
        min_value=0.0,
        max_value=8.0,
        value=float(sample.get("trbc") or 3.9),
        step=0.01,
    )
    mcv = st.number_input(
        "MCV (fL)",
        min_value=40.0,
        max_value=140.0,
        value=float(sample.get("mcv") or 78.0),
        step=0.1,
    )
    mch = st.number_input(
        "MCH (pg)",
        min_value=10.0,
        max_value=40.0,
        value=float(sample.get("mch") or 25.0),
        step=0.1,
    )
    mchc = st.number_input(
        "MCHC (g/dL)",
        min_value=20.0,
        max_value=40.0,
        value=float(sample.get("mchc") or 31.5),
        step=0.1,
    )
    rdw = st.number_input(
        "RDW (%)",
        min_value=5.0,
        max_value=30.0,
        value=float(sample.get("rdw") or 15.2),
        step=0.1,
    )
    ret_count = st.number_input(
        "Reticulocyte count",
        min_value=0.0,
        max_value=5.0,
        value=float(sample.get("reticulocyte_count") or 0.05),
        step=0.01,
    )
    ret_pct = st.number_input(
        "Reticulocyte %",
        min_value=0.0,
        max_value=20.0,
        value=float(sample.get("reticulocyte_pct") or 1.2),
        step=0.1,
    )
    serum_iron = st.number_input(
        "Serum iron (µg/dL)",
        min_value=0.0,
        max_value=300.0,
        value=float(sample.get("serum_iron") or 55.0),
        step=1.0,
    )
    tibc = st.number_input(
        "TIBC (µg/dL)",
        min_value=0.0,
        max_value=600.0,
        value=float(sample.get("tibc") or 420.0),
        step=1.0,
    )
    ts = st.number_input(
        "Transferrin saturation (%)",
        min_value=0.0,
        max_value=100.0,
        value=float(sample.get("transferrin_saturation") or 13.1),
        step=0.1,
    )
    tbil = st.number_input(
        "Total bilirubin",
        min_value=0.0,
        max_value=10.0,
        value=float(sample.get("total_bilirubin") or 0.5),
        step=0.01,
    )
    dbil = st.number_input(
        "Direct bilirubin",
        min_value=0.0,
        max_value=10.0,
        value=float(sample.get("direct_bilirubin") or 0.15),
        step=0.01,
    )
    ibil = st.number_input(
        "Indirect bilirubin",
        min_value=0.0,
        max_value=10.0,
        value=float(sample.get("indirect_bilirubin") or 0.35),
        step=0.01,
    )
    urea = st.number_input(
        "Urea (mg/dL)",
        min_value=0.0,
        max_value=100.0,
        value=float(sample.get("urea") or 18.0),
        step=0.1,
    )
    creat = st.number_input(
        "Creatinine (mg/dL)",
        min_value=0.0,
        max_value=10.0,
        value=float(sample.get("creatinine") or 0.7),
        step=0.01,
    )

    st.subheader("Clinical signs")
    pallor = st.selectbox(
        "Pallor",
        ["Yes", "No"],
        index=0 if sample.get("pallor") == "Yes" else 1,
    )
    edema = st.selectbox("Edema", ["Yes", "No"], index=1)
    icterus = st.selectbox("Icterus", ["Yes", "No"], index=1)
    diet = st.selectbox("Dietary habits", ["Vegetarian", "Non-Vegetarian"], index=0)
    iron_supp = st.selectbox("History of iron supplementation", ["Yes", "No"], index=1)
    transfusion = st.selectbox("History of blood transfusion", ["Yes", "No"], index=1)
    fam_hb = st.selectbox("Family history of hemoglobinopathy", ["Yes", "No"], index=1)
    allergy = st.selectbox("History of allergy", ["Yes", "No"], index=1)

    st.subheader("Thyroid baseline (first trimester)")
    tsh_baseline = st.number_input(
        "TSH baseline",
        min_value=0.0,
        max_value=50.0,
        value=float(sample.get("tsh_baseline") or 2.8),
        step=0.1,
    )
    ft3_baseline = st.number_input(
        "FT3 baseline",
        min_value=0.0,
        max_value=30.0,
        value=float(sample.get("ft3_baseline") or 4.6),
        step=0.1,
    )
    ft4_baseline = st.number_input(
        "FT4 baseline",
        min_value=0.0,
        max_value=40.0,
        value=float(sample.get("ft4_baseline") or 14.5),
        step=0.1,
    )
    tpo_baseline = st.number_input(
        "TPO baseline",
        min_value=0.0,
        max_value=1000.0,
        value=float(sample.get("tpo_baseline") or 12.0),
        step=1.0,
    )
    household_income = st.selectbox(
        "Household income category",
        [1, 2, 3],
        index=int(sample.get("household_income") or 2) - 1,
    )
    parity = st.number_input(
        "Parity",
        min_value=0,
        max_value=15,
        value=int(sample.get("parity") or 1),
    )
    family_history_diabetes = st.selectbox(
        "Family history of diabetes (thyroid schema)",
        [0, 1],
        index=int(sample.get("family_history_diabetes") or 0),
    )
    smoking_exposure = st.selectbox(
        "Smoking / secondhand exposure",
        [0, 1],
        index=int(sample.get("smoking_exposure") or 0),
    )
    alcohol_consumption = st.selectbox(
        "Alcohol consumption",
        [0, 1],
        index=int(sample.get("alcohol_consumption") or 0),
    )
    folic_acid_supplementation = st.selectbox(
        "Folic acid supplementation",
        [0, 1, 2],
        index=int(sample.get("folic_acid_supplementation") or 2),
    )
    vd_supplementation = st.selectbox(
        "Vitamin D supplementation",
        [0, 1, 2],
        index=int(sample.get("vd_supplementation") or 1),
    )

    run = st.button("Analyze maternal health", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Build patient_data (same keys as before + thyroid fields already in sample)
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
    "tsh_baseline": tsh_baseline,
    "ft3_baseline": ft3_baseline,
    "ft4_baseline": ft4_baseline,
    "tpo_baseline": tpo_baseline,
    "household_income": household_income,
    "parity": parity,
    "family_history_diabetes": family_history_diabetes,
    "smoking_exposure": smoking_exposure,
    "alcohol_consumption": alcohol_consumption,
    "folic_acid_supplementation": folic_acid_supplementation,
    "vd_supplementation": vd_supplementation,
}

# ---------------------------------------------------------------------------
# Main results area
# ---------------------------------------------------------------------------
if run:
    with st.spinner("Running condition modules…"):
        profile = analyze_patient(patient_data)

    modules = profile["modules"]
    summary = profile["summary"]

    render_profile_summary(summary)
    st.divider()

    st.subheader("Condition modules")
    row1 = st.columns(2)
    row2 = st.columns(2)
    column_slots = [row1[0], row1[1], row2[0], row2[1]]
    for slot, name in zip(column_slots, MODULE_ORDER):
        with slot:
            render_condition_card(name, modules[name])

    st.divider()
    render_probability_comparison(modules)

    st.divider()
    st.warning(profile.get("disclaimer") or "")

else:
    st.info(
        "Enter patient information in the sidebar (or load the sample patient), "
        "then select **Analyze maternal health**."
    )

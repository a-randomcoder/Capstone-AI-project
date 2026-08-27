"""
NineFolds — Streamlit companion interface

A calm, maternal-centered decision-support prototype.
NOT a medical diagnostic system. Does not replace clinical judgment.

Backend integration is unchanged:
  analyze_patient() → generate_maternal_profile() → four independent modules
"""

from __future__ import annotations

import re

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
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="NineFolds",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Theme — soft baby-pink undertone + sage + warm brown
# Fixes: empty top bar, sidebar collapse control must stay usable
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;500;600;700&family=Literata:opsz,wght@7..72,500;7..72,600;7..72,700&display=swap');

    :root {
        --creme: #EEE4DA;
        --dusty: #C8A49F;
        --burgundy: #4D0E13;
        --black: #000000;
    }

    html, body, [class*="css"] {
        font-family: 'Source Sans 3', system-ui, -apple-system, sans-serif;
        color: #000000 !important;
        font-size: 17px;
    }
    .stApp { background: #EEE4DA !important; }

    header[data-testid="stHeader"] {
        background-color: #EEE4DA !important;
        background-image: none !important;
        height: 2.5rem !important;
        min-height: 2.5rem !important;
        border-bottom: none !important;
        box-shadow: none !important;
    }
    header[data-testid="stHeader"] button,
    header[data-testid="stHeader"] svg { color: #000000 !important; }
    [data-testid="stDecoration"] {
        background: transparent !important;
        height: 0 !important;
        display: none !important;
    }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 1080px;
        padding-left: 1.5rem;
        padding-right: 1.5rem;
    }
    div[data-testid="stAppViewContainer"] > .main { background: #EEE4DA !important; }
    section.main > div { padding-top: 0.5rem !important; }

    section[data-testid="stSidebar"] {
        background: #EEE4DA !important;
        border-right: 1px solid #000000 !important;
    }
    section[data-testid="stSidebar"] > div { background: #EEE4DA !important; }
    section[data-testid="stSidebar"] .block-container { padding-top: 1.1rem; }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label { color: #000000 !important; }
    section[data-testid="stSidebar"] .stCaption,
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: #000000 !important; opacity: 0.85; font-size: 0.95rem !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label {
        color: #000000 !important; font-weight: 500 !important; font-size: 1.05rem !important;
        padding: 0.4rem 0.55rem !important; border-radius: 8px;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover { background: #C8A49F !important; }
    section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
        background: #C8A49F !important; color: #4D0E13 !important; font-weight: 600 !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] div[data-testid="stMarkdownContainer"] p {
        color: #000000 !important; font-size: 1.05rem !important;
    }
    section[data-testid="stSidebar"] label p { color: #000000 !important; font-weight: 500 !important; }
    section[data-testid="stSidebar"] input,
    section[data-testid="stSidebar"] select { color: #000000 !important; }

    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapseButton"],
    button[kind="header"],
    button[data-testid="baseButton-header"] {
        visibility: visible !important; opacity: 1 !important;
        pointer-events: auto !important; z-index: 1000 !important;
    }
    [data-testid="collapsedControl"] {
        color: #000000 !important; background: #EEE4DA !important;
        border: 1px solid #000000 !important; border-radius: 8px !important;
    }
    [data-testid="collapsedControl"] svg,
    [data-testid="stSidebarCollapseButton"] svg {
        fill: #000000 !important; color: #000000 !important; stroke: #000000 !important;
    }

    h1, h2, h3 {
        font-family: 'Literata', Georgia, serif !important;
        color: #4D0E13 !important; font-weight: 600 !important;
    }
    h1 { font-size: 2.05rem !important; }
    h2 { font-size: 1.55rem !important; }
    h3 { font-size: 1.25rem !important; }
    p, span, li, label { color: #000000; font-size: 1.05rem; }

    .mdt-card {
        background: #EEE4DA; border: 1px solid #000000; border-radius: 14px;
        padding: 1.35rem 1.5rem; margin-bottom: 1.1rem; box-shadow: none;
    }
    .mdt-card-soft {
        background: #C8A49F; border: 1px solid #000000; border-radius: 14px;
        padding: 1.25rem 1.4rem; margin-bottom: 1.1rem;
    }
    .mdt-card-discuss {
        background: #EEE4DA; border: 1px solid #000000; border-left: 4px solid #4D0E13;
        border-radius: 12px; padding: 1.1rem 1.3rem; margin-bottom: 0.85rem;
    }
    .mdt-card-steady {
        background: #EEE4DA; border: 1px solid #000000; border-left: 4px solid #C8A49F;
        border-radius: 12px; padding: 1.1rem 1.3rem; margin-bottom: 0.85rem;
    }
    .mdt-eyebrow {
        font-size: 0.85rem; letter-spacing: 0.08em; text-transform: uppercase;
        color: #000000 !important; opacity: 0.75; font-weight: 700; margin-bottom: 0.4rem;
    }
    .mdt-title {
        font-family: 'Literata', Georgia, serif; font-size: 1.5rem;
        color: #4D0E13 !important; font-weight: 600; line-height: 1.3; margin-bottom: 0.45rem;
    }
    .mdt-body { color: #000000 !important; font-size: 1.1rem; line-height: 1.6; }
    .mdt-muted { color: #000000 !important; opacity: 0.8; font-size: 1.02rem; line-height: 1.55; }
    .mdt-greeting {
        font-family: 'Literata', Georgia, serif; font-size: 2.15rem;
        color: #4D0E13 !important; font-weight: 600; line-height: 1.25; margin-bottom: 0.4rem;
    }
    .mdt-tagline { color: #000000 !important; opacity: 0.85; font-size: 1.2rem; margin-bottom: 1.25rem; }

    .mdt-bar-track { background: #C8A49F; border-radius: 999px; height: 11px; width: 100%; overflow: hidden; margin: 0.4rem 0 0.2rem 0; }
    .mdt-bar-fill, .mdt-bar-fill-discuss { background: #4D0E13; height: 100%; border-radius: 999px; }
    .mdt-bar-label { display: flex; justify-content: space-between; font-size: 0.98rem; color: #000000 !important; }

    .mdt-factor-row { display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.5rem; }
    .mdt-factor-name { min-width: 150px; font-size: 1rem; color: #000000 !important; font-weight: 500; }
    .mdt-factor-track { flex: 1; background: #C8A49F; border-radius: 999px; height: 9px; overflow: hidden; }
    .mdt-factor-fill { background: #4D0E13; height: 100%; border-radius: 999px; }

    .mdt-chip, .mdt-chip-earth {
        display: inline-block; background: #C8A49F; color: #000000 !important;
        border: 1px solid #000000; border-radius: 999px; padding: 0.32rem 0.9rem;
        font-size: 0.95rem; font-weight: 600; margin-right: 0.45rem; margin-bottom: 0.45rem;
    }
    .mdt-limit, .mdt-proto {
        background: #C8A49F; border: 1px solid #000000; border-radius: 12px;
        padding: 1rem 1.2rem; font-size: 1rem; color: #000000 !important; line-height: 1.55; margin: 1rem 0;
    }
    .mdt-proto { margin-bottom: 1.1rem; }

    .stButton > button {
        border-radius: 10px !important; font-weight: 600 !important;
        font-family: 'Source Sans 3', system-ui, sans-serif !important; font-size: 1.05rem !important;
        border: 1px solid #000000 !important; color: #000000 !important;
        background: #EEE4DA !important; padding: 0.45rem 1rem !important; box-shadow: none !important;
    }
    .stButton > button:hover {
        border-color: #4D0E13 !important; color: #4D0E13 !important; background: #C8A49F !important;
    }
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="baseButton-primary"],
    button[kind="primary"] {
        background-color: #4D0E13 !important; border-color: #4D0E13 !important; color: #EEE4DA !important;
    }
    .stButton > button[kind="primary"]:hover,
    button[kind="primary"]:hover {
        background-color: #4D0E13 !important; border-color: #000000 !important; color: #EEE4DA !important; opacity: 0.92;
    }
    .stButton > button[kind="primary"] p,
    .stButton > button[kind="primary"] span,
    button[kind="primary"] p,
    button[kind="primary"] span { color: #EEE4DA !important; }

    div[data-testid="stExpander"] { background: #EEE4DA; border: 1px solid #000000; border-radius: 12px; }
    div[data-testid="stExpander"] summary,
    div[data-testid="stExpander"] p { color: #000000 !important; font-size: 1.05rem !important; }
    div[data-testid="stMetricValue"] { font-size: 1.3rem !important; color: #000000 !important; font-weight: 600; }
    div[data-testid="stMetricLabel"] { color: #000000 !important; font-size: 0.95rem !important; }
    hr { border-color: #000000; margin: 1.4rem 0; }
    .stMarkdown, .stMarkdown p, .stMarkdown li { color: #000000 !important; font-size: 1.08rem; }
    .stCaption, [data-testid="stCaptionContainer"] { color: #000000 !important; opacity: 0.85; font-size: 1rem !important; }

    .mdt-progress-track {
        background: #C8A49F; border-radius: 999px; height: 9px; width: 100%; max-width: 300px;
        overflow: hidden; margin: 0.45rem 0 0.85rem 0;
    }
    .mdt-progress-fill { background: #4D0E13; height: 100%; border-radius: 999px; }
    .mdt-measures { display: flex; flex-wrap: wrap; gap: 1.1rem 1.85rem; margin-top: 0.55rem; }
    .mdt-measure-k {
        font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.06em;
        color: #000000 !important; opacity: 0.75; font-weight: 700;
    }
    .mdt-measure-v { font-size: 1.2rem; font-weight: 600; color: #000000 !important; }

    .mdt-side-brand {
        font-family: 'Literata', Georgia, serif; font-size: 1.35rem; font-weight: 600;
        color: #4D0E13 !important; margin-bottom: 0.2rem;
    }
    .mdt-side-tag { font-size: 0.98rem; color: #000000 !important; opacity: 0.85; margin-bottom: 0.85rem; }
    .mdt-side-label {
        font-size: 0.8rem; letter-spacing: 0.08em; text-transform: uppercase; font-weight: 700;
        color: #000000 !important; opacity: 0.7; margin: 0.9rem 0 0.4rem 0;
    }
    div[data-testid="stAlert"] {
        background: #C8A49F !important; color: #000000 !important; border: 1px solid #000000 !important;
    }

    .mdt-record {
        background: #FFFDF8;
        border: 1px solid #000000;
        border-radius: 12px;
        padding: 0.25rem 1.2rem;
        margin: 0.55rem 0 1rem 0;
    }
    .mdt-record-row {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        gap: 1.5rem;
        padding: 0.72rem 0;
        border-bottom: 1px solid #000000;
    }
    .mdt-record-row:last-child {
        border-bottom: none;
    }
    .mdt-record-label {
        color: #000000 !important;
        font-weight: 500;
        font-size: 1.05rem;
        text-align: left;
    }
    .mdt-record-value {
        color: #000000 !important;
        font-weight: 600;
        font-size: 1.05rem;
        text-align: right;
        white-space: nowrap;
    }

    </style>
    """,
    unsafe_allow_html=True,
)




# ---------------------------------------------------------------------------
# Constants & helpers (display only — backend keys unchanged)
# ---------------------------------------------------------------------------
MODULE_ORDER = ["Anemia", "Preeclampsia", "GDM", "Thyroid"]

MODULE_DISPLAY = {
    "Anemia": {
        "title": "Anemia",
        "subtitle": "Blood-related pattern",
        "plain": {
            "Normal": "Your blood-related measurements currently fall in a pattern the model associates with the normal range.",
            "Mild": "Your blood-related measurements show a mild pattern. This is something many pregnancies manage with guidance from a clinician.",
            "Moderate": "Your blood-related measurements show a moderate pattern. It is worth discussing with your care team so they can interpret it in context.",
            "Severe": "Your blood-related measurements show a more pronounced pattern. Please share this with your clinician promptly for their interpretation.",
        },
        "limitation": "Hb, PCV, and WHO rule-based labels are excluded from the model because they leak the training target. This module does not diagnose anemia.",
    },
    "Preeclampsia": {
        "title": "Preeclampsia-related indicators",
        "subtitle": "Blood pressure and related signals",
        "plain": {
            "high risk (proxy)": "Some measurements in this record align with a higher-risk pattern on a proxy label (not confirmed clinical preeclampsia). Please review these numbers with your clinician.",
            "low/mid risk (proxy)": "Current measurements sit in a lower-to-mid pattern on the proxy label. This is not a clinical clearance — only your care team can interpret risk fully.",
        },
        "limitation": "The model uses a RiskLevel-derived proxy from a public dataset. It is not confirmed clinical preeclampsia and is not a diagnosis.",
    },
    "GDM": {
        "title": "Gestational diabetes (GDM)",
        "subtitle": "Glucose-related pattern",
        "plain": {
            "GDM likely (synthetic model)": "The synthetic model places this profile toward a gestational-diabetes-like pattern. Treat this as research output only — it is not clinically validated.",
            "GDM unlikely (synthetic model)": "The synthetic model does not place this profile toward an elevated GDM pattern. This is research output only and is not clinically validated.",
        },
        "limitation": "Trained on synthetic data (~65% prevalence in training set). Not clinically validated. Not a diagnostic test.",
    },
    "Thyroid": {
        "title": "Thyroid",
        "subtitle": "Later dysfunction indication",
        "plain": {
            "later thyroid dysfunction risk elevated": "Based on first-trimester baseline labs, the model suggests a higher chance of later SCH-consistent findings. Discuss thyroid follow-up with your clinician.",
            "later thyroid dysfunction risk not elevated": "Based on first-trimester baseline labs, the model does not suggest an elevated chance of later SCH-consistent findings. Routine clinical follow-up still applies.",
        },
        "limitation": "Looks at patterns related to later thyroid findings from baseline information. Not a diagnosis.",
    },
}


def _pct(p: Optional[float]) -> str:
    if p is None:
        return "—"
    return f"{100.0 * float(p):.1f}%"


def _is_discuss(key: str, result: Dict[str, Any]) -> bool:
    """Whether this module result is worth discussing (flagged side)."""
    status = result.get("status") or ""
    if status != "ok":
        return False
    pred = str(result.get("prediction") or "").lower()
    normalish = {
        "normal",
        "no gdm",
        "gdm unlikely (synthetic model)",
        "low/mid risk (proxy)",
        "later thyroid dysfunction risk not elevated",
    }
    return pred not in normalish


def _plain_text(key: str, result: Dict[str, Any]) -> str:
    pred = str(result.get("prediction") or "")
    mapping = MODULE_DISPLAY.get(key, {}).get("plain", {})
    if pred in mapping:
        return mapping[pred]
    if result.get("status") == "missing_features":
        return "Some required information is missing, so this module could not complete a full reading."
    if result.get("status") == "missing_model":
        return "This module’s model file is not available in the current environment."
    return result.get("notes") or pred or "No reading available yet."


def _friendly_prediction(key: str, result: Dict[str, Any]) -> str:
    pred = str(result.get("prediction") or "—")
    friendly = {
        "Moderate": "Moderate pattern",
        "Mild": "Mild pattern",
        "Severe": "More pronounced pattern",
        "Normal": "Within typical pattern",
        "high risk (proxy)": "Worth discussing with your clinician",
        "low/mid risk (proxy)": "Lower-to-mid pattern on proxy label",
        "GDM likely (synthetic model)": "Elevated pattern (synthetic model)",
        "GDM unlikely (synthetic model)": "No elevated pattern (synthetic model)",
        "later thyroid dysfunction risk elevated": "Later-dysfunction indication elevated",
        "later thyroid dysfunction risk not elevated": "Later-dysfunction indication not elevated",
        "Insufficient input": "Incomplete information",
        "Model missing": "Model unavailable",
    }
    return friendly.get(pred, pred)


def _bar_html(label: str, fraction: float, discuss: bool = False) -> str:
    pct = max(0.0, min(100.0, 100.0 * float(fraction)))
    cls = "mdt-bar-fill-discuss" if discuss else "mdt-bar-fill"
    return f"""
    <div class="mdt-bar-label"><span>{label}</span><span>{pct:.1f}%</span></div>
    <div class="mdt-bar-track"><div class="{cls}" style="width:{pct:.1f}%"></div></div>
    """


def _factor_bars_html(factors: List[Dict[str, Any]], top_k: int = 5) -> str:
    if not factors:
        return ""
    vals = [abs(float(f.get("shap_contribution") or 0)) for f in factors[:top_k]]
    mx = max(vals) if vals else 1.0
    if mx <= 0:
        mx = 1.0
    parts = []
    for f in factors[:top_k]:
        name = str(f.get("feature", "")).replace("num__", "").replace("cat__", "")
        val = abs(float(f.get("shap_contribution") or 0))
        pct = 100.0 * val / mx
        parts.append(
            f"""
            <div class="mdt-factor-row">
              <div class="mdt-factor-name">{name}</div>
              <div class="mdt-factor-track"><div class="mdt-factor-fill" style="width:{pct:.1f}%"></div></div>
            </div>
            """
        )
    return "".join(parts)


# ---------------------------------------------------------------------------
# Session defaults
# ---------------------------------------------------------------------------
if "page" not in st.session_state:
    st.session_state["page"] = "Home"
if "profile" not in st.session_state:
    st.session_state["profile"] = None
if "patient_name" not in st.session_state:
    st.session_state["patient_name"] = " "
if "role" not in st.session_state:
    st.session_state["role"] = "Patient"


# ---------------------------------------------------------------------------
# Update-information form widget keys & defaults
# ---------------------------------------------------------------------------
# Keys must match collect_patient_inputs(). Clearing sets these before widgets render.
FORM_DEFAULTS: Dict[str, Any] = {
    "fi_name": "",
    "fi_age": 0.0,
    "fi_ethnicity": "Asian",
    "fi_gestational_age": 0.0,
    "fi_booking_ga": 0.0,
    "fi_bmi": 0.0,
    "fi_obs_score": 0,
    "fi_lmp_known": "Yes",
    "fi_usg": "Yes",
    "fi_sbp": 0,
    "fi_dbp": 0,
    "fi_pulse": 0,
    "fi_body_temp": 0.0,
    "fi_blood_sugar": 0.0,
    "fi_early_rbs": 0.0,
    "fi_early_ppbs": 0.0,
    "fi_early_hba1c": 0.0,
    "fi_early_ogtt": 0,
    "fi_family_history_dm": 0,
    "fi_previous_gdm": 0,
    "fi_pcos": 0,
    "fi_previous_macrosomia": 0,
    "fi_trbc": 0.0,
    "fi_mcv": 0.0,
    "fi_mch": 0.0,
    "fi_mchc": 0.0,
    "fi_rdw": 0.0,
    "fi_ret_count": 0.0,
    "fi_ret_pct": 0.0,
    "fi_serum_iron": 0.0,
    "fi_tibc": 0.0,
    "fi_ts": 0.0,
    "fi_tbil": 0.0,
    "fi_dbil": 0.0,
    "fi_ibil": 0.0,
    "fi_urea": 0.0,
    "fi_creat": 0.0,
    "fi_pallor": "No",
    "fi_edema": "No",
    "fi_icterus": "No",
    "fi_diet": "Vegetarian",
    "fi_iron_supp": "No",
    "fi_transfusion": "No",
    "fi_fam_hb": "No",
    "fi_allergy": "No",
    "fi_tsh": 0.0,
    "fi_ft3": 0.0,
    "fi_ft4": 0.0,
    "fi_tpo": 0.0,
    "fi_household_income": 1,
    "fi_parity": 0,
    "fi_family_history_diabetes": 0,
    "fi_smoking": 0,
    "fi_alcohol": 0,
    "fi_folic": 0,
    "fi_vd": 0,
}


def _ensure_form_defaults() -> None:
    for k, v in FORM_DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v


def clear_patient_information() -> None:
    """Reset every Update Information form widget to its original default."""
    st.session_state["use_sample"] = False
    for k, v in FORM_DEFAULTS.items():
        st.session_state[k] = v
    st.session_state["patient_name"] = " "


def load_sample_information() -> None:
    """Fill form widget keys from sample_patient()."""
    st.session_state["use_sample"] = True
    s = sample_patient()
    st.session_state["fi_name"] = ""
    st.session_state["fi_age"] = float(s.get("age") or FORM_DEFAULTS["fi_age"])
    eth = s.get("ethnicity", "Asian")
    st.session_state["fi_ethnicity"] = eth if eth in ["Asian", "Caucasian", "African", "Hispanic", "Other"] else "Asian"
    st.session_state["fi_gestational_age"] = float(s.get("gestational_age") or FORM_DEFAULTS["fi_gestational_age"])
    st.session_state["fi_booking_ga"] = float(s.get("booking_gestational_age") or FORM_DEFAULTS["fi_booking_ga"])
    st.session_state["fi_bmi"] = float(s.get("pre_pregnancy_bmi") or FORM_DEFAULTS["fi_bmi"])
    st.session_state["fi_obs_score"] = int(s.get("obs_score_l") or FORM_DEFAULTS["fi_obs_score"])
    st.session_state["fi_lmp_known"] = s.get("lmp_known") or "Yes"
    st.session_state["fi_usg"] = s.get("usg") or "Yes"
    st.session_state["fi_sbp"] = int(s.get("systolic_bp") or FORM_DEFAULTS["fi_sbp"])
    st.session_state["fi_dbp"] = int(s.get("diastolic_bp") or FORM_DEFAULTS["fi_dbp"])
    st.session_state["fi_pulse"] = int(s.get("pulse") or FORM_DEFAULTS["fi_pulse"])
    st.session_state["fi_body_temp"] = float(s.get("body_temp") or FORM_DEFAULTS["fi_body_temp"])
    st.session_state["fi_blood_sugar"] = float(s.get("blood_sugar") or FORM_DEFAULTS["fi_blood_sugar"])
    st.session_state["fi_early_rbs"] = float(s.get("early_rbs_mgdl") or FORM_DEFAULTS["fi_early_rbs"])
    st.session_state["fi_early_ppbs"] = float(s.get("early_ppbs_mgdl") or FORM_DEFAULTS["fi_early_ppbs"])
    st.session_state["fi_early_hba1c"] = float(s.get("early_hba1c_percent") or FORM_DEFAULTS["fi_early_hba1c"])
    st.session_state["fi_early_ogtt"] = int(s.get("early_ogtt_performed") or 0)
    st.session_state["fi_family_history_dm"] = int(s.get("family_history_dm") or 0)
    st.session_state["fi_previous_gdm"] = int(s.get("previous_gdm") or 0)
    st.session_state["fi_pcos"] = int(s.get("pcos") or 0)
    st.session_state["fi_previous_macrosomia"] = int(s.get("previous_macrosomia") or 0)
    st.session_state["fi_trbc"] = float(s.get("trbc") or FORM_DEFAULTS["fi_trbc"])
    st.session_state["fi_mcv"] = float(s.get("mcv") or FORM_DEFAULTS["fi_mcv"])
    st.session_state["fi_mch"] = float(s.get("mch") or FORM_DEFAULTS["fi_mch"])
    st.session_state["fi_mchc"] = float(s.get("mchc") or FORM_DEFAULTS["fi_mchc"])
    st.session_state["fi_rdw"] = float(s.get("rdw") or FORM_DEFAULTS["fi_rdw"])
    st.session_state["fi_ret_count"] = float(s.get("reticulocyte_count") or FORM_DEFAULTS["fi_ret_count"])
    st.session_state["fi_ret_pct"] = float(s.get("reticulocyte_pct") or FORM_DEFAULTS["fi_ret_pct"])
    st.session_state["fi_serum_iron"] = float(s.get("serum_iron") or FORM_DEFAULTS["fi_serum_iron"])
    st.session_state["fi_tibc"] = float(s.get("tibc") or FORM_DEFAULTS["fi_tibc"])
    st.session_state["fi_ts"] = float(s.get("transferrin_saturation") or FORM_DEFAULTS["fi_ts"])
    st.session_state["fi_tbil"] = float(s.get("total_bilirubin") or FORM_DEFAULTS["fi_tbil"])
    st.session_state["fi_dbil"] = float(s.get("direct_bilirubin") or FORM_DEFAULTS["fi_dbil"])
    st.session_state["fi_ibil"] = float(s.get("indirect_bilirubin") or FORM_DEFAULTS["fi_ibil"])
    st.session_state["fi_urea"] = float(s.get("urea") or FORM_DEFAULTS["fi_urea"])
    st.session_state["fi_creat"] = float(s.get("creatinine") or FORM_DEFAULTS["fi_creat"])
    st.session_state["fi_pallor"] = s.get("pallor") or "No"
    st.session_state["fi_edema"] = s.get("edema") or "No"
    st.session_state["fi_icterus"] = s.get("icterus") or "No"
    st.session_state["fi_diet"] = s.get("dietary_habits") or "Vegetarian"
    st.session_state["fi_iron_supp"] = s.get("history_iron_supplementation") or "No"
    st.session_state["fi_transfusion"] = s.get("history_blood_transfusion") or "No"
    st.session_state["fi_fam_hb"] = s.get("family_history_hemoglobinopathy") or "No"
    st.session_state["fi_allergy"] = s.get("history_allergy") or "No"
    st.session_state["fi_tsh"] = float(s.get("tsh_baseline") or FORM_DEFAULTS["fi_tsh"])
    st.session_state["fi_ft3"] = float(s.get("ft3_baseline") or FORM_DEFAULTS["fi_ft3"])
    st.session_state["fi_ft4"] = float(s.get("ft4_baseline") or FORM_DEFAULTS["fi_ft4"])
    st.session_state["fi_tpo"] = float(s.get("tpo_baseline") or FORM_DEFAULTS["fi_tpo"])
    st.session_state["fi_household_income"] = int(s.get("household_income") or 2)
    st.session_state["fi_parity"] = int(s.get("parity") or 1)
    st.session_state["fi_family_history_diabetes"] = int(s.get("family_history_diabetes") or 0)
    st.session_state["fi_smoking"] = int(s.get("smoking_exposure") or 0)
    st.session_state["fi_alcohol"] = int(s.get("alcohol_consumption") or 0)
    st.session_state["fi_folic"] = int(s.get("folic_acid_supplementation") or 1)
    st.session_state["fi_vd"] = int(s.get("vd_supplementation") or 1)



# ---------------------------------------------------------------------------
# Sidebar navigation + role
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        '<div class="mdt-side-brand">NineFolds</div>'
        '<div class="mdt-side-tag">Your pregnancy, your pace.</div>',
        unsafe_allow_html=True,
    )

    # Subtle admin / technical entry (evaluators only — not patient navigation)
    if st.button("⚙", key="admin_icon_btn", help="Admin / System Information"):
        st.session_state["page"] = "Admin"
        st.session_state["_force_nav"] = "Admin"
        st.rerun()

    st.markdown('<div class="mdt-side-label">Patient</div>', unsafe_allow_html=True)
    role_label = st.radio(
        "I am using this as",
        ["Patient", "Clinician (prototype)"],
        index=0 if st.session_state["role"] == "Patient" else 1,
        label_visibility="collapsed",
        key="role_radio",
    )
    new_role = "Patient" if role_label.startswith("Patient") else "Clinician"
    if new_role != st.session_state["role"]:
        st.session_state["role"] = new_role
        st.session_state["page"] = "Home" if new_role == "Patient" else "Clinician home"

    st.markdown("---")

    if st.session_state["role"] == "Patient":
        pages = [
            "Home",
            "My health",
            "Update information",
            "Health records",
            "Privacy & access",
            "About & limitations",
        ]
        st.markdown('<div class="mdt-side-label">Your space</div>', unsafe_allow_html=True)
    else:
        pages = [
            "Clinician home",
            "Patient list",
            "About & limitations",
        ]
        st.markdown('<div class="mdt-side-label">Clinician space</div>', unsafe_allow_html=True)

    # Apply pending navigation from buttons (must run before radio is created)
    if "_force_nav" in st.session_state:
        dest = st.session_state.pop("_force_nav")
        if dest == "Admin" or dest in pages:
            st.session_state["page"] = dest
            if dest in pages:
                st.session_state["nav_radio"] = dest

    current = st.session_state.get("page", pages[0])
    # Admin is outside patient/clinician nav radios — keep it without resetting
    if current not in pages and current != "Admin":
        current = pages[0]
        st.session_state["page"] = current
    if current == "Admin":
        # Keep radio on first page without changing session page
        radio_index = 0
    else:
        radio_index = pages.index(current)
    page = st.radio(
        "Navigate",
        pages,
        index=radio_index,
        label_visibility="collapsed",
        key="nav_radio",
    )
    if current != "Admin" and page != st.session_state.get("page"):
        st.session_state["page"] = page
    elif current == "Admin" and page != pages[0]:
        # User chose a real nav item while on Admin — leave Admin
        st.session_state["page"] = page

    st.markdown("---")
    st.caption(
        "Student research prototype · Not a diagnostic device · "
        "Does not replace clinical judgment."
    )


# ---------------------------------------------------------------------------
# Shared: collect patient inputs (used by Update information & analysis)
# ---------------------------------------------------------------------------
def collect_patient_inputs() -> Dict[str, Any]:
    """Form for updating information. Widget values live in st.session_state via keys."""
    _ensure_form_defaults()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### About you")
        name = st.text_input(
            "Preferred name (optional)",
            placeholder="How should we greet you?",
            key="fi_name",
        )
        if name and str(name).strip():
            st.session_state["patient_name"] = str(name).strip()

        age = st.number_input("Age (years)", min_value=0.0, max_value=60.0, step=1.0, key="fi_age")
        ethnicity = st.selectbox(
            "Ethnicity",
            ["Asian", "Caucasian", "African", "Hispanic", "Other"],
            key="fi_ethnicity",
        )

        st.markdown("##### Pregnancy")
        gestational_age = st.number_input(
            "Current gestational age (weeks)", min_value=0.0, max_value=45.0, step=0.5, key="fi_gestational_age"
        )
        booking_ga = st.number_input(
            "Booking gestational age (weeks)", min_value=0.0, max_value=45.0, step=0.5, key="fi_booking_ga"
        )
        bmi = st.number_input("Pre-pregnancy BMI", min_value=0.0, max_value=60.0, step=0.1, key="fi_bmi")
        obs_score = st.number_input("Obstetric score (L)", min_value=0, max_value=10, key="fi_obs_score")
        lmp_known = st.selectbox("LMP known", ["Yes", "No"], key="fi_lmp_known")
        usg = st.selectbox("Ultrasound (USG) available", ["Yes", "No"], key="fi_usg")

        st.markdown("##### Vitals")
        sbp = st.number_input("Systolic blood pressure", min_value=0, max_value=250, key="fi_sbp")
        dbp = st.number_input("Diastolic blood pressure", min_value=0, max_value=150, key="fi_dbp")
        pulse = st.number_input("Pulse / heart rate", min_value=0, max_value=200, key="fi_pulse")
        body_temp = st.number_input("Body temperature (°F)", min_value=0.0, max_value=110.0, step=0.1, key="fi_body_temp")

    with c2:
        st.markdown("##### Glucose-related")
        blood_sugar = st.number_input("Blood sugar (module unit)", min_value=0.0, max_value=50.0, step=0.1, key="fi_blood_sugar")
        early_rbs = st.number_input("Early random blood sugar (mg/dL)", min_value=0.0, max_value=400.0, step=1.0, key="fi_early_rbs")
        early_ppbs = st.number_input("Early post-prandial blood sugar (mg/dL)", min_value=0.0, max_value=400.0, step=1.0, key="fi_early_ppbs")
        early_hba1c = st.number_input("Early HbA1c (%)", min_value=0.0, max_value=15.0, step=0.1, key="fi_early_hba1c")
        early_ogtt_performed = st.selectbox("Early OGTT performed", [0, 1], key="fi_early_ogtt")
        family_history_dm = st.selectbox("Family history of diabetes", [0, 1], key="fi_family_history_dm")
        previous_gdm = st.selectbox("Previous gestational diabetes", [0, 1], key="fi_previous_gdm")
        pcos = st.selectbox("PCOS", [0, 1], key="fi_pcos")
        previous_macrosomia = st.selectbox("Previous macrosomia", [0, 1], key="fi_previous_macrosomia")

        st.markdown("##### Blood counts")
        st.caption("Hemoglobin and PCV are not used by the anemia model.")
        trbc = st.number_input("Total RBC (10^6/µL)", min_value=0.0, max_value=8.0, step=0.01, key="fi_trbc")
        mcv = st.number_input("MCV (fL)", min_value=0.0, max_value=140.0, step=0.1, key="fi_mcv")
        mch = st.number_input("MCH (pg)", min_value=0.0, max_value=40.0, step=0.1, key="fi_mch")
        mchc = st.number_input("MCHC (g/dL)", min_value=0.0, max_value=40.0, step=0.1, key="fi_mchc")
        rdw = st.number_input("RDW (%)", min_value=0.0, max_value=30.0, step=0.1, key="fi_rdw")
        ret_count = st.number_input("Reticulocyte count", min_value=0.0, max_value=5.0, step=0.01, key="fi_ret_count")
        ret_pct = st.number_input("Reticulocyte %", min_value=0.0, max_value=20.0, step=0.1, key="fi_ret_pct")
        serum_iron = st.number_input("Serum iron (µg/dL)", min_value=0.0, max_value=300.0, step=1.0, key="fi_serum_iron")
        tibc = st.number_input("TIBC (µg/dL)", min_value=0.0, max_value=600.0, step=1.0, key="fi_tibc")
        ts = st.number_input("Transferrin saturation (%)", min_value=0.0, max_value=100.0, step=0.1, key="fi_ts")
        tbil = st.number_input("Total bilirubin", min_value=0.0, max_value=10.0, step=0.01, key="fi_tbil")
        dbil = st.number_input("Direct bilirubin", min_value=0.0, max_value=10.0, step=0.01, key="fi_dbil")
        ibil = st.number_input("Indirect bilirubin", min_value=0.0, max_value=10.0, step=0.01, key="fi_ibil")
        urea = st.number_input("Urea (mg/dL)", min_value=0.0, max_value=100.0, step=0.1, key="fi_urea")
        creat = st.number_input("Creatinine (mg/dL)", min_value=0.0, max_value=10.0, step=0.01, key="fi_creat")

    st.markdown("##### Clinical signs & history")
    c3, c4, c5 = st.columns(3)
    with c3:
        pallor = st.selectbox("Pallor", ["Yes", "No"], key="fi_pallor")
        edema = st.selectbox("Edema", ["Yes", "No"], key="fi_edema")
        icterus = st.selectbox("Icterus", ["Yes", "No"], key="fi_icterus")
    with c4:
        diet = st.selectbox("Dietary habits", ["Vegetarian", "Non-Vegetarian"], key="fi_diet")
        iron_supp = st.selectbox("Iron supplementation history", ["Yes", "No"], key="fi_iron_supp")
        transfusion = st.selectbox("Blood transfusion history", ["Yes", "No"], key="fi_transfusion")
    with c5:
        fam_hb = st.selectbox("Family history of hemoglobinopathy", ["Yes", "No"], key="fi_fam_hb")
        allergy = st.selectbox("History of allergy", ["Yes", "No"], key="fi_allergy")

    st.markdown("##### Thyroid baseline (first trimester)")
    t1, t2, t3, t4 = st.columns(4)
    with t1:
        tsh_baseline = st.number_input("TSH", min_value=0.0, max_value=50.0, step=0.1, key="fi_tsh")
        ft3_baseline = st.number_input("FT3", min_value=0.0, max_value=30.0, step=0.1, key="fi_ft3")
    with t2:
        ft4_baseline = st.number_input("FT4", min_value=0.0, max_value=40.0, step=0.1, key="fi_ft4")
        tpo_baseline = st.number_input("TPO", min_value=0.0, max_value=1000.0, step=1.0, key="fi_tpo")
    with t3:
        household_income = st.selectbox("Household income category", [1, 2, 3], key="fi_household_income")
        parity = st.number_input("Parity", min_value=0, max_value=15, key="fi_parity")
        family_history_diabetes = st.selectbox("Family history of diabetes", [0, 1], key="fi_family_history_diabetes")
    with t4:
        smoking_exposure = st.selectbox("Smoking / secondhand exposure", [0, 1], key="fi_smoking")
        alcohol_consumption = st.selectbox("Alcohol consumption", [0, 1], key="fi_alcohol")
        folic_acid_supplementation = st.selectbox("Folic acid supplementation", [0, 1, 2], key="fi_folic")
        vd_supplementation = st.selectbox("Vitamin D supplementation", [0, 1, 2], key="fi_vd")

    return {
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



def render_condition_module(key: str, result: Dict[str, Any]) -> None:
    """Patient-facing condition card — no confidence %, SHAP, or ML jargon."""
    meta = MODULE_DISPLAY.get(key, {})
    title = meta.get("title", key)
    discuss = _is_discuss(key, result)
    friendly = _friendly_prediction(key, result)
    plain = _plain_text(key, result)
    data = st.session_state.get("last_patient_data") or {}

    card_cls = "mdt-card-discuss" if discuss else "mdt-card-steady"
    # Single complete HTML block — avoids empty bordered rectangle from split open/close divs
    st.markdown(
        f'<div class="{card_cls}">'
        f'<div class="mdt-eyebrow">{title}</div>'
        f'<div class="mdt-title">{friendly}</div>'
        f'<div class="mdt-body">{plain}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )

    with st.expander("View details"):
        st.markdown("**What this means**")
        st.write(plain)

        st.markdown("**What we noticed**")
        noticed = []
        if key == "Anemia":
            pairs = [("TRBC", "trbc"), ("MCV", "mcv"), ("MCH", "mch"), ("RDW", "rdw"), ("Serum iron", "serum_iron"), ("Pallor", "pallor")]
        elif key == "Preeclampsia":
            pairs = [("Systolic BP", "systolic_bp"), ("Diastolic BP", "diastolic_bp"), ("Pulse", "pulse"), ("Blood sugar", "blood_sugar"), ("Age", "age")]
        elif key == "GDM":
            pairs = [("Early RBS", "early_rbs_mgdl"), ("Early PPBS", "early_ppbs_mgdl"), ("Early HbA1c", "early_hba1c_percent"), ("BMI", "pre_pregnancy_bmi"), ("Previous GDM", "previous_gdm")]
        elif key == "Thyroid":
            pairs = [("TSH", "tsh_baseline"), ("FT3", "ft3_baseline"), ("FT4", "ft4_baseline"), ("TPO", "tpo_baseline"), ("Parity", "parity")]
        else:
            pairs = []
        for label, field in pairs:
            if data.get(field) not in (None, ""):
                noticed.append(f"{label}: {data.get(field)}")
        if noticed:
            for line in noticed:
                st.markdown(f"- {line}")
        else:
            st.caption("No specific measurements for this area were entered yet.")

        st.markdown("**What to discuss with your doctor**")
        if discuss:
            st.write(
                "Bring this reading and the measurements above to your next visit. "
                "Ask how it fits with your full clinical picture — this tool cannot decide that for you."
            )
        else:
            st.write(
                "You can still share this profile if helpful. A steady reading here does not replace routine prenatal care."
            )

        st.markdown("**Important limitation**")
        lim = meta.get("limitation") or ""
        lim = lim.replace("Modest discrimination (ROC-AUC ≈ 0.66). ", "")
        st.write(lim or "This is a research screening indication, not a diagnosis.")



def render_calm_summary(profile: Dict[str, Any]) -> None:
    summary = profile.get("summary") or {}
    modules = profile.get("modules") or {}
    flagged = summary.get("flagged_conditions") or []
    typical = summary.get("within_typical_range_or_proxy_low") or []
    pending = summary.get("pending_or_incomplete") or []

    st.markdown("#### Here’s where things stand")
    st.caption(
        "Independent module readings only. No combined clinical risk percentage is calculated."
    )

    if flagged:
        st.markdown('<div class="mdt-eyebrow">Things worth discussing</div>', unsafe_allow_html=True)
        for item in flagged:
            # item like "Anemia: Moderate"
            st.markdown(
                f'<div class="mdt-card-discuss"><div class="mdt-body">{item}</div></div>',
                unsafe_allow_html=True,
            )
    if typical:
        st.markdown('<div class="mdt-eyebrow">Looking steady</div>', unsafe_allow_html=True)
        for item in typical:
            st.markdown(
                f'<div class="mdt-card-steady"><div class="mdt-body">{item}</div></div>',
                unsafe_allow_html=True,
            )
    if pending:
        st.markdown('<div class="mdt-eyebrow">Information we still need</div>', unsafe_allow_html=True)
        for item in pending:
            st.markdown(f"- {item}")

    if not flagged and not typical and not pending:
        st.info("Run an assessment from **Update information** to see your profile here.")


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def page_home() -> None:
    name = str(st.session_state.get("patient_name") or "").strip()

    greeting = f"Welcome back, {name}" if name else "Welcome back"

    st.markdown(
        f'<div class="mdt-greeting">{greeting}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="mdt-tagline">Your pregnancy, your health, your pace.</div>',
        unsafe_allow_html=True,
    )

    profile = st.session_state.get("profile")
    ga = None
    if profile and st.session_state.get("last_patient_data"):
        ga = st.session_state["last_patient_data"].get("gestational_age")

    chips = []
    if ga is not None:
        chips.append(f'<span class="mdt-chip">Week {ga:g}</span>')
    chips.append('<span class="mdt-chip mdt-chip-earth">Student AI/ML prototype</span>')
    st.markdown("".join(chips), unsafe_allow_html=True)

    # Soft pregnancy progress (snapshot only — no invented longitudinal data)
    if ga is not None:
        try:
            pct = max(0.0, min(100.0, (float(ga) / 40.0) * 100.0))
        except (TypeError, ValueError):
            pct = 0.0
        data = st.session_state.get("last_patient_data") or {}
        age = data.get("age")
        bmi = data.get("pre_pregnancy_bmi")
        sbp = data.get("systolic_bp")
        dbp = data.get("diastolic_bp")
        measures = []
        if age is not None:
            measures.append(f'<div><div class="mdt-measure-k">Age</div><div class="mdt-measure-v">{age}</div></div>')
        if bmi is not None:
            measures.append(f'<div><div class="mdt-measure-k">BMI</div><div class="mdt-measure-v">{bmi}</div></div>')
        if sbp is not None and dbp is not None:
            measures.append(
                f'<div><div class="mdt-measure-k">Blood pressure</div>'
                f'<div class="mdt-measure-v">{sbp} / {dbp}</div></div>'
            )
        st.markdown(
            f'''
            <div class="mdt-card-soft">
              <div class="mdt-eyebrow">Your pregnancy</div>
              <div class="mdt-title" style="font-size:1.15rem;margin-bottom:0.25rem;">Week {ga:g}</div>
              <div class="mdt-progress-track"><div class="mdt-progress-fill" style="width:{pct:.1f}%;"></div></div>
              <div class="mdt-measures">{"".join(measures)}</div>
            </div>
            ''',
            unsafe_allow_html=True,
        )

    st.markdown("")
    st.markdown(
        '<div class="mdt-card-soft">'
        '<div class="mdt-body">'
        "This space is meant to help you understand patterns in your measurements — "
        "not to diagnose, scare, or decide for you. Take what is useful to your clinician. "
        "Leave the rest."
        "</div></div>",
        unsafe_allow_html=True,
    )

    if profile:
        render_calm_summary(profile)
        st.markdown("")
        if st.button("View my health", type="primary"):
            st.session_state["page"] = "My health"
            st.session_state["_force_nav"] = "My health"
            st.rerun()
    else:
        st.markdown(
            '<div class="mdt-card">'
            '<div class="mdt-title">When you are ready</div>'
            '<div class="mdt-body">'
            "Add or update your information, then generate a maternal health profile. "
            "You can load a sample profile first if you only want to explore the interface."
            "</div></div>",
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Update my information", use_container_width=True):
                st.session_state["page"] = "Update information"
                st.session_state["_force_nav"] = "Update information"
                st.rerun()
        with c2:
            if st.button("Load sample & view profile", use_container_width=True):
                st.session_state["use_sample"] = True
                patient = sample_patient()
                with st.spinner("Preparing your profile…"):
                    try:
                        st.session_state["profile"] = analyze_patient(patient)
                        st.session_state["last_patient_data"] = patient
                    except Exception as exc:
                        st.error("Could not generate the sample profile. Check model artifacts.")
                        st.caption(str(exc))
                        st.stop()
                st.session_state["page"] = "My health"
                st.session_state["_force_nav"] = "My health"
                st.rerun()

    st.markdown(
        """
        <div class="mdt-limit">
        <strong>Before you read any result</strong><br/>
        This is a student AI/ML decision-support prototype. It does not diagnose disease,
        prescribe treatment, or replace clinical judgment. No overall pregnancy risk score is produced.
        Anemia excludes Hb/PCV (leakage). Preeclampsia uses a proxy label. GDM uses synthetic data.
        Thyroid predicts later SCH-consistent risk from first-trimester labs.
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_my_health() -> None:
    st.markdown('<div class="mdt-greeting">My health</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="mdt-tagline">A quieter look at each area, one at a time.</div>',
        unsafe_allow_html=True,
    )

    profile = st.session_state.get("profile")
    if not profile:
        st.info(
            "No profile yet. Go to **Update information** or load a sample from **Home**."
        )
        return

    render_calm_summary(profile)
    st.markdown("---")
    st.markdown("#### Area by area")

    modules = profile.get("modules") or {}
    for key in MODULE_ORDER:
        if key in modules:
            render_condition_module(key, modules[key])

    st.markdown(
        f'<div class="mdt-limit">{profile.get("disclaimer") or ""}</div>',
        unsafe_allow_html=True,
    )


def page_update_information() -> None:
    st.markdown('<div class="mdt-greeting">Update information</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="mdt-tagline">Enter what you have. You do not need every field to explore.</div>',
        unsafe_allow_html=True,
    )

    b1, b2 = st.columns([1, 1])
    with b1:
        st.button(
            "Fill with sample values",
            use_container_width=True,
            on_click=load_sample_information,
            key="btn_fill_sample",
        )
    with b2:
        st.button(
            "Clear values",
            use_container_width=True,
            on_click=clear_patient_information,
            key="btn_clear_values",
        )

    if st.session_state.get("use_sample"):
        st.caption("Sample values are active — you can edit them before generating.")
    else:
        st.caption("Working with your own values (or defaults).")

    patient_data = collect_patient_inputs()

    st.markdown("")
    if st.button("Generate my health profile", type="primary", use_container_width=True):
        with st.spinner("Listening to your measurements…"):
            try:
                profile = analyze_patient(patient_data)
            except Exception as exc:
                st.error(
                    "Could not generate your health profile. "
                    "Please confirm model files are available, then try again."
                )
                st.caption(str(exc))
                return
            st.session_state["profile"] = profile
            st.session_state["last_patient_data"] = patient_data
        st.success("Profile ready.")
        st.session_state["page"] = "My health"
        st.session_state["_force_nav"] = "My health"
        st.rerun()



def _fmt_record_value(val: Any, unit: str = "") -> str:
    if val is None or val == "":
        return "Not provided"
    s = str(val).strip()
    return f"{s} {unit}".strip() if unit else s


def _render_record_table(rows: List[tuple]) -> None:
    """Patient-facing label / value list. rows = (label, value, unit)."""
    parts = ['<div class="mdt-record">']
    for label, val, unit in rows:
        display = _fmt_record_value(val, unit)
        parts.append(
            '<div class="mdt-record-row">'
            f'<div class="mdt-record-label">{label}</div>'
            f'<div class="mdt-record-value">{display}</div>'
            "</div>"
        )
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)



# ---------------------------------------------------------------------------
# Document upload: CSV/Excel + image OCR + PDF text extraction
# ---------------------------------------------------------------------------
# Maps spreadsheet / OCR labels → (display label, form widget key or None, unit, cast)
# form keys must match collect_patient_inputs() / FORM_DEFAULTS when present.
_DOC_COLUMN_MAP = {
    "age": ("Age", "fi_age", "years", "float"),
    "maternal age": ("Age", "fi_age", "years", "float"),
    "gestational age": ("Gestational age", "fi_gestational_age", "weeks", "float"),
    "gestational_age": ("Gestational age", "fi_gestational_age", "weeks", "float"),
    "gestationalage": ("Gestational age", "fi_gestational_age", "weeks", "float"),
    "ga": ("Gestational age", "fi_gestational_age", "weeks", "float"),
    "gestational age (weeks)": ("Gestational age", "fi_gestational_age", "weeks", "float"),
    "booking gestational age": ("Booking gestational age", "fi_booking_ga", "weeks", "float"),
    "booking_gestational_age": ("Booking gestational age", "fi_booking_ga", "weeks", "float"),
    "bmi": ("Pre-pregnancy BMI", "fi_bmi", "", "float"),
    "pre-pregnancy bmi": ("Pre-pregnancy BMI", "fi_bmi", "", "float"),
    "pre_pregnancy_bmi": ("Pre-pregnancy BMI", "fi_bmi", "", "float"),
    "prepregnancy bmi": ("Pre-pregnancy BMI", "fi_bmi", "", "float"),
    "parity": ("Parity", "fi_parity", "", "int"),
    "systolicbp": ("Systolic BP", "fi_sbp", "mmHg", "int"),
    "systolic bp": ("Systolic BP", "fi_sbp", "mmHg", "int"),
    "systolic blood pressure": ("Systolic BP", "fi_sbp", "mmHg", "int"),
    "systolic_bp": ("Systolic BP", "fi_sbp", "mmHg", "int"),
    "sbp": ("Systolic BP", "fi_sbp", "mmHg", "int"),
    "diastolicbp": ("Diastolic BP", "fi_dbp", "mmHg", "int"),
    "diastolic bp": ("Diastolic BP", "fi_dbp", "mmHg", "int"),
    "diastolic blood pressure": ("Diastolic BP", "fi_dbp", "mmHg", "int"),
    "diastolic_bp": ("Diastolic BP", "fi_dbp", "mmHg", "int"),
    "dbp": ("Diastolic BP", "fi_dbp", "mmHg", "int"),
    "bs": ("Blood sugar", "fi_blood_sugar", "", "float"),
    "blood sugar": ("Blood sugar", "fi_blood_sugar", "", "float"),
    "blood_sugar": ("Blood sugar", "fi_blood_sugar", "", "float"),
    "blood glucose": ("Blood sugar", "fi_blood_sugar", "", "float"),
    "glucose": ("Blood sugar", "fi_blood_sugar", "", "float"),
    "bodytemp": ("Body temperature", "fi_body_temp", "°F", "float"),
    "body temp": ("Body temperature", "fi_body_temp", "°F", "float"),
    "body temperature": ("Body temperature", "fi_body_temp", "°F", "float"),
    "body_temp": ("Body temperature", "fi_body_temp", "°F", "float"),
    "temperature": ("Body temperature", "fi_body_temp", "°F", "float"),
    "heartrate": ("Heart rate", "fi_pulse", "bpm", "int"),
    "heart rate": ("Heart rate", "fi_pulse", "bpm", "int"),
    "heart_rate": ("Heart rate", "fi_pulse", "bpm", "int"),
    "pulse": ("Heart rate", "fi_pulse", "bpm", "int"),
    "tsh": ("TSH", "fi_tsh", "", "float"),
    "tsh baseline": ("TSH", "fi_tsh", "", "float"),
    "tsh_baseline": ("TSH", "fi_tsh", "", "float"),
    "ft4": ("FT4", "fi_ft4", "", "float"),
    "free t4": ("FT4", "fi_ft4", "", "float"),
    "free thyroxine": ("FT4", "fi_ft4", "", "float"),
    "ft4 baseline": ("FT4", "fi_ft4", "", "float"),
    "ft4_baseline": ("FT4", "fi_ft4", "", "float"),
    "ft3": ("FT3", "fi_ft3", "", "float"),
    "ft3 baseline": ("FT3", "fi_ft3", "", "float"),
    "ft3_baseline": ("FT3", "fi_ft3", "", "float"),
    "tpo": ("TPO", "fi_tpo", "", "float"),
    "tpo baseline": ("TPO", "fi_tpo", "", "float"),
    "tpo_baseline": ("TPO", "fi_tpo", "", "float"),
    "trbc": ("TRBC", "fi_trbc", "", "float"),
    "mcv": ("MCV", "fi_mcv", "fL", "float"),
    "mch": ("MCH", "fi_mch", "pg", "float"),
    "mchc": ("MCHC", "fi_mchc", "g/dL", "float"),
    "rdw": ("RDW", "fi_rdw", "%", "float"),
    "serum iron": ("Serum iron", "fi_serum_iron", "µg/dL", "float"),
    "serum_iron": ("Serum iron", "fi_serum_iron", "µg/dL", "float"),
    "tibc": ("TIBC", "fi_tibc", "µg/dL", "float"),
    "early hba1c": ("Early HbA1c", "fi_early_hba1c", "%", "float"),
    "early_hba1c": ("Early HbA1c", "fi_early_hba1c", "%", "float"),
    "early_hba1c_percent": ("Early HbA1c", "fi_early_hba1c", "%", "float"),
    "hba1c": ("Early HbA1c", "fi_early_hba1c", "%", "float"),
    "early rbs": ("Early RBS", "fi_early_rbs", "mg/dL", "float"),
    "early_rbs": ("Early RBS", "fi_early_rbs", "mg/dL", "float"),
    "early_rbs_mgdl": ("Early RBS", "fi_early_rbs", "mg/dL", "float"),
    "early ppbs": ("Early PPBS", "fi_early_ppbs", "mg/dL", "float"),
    "early_ppbs": ("Early PPBS", "fi_early_ppbs", "mg/dL", "float"),
    "early_ppbs_mgdl": ("Early PPBS", "fi_early_ppbs", "mg/dL", "float"),
    # Display-only (no form widget / excluded from anemia model):
    "hemoglobin": ("Hemoglobin", None, "g/dL", "float"),
    "haemoglobin": ("Hemoglobin", None, "g/dL", "float"),
    "hb": ("Hemoglobin", None, "g/dL", "float"),
}


def _norm_col(name: Any) -> str:
    s = str(name).strip().lower().replace("_", " ")
    while "  " in s:
        s = s.replace("  ", " ")
    ns = s.replace(" ", "")
    if ns in {
        "systolicbp", "diastolicbp", "bodytemp", "heartrate",
        "gestationalage", "bloodsugar", "bloodglucose",
    }:
        return ns
    return s


def _cast_doc_value(raw: Any, kind: str) -> Any:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    s = str(raw).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None
    # strip common units glued to numbers
    s_clean = re.sub(
        r"(mg/dl|g/dl|mmhg|bpm|mIU/L|miu/l|µiu/ml|uiu/ml|%|weeks?|yrs?|years?)$",
        "",
        s,
        flags=re.I,
    ).strip()
    try:
        if kind == "int":
            return int(float(s_clean))
        if kind == "float":
            return float(s_clean)
        return s
    except (TypeError, ValueError):
        return None


def _is_ambiguous_number(text: str) -> bool:
    t = str(text)
    if "?" in t or re.search(r"\d[oO]\d|\d[lI]\d", t):
        return True
    if re.search(r"[^\d.\s/\-]", t) and re.search(r"\d", t):
        # mixed digits with letters often OCR noise
        if re.search(r"[A-Za-z]", t) and not re.search(
            r"(mg|dl|mm|hg|bpm|g/|mIU|weeks|years)", t, re.I
        ):
            return True
    return False


def _read_tabular_upload(uploaded_file) -> Optional[pd.DataFrame]:
    name = (uploaded_file.name or "").lower()
    try:
        uploaded_file.seek(0)
        if name.endswith(".csv"):
            return pd.read_csv(uploaded_file)
        if name.endswith(".xlsx") or name.endswith(".xls"):
            return pd.read_excel(uploaded_file)
    except Exception:
        return None
    return None


def extract_fields_from_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    mapped: List[Dict[str, Any]] = []
    other: List[Dict[str, Any]] = []
    needs_review: List[Dict[str, Any]] = []
    if df is None or df.empty:
        return {"mapped": mapped, "other": other, "needs_review": needs_review, "count": 0}

    row = df.iloc[0]
    used_form_keys = set()
    used_labels = set()

    for col in df.columns:
        raw = row[col]
        if raw is None or (isinstance(raw, float) and pd.isna(raw)):
            continue
        if str(raw).strip() == "":
            continue

        key = _norm_col(col)
        key_ns = key.replace(" ", "")
        spec = _DOC_COLUMN_MAP.get(key) or _DOC_COLUMN_MAP.get(key_ns)
        if spec is None:
            key2 = str(col).strip().lower().replace("_", " ")
            while "  " in key2:
                key2 = key2.replace("  ", " ")
            spec = _DOC_COLUMN_MAP.get(key2)

        if spec is None:
            other.append({"label": str(col), "value": str(raw).strip()})
            continue

        label, form_key, unit, kind = spec
        if form_key and form_key in used_form_keys:
            continue
        if label in used_labels:
            continue
        if _is_ambiguous_number(str(raw)):
            needs_review.append({"label": label, "value": str(raw).strip()})
            continue
        val = _cast_doc_value(raw, kind)
        if val is None:
            needs_review.append({"label": label, "value": str(raw).strip()})
            continue
        if form_key:
            used_form_keys.add(form_key)
        used_labels.add(label)
        display = f"{val} {unit}".strip() if unit else str(val)
        mapped.append(
            {
                "label": label,
                "value": val,
                "unit": unit,
                "form_key": form_key,
                "display": display,
            }
        )

    return {
        "mapped": mapped,
        "other": other,
        "needs_review": needs_review,
        "count": len(mapped),
    }


# Label patterns for OCR / PDF free text (order matters for longer phrases first)
_TEXT_FIELD_PATTERNS = [
    (r"(?:pre[-\s]?pregnancy\s*)?bmi", "Pre-pregnancy BMI", "fi_bmi", "", "float"),
    (r"gestational\s*age(?:\s*\(?\s*weeks?\s*\)?)?|\bga\b", "Gestational age", "fi_gestational_age", "weeks", "float"),
    (r"booking\s*gestational\s*age", "Booking gestational age", "fi_booking_ga", "weeks", "float"),
    (r"maternal\s*age|\bage\b", "Age", "fi_age", "years", "float"),
    (r"systolic(?:\s*blood)?\s*pressure|\bsystolic\s*bp\b|\bsbp\b", "Systolic BP", "fi_sbp", "mmHg", "int"),
    (r"diastolic(?:\s*blood)?\s*pressure|\bdiastolic\s*bp\b|\bdbp\b", "Diastolic BP", "fi_dbp", "mmHg", "int"),
    (r"blood\s*pressure|\bbp\b", "Blood pressure", None, "mmHg", "bp_pair"),
    (r"blood\s*sugar|blood\s*glucose|\bglucose\b|\bbs\b", "Blood sugar", "fi_blood_sugar", "", "float"),
    (r"body\s*temp(?:erature)?|\btemperature\b", "Body temperature", "fi_body_temp", "°F", "float"),
    (r"heart\s*rate|\bpulse\b", "Heart rate", "fi_pulse", "bpm", "int"),
    (r"\btsh\b", "TSH", "fi_tsh", "", "float"),
    (r"free\s*t4|free\s*thyroxine|\bft4\b", "FT4", "fi_ft4", "", "float"),
    (r"\bft3\b", "FT3", "fi_ft3", "", "float"),
    (r"\btpo\b", "TPO", "fi_tpo", "", "float"),
    (r"ha?emoglobin|\bhb\b", "Hemoglobin", None, "g/dL", "float"),
    (r"\btrbc\b|total\s*rbc", "TRBC", "fi_trbc", "", "float"),
    (r"\bmcv\b", "MCV", "fi_mcv", "fL", "float"),
    (r"\bmchc\b", "MCHC", "fi_mchc", "g/dL", "float"),
    (r"\bmch\b", "MCH", "fi_mch", "pg", "float"),
    (r"\brdw\b", "RDW", "fi_rdw", "%", "float"),
    (r"serum\s*iron", "Serum iron", "fi_serum_iron", "µg/dL", "float"),
    (r"\btibc\b", "TIBC", "fi_tibc", "µg/dL", "float"),
    (r"hba1c", "Early HbA1c", "fi_early_hba1c", "%", "float"),
    (r"\bparity\b", "Parity", "fi_parity", "", "int"),
]


def extract_fields_from_text(text: str) -> Dict[str, Any]:
    """Parse free text (OCR / PDF) for label–value pairs. Never invents values."""
    mapped: List[Dict[str, Any]] = []
    other: List[Dict[str, Any]] = []
    needs_review: List[Dict[str, Any]] = []
    if not text or not str(text).strip():
        return {"mapped": mapped, "other": other, "needs_review": needs_review, "count": 0}

    used_labels = set()
    used_form_keys = set()
    # Normalize whitespace
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in str(text).splitlines()]
    blob = "\n".join(lines)

    # Blood pressure pair: 130/80 or 130 / 80
    bp = re.search(
        r"(?:blood\s*pressure|b\.?p\.?)\s*[:\-]?\s*(\d{2,3})\s*/\s*(\d{2,3})",
        blob,
        flags=re.I,
    )
    if bp:
        sbp, dbp = int(bp.group(1)), int(bp.group(2))
        mapped.append(
            {
                "label": "Systolic BP",
                "value": sbp,
                "unit": "mmHg",
                "form_key": "fi_sbp",
                "display": f"{sbp} mmHg",
            }
        )
        mapped.append(
            {
                "label": "Diastolic BP",
                "value": dbp,
                "unit": "mmHg",
                "form_key": "fi_dbp",
                "display": f"{dbp} mmHg",
            }
        )
        used_labels.update({"Systolic BP", "Diastolic BP"})
        used_form_keys.update({"fi_sbp", "fi_dbp"})

    for pattern, label, form_key, unit, kind in _TEXT_FIELD_PATTERNS:
        if label in used_labels:
            continue
        if kind == "bp_pair":
            continue  # handled above
        # value after label with : or = or whitespace
        rx = re.compile(
            r"(?:" + pattern + r")[\s:\-–.]*([0-9]+(?:\.[0-9]+)?)",
            flags=re.I,
        )
        m = rx.search(blob)
        if not m:
            continue
        raw_val = m.group(1)
        # Only judge the numeric token — not the label text
        if _is_ambiguous_number(raw_val):
            needs_review.append({"label": label, "value": raw_val})
            continue
        val = _cast_doc_value(raw_val, kind)
        if val is None:
            needs_review.append({"label": label, "value": str(raw_val)})
            continue
        if form_key and form_key in used_form_keys:
            continue
        if form_key:
            used_form_keys.add(form_key)
        used_labels.add(label)
        display = f"{val} {unit}".strip() if unit else str(val)
        mapped.append(
            {
                "label": label,
                "value": val,
                "unit": unit,
                "form_key": form_key,
                "display": display,
            }
        )

    return {
        "mapped": mapped,
        "other": other,
        "needs_review": needs_review,
        "count": len(mapped),
    }


def _ocr_image_bytes(data: bytes) -> tuple:
    """
    OCR image bytes. Returns (text, error_message).
    error_message is set if OCR unavailable or failed.
    """
    try:
        from PIL import Image
        import pytesseract
        import io
    except Exception:
        return "", "Image text extraction is not available in this environment yet."

    try:
        img = Image.open(io.BytesIO(data))
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        text = pytesseract.image_to_string(img)
        return text or "", None
    except Exception:
        return "", "We couldn't read this file. Please try another document."


def _extract_pdf_text(data: bytes) -> tuple:
    """Extract text from PDF. Returns (text, error_message)."""
    import io

    text_parts = []
    try:
        import pdfplumber

        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page in pdf.pages[:10]:
                t = page.extract_text() or ""
                if t.strip():
                    text_parts.append(t)
    except Exception:
        try:
            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(data))
            for page in reader.pages[:10]:
                t = page.extract_text() or ""
                if t.strip():
                    text_parts.append(t)
        except Exception:
            return "", "We couldn't read this file. Please try another document."

    text = "\n".join(text_parts).strip()
    if text:
        return text, None

    # Scanned PDF: try OCR on first pages via pdfplumber images if possible
    try:
        import pdfplumber
        from PIL import Image
        import pytesseract

        ocr_bits = []
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page in pdf.pages[:3]:
                try:
                    pil = page.to_image(resolution=200).original
                    ocr_bits.append(pytesseract.image_to_string(pil) or "")
                except Exception:
                    continue
        text = "\n".join(ocr_bits).strip()
        if text:
            return text, None
    except Exception:
        pass

    return "", "We couldn't identify any readable health information in this document."


def apply_extracted_to_form(mapped: List[Dict[str, Any]]) -> int:
    """Write extracted values into form widget keys. Only keys present in mapped."""
    n = 0
    for item in mapped:
        fk = item.get("form_key")
        val = item.get("value")
        if fk and val is not None:
            st.session_state[fk] = val
            n += 1
    st.session_state["use_sample"] = False
    return n


def process_uploaded_document(uploaded_file) -> Dict[str, Any]:
    """
    Process one UploadedFile (CSV/Excel/image/PDF).
    Returns session record with mapped / other / needs_review.
    Never invents values.
    """
    name = uploaded_file.name or "document"
    lower = name.lower()
    rec: Dict[str, Any] = {
        "name": name,
        "type": getattr(uploaded_file, "type", None) or "file",
        "size": getattr(uploaded_file, "size", None),
        "status": "unsupported",
        "mapped": [],
        "other": [],
        "needs_review": [],
        "count": 0,
        "message": "",
    }

    # ---- Tabular ----
    if lower.endswith(".csv") or lower.endswith(".xlsx") or lower.endswith(".xls"):
        df = _read_tabular_upload(uploaded_file)
        if df is None:
            rec["status"] = "error"
            rec["message"] = "We couldn't read this document. Please check the file format and try again."
            return rec
        result = extract_fields_from_dataframe(df)
        rec["mapped"] = result["mapped"]
        rec["other"] = result["other"]
        rec["needs_review"] = result.get("needs_review") or []
        rec["count"] = result["count"]
        if result["count"] == 0:
            rec["status"] = "no_fields"
            rec["message"] = (
                "We could read the document, but we couldn't identify any fields "
                "that match the information used in this prototype."
            )
            if not rec["other"]:
                try:
                    preview = []
                    row0 = df.iloc[0]
                    for c in list(df.columns)[:12]:
                        v = row0[c]
                        if v is not None and not (isinstance(v, float) and pd.isna(v)):
                            preview.append({"label": str(c), "value": str(v)})
                    rec["other"] = preview
                except Exception:
                    pass
        else:
            rec["status"] = "ok"
            rec["message"] = f"We found {result['count']} piece(s) of information in this document."
        return rec

    # ---- Images (OCR) ----
    if lower.endswith((".png", ".jpg", ".jpeg")):
        try:
            data = uploaded_file.getvalue()
        except Exception:
            uploaded_file.seek(0)
            data = uploaded_file.read()
        text, err = _ocr_image_bytes(data)
        if err and not text:
            rec["status"] = "error"
            rec["message"] = err
            return rec
        result = extract_fields_from_text(text)
        rec["mapped"] = result["mapped"]
        rec["other"] = result["other"]
        rec["needs_review"] = result.get("needs_review") or []
        rec["count"] = result["count"]
        if result["count"] == 0:
            rec["status"] = "no_fields"
            rec["message"] = "We couldn't identify any readable health information in this document."
        else:
            rec["status"] = "ok"
            rec["message"] = f"We found {result['count']} piece(s) of information in this document."
        if result.get("needs_review"):
            rec["message"] += " Some information could not be read confidently."
        return rec

    # ---- PDF ----
    if lower.endswith(".pdf"):
        try:
            data = uploaded_file.getvalue()
        except Exception:
            uploaded_file.seek(0)
            data = uploaded_file.read()
        text, err = _extract_pdf_text(data)
        if err and not text:
            rec["status"] = "error"
            rec["message"] = err
            return rec
        result = extract_fields_from_text(text)
        rec["mapped"] = result["mapped"]
        rec["other"] = result["other"]
        rec["needs_review"] = result.get("needs_review") or []
        rec["count"] = result["count"]
        if result["count"] == 0:
            rec["status"] = "no_fields"
            rec["message"] = "We couldn't identify any readable health information in this document."
        else:
            rec["status"] = "ok"
            rec["message"] = f"We found {result['count']} piece(s) of information in this document."
        if result.get("needs_review"):
            rec["message"] += " Some information could not be read confidently."
        return rec

    rec["status"] = "unsupported"
    rec["message"] = (
        "This prototype extracts fields from CSV, Excel, PDF, and image (PNG/JPG) files. "
        "This file type is stored for this session only and is not parsed."
    )
    return rec



def page_health_records() -> None:
    st.markdown('<div class="mdt-greeting">Health records</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="mdt-tagline">A place for your history — designed for later, sketched for now.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="mdt-proto">
        <strong>Prototype layout</strong> — There is no persistent patient database in this student build.
        The sections below show how a future “My records” area could be organized.
        Nothing here claims production security or clinical-grade storage.
        </div>
        """,
        unsafe_allow_html=True,
    )

    tabs = st.tabs(
        [
            "Pregnancy",
            "Lab results",
            "Assessments",
            "Appointments",
            "Documents",
            "Timeline",
        ]
    )
    with tabs[0]:
        st.markdown("##### Pregnancy details")
        data = st.session_state.get("last_patient_data") or {}
        if not data:
            st.caption("No pregnancy details loaded yet.")
        else:
            _render_record_table(
                [
                    ("Gestational age", data.get("gestational_age"), "weeks"),
                    ("Booking gestational age", data.get("booking_gestational_age"), "weeks"),
                    ("Pre-pregnancy BMI", data.get("pre_pregnancy_bmi"), ""),
                    ("Parity", data.get("parity"), ""),
                ]
            )
        st.caption("Current session snapshot — not a longitudinal clinical history.")

    with tabs[1]:
        st.markdown("##### Lab results")
        st.caption("Laboratory values from information entered in this session.")
        d = st.session_state.get("last_patient_data") or {}
        if not d:
            st.caption("No laboratory results have been added yet.")
        else:
            _render_record_table(
                [
                    ("TRBC", d.get("trbc"), ""),
                    ("MCV", d.get("mcv"), "fL"),
                    ("MCH", d.get("mch"), "pg"),
                    ("MCHC", d.get("mchc"), "g/dL"),
                    ("RDW", d.get("rdw"), "%"),
                    ("Serum iron", d.get("serum_iron"), "µg/dL"),
                    ("TIBC", d.get("tibc"), "µg/dL"),
                    ("Transferrin saturation", d.get("transferrin_saturation"), "%"),
                    ("Total bilirubin", d.get("total_bilirubin"), ""),
                    ("Early RBS", d.get("early_rbs_mgdl"), "mg/dL"),
                    ("Early PPBS", d.get("early_ppbs_mgdl"), "mg/dL"),
                    ("Early HbA1c", d.get("early_hba1c_percent"), "%"),
                    ("TSH baseline", d.get("tsh_baseline"), ""),
                    ("FT3 baseline", d.get("ft3_baseline"), ""),
                    ("FT4 baseline", d.get("ft4_baseline"), ""),
                    ("TPO baseline", d.get("tpo_baseline"), ""),
                ]
            )


    with tabs[2]:
        st.markdown("##### Previous model assessments")
        if st.session_state.get("profile"):
            st.caption("Current session assessment only — not a saved history.")
            render_calm_summary(st.session_state["profile"])
        else:
            st.caption("No assessments in this session yet.")

    with tabs[3]:
        st.markdown("##### Appointments")
        st.caption("Future: upcoming visits, who you will see, what to bring.")
        st.info("No appointments stored in this prototype.")

    with tabs[4]:
        st.markdown("##### Documents & reports (prototypehead -5 maternal_health_risk.csv)")
        st.caption(
            "Upload a spreadsheet (CSV or Excel) to extract values for this session. "
            "Files are not stored in a clinical record system."
        )
        if "uploaded_docs" not in st.session_state:
            st.session_state["uploaded_docs"] = []

        uploaded = st.file_uploader(
            "Upload a document",
            type=["pdf", "csv", "xlsx", "xls", "png", "jpg", "jpeg", "txt"],
            accept_multiple_files=True,
            key="doc_uploader",
        )

        if uploaded:
            existing = {d.get("name") for d in st.session_state["uploaded_docs"]}
            for f in uploaded:
                if f.name in existing:
                    continue
                rec = process_uploaded_document(f)
                st.session_state["uploaded_docs"].append(rec)
                # keep latest extraction available for "Use this information"
                st.session_state["latest_extraction"] = rec

        docs = st.session_state.get("uploaded_docs") or []
        if not docs:
            st.info("No documents uploaded yet.")
        else:
            for i, d in enumerate(docs):
                st.markdown(f"**{d.get('name')}**")
                st.caption("Session only · not saved to a clinical record")

                status = d.get("status")
                msg = d.get("message") or ""
                if status == "ok":
                    st.success(msg)
                    mapped = d.get("mapped") or []
                    if mapped:
                        st.markdown("###### Information found")
                        table_df = pd.DataFrame(
                            [
                                {"Information": m["label"], "Value": m["display"]}
                                for m in mapped
                            ]
                        )
                        st.dataframe(table_df, use_container_width=True, hide_index=True)

                        def _make_use_cb(mapped_items):
                            def _cb():
                                apply_extracted_to_form(mapped_items)
                                st.session_state["page"] = "Update information"
                                st.session_state["_force_nav"] = "Update information"
                            return _cb

                        st.button(
                            "Use this information",
                            key=f"use_extract_{i}",
                            type="primary",
                            on_click=_make_use_cb(mapped),
                        )
                        st.caption(
                            "Only the fields found above will be filled on Update information. "
                            "Other fields stay as they are."
                        )

                    needs = d.get("needs_review") or []
                    if needs:
                        st.caption("Some information could not be read confidently.")
                        with st.expander("Needs review"):
                            nr_df = pd.DataFrame(
                                [{"Information": n["label"], "Extracted text": n["value"]} for n in needs]
                            )
                            st.dataframe(nr_df, use_container_width=True, hide_index=True)

                    other = d.get("other") or []
                    if other:
                        with st.expander("Other information in this document"):
                            other_df = pd.DataFrame(
                                [{"Information": o["label"], "Value": o["value"]} for o in other]
                            )
                            st.dataframe(other_df, use_container_width=True, hide_index=True)

                elif status == "no_fields":
                    st.info(msg)
                    needs = d.get("needs_review") or []
                    if needs:
                        with st.expander("Needs review"):
                            nr_df = pd.DataFrame(
                                [{"Information": n["label"], "Extracted text": n["value"]} for n in needs]
                            )
                            st.dataframe(nr_df, use_container_width=True, hide_index=True)
                    other = d.get("other") or []
                    if other:
                        with st.expander("Other information in this document"):
                            other_df = pd.DataFrame(
                                [{"Information": o["label"], "Value": o["value"]} for o in other]
                            )
                            st.dataframe(other_df, use_container_width=True, hide_index=True)

                elif status == "error":
                    st.warning(msg)

                else:
                    # unsupported type
                    st.caption(msg or "Stored for this session only.")

                st.markdown("---")

            if st.button("Clear uploaded documents from this session"):
                st.session_state["uploaded_docs"] = []
                st.session_state["latest_extraction"] = None
                st.rerun()

    with tabs[5]:
        st.markdown("##### Pregnancy timeline")
        st.caption(
            "Future: a calm timeline of labs, assessments, and visits. "
            "No longitudinal data is invented here."
        )
        if st.session_state.get("last_patient_data"):
            ga = st.session_state["last_patient_data"].get("gestational_age")
            st.write(f"Current snapshot only — week {ga}.")
        else:
            st.write("No timeline events yet.")


def page_privacy() -> None:
    st.markdown('<div class="mdt-greeting">Privacy & access</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="mdt-tagline">Your records should feel like yours.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="mdt-proto">
        <strong>Prototype concept only.</strong> This page describes the access model we want
        the product to support later. It does <em>not</em> implement secure authentication,
        encryption, consent enforcement, or audit logging. Do not enter real patient identifiers
        into this student prototype.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="mdt-card">'
        '<div class="mdt-title">Who can see my information?</div>'
        '<div class="mdt-body">'
        "In a full product, you would grant access deliberately — by clinician, by category "
        "(pregnancy notes, labs, assessments), and with an expiry you control. "
        "You would also be able to revoke access and review an access history."
        "</div></div>",
        unsafe_allow_html=True,
    )

    st.markdown("#### Doctor access (illustrative)")
    st.markdown(
        """
        <div class="mdt-card-steady">
        <div class="mdt-eyebrow">Example — not live data</div>
        <div class="mdt-title">Dr. Example, antenatal clinic</div>
        <div class="mdt-muted">Access: pregnancy details + lab summaries<br/>
        Granted: — &nbsp;·&nbsp; Expires: —</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.button("Manage access (not connected)", disabled=True)
    st.button("View access history (not connected)", disabled=True)

    st.markdown("#### Principles we want to keep")
    st.markdown(
        """
        - **You own the narrative** of your record in this product vision.
        - **Access is explicit** — not assumed because someone has a login.
        - **Sharing is granular** — labs, assessments, and notes can be separated later.
        - **Revocation is normal** — changing care teams should not leave open doors.
        - **This build is honest** — none of the above is enforced by backend security yet.
        """
    )


def page_about() -> None:
    st.markdown('<div class="mdt-greeting">About & limitations</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="mdt-tagline">Supporting healthier pregnancies through intelligent insights</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="mdt-card">
        <div class="mdt-body">
        NineFolds is a <strong>student-built AI/ML decision-support prototype</strong>
        designed to help expectant mothers and healthcare professionals explore important aspects
        of maternal health through data-driven insights.
        </div>
        <div class="mdt-body" style="margin-top:0.75rem;">
        The platform brings together multiple maternal health areas in one place, helping users
        understand potential concerns related to anemia, preeclampsia, gestational diabetes,
        and thyroid health.
        </div>
        <div class="mdt-body" style="margin-top:0.75rem;">
        Our goal is simple: make health information easier to understand, encourage timely
        attention to possible concerns, and support meaningful conversations between mothers
        and healthcare professionals.
        </div>
        <div class="mdt-body" style="margin-top:0.75rem;">
        NineFolds is designed to <strong>complement healthcare—not replace it</strong>.
        Every result should be understood as an indication that may deserve attention,
        rather than a medical conclusion.
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### What the platform provides")
    st.markdown(
        """
        <div class="mdt-card-steady">
        <div class="mdt-body">
        <ul style="margin:0; padding-left:1.2rem;">
        <li><strong>Anemia insights</strong> — helps identify patterns that may indicate a possible risk of anemia.</li>
        <li><strong>Preeclampsia insights</strong> — highlights patterns associated with increased pregnancy-related risk.</li>
        <li><strong>Gestational diabetes insights</strong> — provides an indication of possible risk related to blood-sugar health during pregnancy.</li>
        <li><strong>Thyroid insights</strong> — highlights patterns that may suggest a need for further thyroid-related attention.</li>
        </ul>
        </div>
        <div class="mdt-muted" style="margin-top:0.85rem;">
        Each area is presented independently so that users can understand individual health concerns
        without reducing the complexity of pregnancy to a single overall risk score.
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Important limitations")
    st.markdown(
        """
        <div class="mdt-card-discuss">
        <div class="mdt-body">
        NineFolds is an educational and decision-support prototype,
        <strong>not a medical diagnostic system</strong>.
        </div>
        <div class="mdt-body" style="margin-top:0.75rem;">
        <ul style="margin:0; padding-left:1.2rem;">
        <li>The results are not medical diagnoses.</li>
        <li>A higher-risk result does not mean that a condition is definitely present.</li>
        <li>A lower-risk result does not guarantee that a condition is absent.</li>
        <li>The platform should not be used to make decisions about medication, treatment, or emergency care.</li>
        <li>Results depend on the quality and completeness of the information provided.</li>
        <li>The prototype has not been clinically validated for use as a replacement for professional medical assessment.</li>
        <li>Pregnancy is highly individual, and important factors may not be captured by the information available to the system.</li>
        </ul>
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Module-specific notes")
    st.markdown(
        """
        | Module | What to remember |
        | --- | --- |
        | **Anemia** | Hb / PCV / who_pred excluded because of target leakage. Macro F1 is modest; severe class is rare. |
        | **Preeclampsia** | Target is a **RiskLevel-derived proxy**, not confirmed clinical preeclampsia. |
        | **GDM** | Model trained on **synthetic** data (~65% prevalence). Not clinically validated. |
        | **Thyroid** | Later SCH-consistent risk among baseline-negative women; single-center; ROC-AUC ≈ 0.66. |
        """
    )

    st.markdown("#### Use it responsibly")
    st.markdown(
        """
        <div class="mdt-card">
        <div class="mdt-body">
        Think of NineFolds as a <strong>conversation starter</strong>, not a final answer.
        </div>
        <div class="mdt-body" style="margin-top:0.75rem;">
        If the platform highlights a potential concern, discuss it with your doctor or qualified
        healthcare professional. Continue regular antenatal check-ups and follow the advice
        provided by your healthcare team.
        </div>
        <div class="mdt-body" style="margin-top:0.75rem;">
        For emergencies or concerning symptoms, seek medical care immediately rather than
        relying on this platform.
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Your privacy matters")
    st.markdown(
        """
        <div class="mdt-limit">
        Please do not enter personally identifiable information or identifiable real-patient data
        into this student prototype.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Our vision")
    st.markdown(
        """
        <div class="mdt-card-soft">
        <div class="mdt-body">
        We believe technology can make maternal health information more accessible, understandable,
        and supportive.
        </div>
        <div class="mdt-body" style="margin-top:0.75rem;">
        NineFolds is our step toward creating a future where data and AI can support
        mothers and healthcare professionals with meaningful insights—while keeping human care
        and clinical judgment at the heart of every pregnancy journey.
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_clinician_home() -> None:
    st.markdown('<div class="mdt-greeting">Clinician workspace</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="mdt-tagline">A separate door — still a prototype sketch.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="mdt-proto">
        <strong>Prototype only.</strong> There is no authentication, authorization, or patient roster backend.
        This screen shows how a future clinician view might be structured after a patient explicitly grants access.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="mdt-card">
        <div class="mdt-title">What clinicians would see later</div>
        <div class="mdt-body">
        Authorized patients · pregnancy overview · labs · model assessments · notes · timeline.
        Access would depend on patient consent, not on simply choosing this role in a demo.
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_patient_list() -> None:
    st.markdown('<div class="mdt-greeting">Patients</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="mdt-proto">
        No patient directory is connected. In a real system this list would only include people
        who granted this clinician access, with scope and expiry visible.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info("Patient list is empty in this prototype.")


def page_admin() -> None:
    """Technical / evaluation view for project evaluators. Not patient-facing."""
    st.markdown(
        '<div class="mdt-greeting">Admin / System Information</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="mdt-tagline">Technical reference for evaluators — not a patient screen.</div>',
        unsafe_allow_html=True,
    )

    if st.button("← Back to Home", key="admin_back_home"):
        st.session_state["page"] = "Home"
        st.session_state["_force_nav"] = "Home"
        st.rerun()

    st.markdown(
        """
        <div class="mdt-proto">
        <strong>Student prototype only.</strong> This view is for project evaluators and
        technical reviewers. It is <em>not</em> authenticated, encrypted, or production-grade.
        Metrics shown below are taken only from repository metadata and evaluation documentation
        that already exist in this project — nothing is invented at display time.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="mdt-card">'
        '<div class="mdt-title">NineFolds</div>'
        '<div class="mdt-body">'
        "Student AI/ML decision-support prototype. Four independent condition modules run side by side. "
        "No combined clinical overall pregnancy risk score is computed. "
        "It is not a diagnostic device and does not replace clinical judgment."
        "</div></div>",
        unsafe_allow_html=True,
    )

    st.markdown("#### System overview")
    st.markdown(
        """
        Patient information is mapped through a common schema, then scored independently by:

        1. **Anemia** — Track B severity pattern (multiclass)
        2. **Preeclampsia** — RiskLevel-derived **proxy** (binary)
        3. **Gestational diabetes (GDM)** — synthetic-data model (binary)
        4. **Thyroid** — later SCH-consistent dysfunction risk from first-trimester baseline (binary)

        Official held-out metrics below come from training metadata / results files under `models/`.
        External-file evaluation scripts under `evaluation/` score frozen artifacts only and do not retrain.
        """
    )

    st.markdown("---")
    st.markdown("#### Modules")

    # --- 1. Anemia ---
    st.markdown(
        '<div class="mdt-card-discuss">'
        '<div class="mdt-eyebrow">Module</div>'
        '<div class="mdt-title">1. Anemia</div>'
        '<div class="mdt-body">Logistic Regression (Track B) · held-out metrics from model metadata</div>'
        "</div>",
        unsafe_allow_html=True,
    )
    with st.expander("Technical details — Anemia", expanded=False):
        _render_record_table(
            [
                ("Model type", "Logistic Regression (Track B)", ""),
                (
                    "Dataset / evaluation data",
                    "Project CBC / anemia training set (held-out split documented in model metadata)",
                    "",
                ),
                ("Evaluation samples", "Not available in committed metadata.", ""),
                (
                    "Dataset note",
                    "The severe class is rare in the training data (~9 samples noted in training documentation).",
                    "",
                ),
                (
                    "Evaluation method",
                    "Official held-out test metrics from models/anemia/anemia_model_metadata.json (RANDOM_STATE=42).",
                    "",
                ),
                (
                    "Source in repository",
                    "models/anemia/anemia_model_metadata.json · evaluation/README.md",
                    "",
                ),
            ]
        )
        st.markdown("**Held-out metrics (from repository artifacts)**")
        _render_record_table(
            [
                ("Accuracy", "0.785", ""),
                ("Balanced accuracy", "0.696", ""),
                ("F1 (macro)", "0.675", ""),
                ("F1 (weighted)", "0.788", ""),
            ]
        )
        st.markdown(
            '<div class="mdt-limit"><strong>Important limitations</strong><br/>'
            "Hb, PCV, and who_pred are excluded from model inputs because of target leakage. "
            "This module does not diagnose anemia. Macro F1 is modest; the severe class is rare."
            "</div>",
            unsafe_allow_html=True,
        )

    # --- 2. Preeclampsia ---
    st.markdown(
        '<div class="mdt-card-discuss">'
        '<div class="mdt-eyebrow">Module</div>'
        '<div class="mdt-title">2. Preeclampsia</div>'
        '<div class="mdt-body">XGBoost · RiskLevel-derived proxy · metrics from preeclampsia_results.md</div>'
        "</div>",
        unsafe_allow_html=True,
    )
    with st.expander("Technical details — Preeclampsia", expanded=False):
        _render_record_table(
            [
                (
                    "Model type",
                    "XGBoost (chosen among Logistic Regression, Random Forest, XGBoost)",
                    "",
                ),
                (
                    "Dataset / evaluation data",
                    "UCI Maternal Health Risk Dataset (N = 1,014 records)",
                    "",
                ),
                (
                    "Evaluation samples",
                    "Internal train/test split (80/20) on ~1,014 records (documented in results file)",
                    "",
                ),
                (
                    "Evaluation method",
                    "Official comparison table in models/preeclampsia/preeclampsia_results.md. "
                    "XGBoost selected for highest ROC-AUC and F1 among the three baselines on this split.",
                    "",
                ),
                (
                    "Source in repository",
                    "models/preeclampsia/preeclampsia_results.md · evaluation/evaluate_preeclampsia.py",
                    "",
                ),
            ]
        )
        st.markdown("**Held-out metrics (from repository artifacts)**")
        _render_record_table(
            [
                ("Precision", "0.907", ""),
                ("Recall", "0.907", ""),
                ("F1", "0.907", ""),
                ("ROC-AUC", "0.970", ""),
            ]
        )
        st.markdown(
            '<div class="mdt-limit"><strong>Important limitations</strong><br/>'
            "Target is a RiskLevel-derived <strong>proxy</strong> label (high risk vs low/mid), "
            "not confirmed clinical preeclampsia. Single snapshot per patient (no true longitudinal BP trend). "
            "Single-source dataset; no external clinical validation in this prototype."
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="mdt-limit"><strong>Cross-validation note</strong><br/>'
            "A temporary 5-fold CV experiment for preeclampsia (Random Forest mean ± std) was mentioned "
            "in project discussion, but the corresponding experiment script / output is <em>not</em> "
            "present in the current repository tree. Those CV numbers are therefore <strong>not</strong> "
            "displayed here. The production artifact documented in-repo is the XGBoost model and the "
            "metrics table in <code>preeclampsia_results.md</code>."
            "</div>",
            unsafe_allow_html=True,
        )

    # --- 3. GDM ---
    st.markdown(
        '<div class="mdt-card-discuss">'
        '<div class="mdt-eyebrow">Module</div>'
        '<div class="mdt-title">3. Gestational diabetes (GDM)</div>'
        '<div class="mdt-body">Random Forest pipeline · synthetic training data · held-out metrics not in committed metadata</div>'
        "</div>",
        unsafe_allow_html=True,
    )
    with st.expander("Technical details — Gestational diabetes (GDM)", expanded=False):
        _render_record_table(
            [
                ("Model type", "Random Forest pipeline", ""),
                (
                    "Dataset / evaluation data",
                    "Synthetic training data (n ≈ 10,000; ~65% GDM prevalence in training set)",
                    "",
                ),
                (
                    "Evaluation samples",
                    "Not available in committed model metadata.",
                    "",
                ),
                (
                    "Evaluation method",
                    "Frozen artifact under models/gdm/. Official notebook metrics are referenced in "
                    "evaluation notes; no numeric held-out table is stored in model metadata JSON in this repo.",
                    "",
                ),
                (
                    "Source in repository",
                    "evaluation/evaluate_gdm.py · evaluation/README.md · README.md",
                    "",
                ),
            ]
        )
        st.markdown("**Held-out metrics (from repository artifacts)**")
        st.caption("Not available in committed model metadata.")
        st.markdown(
            '<div class="mdt-limit"><strong>Important limitations</strong><br/>'
            "Trained on <strong>synthetic</strong> data with elevated prevalence (~65%). "
            "Not clinically validated. Not a diagnostic test."
            "</div>",
            unsafe_allow_html=True,
        )

    # --- 4. Thyroid ---
    st.markdown(
        '<div class="mdt-card-discuss">'
        '<div class="mdt-eyebrow">Module</div>'
        '<div class="mdt-title">4. Thyroid</div>'
        '<div class="mdt-body">Random Forest · later SCH-consistent risk · metrics from thyroid feature metadata</div>'
        "</div>",
        unsafe_allow_html=True,
    )
    with st.expander("Technical details — Thyroid", expanded=False):
        _render_record_table(
            [
                ("Model type", "Random Forest", ""),
                (
                    "Dataset / evaluation data",
                    "Single-center retrospective baseline labs → later SCH-consistent status",
                    "",
                ),
                (
                    "Evaluation samples",
                    "Not available in committed model metadata.",
                    "",
                ),
                (
                    "Evaluation method",
                    "Official held-out metrics from models/thyroid/thyroid_feature_metadata.json (RANDOM_STATE=42).",
                    "",
                ),
                (
                    "Source in repository",
                    "models/thyroid/thyroid_feature_metadata.json · evaluation/README.md",
                    "",
                ),
            ]
        )
        st.markdown("**Held-out metrics (from repository artifacts)**")
        _render_record_table(
            [
                ("ROC-AUC", "0.6588", ""),
                ("PR-AUC", "0.2033", ""),
                ("F1", "0.2759", ""),
            ]
        )
        st.markdown(
            '<div class="mdt-limit"><strong>Important limitations</strong><br/>'
            "Predicts later SCH-consistent risk among baseline-negative women from first-trimester labs. "
            "Single-center; modest discrimination (ROC-AUC ≈ 0.66). Not a diagnosis."
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(
        """
        <div class="mdt-proto">
        <strong>Patient UI boundary</strong> — Patient-facing pages (Home, My health, View details, etc.)
        intentionally omit model names, confidence scores, accuracy, F1, ROC-AUC, and other ML metrics.
        Technical detail belongs only on this Admin page.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
page = st.session_state.get("page", "Home")

if page == "Home":
    page_home()
elif page == "My health":
    page_my_health()
elif page == "Update information":
    page_update_information()
elif page == "Health records":
    page_health_records()
elif page == "Privacy & access":
    page_privacy()
elif page == "About & limitations":
    page_about()
elif page == "Clinician home":
    page_clinician_home()
elif page == "Patient list":
    page_patient_list()
elif page == "Admin":
    page_admin()
else:
    page_home()

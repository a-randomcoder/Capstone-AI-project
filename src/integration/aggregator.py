"""
Maternal Health Profile aggregator.

Calls all four condition modules and returns a transparent combined profile.
Does NOT invent an overall pregnancy-risk percentage or clinical score.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.predict_anemia import predict_anemia
from src.models.predict_preeclampsia import predict_preeclampsia
from src.models.predict_gdm import predict_gdm
from src.models.predict_thyroid import predict_thyroid


def generate_maternal_profile(patient_data: dict) -> Dict[str, Any]:
    """
    Run all four modules and assemble a maternal health profile.
    """
    results = {
        "Anemia": predict_anemia(patient_data),
        "Preeclampsia": predict_preeclampsia(patient_data),
        "GDM": predict_gdm(patient_data),
        "Thyroid": predict_thyroid(patient_data),
    }

    flagged: List[str] = []
    pending: List[str] = []
    ok: List[str] = []

    normalish = {
        "normal",
        "no gdm",
        "gdm unlikely (synthetic model)",
        "low/mid risk (proxy)",
        "pending",
        "insufficient input",
    }

    for name, r in results.items():
        status = r.get("status", "ok")
        pred = str(r.get("prediction", "")).strip()
        if status == "pending":
            pending.append(name)
        elif status == "missing_features":
            pending.append(f"{name} (incomplete input)")
        elif pred.lower() in normalish or pred == "":
            ok.append(f"{name}: {pred}")
        else:
            flagged.append(f"{name}: {pred}")

    summary = {
        "flagged_conditions": flagged,
        "within_typical_range_or_proxy_low": ok,
        "pending_or_incomplete": pending,
        "note": (
            "This profile lists independent module outputs only. "
            "No combined clinical risk score is computed."
        ),
    }

    return {
        "modules": results,
        "summary": summary,
        "disclaimer": (
            "Maternal Digital Twin is a student AI/ML decision-support prototype. "
            "It does not diagnose disease, prescribe treatment, or replace clinical judgment. "
            "Anemia excludes Hb/PCV due to leakage. Preeclampsia uses a RiskLevel proxy label. "
            "GDM uses a synthetic dataset/model. Thyroid is pending."
        ),
    }


if __name__ == "__main__":
    import json
    from src.integration.schemas import sample_patient

    profile = generate_maternal_profile(sample_patient())
    print(json.dumps(profile, indent=2, default=str))

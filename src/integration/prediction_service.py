"""Thin entry point used by the Streamlit frontend."""

from __future__ import annotations

from typing import Any, Dict

from src.integration.aggregator import generate_maternal_profile


def analyze_patient(patient_data: dict) -> Dict[str, Any]:
    """Run the full maternal health profile pipeline."""
    return generate_maternal_profile(patient_data)

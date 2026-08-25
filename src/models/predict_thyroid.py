"""
Thyroid dysfunction module — PLACEHOLDER.

Teammate will replace this with a real predict_thyroid() when artifacts arrive.
Interface matches the other modules so the aggregator and dashboard need no changes.
"""

from __future__ import annotations

from typing import Any, Dict


def predict_thyroid(patient_data: dict, top_k_factors: int = 5) -> Dict[str, Any]:
    """
    Placeholder until the thyroid module is delivered.

    Expected future return shape (same as other modules):
      condition, prediction, probability, class_probabilities,
      important_factors, notes, status
    """
    return {
        "condition": "Thyroid",
        "prediction": "Pending",
        "probability": None,
        "class_probabilities": {},
        "important_factors": [],
        "notes": (
            "Thyroid module not yet integrated. Drop model artifacts under "
            "models/thyroid/ and replace this function body. Research prototype only."
        ),
        "status": "pending",
    }


if __name__ == "__main__":
    import json

    print(json.dumps(predict_thyroid({}), indent=2))

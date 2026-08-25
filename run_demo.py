#!/usr/bin/env python3
"""Quick CLI demo of the integrated maternal health profile."""

from src.integration.schemas import sample_patient
from src.integration.aggregator import generate_maternal_profile
import json


def main():
    patient = sample_patient()
    profile = generate_maternal_profile(patient)
    print("=" * 60)
    print("MATERNAL DIGITAL TWIN — DEMO PROFILE")
    print("=" * 60)
    for name, r in profile["modules"].items():
        print(f"\n[{name}]  status={r['status']}")
        print(f"  prediction : {r['prediction']}")
        print(f"  probability: {r['probability']}")
        if r.get("class_probabilities"):
            print(f"  class_probs: {r['class_probabilities']}")
        if r.get("important_factors"):
            print("  top factors:")
            for f in r["important_factors"][:3]:
                print(f"    - {f['feature']}: {f['shap_contribution']}")
    print("\n" + "-" * 60)
    print("SUMMARY")
    print(json.dumps(profile["summary"], indent=2))
    print("\n" + profile["disclaimer"])


if __name__ == "__main__":
    main()

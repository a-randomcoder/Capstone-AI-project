#!/usr/bin/env python3
"""
Internal single-patient manual test harness.

Enter feature values interactively (or use --sample) and call the same
production predict_* wrappers used by the app. Does NOT retrain models.
Not for clinical use.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.integration.schemas import sample_patient  # noqa: E402
from src.models.predict_anemia import predict_anemia  # noqa: E402
from src.models.predict_gdm import predict_gdm  # noqa: E402
from src.models.predict_preeclampsia import predict_preeclampsia  # noqa: E402
from src.models.predict_thyroid import predict_thyroid  # noqa: E402

MODULES = {
    "anemia": predict_anemia,
    "preeclampsia": predict_preeclampsia,
    "gdm": predict_gdm,
    "thyroid": predict_thyroid,
    "all": None,
}


def prompt_float(label: str, default: float) -> float:
    raw = input(f"  {label} [{default}]: ").strip()
    return float(raw) if raw else float(default)


def prompt_str(label: str, default: str) -> str:
    raw = input(f"  {label} [{default}]: ").strip()
    return raw if raw else default


def prompt_int(label: str, default: int) -> int:
    raw = input(f"  {label} [{default}]: ").strip()
    return int(raw) if raw else int(default)


def interactive_patient(base: dict) -> dict:
    print("\nEnter patient values (press Enter to keep default).\n")
    p = dict(base)
    p["age"] = prompt_float("age", p.get("age", 28))
    p["ethnicity"] = prompt_str("ethnicity", p.get("ethnicity", "Asian"))
    p["gestational_age"] = prompt_float("gestational_age / POG weeks", p.get("gestational_age", 24))
    p["booking_gestational_age"] = prompt_float(
        "booking_gestational_age", p.get("booking_gestational_age", 10)
    )
    p["pre_pregnancy_bmi"] = prompt_float("pre_pregnancy_bmi", p.get("pre_pregnancy_bmi", 26.5))
    p["systolic_bp"] = prompt_float("systolic_bp", p.get("systolic_bp", 135))
    p["diastolic_bp"] = prompt_float("diastolic_bp", p.get("diastolic_bp", 88))
    p["pulse"] = prompt_float("pulse / heart rate", p.get("pulse", 86))
    p["body_temp"] = prompt_float("body_temp °F", p.get("body_temp", 98.2))
    p["blood_sugar"] = prompt_float("blood_sugar (PE unit)", p.get("blood_sugar", 12.0))
    p["early_rbs_mgdl"] = prompt_float("early_rbs_mgdl", p.get("early_rbs_mgdl", 118))
    p["early_ppbs_mgdl"] = prompt_float("early_ppbs_mgdl", p.get("early_ppbs_mgdl", 145))
    p["early_hba1c_percent"] = prompt_float("early_hba1c_percent", p.get("early_hba1c_percent", 5.6))
    p["tsh_baseline"] = prompt_float("tsh_baseline", p.get("tsh_baseline", 2.8))
    p["ft3_baseline"] = prompt_float("ft3_baseline", p.get("ft3_baseline", 4.6))
    p["ft4_baseline"] = prompt_float("ft4_baseline", p.get("ft4_baseline", 14.5))
    p["tpo_baseline"] = prompt_float("tpo_baseline", p.get("tpo_baseline", 12.0))
    return p


def run_module(name: str, patient: dict) -> dict:
    fn = MODULES[name]
    return fn(patient)


def main():
    parser = argparse.ArgumentParser(description="Manual single-patient prediction test")
    parser.add_argument(
        "--module",
        choices=list(MODULES.keys()),
        default="all",
        help="Which module to run (default: all)",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Use built-in sample_patient() without interactive prompts",
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="Optional path to a JSON file with patient_data fields",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("MANUAL SINGLE-PATIENT TEST (internal)")
    print("Uses production predict_* wrappers. Does not retrain.")
    print("Not a clinical diagnostic tool.")
    print("=" * 60)

    if args.json:
        patient = json.loads(Path(args.json).read_text())
    elif args.sample:
        patient = sample_patient()
        print("Using sample_patient() defaults.")
    else:
        patient = interactive_patient(sample_patient())

    names = ["anemia", "preeclampsia", "gdm", "thyroid"] if args.module == "all" else [args.module]

    results = {}
    for name in names:
        print(f"\n--- {name.upper()} ---")
        try:
            res = run_module(name, patient)
            results[name] = res
            print(f"  prediction : {res.get('prediction')}")
            print(f"  probability: {res.get('probability')}")
            print(f"  status     : {res.get('status')}")
            cp = res.get("class_probabilities") or {}
            if cp:
                print("  class probabilities:")
                for k, v in cp.items():
                    print(f"    {k}: {v}")
            factors = res.get("important_factors") or []
            if factors:
                print("  top factors:")
                for f in factors[:5]:
                    print(f"    - {f.get('feature')}: {f.get('shap_contribution')}")
            if res.get("notes"):
                print(f"  notes: {res['notes'][:160]}...")
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR: {exc}")
            results[name] = {"error": str(exc)}

    out_path = ROOT / "evaluation" / "results" / "manual_last_run.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"patient": patient, "results": results}, indent=2, default=str))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

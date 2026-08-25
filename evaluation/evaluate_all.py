#!/usr/bin/env python3
"""
Run external-data evaluation for all four condition modules.

Each --*-data argument is optional. Modules without a data path are skipped.
Does NOT retrain any model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.evaluate_anemia import run as run_anemia  # noqa: E402
from evaluation.evaluate_gdm import run as run_gdm  # noqa: E402
from evaluation.evaluate_preeclampsia import run as run_pe  # noqa: E402
from evaluation.evaluate_thyroid import run as run_thyroid  # noqa: E402


def main():
    p = argparse.ArgumentParser(
        description="Evaluate all frozen maternal-health modules on external test files"
    )
    p.add_argument("--anemia-data", default=None, help="CSV/Excel for anemia Track B evaluation")
    p.add_argument("--preeclampsia-data", default=None, help="CSV/Excel for preeclampsia proxy evaluation")
    p.add_argument("--gdm-data", default=None, help="CSV/Excel for GDM evaluation")
    p.add_argument("--thyroid-data", default=None, help="CSV/Excel for thyroid evaluation")
    p.add_argument("--anemia-target", default="Severity of anemia (on the basis of Hb)")
    p.add_argument("--preeclampsia-target", default="RiskLevel")
    p.add_argument("--gdm-target", default="gdm_outcome")
    p.add_argument("--thyroid-target", default="thyroid_dysfunction_later")
    args = p.parse_args()

    jobs = []
    if args.anemia_data:
        jobs.append(("Anemia", lambda: run_anemia(Path(args.anemia_data), args.anemia_target)))
    if args.preeclampsia_data:
        jobs.append(
            ("Preeclampsia", lambda: run_pe(Path(args.preeclampsia_data), args.preeclampsia_target))
        )
    if args.gdm_data:
        jobs.append(("GDM", lambda: run_gdm(Path(args.gdm_data), args.gdm_target)))
    if args.thyroid_data:
        jobs.append(("Thyroid", lambda: run_thyroid(Path(args.thyroid_data), args.thyroid_target)))

    if not jobs:
        p.error("Provide at least one of --anemia-data, --preeclampsia-data, --gdm-data, --thyroid-data")

    print("=" * 60)
    print("EXTERNAL TEST EVALUATION (frozen models only — no retrain)")
    print("=" * 60)
    failures = []
    for name, fn in jobs:
        print(f"\n>>> {name}")
        try:
            out = fn()
            print(f"    OK -> {out}")
        except Exception as exc:  # noqa: BLE001
            failures.append((name, str(exc)))
            print(f"    FAILED: {exc}")

    print("\n" + "=" * 60)
    if failures:
        print(f"Completed with {len(failures)} failure(s):")
        for name, msg in failures:
            print(f"  - {name}: {msg}")
        sys.exit(1)
    print("All requested evaluations completed.")


if __name__ == "__main__":
    main()

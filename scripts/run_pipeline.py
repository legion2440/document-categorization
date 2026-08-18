#!/usr/bin/env python3
"""Run the reproducible Revision 2 design/validation workflow without touching final test data."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args: str) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-download-models", action="store_true")
    parser.add_argument("--skip-prepare-data", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--optimize", action="store_true")
    args = parser.parse_args()

    py = sys.executable
    print(
        "Revision 2 pipeline: preprocessing -> train/validation preflight -> training -> "
        "calibration -> freeze -> validation verification. Held-out test is never read."
    )

    if not args.skip_download_models:
        run(py, "scripts/download_models.py")
    if not args.skip_prepare_data:
        run(py, "scripts/prepare_data.py")
    run(py, "scripts/preflight_revision2.py")

    if not args.skip_training:
        run(py, "scripts/train_revision2.py")
        run(py, "scripts/calibrate_validation.py")
        run(py, "scripts/freeze_production.py")
        run(py, "scripts/verify_production_validation.py")
        if args.optimize:
            run(py, "scripts/optimize_model.py")

    run(py, "scripts/validate_agent_contracts.py")
    run(py, "scripts/validate_project.py")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""One-command orchestration for the reproducible audit workflow."""
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
    parser.add_argument("--optimize", action="store_true")
    args = parser.parse_args()

    py = sys.executable
    if not args.skip_download_models:
        run(py, "scripts/download_models.py")
    if not args.skip_prepare_data:
        run(py, "scripts/prepare_data.py")
    run(py, "scripts/train.py")
    run(py, "scripts/evaluate.py")
    if args.optimize:
        run(py, "scripts/optimize_model.py")
    run(py, "scripts/validate_project.py")


if __name__ == "__main__":
    main()

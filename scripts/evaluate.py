#!/usr/bin/env python3
"""Show preserved final evaluation evidence without re-reading the consumed test split."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Backward-compatible statistical helpers used by the test suite and by external
# callers. Importing them does not execute the guarded Revision 2 evaluator.
from scripts.evaluate_revision2 import _cluster_bootstrap_improvement, _mcnemar_exact

METRICS = ROOT / "reports/performance_metrics.json"
REVISION1 = ROOT / "reports/revision1_performance_metrics.json"
REVISION2 = ROOT / "reports/revision2/performance_metrics.json"
MARKER = ROOT / "reports/revision2/final_test_consumed.json"


def main() -> int:
    if not MARKER.exists():
        raise SystemExit(
            "Final test evidence has not been consumed in this checkout. "
            "Use scripts/evaluate_revision2.py only under the registered one-time protocol."
        )

    marker = json.loads(MARKER.read_text(encoding="utf-8"))
    if marker.get("status") != "completed" or marker.get("test_split_read") is not True:
        raise SystemExit(
            "The irreversible Revision 2 final-test marker exists but is not completed. "
            "The protocol forbids another held-out test run."
        )
    if marker.get("third_final_test_forbidden") is not True:
        raise ValueError("Final-test marker does not enforce the no-third-run policy")

    missing = [str(path) for path in (REVISION1, REVISION2, METRICS) if not path.exists()]
    if missing:
        raise FileNotFoundError("Final evaluation evidence is incomplete: " + ", ".join(missing))

    metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    if metrics.get("revision") != 2 or metrics.get("final_test_attempt_number") != 2:
        raise ValueError("Top-level performance_metrics.json is not the preserved Revision 2 result")

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print("Final test is already consumed; no test data was read by scripts/evaluate.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

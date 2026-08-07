import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_agent_contracts_are_consistent():
    result = subprocess.run([sys.executable, "scripts/validate_agent_contracts.py"], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_pretraining_project_validation_has_no_structural_failures():
    result = subprocess.run([sys.executable, "scripts/validate_project.py"], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "[WARN] training not completed" in result.stdout

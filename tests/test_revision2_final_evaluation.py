from __future__ import annotations

import json

import pytest

import scripts.evaluate_revision2 as evaluation


def _redirect_guard_paths(monkeypatch, tmp_path):
    revision2_dir = tmp_path / "revision2"
    marker = revision2_dir / "final_test_consumed.json"
    metrics = revision2_dir / "performance_metrics.json"
    monkeypatch.setattr(evaluation, "REVISION2_DIR", revision2_dir)
    monkeypatch.setattr(evaluation, "CONSUMPTION_MARKER", marker)
    monkeypatch.setattr(evaluation, "REVISION2_METRICS", metrics)
    return marker, metrics


def test_revision2_final_guard_allows_first_attempt(monkeypatch, tmp_path):
    _redirect_guard_paths(monkeypatch, tmp_path)
    evaluation._assert_final_test_available()


def test_revision2_final_guard_rejects_existing_consumption_marker(monkeypatch, tmp_path):
    marker, _ = _redirect_guard_paths(monkeypatch, tmp_path)
    marker.parent.mkdir(parents=True)
    marker.write_text(json.dumps({"status": "started"}), encoding="utf-8")
    with pytest.raises(SystemExit, match="already been consumed"):
        evaluation._assert_final_test_available()


def test_revision2_final_guard_rejects_existing_metrics(monkeypatch, tmp_path):
    _, metrics = _redirect_guard_paths(monkeypatch, tmp_path)
    metrics.parent.mkdir(parents=True)
    metrics.write_text(json.dumps({"revision": 2}), encoding="utf-8")
    with pytest.raises(SystemExit, match="metrics already exist"):
        evaluation._assert_final_test_available()

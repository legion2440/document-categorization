#!/usr/bin/env python3
"""Validate repo-local agent navigation metadata against repository reality."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
AGENT = ROOT / "agent"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate_schema(document: Path, schema: Path) -> list[str]:
    validator = Draft202012Validator(load_json(schema))
    return [f"{document.relative_to(ROOT)}: {e.message}" for e in validator.iter_errors(load_json(document))]


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []
    errors += validate_schema(AGENT / "methodology.json", AGENT / "schemas/methodology.schema.json")
    errors += validate_schema(AGENT / "module-index.json", AGENT / "schemas/module-index.schema.json")
    errors += validate_schema(AGENT / "dependency-graph.json", AGENT / "schemas/dependency-graph.schema.json")

    index = load_json(AGENT / "module-index.json")
    modules = index["modules"]
    known = set(modules)
    manifest_dir = AGENT / "modules"
    manifest_paths = {p.stem: p for p in manifest_dir.glob("*.json")}

    for module_id, meta in modules.items():
        manifest_path = ROOT / meta["manifest"]
        if not manifest_path.exists():
            errors.append(f"missing manifest for module {module_id}: {meta['manifest']}")
            continue
        errors += validate_schema(manifest_path, AGENT / "schemas/module-manifest.schema.json")
        manifest = load_json(manifest_path)
        if manifest.get("module_id") != module_id:
            errors.append(f"manifest module_id mismatch for {module_id}")
        if meta["status"] == "implemented":
            for path in meta["roots"] + meta["entrypoints"]:
                if not (ROOT / path).exists():
                    errors.append(f"implemented module {module_id} path missing: {path}")
        for dep in meta["dependencies"]:
            if dep not in known:
                errors.append(f"module {module_id} references unknown dependency {dep}")

        for item in manifest.get("paths", []):
            path = ROOT / item["path"]
            lifecycle = item["lifecycle"]
            if lifecycle == "planned" and path.exists():
                errors.append(f"planned path already exists: {item['path']}")
            elif lifecycle in {"implemented", "deprecated"} and not path.exists():
                errors.append(f"{lifecycle} path missing: {item['path']}")
            if lifecycle != "planned" and item["provenance"] == "generated":
                generator = ROOT / item["generator"]
                if not generator.exists():
                    errors.append(f"generator missing for {item['path']}: {item['generator']}")
                elif item.get("check"):
                    result = subprocess.run(item["check"], shell=True, cwd=ROOT, capture_output=True, text=True)
                    if result.returncode:
                        errors.append(f"generated check failed for {item['path']}: {result.stderr.strip()}")

    for manifest_id in manifest_paths:
        if manifest_id not in known:
            errors.append(f"orphan manifest: agent/modules/{manifest_id}.json")

    graph = load_json(AGENT / "dependency-graph.json")
    graph_edges = {tuple(edge) for edge in graph["edges"]}
    index_edges = {(module, dep) for module, meta in modules.items() for dep in meta["dependencies"]}
    if graph_edges != index_edges:
        errors.append(f"dependency graph drift: graph={sorted(graph_edges)} index={sorted(index_edges)}")
    for left, right in graph_edges:
        if left not in known or right not in known:
            errors.append(f"dependency edge references unknown module: {left} -> {right}")

    index_size = (AGENT / "module-index.json").stat().st_size
    if index_size > 12 * 1024:
        warnings.append(f"module-index.json exceeds 12 KB warning budget: {index_size} bytes")
    for path in manifest_paths.values():
        if path.stat().st_size > 10 * 1024:
            warnings.append(f"{path.name} exceeds 10 KB warning budget: {path.stat().st_size} bytes")

    for warning in warnings:
        print(f"[WARN] {warning}")
    if errors:
        for error in errors:
            print(f"[FAIL] {error}")
        return 1
    print(f"[PASS] agent contracts: {len(modules)} modules, {len(graph_edges)} dependency edges")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

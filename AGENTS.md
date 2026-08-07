# Agent navigation

1. Read `agent/module-index.json`.
2. Identify the module that owns the requested feature using roots and entrypoints.
3. Read only `agent/modules/<module-id>.json` for the owning module.
4. Inspect the relevant production code/configuration.
5. Read `docs/architecture.md` only when architectural context is required.
6. Treat README and agent metadata as navigation, not implementation evidence.
7. For runtime claims, inspect generated reports/checkpoints or run the relevant verifier.
8. Before changing cross-module contracts, inspect `agent/dependency-graph.json`.
9. Run affected tests/domain validators after changes.
10. Run `python scripts/validate_agent_contracts.py` when agent metadata changes.
11. Do not read every module manifest unless the task is repository-wide.
12. If ownership is ambiguous, inspect roots and dependency edges; do not guess.

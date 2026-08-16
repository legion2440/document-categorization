import numpy as np
import pandas as pd

from scripts.evaluate import _cluster_bootstrap_improvement, _mcnemar_exact


def test_mcnemar_exact_counts_discordant_outcomes():
    transformer = np.asarray([True, True, False, True, False, True])
    baseline = np.asarray([True, False, True, False, False, True])
    result = _mcnemar_exact(transformer, baseline)
    assert result["transformer_only_correct"] == 2
    assert result["baseline_only_correct"] == 1
    assert result["discordant_pairs"] == 3
    assert 0.0 <= result["exact_two_sided_p_value"] <= 1.0


def test_cluster_bootstrap_is_deterministic_and_clustered_by_pair_id():
    frame = pd.DataFrame({"pair_id": ["a", "a", "b", "b", "c", "c"]})
    transformer = np.asarray([True, True, True, False, True, True])
    baseline = np.asarray([True, False, False, False, True, False])
    first = _cluster_bootstrap_improvement(frame, transformer, baseline, iterations=50, seed=7)
    second = _cluster_bootstrap_improvement(frame, transformer, baseline, iterations=50, seed=7)
    assert first == second
    assert first["cluster"] == "pair_id"
    assert first["clusters"] == 3
    assert first["iterations"] == 50

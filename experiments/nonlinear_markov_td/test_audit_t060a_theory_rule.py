from __future__ import annotations

import numpy as np

from experiments.nonlinear_markov_td.audit_t060a_theory_rule import (
    cluster_bootstrap,
    summarize_family,
    theory_action,
)


def test_theory_action_is_nonincreasing_with_correlation() -> None:
    rhos = (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0)
    for overhead in (8, 32):
        actions = [theory_action(overhead=overhead, rho=rho) for rho in rhos]
        assert all(first >= second for first, second in zip(actions, actions[1:]))
        assert actions[0] == 16
        assert actions[-1] == 1


def test_cluster_bootstrap_resamples_complete_seed_columns() -> None:
    proposed = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])
    baseline = 2.0 * proposed
    values = cluster_bootstrap(
        proposed, baseline, replicates=100, seed=7, batch_size=17
    )
    assert np.allclose(values, np.log(0.5))
    summary = summarize_family(
        proposed,
        baseline,
        bootstrap_replicates=100,
        bootstrap_seed=7,
    )
    assert np.isclose(summary["ratio"], 0.5)
    assert summary["strict_cell_fraction"] == 1.0

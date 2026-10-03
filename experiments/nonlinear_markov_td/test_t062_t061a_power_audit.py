from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.nonlinear_markov_td.t062_t061a_power_audit import (
    cluster_influence,
    matrices,
    projected_breadth,
    required_seed_count,
)


def test_matrices_preserve_complete_master_seed_columns() -> None:
    rows = []
    for seed in (11, 12, 13):
        for rho in (0.0, 0.5):
            rows.append(
                {
                    "master_seed": seed,
                    "game": "test",
                    "rho": rho,
                    "overhead": 8,
                    "delay": 0,
                    "controller_risk": seed + rho,
                    "strong_risk": 2 * seed + rho,
                }
            )
    first, second, cells, seeds = matrices(pd.DataFrame(rows))
    assert first.shape == (2, 3)
    assert second.shape == (2, 3)
    assert len(cells) == 2
    np.testing.assert_array_equal(seeds, [11, 12, 13])


def test_constant_ratio_has_zero_cluster_influence() -> None:
    second = np.arange(1.0, 13.0).reshape(3, 4)
    first = 0.8 * second
    result = cluster_influence(first, second)
    assert abs(result["point_ratio"] - 0.8) < 1e-14
    assert result["influence_standard_deviation"] < 1e-14


def test_required_count_increases_with_cluster_noise() -> None:
    low = required_seed_count(
        observed_ratio=0.8,
        threshold=0.95,
        influence_sd_upper=0.2,
        upper_quantile=0.95,
    )
    high = required_seed_count(
        observed_ratio=0.8,
        threshold=0.95,
        influence_sd_upper=0.4,
        upper_quantile=0.95,
    )
    assert high["required_seeds"] > low["required_seeds"]


def test_projected_breadth_is_deterministic() -> None:
    second = np.ones((5, 8))
    first = 0.8 * second
    first_result = projected_breadth(
        first,
        second,
        sample_sizes=(16,),
        replicates=100,
        seed=7,
    )
    second_result = projected_breadth(
        first,
        second,
        sample_sizes=(16,),
        replicates=100,
        seed=7,
    )
    assert first_result == second_result
    assert first_result["16"]["one_sided_025_lower"] == 1.0

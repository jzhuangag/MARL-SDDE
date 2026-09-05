import numpy as np

from experiments.dependence_delay_linear.run_t054_paired_power_audit import (
    DEFAULT_ENDPOINTS,
    DEFAULT_THEORY,
    assurance_seed_count,
    execute,
    seed_cluster_influence,
    vectorized_cluster_bootstrap,
)


def test_seed_cluster_influence_is_zero_for_constant_ratio():
    strong = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    controller = 0.8 * strong
    result = seed_cluster_influence(controller, strong)
    assert abs(result["ratio"] - 0.8) < 1e-15
    np.testing.assert_allclose(result["influence"], 0.0, atol=1e-15)


def test_vectorized_bootstrap_is_deterministic():
    controller = np.array([[1.0, 2.0, 4.0], [2.0, 3.0, 8.0]])
    strong = np.array([[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
    first = vectorized_cluster_bootstrap(
        controller, strong, replicates=200, seed=17, batch_size=31
    )
    second = vectorized_cluster_bootstrap(
        controller, strong, replicates=200, seed=17, batch_size=31
    )
    assert np.array_equal(first, second)


def test_assurance_seed_count_increases_with_uncertainty():
    small = assurance_seed_count(
        theory_ratio=0.86,
        gate_ratio=0.97,
        influence_sd_upper=0.2,
        assurance=0.95,
    )
    large = assurance_seed_count(
        theory_ratio=0.86,
        gate_ratio=0.97,
        influence_sd_upper=0.4,
        assurance=0.95,
    )
    assert large > small >= 1


def test_execute_end_to_end_on_frozen_inputs():
    result = execute(bootstrap_replicates=100, bootstrap_seed=7)
    assert result["endpoints_sha256"]
    assert result["theory_sha256"]
    assert result["endpoints_path"] == DEFAULT_ENDPOINTS.as_posix()
    assert result["theory_path"] == DEFAULT_THEORY.as_posix()
    assert result["reuse_T053A_seeds"] is False
    assert result["formal_authorized"] is False
    assert sorted(result["task_results"]) == [
        "cliffwalking",
        "frozenlake8x8",
        "taxi",
    ]

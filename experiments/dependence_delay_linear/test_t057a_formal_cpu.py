import numpy as np
import pytest

from experiments.dependence_delay_linear.analyze_t057a_formal_cpu import (
    active_fraction_inference,
    cluster_bootstrap_log_statistic,
    ratio_inference,
)
from experiments.dependence_delay_linear.run_t053a_sampled_cpu_pilot import (
    load_config as load_t053a,
)
from experiments.dependence_delay_linear.run_t055a_confirmation_cpu_pilot import (
    load_config as load_t055a,
)
from experiments.dependence_delay_linear.run_t057a_formal_cpu import load_config


def test_t057a_seed_registry_is_new_and_exact():
    config = load_config()
    seeds = config["pilot_seeds"]
    assert seeds == list(range(202608036001, 202608036257))
    assert len(seeds) == len(set(seeds)) == 256
    assert not set(seeds).intersection(load_t053a()["pilot_seeds"])
    assert not set(seeds).intersection(load_t055a()["pilot_seeds"])


def test_t057a_is_cpu_only_and_preserves_the_controller():
    config = load_config()
    pilot = load_t055a()
    for key in ("tasks", "kernel_sha256", "grid", "learning", "probe", "comparators"):
        assert config[key] == pilot[key]
    assert config["expected_workload"]["endpoints"] == 84 * 256
    assert config["expected_workload"]["long_learning_trajectories"] == 84 * 256 * 17
    assert config["expected_workload"]["recommended_hardware"] == "local CPU"
    assert not config["authorization"]["gpu"]
    assert not config["authorization"]["hpc4"]


def test_cluster_bootstrap_is_deterministic_and_clustered():
    numerator = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])
    denominator = np.array([[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]])
    first = cluster_bootstrap_log_statistic(
        numerator, denominator, replicates=200, seed=19, batch_size=17
    )
    second = cluster_bootstrap_log_statistic(
        numerator, denominator, replicates=200, seed=19, batch_size=31
    )
    assert np.array_equal(first, second)


def test_formal_ratio_and_breadth_gates_use_confidence_bounds():
    numerator = np.full((5, 20), 0.8)
    denominator = np.ones((5, 20))
    ratio = ratio_inference(
        numerator,
        denominator,
        upper_quantile=0.95,
        threshold=0.95,
        replicates=100,
        seed=23,
    )
    breadth = active_fraction_inference(
        numerator,
        denominator,
        lower_quantile=0.05,
        threshold=0.60,
        replicates=100,
        seed=23,
    )
    assert ratio["pass"] and ratio["one_sided_upper_ratio"] == pytest.approx(0.8)
    assert breadth["pass"] and breadth["one_sided_lower_fraction"] == 1.0

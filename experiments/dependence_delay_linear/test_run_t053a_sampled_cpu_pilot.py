from math import prod

from experiments.dependence_delay_linear.run_t053a_sampled_cpu_pilot import (
    load_config,
    scenario_rows,
    stable_seed,
)


def test_t053a_registry_is_frozen_and_cpu_only():
    config = load_config()
    assert config["analysis"]["uses_prior_outcome_rows"] is False
    assert config["expected_workload"]["recommended_hardware"] == "local CPU"
    assert not config["authorization"]["formal"]
    assert not config["authorization"]["gpu"]
    assert not config["authorization"]["hpc4"]
    assert len(config["pilot_seeds"]) == len(set(config["pilot_seeds"])) == 8


def test_t053a_grid_and_endpoint_counts_are_static():
    config = load_config()
    cells = prod(
        [
            len(config["tasks"]),
            len(config["grid"]["delays"]),
            len(config["grid"]["message_overheads"]),
            len(config["grid"]["correlations"]),
        ]
    )
    assert cells == len(scenario_rows(config)) == 84
    assert cells * len(config["pilot_seeds"]) == 672


def test_t053a_seed_derivation_is_stable_and_stream_separated():
    first = stable_seed(202608031001, "cell", "probe")
    assert first == stable_seed(202608031001, "cell", "probe")
    assert first != stable_seed(202608031001, "cell", "learning")
    assert first != stable_seed(202608031002, "cell", "probe")


def test_t053a_gates_are_complete_and_unchanged():
    gates = load_config()["mandatory_gates"]
    assert list(gates) == [f"P{index}" for index in range(1, 13)]
    assert "<= 0.95" in gates["P3"]
    assert "<= 0.97" in gates["P4"]
    assert "<= 0.97" in gates["P5"]
    assert "0.60" in gates["P6"]
    assert "<= 1.05" in gates["P7"]
    assert "<= 1.20" in gates["P8"]

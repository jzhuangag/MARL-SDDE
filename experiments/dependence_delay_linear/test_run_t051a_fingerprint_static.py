from math import prod

from experiments.dependence_delay_linear.run_t051a_fingerprint_static import load_config


def test_t051a_static_registry_and_taint_boundary():
    config = load_config()
    assert config["analysis"]["uses_t049a_outcome_rows"] is False
    assert config["expected_workload"]["sampled_trajectories"] == 0
    assert not any(
        config["authorization"][key]
        for key in ("sampled_cpu_pilot", "formal", "gpu", "hpc4")
    )
    expected = prod(
        [
            len(config["tasks"]),
            len(config["grid"]["message_overheads"]),
            len(config["grid"]["correlations"]),
            len(config["grid"]["delays"]),
        ]
    )
    assert expected == config["expected_workload"]["cells"] == 126


def test_t051a_gate_thresholds_are_frozen_without_outcomes():
    gates = load_config()["mandatory_gates"]
    assert list(gates) == [f"S{index}" for index in range(1, 13)]
    assert ">= 0.05" in gates["S4"]
    assert ">= 0.05" in gates["S5"]
    assert "0.70" in gates["S6"]
    assert "<= 1.05" in gates["S7"]


def test_t051a_probe_and_horizon_rules_are_explicit():
    config = load_config()
    assert config["probe"]["blocks"] == 96
    assert config["probe"]["q_probe"] == 2
    assert config["probe"]["maximum_independent_path_collision"] == 0.01
    assert config["learning_horizon_rule"]["contraction_target"] == 0.0001
    assert config["learning_horizon_rule"]["pr_burn_fraction"] == 0.5

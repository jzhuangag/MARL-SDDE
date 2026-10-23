from math import prod

from experiments.dependence_delay_linear.run_t052a_exact_fingerprint_static import (
    load_config,
)


def test_t052a_registry_is_outcome_free_and_static():
    config = load_config()
    assert config["analysis"]["uses_t049a_outcome_rows"] is False
    assert config["analysis"]["uses_t051a_result_rows"] is False
    assert not any(
        config["authorization"][key]
        for key in ("sampled_cpu_pilot", "formal", "gpu", "hpc4")
    )
    cells = prod(
        [
            len(config["tasks"]),
            len(config["grid"]["message_overheads"]),
            len(config["grid"]["correlations"]),
            len(config["grid"]["delays"]),
        ]
    )
    assert cells == config["expected_workload"]["cells"] == 126


def test_t052a_keeps_t051a_design_but_freezes_new_analysis():
    config = load_config()
    assert config["probe"]["blocks"] == 96
    assert config["probe"]["q_probe"] == 2
    assert config["probe"]["maximum_independent_path_collision"] == 0.01
    assert config["learning_horizon_rule"]["contraction_target"] == 0.0001
    assert config["grid"]["participation_catalogue"] == [1, 4, 16]
    assert "exact Binomial" in config["probe"]["decision_law"]


def test_t052a_gate_thresholds_are_frozen():
    gates = load_config()["mandatory_gates"]
    assert list(gates) == [f"B{index}" for index in range(1, 13)]
    assert ">= 0.05" in gates["B4"]
    assert ">= 0.05" in gates["B5"]
    assert "0.70" in gates["B6"]
    assert "<= 1.05" in gates["B7"]

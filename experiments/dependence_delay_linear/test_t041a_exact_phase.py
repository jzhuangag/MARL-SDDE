import numpy as np

from experiments.dependence_delay_linear.run_t041a_exact_phase import (
    DEFAULT_CONFIG,
    estimate,
    evaluate_action,
    load_config,
    scenario_rows,
    static_validate,
)


def test_frozen_configuration_has_expected_workload() -> None:
    config = load_config()
    assert static_validate(config) == {
        "experiment_id": "T-041A",
        "scenarios": 270,
        "actions": 6,
        "rows": 1620,
        "gpu": False,
    }


def test_scenario_identifiers_are_unique() -> None:
    scenarios = scenario_rows(load_config())
    assert len({scenario["scenario_id"] for scenario in scenarios}) == len(scenarios)


def test_static_estimate_remains_cpu_only() -> None:
    result = estimate(load_config())
    assert result["recommended_hardware"] == "local CPU"
    assert result["largest_horizon"] <= 96


def test_registered_result_directory_is_not_a_required_source_input() -> None:
    config = load_config()
    assert "result" not in config
    assert config["kind"] == "prospective_exact_cpu_phase_map"


def test_unregistered_toy_action_is_finite_and_scalar_consistent() -> None:
    scenario = {
        "scenario_id": "toy-not-registered",
        "family": "toy",
        "matrix_name": "toy",
        "matrix": [[1.0]],
        "rho": 0.25,
        "delay": 1,
        "markov_lambda": 0.5,
        "step_size": 0.05,
        "initial_scale": 1.0,
        "message_budget": 30,
        "environment_budget": 30,
    }
    row = evaluate_action(
        scenario,
        q=2,
        stride=1,
        overhead=1,
        private_variance_floor=1e-6,
    )
    assert np.isfinite(row["risk"])
    assert row["risk"] == row["scalar_reference_risk"]


def test_config_contains_all_frozen_gate_ids() -> None:
    config = load_config(DEFAULT_CONFIG)
    assert set(config["mandatory_gates"]) == {f"P{index}" for index in range(1, 11)}

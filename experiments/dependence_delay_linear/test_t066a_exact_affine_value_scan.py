import copy

from experiments.dependence_delay_linear.run_t066a_exact_affine_value_scan import (
    evaluate_action,
    load_config,
    scenario_rows,
    sensor_costs,
    validate,
)


def test_frozen_static_workload_and_certificates():
    result = validate(load_config())
    assert result["cells"] == 648
    assert result["actions_per_cell"] == 42
    assert result["action_rows"] == 27_216
    assert len(result["certificates"]) == 9
    assert all(row["margin"] > 0 for row in result["certificates"].values())


def test_sensor_costs_are_positive_and_depend_on_overhead():
    config = load_config()
    low = sensor_costs(config, 4)
    high = sensor_costs(config, 32)
    assert low[0] < high[0]
    assert low[1] == high[1] == 384


def test_action_is_finite_and_respects_both_budgets():
    config = load_config()
    for scenario in scenario_rows(config)[::79]:
        row = evaluate_action(config, scenario, participation=8, gain=0.01)
        assert row["risk"] > 0
        assert row["message_used"] <= row["message_budget"]
        assert row["environment_used"] <= row["environment_budget"]


def test_environment_binding_horizon_decreases_with_q():
    config = load_config()
    scenario = next(row for row in scenario_rows(config) if row["budget_regime"] == "environment_binding")
    q1 = evaluate_action(config, scenario, participation=1, gain=0.01)
    q8 = evaluate_action(config, scenario, participation=8, gain=0.01)
    assert q1["updates"] > q8["updates"]


def test_sampled_outcome_taint_is_rejected():
    config = copy.deepcopy(load_config())
    config["analysis"]["uses_sampled_outcome"] = True
    try:
        validate(config)
    except ValueError as error:
        assert "sampled outcome" in str(error)
    else:
        raise AssertionError("tainted config was accepted")

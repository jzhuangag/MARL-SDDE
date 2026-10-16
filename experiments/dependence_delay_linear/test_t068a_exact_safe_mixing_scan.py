import copy

from experiments.dependence_delay_linear.run_t068a_exact_safe_mixing_scan import (
    execute_policy,
    load_config,
    scenario_rows,
    validate,
)


def test_frozen_workload_is_exact():
    result = validate(load_config())
    assert result["cells"] == 648
    assert result["policies_per_cell"] == 43
    assert result["policy_rows"] == 27_864
    assert result["safe_message_units"] == 18


def test_safe_policy_fully_charges_probe_transitions():
    config = load_config()
    scenario = scenario_rows(config)[137]
    row = execute_policy(config, scenario, policy="safe_oracle")
    assert row["probe_transitions"] == 12
    assert row["learning_transitions"] == 228
    assert row["environment_used"] == row["environment_budget"] == 240
    assert row["message_used"] <= row["message_budget"]
    assert row["checkpoint_safe"]


def test_no_probe_local_uses_every_transition_for_learning():
    config = load_config()
    scenario = scenario_rows(config)[0]
    row = execute_policy(config, scenario, policy="fixed", early_alpha=0.0)
    assert row["probe_transitions"] == 0
    assert row["learning_transitions"] == 240
    assert row["terminal_risk"] == row["terminal_charged_shadow_risk"]


def test_sampled_outcome_taint_is_rejected():
    config = copy.deepcopy(load_config())
    config["analysis"]["uses_sampled_outcome"] = True
    try:
        validate(config)
    except ValueError as error:
        assert "sampled outcomes" in str(error)
    else:
        raise AssertionError("tainted configuration was accepted")

import copy

from experiments.dependence_delay_linear.run_t071a_sampled_observable_graph_pilot import (
    load_config,
    scenarios,
    shift_decision_indices,
    validate,
)


def test_frozen_t071a_workload_and_source():
    result = validate(load_config())
    assert result["cells"] == 432
    assert result["nonstationary_cells"] == 288
    assert result["pilot_seeds"] == 32
    assert result["endpoints"] == 13_824
    assert result["policy_trajectories"] == 82_944


def test_schedule_shift_indices_are_frozen():
    assert shift_decision_indices("stationary") == []
    assert shift_decision_indices("single_switch") == [3]
    assert shift_decision_indices("alternating") == [2, 4]


def test_controller_taint_is_rejected():
    config = copy.deepcopy(load_config())
    config["analysis"]["uses_t070_outcomes_for_controller"] = True
    try:
        validate(config)
    except ValueError as error:
        assert "cannot enter" in str(error)
    else:
        raise AssertionError("tainted controller configuration was accepted")


def test_scenario_ids_are_unique():
    rows = scenarios(load_config())
    assert len({row["cell_id"] for row in rows}) == len(rows)

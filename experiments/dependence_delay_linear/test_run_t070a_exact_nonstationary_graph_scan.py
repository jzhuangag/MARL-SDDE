import copy

from experiments.dependence_delay_linear.run_t070a_exact_nonstationary_graph_scan import (
    load_config,
    scenarios,
    unit_schedule,
    validate,
)


def test_frozen_t070a_workload():
    result = validate(load_config())
    assert result["cells"] == 432
    assert result["nonstationary_cells"] == 288
    assert result["actions_per_recipient"] == 7
    assert result["static_graphs_per_cell"] == 2401
    assert result["evaluated_static_graph_risks"] == 1_037_232


def test_schedule_changes_are_public_and_aligned():
    config = load_config()
    single = unit_schedule(config, "single_switch")
    alternating = unit_schedule(config, "alternating")
    assert (single[:12] == single[0]).all()
    assert (single[12:] == single[12]).all()
    assert (single[0] != single[12]).any()
    assert (alternating[:8] == alternating[0]).all()
    assert (alternating[8:16] == alternating[8]).all()
    assert (alternating[16:] == alternating[0]).all()


def test_sampled_outcome_taint_is_rejected():
    config = copy.deepcopy(load_config())
    config["analysis"]["uses_sampled_outcome"] = True
    try:
        validate(config)
    except ValueError as error:
        assert "sampled outcomes" in str(error)
    else:
        raise AssertionError("tainted configuration was accepted")


def test_cell_ids_are_unique():
    rows = scenarios(load_config())
    assert len({row["cell_id"] for row in rows}) == len(rows)

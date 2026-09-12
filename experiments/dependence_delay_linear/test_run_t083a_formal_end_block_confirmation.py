from experiments.dependence_delay_linear.run_t083a_formal_end_block_confirmation import (
    formal_rows,
    load_config,
    validate,
)


def test_t083a_static_validation_and_seed_isolation():
    result = validate(load_config())
    assert result["cells"] == 432
    assert result["primary_cells"] == 96
    assert result["formal_seeds"] == 128
    assert result["endpoints"] == 55_296
    assert result["prior_seed_overlap"] == 0
    assert result["scientific_outcome_created"] is False


def test_t083a_rows_are_complete_and_unique():
    config = load_config()
    rows = formal_rows(config)
    assert len(rows) == 55_296
    assert len({(row["cell_id"], row["seed"]) for row in rows}) == len(rows)


def test_t083a_freezes_pilot_design_and_stops_gpu():
    config = load_config()
    frozen = config["frozen_sources"]
    assert frozen["controller_changed_after_pilot"] is False
    assert frozen["primary_population_changed_after_pilot"] is False
    assert frozen["gates_changed_after_pilot"] is False
    assert config["authorization"]["gpu"] is False
    assert config["authorization"]["hpc4"] is False

from experiments.dependence_delay_linear.run_t082a_independent_end_block_pilot import (
    load_config,
    pilot_rows,
    validate,
)


def test_t082a_static_validation_and_seed_isolation():
    result = validate(load_config())
    assert result["cells"] == 432
    assert result["primary_cells"] == 96
    assert result["pilot_seeds"] == 64
    assert result["endpoints"] == 27_648
    assert result["old_seed_overlap"] == 0
    assert result["scientific_outcome_created"] is False


def test_t082a_rows_have_complete_seed_cell_cross_product():
    config = load_config()
    rows = pilot_rows(config)
    assert len(rows) == 27_648
    assert len({(row["cell_id"], row["seed"]) for row in rows}) == len(rows)


def test_t082a_stops_formal_and_gpu():
    authorization = load_config()["authorization"]
    assert authorization["formal"] is False
    assert authorization["gpu"] is False
    assert authorization["hpc4"] is False

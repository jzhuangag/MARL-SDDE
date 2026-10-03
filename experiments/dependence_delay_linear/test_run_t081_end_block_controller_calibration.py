import json

from experiments.dependence_delay_linear.run_t081_end_block_controller_calibration import (
    continuous_static_weights,
    is_primary,
    load_config,
    validate,
)


def test_t081_static_validation_and_primary_population():
    result = validate(load_config())
    assert result["cells"] == 432
    assert result["primary_cells"] == 96
    assert result["endpoints"] == 13_824
    assert result["scientific_outcome_created"] is False


def test_t081_continuous_static_weights_are_row_stochastic():
    weights = continuous_static_weights()
    assert len(weights) == 432
    for matrix in weights.values():
        assert matrix.shape == (4, 4)
        assert abs(matrix.sum(axis=1) - 1.0).max() < 1e-9
        assert matrix.min() >= 0.0


def test_t081_primary_rule_uses_only_registered_factors():
    base = {
        "schedule_family": "single_switch",
        "target_scale": "0.3",
        "temporal_correlation": "0",
    }
    assert is_primary(base)
    for key, value in (
        ("schedule_family", "stationary"),
        ("target_scale", "0.1"),
        ("temporal_correlation", "0.9"),
    ):
        changed = dict(base)
        changed[key] = value
        assert not is_primary(changed)


def test_t081_config_declares_taint_and_stops_new_seeds():
    config = load_config()
    assert config["source"]["new_scientific_seeds"] is False
    assert config["authorization"]["new_seed_pilot"] is False
    assert config["authorization"]["gpu"] is False

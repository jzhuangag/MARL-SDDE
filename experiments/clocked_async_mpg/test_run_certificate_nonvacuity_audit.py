from __future__ import annotations

from .run_certificate_nonvacuity_audit import load_config, scenarios, validate_config


def test_frozen_nonvacuity_population_has_expected_shape() -> None:
    config = load_config()
    cases = scenarios(config)
    assert len(cases) == 576
    assert {case["horizon"] for case in cases} == {2, 4, 8}
    assert {case["discount"] for case in cases} == {0.6, 0.8, 0.95}
    assert {case["owner"] for case in cases} == {0, 1}


def test_nonvacuity_thresholds_and_grid_are_outcome_free() -> None:
    config = load_config()
    assert config["trajectory_count_grid"] == [
        64,
        256,
        1024,
        4096,
        16384,
        65536,
        262144,
        1048576,
    ]
    assert config["practical_transition_cap"] == 8192
    assert config["extended_transition_cap"] == 16384
    assert set(config["mandatory_gates"]) == {"N1", "N2", "N3", "N4", "N5", "N6", "N7"}


def test_frozen_scenario_hash_matches() -> None:
    validation = validate_config(load_config())
    assert validation["case_count_matches"]
    assert validation["scenario_hash_matches"]

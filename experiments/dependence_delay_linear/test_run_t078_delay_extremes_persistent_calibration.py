from experiments.dependence_delay_linear.run_t078_delay_extremes_persistent_calibration import (
    load_config, selected_rows, validate,
)


def test_delay_extremes_are_outcome_free_and_complete() -> None:
    config = load_config()
    result = validate(config)
    assert result["outcome_free_delay_selection"] is True
    assert result["cells"] == 288
    assert result["endpoints"] == 9_216
    assert {int(row["delay"]) for row in selected_rows(config)} == {0, 3}


def test_old_seed_calibration_does_not_authorize_escalation() -> None:
    authorization = load_config()["authorization"]
    assert authorization["run_local_CPU_calibration"] is True
    assert all(authorization[key] is False for key in (
        "new_seed_pilot", "formal", "nonlinear", "gpu", "hpc4"))

import copy

from experiments.dependence_delay_linear.run_t079_continuous_static_headroom import (
    load_config,
    validate,
)


def test_t079_frozen_workload_and_scope():
    result = validate(load_config())
    assert result["cells"] == 432
    assert result["nonstationary_cells"] == 288
    assert result["optimizer_starts"] == 10
    assert result["scientific_outcome_created"] is False


def test_t079_rejects_sampled_outcome_static_optimizer():
    config = copy.deepcopy(load_config())
    config["static_optimizer"]["uses_sampled_outcomes"] = True
    try:
        validate(config)
    except ValueError as error:
        assert "sampled outcomes" in str(error)
    else:
        raise AssertionError("tainted optimizer configuration was accepted")


def test_t079_rejects_extra_probe_transitions():
    config = copy.deepcopy(load_config())
    config["dynamic_oracle"]["extra_probe_transitions"] = 1
    try:
        validate(config)
    except ValueError as error:
        assert "extra probes" in str(error)
    else:
        raise AssertionError("extra probe transitions were accepted")

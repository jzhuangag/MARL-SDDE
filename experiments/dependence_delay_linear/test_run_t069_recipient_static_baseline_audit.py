import copy

from experiments.dependence_delay_linear.run_t069_recipient_static_baseline_audit import (
    load_config,
    validate,
)


def test_frozen_t069_workload_and_source():
    result = validate(load_config())
    assert result["cells"] == 648
    assert result["vectors"] == 1296
    assert result["evaluated_risks"] == 839_808


def test_sampled_outcome_taint_is_rejected():
    config = copy.deepcopy(load_config())
    config["analysis"]["uses_sampled_outcome"] = True
    try:
        validate(config)
    except ValueError as error:
        assert "sampled outcomes" in str(error)
    else:
        raise AssertionError("tainted configuration was accepted")

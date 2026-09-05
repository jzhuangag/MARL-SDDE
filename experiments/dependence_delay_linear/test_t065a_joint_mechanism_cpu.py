import copy

import numpy as np
import pytest

from experiments.dependence_delay_linear.run_t065a_joint_mechanism_cpu import (
    _parameters,
    load_config,
    run_endpoint,
    scenario_rows,
    stable_seed,
    validate,
)


def test_frozen_registry_and_workload_are_exact():
    config = load_config()
    result = validate(config)
    assert result["cells"] == 324
    assert result["seeds"] == 64
    assert result["endpoints"] == 20_736


def test_stable_seed_is_repeatable_and_label_separated():
    assert stable_seed(1, "a") == stable_seed(1, "a")
    assert stable_seed(1, "a") != stable_seed(1, "b")


def test_endpoint_is_deterministic_finite_and_fully_charged():
    config = load_config()
    scenario = scenario_rows(config)[137]
    first = run_endpoint(config, scenario, config["pilot_seeds"][0])
    second = run_endpoint(config, scenario, config["pilot_seeds"][0])
    assert first == second
    assert all(
        np.isfinite(first[key])
        for key in ("signal_hat", "noise_hat", "rho_hat", "normalized_regret")
    )
    assert first["noise_hat"] >= 0.0
    assert first["environment_used"] <= first["environment_budget"]
    assert first["message_used"] <= first["message_budget"]
    assert first["sensor_actor_cost"] > 0
    assert first["sensor_message_cost"] > 0


def test_true_oracle_weakly_dominates_observable_action():
    config = load_config()
    for scenario in scenario_rows(config)[::31]:
        row = run_endpoint(config, scenario, config["pilot_seeds"][1])
        assert row["oracle_score"] <= row["observable_true_score"] + 1e-12
        assert row["normalized_regret"] >= -1e-12


def test_outcome_taint_and_unpaid_sensors_are_rejected():
    config = load_config()
    tainted = copy.deepcopy(config)
    tainted["analysis"]["uses_prior_outcome_rows"] = True
    with pytest.raises(ValueError, match="outcome-tainted"):
        validate(tainted)
    unpaid = copy.deepcopy(config)
    unpaid["sensor"]["fully_charged"] = False
    with pytest.raises(ValueError, match="fully charged"):
        validate(unpaid)


def test_parameter_construction_never_uses_unclipped_rho():
    config = load_config()
    scenario = scenario_rows(config)[0]
    assert _parameters(config, scenario, signal=1.0, noise=1.0, rho=2.0).rho_upper == 1.0

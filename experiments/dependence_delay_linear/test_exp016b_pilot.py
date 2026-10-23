"""Implementation and leakage tests for the EXP-016B pilot executor."""

from __future__ import annotations

import inspect

import numpy as np

from run_exp016b_pilot import (
    EXPECTED_CONFIGURATION_SHA256,
    POLICIES,
    execute_affine_td,
    generate_potential_observations,
    information_only_probe_decision,
    load_frozen_bundle,
    maximum_path_length,
    policy_probe_decision,
    registered_probe,
    scenario_lookup,
    simulate_policy,
)


def first_finite_record() -> tuple[dict[str, object], object]:
    bundle = load_frozen_bundle()
    record = next(
        item
        for item in bundle["manifest"]["scenarios"]
        if item["B_value_status"] == "finite" and item["B_id"] < item["B_value"]
    )
    return record, scenario_lookup()[record["scenario_id"]]


def test_frozen_bundle_and_policy_order() -> None:
    bundle = load_frozen_bundle()
    assert bundle["manifest"]["configuration_sha256"] == EXPECTED_CONFIGURATION_SHA256
    assert tuple(bundle["manifest"]["policies"]) == POLICIES


def test_potential_paths_are_deterministic_and_policy_independent() -> None:
    record, scenario = first_finite_record()
    length = maximum_path_length(record, scenario)
    left = generate_potential_observations(11, scenario, "high", "A_gaussian_mechanism", length)
    right = generate_potential_observations(11, scenario, "high", "A_gaussian_mechanism", length)
    assert np.array_equal(left[0], right[0])
    assert np.array_equal(left[1], right[1])


def test_registered_z_decisions_are_mechanical() -> None:
    record, scenario = first_finite_record()
    probe = registered_probe(scenario)
    point = next(
        item
        for item in record["budget_points"]
        if record["B_id"] <= item["scale"] < record["B_value"]
    )
    info = information_only_probe_decision(point["scale"], record["B_id"], probe)
    aware = policy_probe_decision("learning_aware", point["scale"], record, scenario, probe)
    assert info[0] is True
    assert aware[0] is False


def test_policy_decision_signature_has_no_hidden_outcome_input() -> None:
    parameters = set(inspect.signature(information_only_probe_decision).parameters)
    forbidden = {
        "theta_true",
        "regime",
        "outcome_data",
        "downstream_risk",
        "oracle_action",
        "epsilon_safe",
    }
    assert parameters.isdisjoint(forbidden)
    source = inspect.getsource(information_only_probe_decision)
    assert "B_value" not in source
    assert "epsilon_safe" not in source
    assert "downstream" not in source


def test_layer_a_and_b_runs_are_finite_and_fully_charged() -> None:
    record, scenario = first_finite_record()
    length = maximum_path_length(record, scenario)
    point = next(item for item in record["budget_points"] if item["name"] == "Z_midpoint")
    for layer in ("A_gaussian_mechanism", "B_affine_markov_td_transfer"):
        common, epsilon = generate_potential_observations(12, scenario, "high", layer, length)
        for policy in POLICIES:
            result = simulate_policy(
                12, layer, record, scenario, point, "high", policy, common, epsilon
            )
            assert result["finite"]
            assert result["dual_budget_valid"]
            assert result["messages_used"] <= result["message_budget"]
            assert result["environment_used"] <= result["environment_budget"]


def test_affine_queue_applies_exact_delay() -> None:
    observations = np.zeros(11, dtype=float)
    error, teacher, applied = execute_affine_td(observations, delay=4)
    assert applied == 7
    assert np.isfinite(error)
    assert np.isfinite(teacher)


def test_zero_update_layer_a_is_finite_initial_error() -> None:
    record, scenario = first_finite_record()
    length = maximum_path_length(record, scenario)
    common, epsilon = generate_potential_observations(
        13, scenario, "high", "A_gaussian_mechanism", length
    )
    point = next(item for item in record["budget_points"] if item["name"] == "at_B_id")
    result = simulate_policy(
        13,
        "A_gaussian_mechanism",
        record,
        scenario,
        point,
        "high",
        "information_only",
        common,
        epsilon,
    )
    assert result["finite"]
    if result["usable_updates_after_delay"] == 0:
        assert result["terminal_learning_risk"] == 1.0

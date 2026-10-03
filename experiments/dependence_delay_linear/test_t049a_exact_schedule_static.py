"""Static preregistration tests for T-049A."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from experiments.dependence_delay_linear.run_t049a_exact_schedule_static import (
    build_tasks,
    estimate,
    load_config,
    make_schedule,
    mixing_burn_in,
    scenario_rows,
    schedule_specifications,
    slice_lag_covariances,
    static_validate,
)
from experiments.dependence_delay_linear.t037_vector_markov_phase import (
    delayed_vector_companion,
)
from experiments.dependence_delay_linear.t049_standard_task_exact import (
    exact_gradient_lag_covariances,
)


@pytest.fixture(scope="module")
def config() -> dict[str, object]:
    return load_config()


@pytest.fixture(scope="module")
def tasks(config: dict[str, object]) -> dict[str, dict[str, object]]:
    return build_tasks(config)


def test_frozen_workload(config: dict[str, object], tasks: dict[str, object]) -> None:
    result = static_validate(config, tasks)
    assert result["tasks"] == 3
    assert result["base_scenarios"] == 36
    assert result["rho_cells"] == 252
    assert result["schedules_per_cell"] == 9
    assert result["rows"] == 2268


def test_estimated_maximum_horizon(
    config: dict[str, object], tasks: dict[str, object]
) -> None:
    result = estimate(config, tasks)
    assert result["maximum_full_updates"] == 256
    assert result["recommended_hardware"] == "local CPU"


def test_schedule_library_is_frozen_and_unique(config: dict[str, object]) -> None:
    specifications = schedule_specifications(config)
    names = [specification["name"] for specification in specifications]
    assert len(names) == len(set(names)) == 9
    assert names[:3] == ["fixed-q1", "fixed-q4", "fixed-q16"]


def test_two_stage_schedule_is_maximal_and_ordered(
    config: dict[str, object],
) -> None:
    specification = next(
        item
        for item in schedule_specifications(config)
        if item["name"] == "q1-to-q16-f0.5"
    )
    schedule = make_schedule(
        specification=specification,
        message_budget=1_000,
        environment_budget=10_000,
        message_overhead=8,
        delay=4,
    )
    assert schedule[0] == 1
    assert schedule[-1] == 16
    assert sum(8 + q for q in schedule) <= 1_000
    extended_cost = sum(8 + q for q in schedule) + 8 + 16
    assert extended_cost > 1_000


def test_fixed_schedule_uses_exact_budget() -> None:
    specification = {
        "name": "fixed-q4",
        "kind": "fixed",
        "q_first": 4,
        "q_second": 4,
        "fraction": 1.0,
    }
    schedule = make_schedule(
        specification=specification,
        message_budget=120,
        environment_budget=100,
        message_overhead=8,
        delay=2,
    )
    assert schedule == (4,) * 10


def test_lag_slice_preserves_orientation(tasks: dict[str, object]) -> None:
    task = tasks["frozenlake8x8"]
    full = exact_gradient_lag_covariances(task, horizon=12)
    sliced = slice_lag_covariances(full, full_horizon=12, horizon=5)
    direct = exact_gradient_lag_covariances(task, horizon=5)
    assert np.array_equal(sliced, direct)


def test_probe_burn_in_is_public_and_finite(tasks: dict[str, object]) -> None:
    values = [mixing_burn_in(task, 1e-4) for task in tasks.values()]
    assert all(value >= 1 for value in values)
    assert max(values) < 1_000


def test_registered_delayed_companions_are_strictly_stable(
    config: dict[str, object], tasks: dict[str, object]
) -> None:
    for task in tasks.values():
        step_size = (
            config["estimator"]["step_multiplier"]
            * (1.0 - task["mixing_slem"])
            / task["drift_norm"]
        )
        for delay in config["grid"]["delay"]:
            companion = delayed_vector_companion(task["drift"], step_size, delay)
            assert np.max(np.abs(np.linalg.eigvals(companion))) < 1.0


def test_no_outcome_or_compute_leakage(config: dict[str, object]) -> None:
    assert config["analysis"]["T043A_T045A_results_as_inputs"] is False
    assert config["sampled_learning_trajectory_authorized"] is False
    assert config["formal_or_gpu_authorized"] is False
    assert config["expected_workload"]["recommended_hardware"] == "local CPU"


def test_invalid_workload_is_rejected(
    config: dict[str, object], tasks: dict[str, object]
) -> None:
    broken = copy.deepcopy(config)
    broken["expected_workload"]["rows"] += 1
    with pytest.raises(ValueError, match="row count"):
        static_validate(broken, tasks)


def test_scenario_identifiers_are_unique(config: dict[str, object]) -> None:
    scenarios = scenario_rows(config)
    identifiers = [scenario["scenario_id"] for scenario in scenarios]
    assert len(identifiers) == len(set(identifiers))

import json
from pathlib import Path

import numpy as np

from experiments.dependence_delay_linear.run_t043a_standard_task_static import (
    DEFAULT_CONFIG,
    build_tasks,
    estimate,
    load_config,
    scenario_rows,
    static_validate,
)


def test_frozen_configuration_and_workload() -> None:
    config = load_config()
    validated = static_validate(config)
    assert validated["tasks"] == 2
    assert validated["scenarios"] == 144
    assert validated["rows"] == 1296
    assert validated["sampled_trajectories"] == 0


def test_public_task_kernels_and_features_are_exact() -> None:
    tasks = build_tasks(load_config())
    for task in tasks.values():
        np.testing.assert_allclose(task["continuing_transition"].sum(axis=1), 1.0)
        np.testing.assert_allclose(
            task["stationary"] @ task["continuing_transition"], task["stationary"]
        )
        np.testing.assert_allclose(
            task["features"].T @ (task["stationary"][:, None] * task["features"]),
            np.eye(task["features"].shape[1]),
            atol=1e-10,
        )
        assert task["drift_minimum"] > 0.0
        assert 0.0 <= task["mixing_slem"] < 1.0
        assert task["single_agent_noise_second"] > 0.0


def test_every_scenario_id_is_unique() -> None:
    scenarios = scenario_rows(load_config())
    assert len({row["scenario_id"] for row in scenarios}) == len(scenarios)


def test_estimate_is_local_cpu_and_matches_horizon() -> None:
    result = estimate(load_config())
    assert result["maximum_scalar_horizon"] == 256
    assert result["recommended_hardware"] == "local CPU"


def test_no_sampled_or_gpu_authorization() -> None:
    config = json.loads(Path(DEFAULT_CONFIG).read_text(encoding="utf-8"))
    assert config["sampled_learning_trajectory_authorized"] is False
    assert config["formal_or_gpu_authorized"] is False
    assert config["analysis"]["no_seeds_or_confidence_intervals"] is True


def test_result_directory_is_not_a_required_source_input() -> None:
    source = Path(__file__).with_name("run_t043a_standard_task_static.py").read_text(
        encoding="utf-8"
    )
    assert "results/t043a_standard_task_static" not in source
    assert "read_csv" not in source

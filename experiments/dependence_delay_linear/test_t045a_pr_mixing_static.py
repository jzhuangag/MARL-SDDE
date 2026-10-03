import json
from pathlib import Path

from experiments.dependence_delay_linear.run_t045a_pr_mixing_static import (
    DEFAULT_CONFIG,
    build_registered_tasks,
    estimate,
    load_config,
    registered_scenarios,
    static_validate,
)


def test_frozen_workload_and_cpu_scope() -> None:
    result = static_validate(load_config())
    assert result["scenarios"] == 144
    assert result["actions_per_scenario"] == 12
    assert result["rows"] == 1728
    assert result["sampled_trajectories"] == 0
    assert result["recommended_hardware"] == "local CPU"


def test_task_kernels_match_t043a_hashes() -> None:
    tasks = build_registered_tasks(load_config())
    assert tasks["frozenlake4x4"]["kernel_sha256"] == "ee41ccb973511d1d3dd6ccde80203756358d16a00a127f155cd6450569fc5d88"
    assert tasks["cliffwalking"]["kernel_sha256"] == "95f5b560632799694042c7e4652be20bdd2ffaf0803b3f0e77d024779cc8f633"


def test_scenario_ids_and_cells_are_unchanged_from_t043a() -> None:
    scenarios = registered_scenarios(load_config())
    assert len({row["scenario_id"] for row in scenarios}) == 144
    assert {row["rho"] for row in scenarios} == {0.0, 0.1, 0.9, 1.0}
    assert {row["delay"] for row in scenarios} == {0, 1, 3}


def test_estimated_horizon_is_frozen() -> None:
    assert estimate(load_config())["maximum_scalar_horizon"] == 256


def test_t043_results_and_gpu_are_forbidden() -> None:
    config = json.loads(Path(DEFAULT_CONFIG).read_text(encoding="utf-8"))
    assert config["analysis"]["T043A_results_as_inputs"] is False
    assert config["sampled_learning_trajectory_authorized"] is False
    assert config["formal_or_gpu_authorized"] is False


def test_runner_does_not_read_t043_result_artifacts() -> None:
    source = Path(__file__).with_name("run_t045a_pr_mixing_static.py").read_text(
        encoding="utf-8"
    )
    assert "t043a_standard_task_static/rows.csv" not in source
    assert "read_csv" not in source

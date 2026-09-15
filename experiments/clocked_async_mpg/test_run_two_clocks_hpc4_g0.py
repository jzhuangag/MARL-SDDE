from __future__ import annotations

import pytest

from .run_two_clocks_hpc4_g0 import _assert_outcome_free, validate_summary


def _task(name: str) -> dict[str, object]:
    return {
        "task": name,
        "status": "pass",
        "agents": 4,
        "environment_transitions": 1,
        "actor_transitions": 4,
        "policy": {"distinct_actor_objects": True},
    }


def _valid_summary() -> dict[str, object]:
    return {
        "scope": "outcome-free Two Clocks HPC4 G0",
        "scientific_outcome_generated": False,
        "tasks": [_task("mamujoco_ant_4x2"), _task("smacv2_terran_5v5")],
        "invariants": {"cuda": True, "teardown": True},
    }


def test_valid_outcome_free_summary_passes() -> None:
    validate_summary(_valid_summary())


@pytest.mark.parametrize(
    "key", ["reward", "return", "episode_reward", "win_rate", "final_return"]
)
def test_scientific_outcome_fields_are_rejected(key: str) -> None:
    summary = _valid_summary()
    summary["nested"] = [{key: 1.0}]
    with pytest.raises(RuntimeError, match="prohibited G0 outcome key"):
        _assert_outcome_free(summary)


def test_failed_task_is_rejected() -> None:
    summary = _valid_summary()
    summary["tasks"][0]["status"] = "fail"  # type: ignore[index]
    with pytest.raises(RuntimeError, match="did not pass"):
        validate_summary(summary)


def test_wrong_task_set_is_rejected() -> None:
    summary = _valid_summary()
    summary["tasks"][1]["task"] = "unregistered"  # type: ignore[index]
    with pytest.raises(RuntimeError, match="unexpected G0 task set"):
        validate_summary(summary)

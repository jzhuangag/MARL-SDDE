import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
MODULE_PATH = EXPERIMENTS / "harl_online_runtime_bridge_v2.py"
SPEC = importlib.util.spec_from_file_location("harl_online_runtime_bridge_v2", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class _PinnedRunner:
    args = {"algo": "mappo", "env": "mamujoco", "exp_name": "qualification"}


def test_environment_name_uses_pinned_harl_args_mapping():
    assert MODULE.runner_environment_name(_PinnedRunner()) == "mamujoco"


def test_unavailable_action_representations_match_pinned_harl_contract():
    assert MODULE.available_actions_for_agent(None, 0) is None
    one_dimensional_none = np.asarray([None, None], dtype=object)
    assert MODULE.available_actions_for_agent(one_dimensional_none, 1) is None
    mask = np.arange(24).reshape(2, 3, 4)
    assert np.array_equal(
        MODULE.available_actions_for_agent(mask, 1),
        mask[:, 1],
    )


def test_empty_or_invalid_action_mask_is_rejected():
    with pytest.raises(ValueError, match="cannot be empty"):
        MODULE.available_actions_for_agent(np.asarray([]), 0)
    with pytest.raises(ValueError, match="include agent axis"):
        MODULE.available_actions_for_agent(np.asarray([1.0, 0.0]), 0)

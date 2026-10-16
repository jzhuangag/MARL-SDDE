import importlib.util
import sys
from pathlib import Path

import pytest


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
MODULE_PATH = EXPERIMENTS / "harl_online_runtime_bridge.py"
SPEC = importlib.util.spec_from_file_location("harl_online_runtime_bridge", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class _PinnedRunner:
    args = {"algo": "mappo", "env": "mamujoco", "exp_name": "qualification"}


def test_environment_name_uses_pinned_harl_args_mapping():
    assert MODULE.runner_environment_name(_PinnedRunner()) == "mamujoco"


def test_environment_name_rejects_legacy_nonexistent_attribute_shape():
    class LegacyShape:
        env_name = "mamujoco"

    with pytest.raises(TypeError, match="mapping-valued args"):
        MODULE.runner_environment_name(LegacyShape())


@pytest.mark.parametrize("args", [{}, {"env": ""}, {"env": None}])
def test_environment_name_requires_nonempty_registered_environment(args):
    runner = type("Runner", (), {"args": args})()
    with pytest.raises(ValueError, match="nonempty env"):
        MODULE.runner_environment_name(runner)

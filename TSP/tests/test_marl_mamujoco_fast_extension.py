import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))
MODULE_PATH = EXPERIMENTS / "analyze_marl_mamujoco_fast_extension.py"
SPEC = importlib.util.spec_from_file_location("mamujoco_fast_analyzer", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_frozen_fast_extension_configuration_is_complete_and_disjoint() -> None:
    config = json.loads(
        (EXPERIMENTS / "marl_mamujoco_fast_extension.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["task"]["environment"] == "mamujoco"
    assert config["task"]["scenario"] == "HalfCheetah-v2"
    assert config["task"]["agent_conf"] == "2x3"
    assert config["methods"] == [
        "fixed_q1",
        "fixed_q2",
        "fixed_q4",
        "fixed_q8",
        "lyapunov_probe_commit",
    ]
    assert config["controller"]["candidate_q"] == [1, 2, 4, 8]
    training = set(config["seed_registry"]["development_training"])
    probe = set(config["seed_registry"]["development_probe"])
    assert training.isdisjoint(probe)
    assert config["planned_runs"] == 40
    probe_messages = config["controller"]["probe_blocks"] * (
        config["task"]["server_overhead"]
        + config["controller"]["probe_q"] * config["task"]["rollout_length"]
    )
    assert probe_messages / config["task"]["message_budget"] <= 0.01


def test_relative_gain_handles_negative_and_positive_returns() -> None:
    assert MODULE.relative_gain(-9.0, -10.0) == 0.1
    assert MODULE.relative_gain(11.0, 10.0) == 0.1

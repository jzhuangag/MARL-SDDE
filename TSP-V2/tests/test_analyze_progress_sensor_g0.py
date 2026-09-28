import importlib.util
import json
import sys
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "analyze_progress_sensor_g0.py"
)
SPEC = importlib.util.spec_from_file_location("analyze_progress_sensor_g0", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_complete_fixture_passes(tmp_path):
    tasks = [
        ("pettingzoo_mpe", "simple_spread_v2"),
        ("mamujoco", "HalfCheetah-v2/2x3"),
    ]
    for env, task in tasks:
        for coupling in ["independent", "shared"]:
            for seed in [140101, 140102]:
                directory = tmp_path / f"{env}-{coupling}-{seed}"
                directory.mkdir()
                selected = 8 if coupling == "independent" else 2
                record = {
                    "experiment_id": "TSP-V2-MARL-PROGRESS-G0",
                    "environment": env,
                    "task": task,
                    "coupling": coupling,
                    "seed": seed,
                    "candidate_q": [1, 2, 4, 8],
                    "micro_updates": 24,
                    "selected_q_diagnostic": selected,
                    "evaluation_return_computed": False,
                    "results": {
                        str(q): {
                            "initial_parameter_sha256": "same",
                            "reward_trace": [float(i + q) for i in range(24)],
                            "charged_messages": 100 + q,
                            "charged_environment_ticks": 24,
                        }
                        for q in [1, 2, 4, 8]
                    },
                }
                (directory / "progress_sensor_g0.json").write_text(
                    json.dumps(record), encoding="utf-8"
                )
    config = {
        "experiment_id": "TSP-V2-MARL-PROGRESS-G0",
        "tasks": [
            {
                "environment": env,
                "task": task,
                "couplings": ["independent", "shared"],
                "seeds": [140101, 140102],
            }
            for env, task in tasks
        ],
        "candidate_q": [1, 2, 4, 8],
        "micro_updates": 24,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    result = MODULE.evaluate(tmp_path, config_path, replay_identical=True)
    assert result["pass"] is True
    assert result["records"] == 8


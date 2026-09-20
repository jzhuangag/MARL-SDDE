import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
sys.path.insert(0, str(EXPERIMENTS))
MODULE_PATH = EXPERIMENTS / "analyze_marl_multitask_phase_headroom.py"
SPEC = importlib.util.spec_from_file_location("multitask_phase_analyzer", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _write_run(
    root: Path,
    *,
    experiment_id: str,
    harl_commit: str,
    task: dict,
    ray: str,
    coupling: str,
    q: int,
    seed: int,
    value: float,
) -> None:
    run = root / f'{task["task_id"]}-{ray}-{coupling}-q{q}-s{seed}'
    run.mkdir(parents=True)
    rollout = int(task["rollout_length"])
    message_budget = int(task["message_budgets"][ray])
    environment_budget = int(task["environment_budget"])
    overhead = int(task["server_overhead"])
    message_cost = overhead + q * rollout
    updates = min(message_budget // message_cost, environment_budget // rollout)
    task_name = (
        f'{task["scenario"]}/{task["agent_conf"]}'
        if task["environment"] == "mamujoco"
        else task["scenario"]
    )
    metadata = {
        "experiment_id": experiment_id,
        "environment": task["environment"],
        "task": task_name,
        "scenario": task["scenario"] if task["environment"] == "pettingzoo_mpe" else None,
        "map_name": None,
        "coupling": coupling,
        "q_rollout_workers": q,
        "rollout_length": rollout,
        "message_budget": message_budget,
        "environment_budget": environment_budget,
        "server_overhead": overhead,
        "message_cost_per_update": message_cost,
        "environment_cost_per_update": rollout,
        "usable_updates": updates,
        "charged_training_actor_transitions": updates * q * rollout,
        "charged_training_messages": updates * message_cost,
        "charged_training_environment_ticks": updates * rollout,
        "seed": seed,
        "harl_commit": harl_commit,
        "upstream_modified": False,
    }
    (run / "tsp_bridge_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    (run / "progress.txt").write_text(
        f"0,{value}\n{updates * q * rollout},{value}\n", encoding="utf-8"
    )


def test_frozen_multitask_analyzer_recovers_phase_and_headroom(tmp_path) -> None:
    config_path = EXPERIMENTS / "marl_multitask_phase_headroom.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    values = {
        ("message_binding", "independent", 1): 8.0,
        ("message_binding", "independent", 8): 12.0,
        ("message_binding", "shared", 1): 12.0,
        ("message_binding", "shared", 8): 8.0,
        ("environment_binding", "independent", 1): 8.0,
        ("environment_binding", "independent", 8): 12.0,
        ("environment_binding", "shared", 1): 10.0,
        ("environment_binding", "shared", 8): 11.0,
    }
    for task in config["tasks"]:
        for ray in task["message_budgets"]:
            for coupling in config["coupling_regimes"]:
                for q in config["fixed_q_endpoints"]:
                    for seed in config["development_seeds"]:
                        _write_run(
                            tmp_path,
                            experiment_id=config["experiment_id"],
                            harl_commit=config["upstream"]["harl_commit"],
                            task=task,
                            ray=ray,
                            coupling=coupling,
                            q=q,
                            seed=seed,
                            value=values[(ray, coupling, q)],
                        )
    result, cells = MODULE.evaluate(
        tmp_path, config_path, replay_identical=True
    )
    assert result["run_count"] == result["expected_run_count"] == 32
    assert len(cells) == 8
    assert result["pass"] is True
    assert result["decision"] == "authorize-full-q-and-controller-freeze"
    assert all(result["gates"].values())
    for metrics in result["task_metrics"].values():
        assert metrics["distinct_oracle_actions"] == [1, 8]
        assert metrics["oracle_relative_headroom"] > 0.02

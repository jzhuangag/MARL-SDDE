import importlib.util
import json
import sys
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "analyze_marl_online_drift_queue.py"
)
SPEC = importlib.util.spec_from_file_location("analyze_marl_online", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _write_run(root, env, task, coupling, method, seed, final):
    path = root / f"{env}-{coupling}-{method}-{seed}"
    path.mkdir()
    metadata = {
        "environment": env,
        "task": task,
        "coupling": coupling,
        "method": method,
        "seed": seed,
        "accounting": {
            "charged_messages": 90,
            "charged_environment_ticks": 80,
            "selected_counts": ({"1": 2, "2": 2} if method == "controller" else {method[-1]: 4}),
        },
        "message_budget": 100,
        "environment_budget": 100,
        "evaluation_used_by_controller": False,
    }
    (path / "metadata.json").write_text(json.dumps(metadata))
    rows = [
        {"budget_fraction": 0.0, "evaluation_mean_step_return": 0.0},
        {"budget_fraction": 1.0, "evaluation_mean_step_return": final},
    ]
    (path / "progress.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )


def test_analyzer_uses_strong_fixed_and_requires_all_cells(tmp_path):
    for env, task in (("mpe", "spread"), ("mamujoco", "halfcheetah")):
        for coupling in ("independent", "shared"):
            for seed in (1, 2):
                for method, final in (
                    ("fixed_q1", 1.0),
                    ("fixed_q2", 2.0),
                    ("fixed_q4", 1.5),
                    ("fixed_q8", 1.0),
                    ("controller", 2.2),
                ):
                    _write_run(tmp_path, env, task, coupling, method, seed, final)
    rows, decision = MODULE.analyze(tmp_path, [1, 2])
    assert len(rows) == 4
    assert all(row["best_fixed_method"] == "fixed_q2" for row in rows)
    assert decision["development_pass"]


def test_auc_extends_last_observation_to_full_budget():
    rows = [
        {"budget_fraction": 0.0, "evaluation_mean_step_return": 0.0},
        {"budget_fraction": 0.5, "evaluation_mean_step_return": 1.0},
    ]
    assert MODULE.normalized_auc(rows, points=3) == 0.75

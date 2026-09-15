import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "experiments" / "analyze_mappo_smacv2_development.py"
SPEC = importlib.util.spec_from_file_location("tsp_smacv2_analyzer", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _write_fixed(
    root: Path,
    coupling: str,
    seed: int,
    q: int,
    win_rate: float,
    team_return: float,
    harl_commit: str,
) -> None:
    run = root / f"fixed-{coupling}-q{q}-{seed}"
    run.mkdir(parents=True)
    rollout_length = 200
    message_budget = 16_000_000
    environment_budget = 3_000_000
    message_cost = 800 + q * rollout_length
    updates = min(
        message_budget // message_cost, environment_budget // rollout_length
    )
    actor_transitions = updates * q * rollout_length
    metadata = {
        "q_rollout_workers": q,
        "rollout_length": rollout_length,
        "coupling": coupling,
        "seed": seed,
        "message_cost_per_update": message_cost,
        "environment_cost_per_update": rollout_length,
        "message_budget": message_budget,
        "environment_budget": environment_budget,
        "usable_updates": updates,
        "charged_training_messages": updates * message_cost,
        "charged_training_environment_ticks": updates * rollout_length,
        "upstream_modified": False,
        "harl_commit": harl_commit,
    }
    (run / "tsp_bridge_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    (run / "progress.txt").write_text(
        f"0,{team_return},{win_rate}\n"
        f"{actor_transitions},{team_return},{win_rate}\n",
        encoding="utf-8",
    )


def _write_controller(
    root: Path,
    coupling: str,
    seed: int,
    selected_q: int,
    win_rate: float,
    team_return: float,
    harl_commit: str,
) -> None:
    run = root / f"controller-{coupling}-{seed}"
    run.mkdir(parents=True)
    probe_messages = 153_600
    probe_environment = 12_800
    training_updates = 100
    training_messages = training_updates * (800 + selected_q * 200)
    training_environment = training_updates * 200
    accounting = {
        "probe_messages": probe_messages,
        "probe_environment_ticks": probe_environment,
        "training_messages": training_messages,
        "training_environment_ticks": training_environment,
        "total_messages": probe_messages + training_messages,
        "total_environment_ticks": probe_environment + training_environment,
    }
    metadata = {
        "coupling": coupling,
        "training_seed": seed,
        "message_budget": 16_000_000,
        "environment_budget": 3_000_000,
        "decision": {
            "selected_q": selected_q,
            "certificate": {"estimate": 0.0, "upper": 0.1},
        },
        "accounting": accounting,
        "selection_overhead_fraction": 1e-4,
        "upstream_modified": False,
        "harl_commit": harl_commit,
    }
    (run / "tsp_probe_commit_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    (run / "charged_progress.csv").write_text(
        "budget_fraction,team_return,win_rate\n"
        f"0,{team_return},{win_rate}\n"
        f"1,{team_return},{win_rate}\n",
        encoding="utf-8",
    )


def test_synthetic_gate_uses_strong_static_and_regime_oracle(tmp_path) -> None:
    config_path = ROOT / "experiments" / "marl_smacv2_development.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    commit = config["upstream"]["harl_commit"]
    for coupling in ("independent", "shared"):
        for seed in config["seed_registry"]["development_training"]:
            if coupling == "independent":
                q1_win, q8_win, controller_win = 0.30, 0.85, 0.84
                selected_q = 8
            else:
                q1_win, q8_win, controller_win = 0.70, 0.25, 0.69
                selected_q = 1
            _write_fixed(tmp_path, coupling, seed, 1, q1_win, q1_win, commit)
            _write_fixed(tmp_path, coupling, seed, 8, q8_win, q8_win, commit)
            _write_controller(
                tmp_path,
                coupling,
                seed,
                selected_q,
                controller_win,
                controller_win,
                commit,
            )
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps({"pass": True, "analysis_replay_byte_identical": True}),
        encoding="utf-8",
    )
    result = MODULE.evaluate_gates(tmp_path, config_path, audit)
    assert result["strong_static_method"] == "fixed_q8"
    assert result["static_oracle_win_rate_auc_headroom_absolute"] == pytest.approx(
        0.225
    )
    assert result[
        "controller_equal_mixture_win_rate_auc_gain_over_strong_static_absolute"
    ] == pytest.approx(0.215)
    assert result[
        "controller_min_regime_win_rate_auc_gap_from_matching_fixed"
    ] == pytest.approx(-0.01)
    assert result["all_mandatory_gates_pass"]
    assert result["decision"] == "authorize-confirmation-preregistration"


def test_win_rate_auc_rejects_out_of_range_values() -> None:
    import pandas as pd

    frame = pd.DataFrame(
        {"budget_fraction": [0.0, 1.0], "win_rate": [0.0, 1.1]}
    )
    with pytest.raises(ValueError, match="win rate"):
        MODULE.auc_on_budget_fraction(frame, "win_rate")

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
TSP_EXPERIMENTS = ROOT.parent / "TSP" / "experiments"
if str(TSP_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(TSP_EXPERIMENTS))

LYAPUNOV_PATH = TSP_EXPERIMENTS / "lyapunov_probe_commit.py"
LYAPUNOV_SPEC = importlib.util.spec_from_file_location("v3_lyapunov", LYAPUNOV_PATH)
LYAPUNOV = importlib.util.module_from_spec(LYAPUNOV_SPEC)
assert LYAPUNOV_SPEC.loader is not None
sys.modules[LYAPUNOV_SPEC.name] = LYAPUNOV
LYAPUNOV_SPEC.loader.exec_module(LYAPUNOV)

ANALYZER_PATH = ROOT / "experiments" / "analyze_marl_mamujoco_calibrated_development.py"
ANALYZER_SPEC = importlib.util.spec_from_file_location("v3_analyzer", ANALYZER_PATH)
ANALYZER = importlib.util.module_from_spec(ANALYZER_SPEC)
assert ANALYZER_SPEC.loader is not None
sys.modules[ANALYZER_SPEC.name] = ANALYZER
ANALYZER_SPEC.loader.exec_module(ANALYZER)


def config() -> dict:
    return json.loads(
        (ROOT / "experiments" / "marl_mamujoco_calibrated_development.json").read_text(
            encoding="utf-8"
        )
    )


def test_registered_catalogue_selects_q8_low_and_q2_high() -> None:
    low_scores = {
        q: LYAPUNOV.message_limited_lyapunov_score(q, 0.0, 800, 200)
        for q in (2, 8)
    }
    high_scores = {
        q: LYAPUNOV.message_limited_lyapunov_score(q, 1.0, 800, 200)
        for q in (2, 8)
    }
    assert min(low_scores, key=low_scores.get) == 8
    assert min(high_scores, key=high_scores.get) == 2


def test_config_is_development_only_and_complete() -> None:
    cfg = config()
    assert cfg["role"].startswith("development crossover")
    assert cfg["controller"]["candidate_q"] == [2, 8]
    assert cfg["fixed_methods"] == ["fixed_q1", "fixed_q2", "fixed_q4", "fixed_q8"]
    assert cfg["planned_new_runs"] == 8
    assert cfg["combined_analysis_runs"] == 40
    assert set(cfg["seed_registry"]["development_training"]).isdisjoint(
        cfg["seed_registry"]["development_probe"]
    )
    assert "confirmation" not in json.dumps(cfg["seed_registry"]).lower()


def synthetic_records(cfg: dict) -> pd.DataFrame:
    rows = []
    fixed_values = {
        "independent": {
            "fixed_q1": 100.0,
            "fixed_q2": 110.0,
            "fixed_q4": 120.0,
            "fixed_q8": 130.0,
        },
        "shared": {
            "fixed_q1": 100.0,
            "fixed_q2": 130.0,
            "fixed_q4": 115.0,
            "fixed_q8": 90.0,
        },
    }
    task = f'{cfg["task"]["scenario"]}/{cfg["task"]["agent_conf"]}'
    for coupling in cfg["coupling_regimes"]:
        for seed in cfg["seed_registry"]["development_training"]:
            for method in cfg["fixed_methods"]:
                rows.append(
                    {
                        "task": task,
                        "coupling": coupling,
                        "method": method,
                        "training_seed": seed,
                        "return_auc": fixed_values[coupling][method],
                        "selected_q": int(method[-1]),
                        "accounting_ok": True,
                        "upstream_modified": False,
                        "harl_commit": cfg["upstream"]["harl_commit"],
                        "probe_message_fraction": 0.0,
                        "selection_overhead_fraction": 0.0,
                    }
                )
            rows.append(
                {
                    "task": task,
                    "coupling": coupling,
                    "method": cfg["controller_method"],
                    "training_seed": seed,
                    "return_auc": 129.0,
                    "selected_q": 8 if coupling == "independent" else 2,
                    "accounting_ok": True,
                    "upstream_modified": False,
                    "harl_commit": cfg["upstream"]["harl_commit"],
                    "probe_message_fraction": 0.005,
                    "selection_overhead_fraction": 0.001,
                }
            )
    return pd.DataFrame(rows)


def test_synthetic_complete_lattice_passes() -> None:
    cfg = config()
    result, cells = ANALYZER.evaluate_records(
        synthetic_records(cfg), cfg, replay_identical=True
    )
    assert result["run_count"] == result["expected_run_count"] == 40
    assert result["independent_selected_q8_fraction"] == 1.0
    assert result["shared_selected_q2_fraction"] == 1.0
    assert result["controller_relative_auc_over_strong_single_fixed_q"] > 0.01
    assert len(cells) == 2
    assert result["pass"] is True


def test_missing_cell_stops() -> None:
    cfg = config()
    records = synthetic_records(cfg).iloc[:-1]
    result, _ = ANALYZER.evaluate_records(records, cfg, replay_identical=True)
    assert result["pass"] is False
    assert result["decision"] == "stop"


import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
TSP_EXPERIMENTS = ROOT.parent / "TSP" / "experiments"
if str(TSP_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(TSP_EXPERIMENTS))

ANALYZER_PATH = (
    ROOT / "experiments" / "analyze_marl_mamujoco_calibrated_confirmation.py"
)
SPEC = importlib.util.spec_from_file_location("v3_confirmation", ANALYZER_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def config() -> dict:
    return {
        "experiment_id": "confirmation",
        "upstream": {"harl_commit": "harl"},
        "task": {"scenario": "HalfCheetah-v2", "agent_conf": "2x3"},
        "coupling_regimes": ["independent", "shared"],
        "methods": [
            "lyapunov_probe_commit",
            "fixed_q1",
            "fixed_q2",
            "fixed_q4",
            "fixed_q8",
        ],
        "controller_method": "lyapunov_probe_commit",
        "seed_registry": {"confirmation_training": list(range(8))},
        "inference": {"student_t_one_sided_critical": 1.894578605061305},
        "mandatory_confirmation_gates": {
            "independent_selected_q8_fraction_min": 0.875,
            "shared_selected_q2_fraction_min": 0.875,
            "independent_vs_q8_lower_min": -0.02,
            "shared_vs_q2_lower_min": -0.02,
            "mixture_vs_q8_lower_min": 0.0,
            "mixture_vs_q8_mean_min": 0.05,
            "mixture_vs_q8_positive_pairs_min": 7,
            "mixture_vs_q8_sign_p_max": 0.05,
            "mixture_final_return_vs_q8_mean_min": -0.02,
            "probe_message_fraction_max": 0.01,
            "selection_overhead_fraction_max": 0.02,
        },
    }


def synthetic_records() -> pd.DataFrame:
    rows = []
    for coupling in ("independent", "shared"):
        for seed in range(8):
            fixed = {
                "fixed_q1": 70.0,
                "fixed_q2": 98.0 if coupling == "shared" else 80.0,
                "fixed_q4": 85.0,
                "fixed_q8": 100.0 if coupling == "independent" else 75.0,
            }
            for method, value in fixed.items():
                rows.append(
                    {
                        "experiment_id": "confirmation",
                        "task": "HalfCheetah-v2/2x3",
                        "coupling": coupling,
                        "method": method,
                        "training_seed": seed,
                        "return_auc": value,
                        "final_return": value,
                        "selected_q": int(method[-1]),
                        "accounting_ok": True,
                        "upstream_modified": False,
                        "harl_commit": "harl",
                        "probe_message_fraction": 0.0,
                        "selection_overhead_fraction": 0.0,
                    }
                )
            controller = 99.5 if coupling == "independent" else 97.5
            rows.append(
                {
                    "experiment_id": "confirmation",
                    "task": "HalfCheetah-v2/2x3",
                    "coupling": coupling,
                    "method": "lyapunov_probe_commit",
                    "training_seed": seed,
                    "return_auc": controller,
                    "final_return": controller,
                    "selected_q": 8 if coupling == "independent" else 2,
                    "accounting_ok": True,
                    "upstream_modified": False,
                    "harl_commit": "harl",
                    "probe_message_fraction": 0.00768,
                    "selection_overhead_fraction": 1e-6,
                }
            )
    return pd.DataFrame(rows)


def test_exact_sign_probability() -> None:
    assert MODULE.exact_one_sided_sign_p(8, 8) == pytest.approx(1 / 256)
    assert MODULE.exact_one_sided_sign_p(7, 8) == pytest.approx(9 / 256)
    assert MODULE.exact_one_sided_sign_p(6, 8) > 0.05


def test_complete_positive_confirmation_passes() -> None:
    result = MODULE.evaluate_records(
        synthetic_records(), config(), replay_identical=True
    )
    assert result["run_count"] == result["expected_run_count"] == 80
    assert result["mixture_vs_q8_positive_pairs"] == 8
    assert result["mixture_vs_q8_sign_p"] == pytest.approx(1 / 256)
    assert result["all_mandatory_gates_pass"] is True
    assert result["decision"] == "admit-confirmed-return-evidence"


def test_missing_record_stops() -> None:
    result = MODULE.evaluate_records(
        synthetic_records().iloc[:-1], config(), replay_identical=True
    )
    assert result["all_mandatory_gates_pass"] is False
    assert result["decision"] == "stop"


def test_replay_gate_is_mandatory() -> None:
    result = MODULE.evaluate_records(
        synthetic_records(), config(), replay_identical=False
    )
    assert result["gates"]["analysis_replay_byte_identical"] is False
    assert result["all_mandatory_gates_pass"] is False

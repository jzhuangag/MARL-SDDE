from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "analyze_marl_online_drift_queue_v2.py"
)
SPEC = importlib.util.spec_from_file_location("analyze_marl_online_v2", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _controller_fixture(root: Path) -> tuple[Path, dict, list[dict]]:
    ledger = [
        {
            "decision": 1,
            "q": 1,
            "messages": 26,
            "environment_ticks": 14,
            "message_remaining": 174,
            "environment_remaining": 86,
            "message_queue": 0.0,
            "environment_queue": 0.0,
            "risk_before": 0.4,
            "risk_after": 0.3,
            "progress": 0.1,
        },
        {
            "decision": 2,
            "q": 2,
            "messages": 36,
            "environment_ticks": 14,
            "message_remaining": 138,
            "environment_remaining": 72,
            "message_queue": 0.0,
            "environment_queue": 0.0,
            "risk_before": 0.5,
            "risk_after": 0.3,
            "progress": 0.2,
        },
    ]
    ledger_path = root / "decision_ledger.jsonl"
    _write_jsonl(ledger_path, ledger)
    metadata = {
        "method": "controller",
        "candidate_q": [1, 2],
        "message_budget": 200,
        "environment_budget": 100,
        "rollout_length": 10,
        "server_overhead": 4,
        "validation_q": 1,
        "validation_horizon": 2,
        "train_updates_per_decision": 1,
        "controller_parameters": {"decisions": 2},
        "decision_ledger": ledger_path.name,
        "decision_ledger_sha256": hashlib.sha256(ledger_path.read_bytes()).hexdigest(),
        "accounting": {
            "charged_messages": 62,
            "charged_environment_ticks": 28,
            "message_remaining": 138,
            "environment_remaining": 72,
            "selected_counts": {"1": 1, "2": 1},
        },
    }
    rows = [
        {
            "budget_fraction": 0.0,
            "evaluation_mean_step_return": 0.0,
            "cumulative_messages": 0,
            "cumulative_environment_ticks": 0,
        },
        {
            "budget_fraction": 0.31,
            "evaluation_mean_step_return": 1.0,
            "cumulative_messages": 62,
            "cumulative_environment_ticks": 28,
        },
    ]
    metadata_path = root / "metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    return metadata_path, metadata, rows


def test_controller_accounting_replays_from_disjoint_ledger(tmp_path):
    metadata_path, metadata, rows = _controller_fixture(tmp_path)
    MODULE.validate_exact_accounting(metadata_path, metadata, rows)


def test_window_sized_counts_cannot_pass_exact_accounting(tmp_path):
    metadata_path, metadata, rows = _controller_fixture(tmp_path)
    metadata["accounting"]["selected_counts"] = {"1": 1, "2": 0}
    with pytest.raises(ValueError, match="selected counts disagree"):
        MODULE.validate_exact_accounting(metadata_path, metadata, rows)


def test_fixed_comparator_reconstructs_both_budgets(tmp_path):
    metadata_path = tmp_path / "metadata.json"
    metadata = {
        "method": "fixed_q2",
        "candidate_q": [1, 2],
        "message_budget": 100,
        "environment_budget": 50,
        "rollout_length": 10,
        "server_overhead": 5,
        "decision_ledger": None,
        "accounting": {
            "charged_messages": 100,
            "charged_environment_ticks": 40,
            "message_remaining": 0,
            "environment_remaining": 10,
            "selected_counts": {"2": 4},
        },
    }
    rows = [
        {
            "budget_fraction": 0.0,
            "evaluation_mean_step_return": 0.0,
            "cumulative_messages": 0,
            "cumulative_environment_ticks": 0,
        },
        {
            "budget_fraction": 1.0,
            "evaluation_mean_step_return": 1.0,
            "cumulative_messages": 100,
            "cumulative_environment_ticks": 40,
        },
    ]
    MODULE.validate_exact_accounting(metadata_path, metadata, rows)

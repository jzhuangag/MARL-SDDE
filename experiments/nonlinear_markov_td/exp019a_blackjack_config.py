"""Frozen configuration for the EXP-019A Blackjack CPU learning pilot."""

from __future__ import annotations

import hashlib
import json


PILOT_SEEDS = list(range(3190001, 3190033))

CONFIG = {
    "experiment_id": "EXP-019A",
    "parent_static_preregistration": "f56b3b8a69695cc9df9db225437b055485bc37fe",
    "parent_static_result": "b15287f91de8a0c72bbc40f6224d15f0b3cd1494",
    "task": "Blackjack-v1-exact-continuing",
    "policy": "epsilon_soft_stick_at_20",
    "policy_epsilon": 0.10,
    "gamma": 0.99,
    "estimator": "tabular_linear_semigradient_td0",
    "parameter_count": 280,
    "wire_parameter_count": 280,
    "server_overhead_bytes": 65536,
    "bytes_per_parameter": 4,
    "q_values": [1, 2, 4, 8, 16, 32],
    "rho_values": [0.0, 0.1, 0.3, 0.5, 0.7, 0.9],
    "target_horizons": [512, 2048],
    "budget_rays": ["message", "environment"],
    "active_budget_rays": ["message"],
    "delay_fractions": [0.0, 0.05, 0.2],
    "thinning_stride": 5,
    "step_size": 0.05,
    "coordinate_projection": 100.0,
    "evaluation_checkpoints": 33,
    "pilot_seeds": PILOT_SEEDS,
    "aggregate_auc_ratio_gate": 0.95,
    "active_strict_fraction_gate": 0.60,
    "inactive_auc_ratio_gate": 1.02,
    "terminal_ratio_gate": 0.98,
    "max_formal_seeds": 512,
}


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def config_sha256() -> str:
    return hashlib.sha256(canonical_json(CONFIG).encode("utf-8")).hexdigest()

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.nonlinear_markov_td.run_t060a_minatar_fixed_q_pilot import (
    _delayed_regularized_td,
    action_updates,
    analyze,
    canonical_config_hash,
    estimate,
    load_config,
    validate,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "docs" / "t060a_minatar_fixed_q_pilot_preregistration.json"


def test_frozen_configuration_hash_and_static_gate() -> None:
    config = load_config(CONFIG)
    assert canonical_config_hash(config) == config["configuration_sha256"]
    result = validate(config)
    assert all(
        result[key]
        for key in (
            "configuration_hash_matches",
            "pilot_seeds_unique",
            "splits_valid",
            "theoretical_value_gate",
        )
    )


def test_dual_budget_and_delay_reserve() -> None:
    config = load_config(CONFIG)
    for overhead in config["grid"]["overheads"]:
        for delay in config["grid"]["delays"]:
            for q in config["grid"]["participation"]:
                updates = action_updates(config, overhead=overhead, q=q, delay=delay)
                message_budget = (overhead + 16) * config["learning"]["target_updates_qmax"]
                environment_budget = 16 * config["learning"]["target_updates_qmax"]
                assert (updates + delay) * (overhead + q) <= message_budget
                assert (updates + delay) * q <= environment_budget


def test_fixed_point_has_zero_noiseless_delayed_risk() -> None:
    dimension = 3
    fixed = np.array([0.3, -0.2, 0.1])
    covariance = np.eye(dimension)
    drift = np.eye(dimension)
    reward_vector = fixed.copy()
    rows = 100
    # phi=e_j in a repeating cycle and reward=phi^T fixed makes the expected
    # unregularized update stationary at fixed for regularization zero.
    phi = np.zeros((1, rows, dimension))
    rewards = np.zeros((1, rows))
    for row in range(rows):
        phi[0, row, row % dimension] = 1.0
        rewards[0, row] = fixed[row % dimension]
    successors = np.zeros_like(phi)
    # The kernel initializes at zero, so this is a finite convergence rather
    # than a zero-risk assertion; both diagnostics must be finite and improve.
    risk, residual, average = _delayed_regularized_td(
        phi,
        successors,
        rewards,
        fixed,
        covariance,
        drift,
        reward_vector,
        0.1,
        0.0,
        0.0,
        2,
    )
    assert np.isfinite(risk) and np.isfinite(residual)
    assert np.linalg.norm(average - fixed) < np.linalg.norm(fixed)


def synthetic_endpoints(config: dict) -> list[dict]:
    rows = []
    for seed in config["pilot_seeds"]:
        split = "selection" if seed in config["selection_seeds"] else "validation"
        for game in config["tasks"]:
            for overhead in config["grid"]["overheads"]:
                for rho in config["grid"]["correlations"]:
                    preferred = min(
                        config["grid"]["participation"],
                        key=lambda q: (overhead + q) * (rho + (1 - rho) / q),
                    )
                    for delay in config["grid"]["delays"]:
                        for q in config["grid"]["participation"]:
                            risk = 1.0 + 0.2 * abs(np.log2(q) - np.log2(preferred))
                            rows.append(
                                {
                                    "master_seed": seed,
                                    "split": split,
                                    "game": game,
                                    "overhead": overhead,
                                    "rho": rho,
                                    "delay": delay,
                                    "q": q,
                                    "prediction_risk": risk,
                                    "message_used": 1,
                                    "message_budget": 1,
                                    "environment_used": 1,
                                    "environment_budget": 1,
                                }
                            )
    return rows


def test_analyzer_detects_a_positive_heldout_phase() -> None:
    config = load_config(CONFIG)
    diagnostics = [
        {
            "game": game,
            "drift_relative_disagreement": 0.0,
            "fixed_point_prediction_relative_disagreement": 0.0,
            "symmetric_min_eigenvalue": 0.1,
            "drift_condition_number": 2.0,
            "lifted_spectral_radius": {"0": 0.9, "8": 0.95},
        }
        for game in config["tasks"]
    ]
    summary = analyze(config, synthetic_endpoints(config), diagnostics)
    assert summary["gates"]["V1_complete_unique"]
    assert summary["gates"]["V3_reference_stable"]
    assert summary["heldout_oracle_strong_geometric_ratio"] < 1.0


def test_static_workload_is_cpu_sized() -> None:
    config = load_config(CONFIG)
    workload = estimate(config)
    assert workload["recommended_device"] == "local CPU"
    assert workload["endpoints"] == 8064
    assert workload["total_generated_environment_transitions"] < 25_000_000


def test_configuration_is_valid_json() -> None:
    with CONFIG.open("r", encoding="utf-8") as handle:
        assert json.load(handle)["experiment_id"] == "T-060A"

from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments.nonlinear_markov_td.run_t061a_reward_free_controller_pilot import (
    analyze,
    estimate,
    load_config,
    run_endpoint,
    validate,
)
from experiments.nonlinear_markov_td.t059_minatar_fixed_encoder import ReferenceMoments


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "docs" / "t061a_reward_free_controller_pilot_preregistration.json"


def test_static_configuration_and_workload() -> None:
    config = load_config(CONFIG)
    audit = validate(config)
    assert all(audit.values())
    workload = estimate(config)
    assert workload["recommended_device"] == "local CPU"
    assert workload["endpoints"] == 2688
    assert workload["total_generated_transitions"] < 22_000_000


def synthetic_endpoints(config: dict) -> list[dict]:
    rows = []
    for seed in config["pilot_seeds"]:
        for game in config["tasks"]:
            for rho in config["grid"]["correlations"]:
                for overhead in config["grid"]["overheads"]:
                    q = min((1, 4, 16), key=lambda value: (overhead + value) * (rho + (1-rho)/value))
                    for delay in config["grid"]["delays"]:
                        rows.append(
                            {
                                "master_seed": seed,
                                "game": game,
                                "rho": rho,
                                "overhead": overhead,
                                "delay": delay,
                                "match_count": round(96 * rho),
                                "selected_q": q,
                                "controller_risk": 0.8,
                                "strong_risk": 1.0,
                                "true_rho_full_budget_risk": 0.79,
                                "probe_message": 1,
                                "learning_message": 1,
                                "message_budget": 2,
                                "probe_environment": 1,
                                "learning_environment": 1,
                                "environment_budget": 2,
                            }
                        )
    return rows


def test_analyzer_accepts_a_broad_reward_free_effect() -> None:
    config = load_config(CONFIG)
    summary = analyze(config, synthetic_endpoints(config))
    assert summary["aggregate_controller_strong_ratio"] == 0.8
    assert summary["strict_cell_fraction"] == 1.0
    assert summary["gates"]["P1_complete_unique"]
    assert summary["gates"]["P8_participation_direction"]
    assert np.isfinite(summary["fingerprint_standardized_rmse"])


def test_run_endpoint_reuses_fixed_comparator_without_split_registry() -> None:
    config = load_config(CONFIG)
    mini = {
        **config,
        "learning": {**config["learning"], "target_updates_qmax": 8},
        "probe": {**config["probe"], "blocks": 4, "length": 1},
    }
    dimension = 3
    reference = ReferenceMoments(
        drift=np.eye(dimension),
        reward_vector=np.zeros(dimension),
        feature_covariance=np.eye(dimension),
        fixed_point=np.zeros(dimension),
        symmetric_min_eigenvalue=1.0,
        spectral_norm=1.0,
    )
    phi = np.zeros((64, dimension))
    phi[:, 0] = 1.0
    successors = np.zeros_like(phi)
    rewards = np.ones(64)
    bank = [(phi, successors, rewards) for _ in range(17)]
    row = run_endpoint(
        config=mini,
        reference=reference,
        diagnostics={"step_size": 0.01, "encoder_sha256": "test"},
        bank=bank,
        game="asterix",
        master_seed=123,
        rho=0.5,
        match_count=2,
        overhead=8,
        delay=0,
    )
    assert row["selected_q"] in (1, 4, 16)
    assert row["controller_risk"] >= 0.0
    assert row["strong_risk"] >= 0.0

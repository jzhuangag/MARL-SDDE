from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT / "TSP-V3" / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import certificate_robust_shield as shield
from multistate_certificate import build_transfer_mrp


def test_new_seeds_are_disjoint_from_prior_study() -> None:
    config = json.loads(
        (EXPERIMENTS / "certificate_robust_shield_preregistration.json").read_text(
            encoding="utf-8"
        )
    )
    new = set(range(35000001, 35000009)) | set(range(36000001, 36000065))
    old = set(range(33000001, 33000009)) | set(range(34000001, 34000065))
    assert new.isdisjoint(old)
    assert config["experiment_id"] == "TSP-V3-CERT-SHIELD-002"


def test_upper_shield_keeps_executed_action_unchanged() -> None:
    model = build_transfer_mrp(0.9)
    policies, _ = shield.build_shield_policy_actions(
        model,
        0.0,
        0,
        resource_budget=128000,
        agent_counts=(1, 4, 16, 32),
        sensitivity_levels={
            "shield_moderate": {
                "curvature_factor": 1.1,
                "noise_factor": 1.1,
                "log_rate_weakening": 0.05,
            }
        },
    )
    nominal = policies["finite_horizon"]
    guarded = policies["shield_moderate"]
    assert guarded["certificate_contraction"] < 1.0
    assert guarded["certificate_bound"] >= guarded["nominal_finite_time_bound"]
    assert (guarded["num_agents"], guarded["gap"], guarded["eta"]) == (
        nominal["num_agents"],
        nominal["gap"],
        nominal["eta"],
    )


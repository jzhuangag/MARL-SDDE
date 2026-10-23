"""Frozen outcome-free configuration for EXP-018A CPU pilot."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
EXPERIMENT = "EXP-018A"
TITLE = "Frozen nonlinear TD gradient correlation-variance identity"
PARENT_HEAD = "4ab73b536b0aa94b6c0ebbcd4d3fc18b48ab80e9"
TASKS = {
    "cartpole": {
        "gym_id": "CartPole-v1",
        "observation_dimension": 4,
        "action_count": 2,
        "discount": 0.97,
    },
    "acrobot": {
        "gym_id": "Acrobot-v1",
        "observation_dimension": 6,
        "action_count": 3,
        "discount": 0.97,
    },
}
MIXING_PROFILES = {
    "fast_regeneration": {"regeneration_probability": 0.20, "lambda_upper": 0.80},
    "slow_regeneration": {"regeneration_probability": 0.05, "lambda_upper": 0.95},
}
CHECKPOINTS = {"init_a": 18_210_101, "init_b": 18_210_102}
Q_LEVELS = (1, 4, 16, 32)
RHO_LEVELS = (0.0, 0.5, 0.9)
MAXIMUM_AGENTS = max(Q_LEVELS)
BLOCK_LENGTH = 64
HIDDEN_WIDTH = 64
PROJECTION_COUNT = 16
PROJECTION_SEED = 18_220_101
PILOT_REPLICATIONS = 64
FORMAL_SEEDS = None
PILOT_THRESHOLDS = {
    "maximum_pairwise_share_absolute_error": 0.10,
    "median_variance_calibration_relative_error": 0.20,
    "p90_variance_calibration_relative_error": 0.50,
    "median_q1_rho_invariance_spread": 0.30,
    "p90_q1_rho_invariance_spread": 0.75,
    "minimum_monotone_q_path_fraction": 0.80,
}


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def deterministic_seed(label: str, index: int) -> int:
    digest = hashlib.sha256(f"{EXPERIMENT}:{label}:{index}".encode("ascii")).digest()
    return 40_000_000 + int.from_bytes(digest[:4], "big") % 50_000_000


PILOT_SEEDS = tuple(deterministic_seed("pilot", index) for index in range(PILOT_REPLICATIONS))
if len(set(PILOT_SEEDS)) != len(PILOT_SEEDS):
    raise RuntimeError("deterministic pilot seed collision")


def variance_factor(q: int, rho: float) -> float:
    if q < 1 or not 0.0 <= rho <= 1.0:
        raise ValueError("invalid q or rho")
    return rho + (1.0 - rho) / float(q)


def expected_rows() -> int:
    return (
        len(PILOT_SEEDS)
        * len(TASKS)
        * len(MIXING_PROFILES)
        * len(CHECKPOINTS)
        * len(RHO_LEVELS)
        * len(Q_LEVELS)
    )


def expected_source_gradient_evaluations() -> int:
    return (
        len(PILOT_SEEDS)
        * len(TASKS)
        * len(MIXING_PROFILES)
        * len(CHECKPOINTS)
        * (MAXIMUM_AGENTS + 1)
    )


def build_static_manifest() -> dict[str, Any]:
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "title": TITLE,
        "parent_head": PARENT_HEAD,
        "execution": "local_CPU_only",
        "tasks": TASKS,
        "mixing_profiles": MIXING_PROFILES,
        "checkpoints": CHECKPOINTS,
        "q_levels": list(Q_LEVELS),
        "rho_levels": list(RHO_LEVELS),
        "block_length": BLOCK_LENGTH,
        "projection_count": PROJECTION_COUNT,
        "projection_seed": PROJECTION_SEED,
        "pilot_seeds": list(PILOT_SEEDS),
        "formal_seeds": FORMAL_SEEDS,
        "expected_rows": expected_rows(),
        "expected_source_gradient_evaluations": expected_source_gradient_evaluations(),
        "dependence_construction": {
            "base_streams": "independent complete Markov streams with independent regeneration clocks",
            "agent_rule": "use common stream 0 with probability sqrt(rho), otherwise the agent-indexed private stream",
            "pairwise_common_probability": "rho",
            "marginal_law": "unchanged because common and private streams are iid",
        },
        "model": {
            "network": "MLP-ReLU-64-64-1",
            "parameter_updates": 0,
            "optimizer": None,
            "preconditioner": False,
            "hessian_inverse": False,
            "covariance_inverse": False,
        },
        "primary_endpoint": "random_projection_variance_ratio_relative_to_q1",
        "theoretical_endpoint": "rho+(1-rho)/q",
        "pilot_thresholds": PILOT_THRESHOLDS,
        "pilot_inference": "descriptive implementation and feasibility gates only",
        "boundaries": [
            "no online participation controller",
            "no delayed-learning or dual-budget claim",
            "no nonlinear convergence claim",
            "no formal seed allocation",
            "a failed mandatory pilot gate forbids formal preregistration",
        ],
    }
    return manifest


STATIC_MANIFEST_HASH = sha256_json(build_static_manifest())

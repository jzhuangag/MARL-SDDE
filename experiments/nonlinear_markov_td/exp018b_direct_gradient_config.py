"""Frozen outcome-free configuration for EXP-018B CPU formal study."""

from __future__ import annotations

import hashlib
from typing import Any

from exp018a_direct_gradient_config import (
    BLOCK_LENGTH,
    CHECKPOINTS,
    HIDDEN_WIDTH,
    MAXIMUM_AGENTS,
    MIXING_PROFILES,
    PROJECTION_COUNT,
    PROJECTION_SEED,
    Q_LEVELS,
    RHO_LEVELS,
    TASKS,
    canonical_json,
    sha256_json,
)


SCHEMA_VERSION = 1
EXPERIMENT = "EXP-018B"
TITLE = "Formal nonlinear TD gradient correlation-variance calibration"
PARENT_HEAD = "c58ed5d60afb2e32b07ab21b6e3c0c5a5450a19a"
FORMAL_REPLICATIONS = 192
BOOTSTRAP_REPLICATIONS = 5_000
BOOTSTRAP_SEED = 18_240_101
FWER_ALPHA = 0.05
ENDPOINT_UPPER_QUANTILE = 0.975
MEDIAN_ERROR_TOLERANCE = 0.20
P90_ERROR_TOLERANCE = 0.50
PAIRWISE_SHARE_ERROR_TOLERANCE = 0.06
PRACTICAL_DIRECTIONAL_SEPARATION = 0.05


def deterministic_seed(index: int) -> int:
    digest = hashlib.sha256(f"{EXPERIMENT}:formal:{index}".encode("ascii")).digest()
    return 90_000_000 + int.from_bytes(digest[:4], "big") % 900_000_000


FORMAL_SEEDS = tuple(deterministic_seed(index) for index in range(FORMAL_REPLICATIONS))
if len(set(FORMAL_SEEDS)) != len(FORMAL_SEEDS):
    raise RuntimeError("deterministic formal seed collision")


def expected_rows() -> int:
    return (
        len(FORMAL_SEEDS)
        * len(TASKS)
        * len(MIXING_PROFILES)
        * len(CHECKPOINTS)
        * len(RHO_LEVELS)
        * len(Q_LEVELS)
    )


def expected_source_gradient_evaluations() -> int:
    return (
        len(FORMAL_SEEDS)
        * len(TASKS)
        * len(MIXING_PROFILES)
        * len(CHECKPOINTS)
        * (MAXIMUM_AGENTS + 1)
    )


def build_static_manifest() -> dict[str, Any]:
    return {
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
        "formal_seeds": list(FORMAL_SEEDS),
        "expected_rows": expected_rows(),
        "expected_source_gradient_evaluations": expected_source_gradient_evaluations(),
        "q1_crn_rule": "private source 1 is reused exactly across all rho rows",
        "primary_endpoints": {
            "median_relative_calibration_error": MEDIAN_ERROR_TOLERANCE,
            "p90_relative_calibration_error": P90_ERROR_TOLERANCE,
        },
        "inference": {
            "cluster": "formal seed",
            "bootstrap_replications": BOOTSTRAP_REPLICATIONS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "one_sided_quantile_per_endpoint": ENDPOINT_UPPER_QUANTILE,
            "familywise_alpha": FWER_ALPHA,
            "multiplicity": "Bonferroni for two co-primary upper bounds",
        },
        "mandatory_reproduction": "projections.csv and path-independent summary.json byte-identical",
        "boundaries": [
            "fixed-parameter gradient calibration only",
            "no online controller",
            "no delay or dual-budget learning claim",
            "no nonlinear convergence claim",
            "formal claim requires all statistical and exact-reproduction gates",
        ],
    }


STATIC_MANIFEST_HASH = sha256_json(build_static_manifest())

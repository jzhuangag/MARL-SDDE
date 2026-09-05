"""Frozen outcome-free configuration for EXP-017A nonlinear GPU pilot."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
EXPERIMENT = "EXP-017A"
TITLE = "Standard nonlinear Markov-TD participation benchmark"
PARENT_HEAD = "a5f25667217bea72cce55a9aa44a6b991f6847f9"
PILOT_SEEDS = (20550101, 20550102)
FORMAL_SEEDS = None
TASKS = {
    "cartpole": {
        "gym_id": "CartPole-v1",
        "observation_dimension": 4,
        "action_count": 2,
        "behavior_policy": "fixed_cartpole_stabilizing_epsilon_0.20",
        "discount": 0.97,
    },
    "acrobot": {
        "gym_id": "Acrobot-v1",
        "observation_dimension": 6,
        "action_count": 3,
        "behavior_policy": "fixed_acrobot_energy_epsilon_0.20",
        "discount": 0.97,
    },
}
MIXING_PROFILES = {
    "fast_regeneration": {
        "joint_regeneration_probability": 0.20,
        "lambda_upper": 0.80,
        "gamma_certificate": 0.20,
    },
    "slow_regeneration": {
        "joint_regeneration_probability": 0.05,
        "lambda_upper": 0.95,
        "gamma_certificate": 0.05,
    },
}
CORRELATIONS = (0.0, 0.5, 0.9)
DELAY_TRACES = {
    "zero": {
        "kind": "constant",
        "description": "synchronous reference",
        "maximum": 0,
    },
    "edge_jitter": {
        "kind": "periodic_heterogeneous",
        "base": (0, 1, 0, 2, 1, 3, 0, 1),
        "spike_period": 31,
        "spike_add": 2,
        "maximum": 5,
        "description": "low-latency heterogeneous edge/metro jitter with rare spikes",
    },
    "wan_bursty": {
        "kind": "periodic_heterogeneous",
        "base": (1, 2, 4, 8, 2, 6, 12, 3),
        "burst_period": 128,
        "burst_start": 32,
        "burst_length": 8,
        "burst_add": 4,
        "maximum": 16,
        "description": "wide-area heterogeneous delays with deterministic burst windows",
    },
}
BUDGETS = {
    "message_binding": {
        "message_bytes": 134_217_728,
        "environment_steps": 4_096,
    },
    "environment_binding": {
        "message_bytes": 536_870_912,
        "environment_steps": 1_024,
    },
}
POLICIES = (
    "oracle_evaluation_only",
    "always_all",
    "fixed_q4",
    "fixed_q16",
    "fixed_q32",
    "single_agent",
    "information_only",
    "learning_aware",
    "no_delay_ablation",
    "correlation_blind_ablation",
    "mixing_blind_ablation",
)
FIXED_Q_POLICIES = ("fixed_q4", "fixed_q16", "fixed_q32")
Q_CANDIDATES = (1, 4, 16, 32)
B_CANDIDATES = (1, 2, 4)
MAXIMUM_AGENTS = 32
BLOCK_SERVER_TICKS = 16
SERVER_OVERHEAD_BYTES = 65_536
FLOAT_BYTES = 4
LEARNING_RATE = 0.002
HIDDEN_WIDTH = 64
EVALUATION_TRANSITIONS = 512
TRAIN_BANK_LENGTH = 4_200
CERTIFICATE_ALPHA = 0.01
PRIMARY_EFFECT_THRESHOLD = 0.02
PILOT_NONINFERIORITY_RATIO = 1.10
PILOT_TAIL_RATIO = 1.10
FORMAL_FWER = 0.05
FORMAL_SIGN_FLIP_REPLICATIONS = 100_000
FORMAL_RESAMPLING_SEED = 20560101


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def delay_value(trace_name: str, tick: int, agent_index: int) -> int:
    """Return the frozen deterministic delay for one contribution."""

    if trace_name == "zero":
        return 0
    spec = DELAY_TRACES[trace_name]
    base = int(spec["base"][(tick + 3 * agent_index) % len(spec["base"])])
    if trace_name == "edge_jitter":
        spike = int((tick + agent_index) % int(spec["spike_period"]) == 0)
        return min(int(spec["maximum"]), base + spike * int(spec["spike_add"]))
    phase = (tick + 5 * agent_index) % int(spec["burst_period"])
    in_burst = int(spec["burst_start"]) <= phase < (
        int(spec["burst_start"]) + int(spec["burst_length"])
    )
    return min(int(spec["maximum"]), base + int(in_burst) * int(spec["burst_add"]))


def trace_summary(trace_name: str, ticks: int = 4096) -> dict[str, float | int]:
    values = sorted(
        delay_value(trace_name, tick, agent)
        for tick in range(ticks)
        for agent in range(MAXIMUM_AGENTS)
    )

    def quantile(probability: float) -> float:
        position = probability * (len(values) - 1)
        low = math.floor(position)
        high = math.ceil(position)
        if low == high:
            return float(values[low])
        weight = position - low
        return float(values[low] * (1.0 - weight) + values[high] * weight)

    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "median": quantile(0.5),
        "p90": quantile(0.9),
        "p99": quantile(0.99),
        "maximum": max(values),
    }


def scenario_count() -> int:
    return (
        len(TASKS)
        * len(MIXING_PROFILES)
        * len(CORRELATIONS)
        * len(DELAY_TRACES)
        * len(BUDGETS)
    )


def expected_runs() -> int:
    return len(PILOT_SEEDS) * scenario_count() * len(POLICIES)


def build_static_manifest() -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "title": TITLE,
        "parent_head": PARENT_HEAD,
        "tasks": TASKS,
        "mixing_profiles": MIXING_PROFILES,
        "correlations": list(CORRELATIONS),
        "cross_agent_dependence": {
            "construction": "each agent selects a common complete trajectory source with probability sqrt(rho), otherwise its private iid source",
            "pairwise_common_source_probability": "rho",
            "marginal_invariance": "common and private source trajectories have the same task/mixing law, so source mixing leaves every single-agent marginal unchanged",
            "observation_noise_added": False,
        },
        "delay_traces": {
            name: {**spec, "summary": trace_summary(name)}
            for name, spec in DELAY_TRACES.items()
        },
        "budgets": BUDGETS,
        "communication_accounting": {
            "per_server_tick": "server_overhead_bytes + q * parameter_count * float_bytes",
            "communication_matched": True,
            "agent_transitions_reported_separately": True,
        },
        "policies": list(POLICIES),
        "best_fixed_q_rule": "pilot reports the lower envelope of q in {4,16,32}; if pilot passes, a later outcome-free formal registry freezes one task/mixing-specific fixed-q choice using pilot only",
        "q_candidates": list(Q_CANDIDATES),
        "b_candidates": list(B_CANDIDATES),
        "pilot_seeds": list(PILOT_SEEDS),
        "formal_seeds": FORMAL_SEEDS,
        "model": {
            "algorithm": "semi-gradient neural TD(0)",
            "network": "MLP-ReLU-64-64-1",
            "optimizer": "plain SGD",
            "learning_rate": LEARNING_RATE,
            "hessian_inverse": False,
            "covariance_matrix_inverse": False,
            "preconditioner": False,
        },
        "metrics": [
            "terminal_MC_prediction_MSE",
            "heldout_mean_squared_Bellman_error",
            "normalized_prediction_error_AUC",
            "message_bytes",
            "environment_steps",
            "agent_transitions",
            "CVaR90_terminal_prediction_error",
            "wall_seconds",
            "controller_wall_seconds",
            "selected_q_b_trajectory",
        ],
        "multiple_comparisons": {
            "formal_only": True,
            "procedure": "one-sided paired seed-block sign-flip maxT with Holm-compatible ordered reporting",
            "familywise_alpha": FORMAL_FWER,
            "replications": FORMAL_SIGN_FLIP_REPLICATIONS,
            "resampling_seed": FORMAL_RESAMPLING_SEED,
            "pilot_inference": "descriptive progression gates only",
        },
        "scientific_boundaries": [
            "known joint mixing from public regeneration certificate only",
            "no unrestricted unknown-mixing claim",
            "no global occupation-optimality claim",
            "no general nonlinear MARL claim",
            "policy evaluation only; no actor-critic claim",
        ],
        "pilot_scenario_count": scenario_count(),
        "pilot_expected_runs": expected_runs(),
        "scientific_outcomes_present": False,
    }
    manifest["static_manifest_sha256"] = sha256_json(manifest)
    return manifest

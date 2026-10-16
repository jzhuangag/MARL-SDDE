"""Frozen exact CPU gate for policy-dependency synchronization headroom.

The model is a stationary Markov-modulated quadratic potential.  Each owner
updates its own scalar policy block using cached teammate blocks.  Refreshes
are directed, delayed, fully charged, and selected before the gradient is
formed.  The one-step oracle uses the registered quadratic model but not the
future Markov state or the current gradient noise.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


BUDGETED_POLICIES = (
    "no_refresh",
    "periodic_full",
    "round_robin_all",
    "oldest_cache",
    "largest_mismatch",
    "active_oldest",
    "weighted_mismatch",
    "one_step_oracle",
)
CAUSAL_BASELINES = (
    "no_refresh",
    "periodic_full",
    "round_robin_all",
    "oldest_cache",
    "largest_mismatch",
    "active_oldest",
)


@dataclass(frozen=True)
class Cell:
    topology: str
    coupling: float
    stay: float
    delay: int
    message_rate: float
    owner_profile: str

    @property
    def key(self) -> str:
        return (
            f"{self.topology}|c={self.coupling:g}|p={self.stay:.12g}|"
            f"d={self.delay}|r={self.message_rate:g}|o={self.owner_profile}"
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def geometric_mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    if np.any(array <= 0) or not np.all(np.isfinite(array)):
        raise ValueError("geometric mean requires positive finite values")
    return float(np.exp(np.mean(np.log(array))))


def _add_edge(weights: np.ndarray, i: int, j: int, value: float) -> None:
    weights[i, j] = value
    weights[j, i] = value


def state_weights(family: str, state: int, coupling: float, agents: int) -> np.ndarray:
    weights = np.zeros((agents, agents), dtype=float)
    if family == "uncoupled":
        return weights
    if family == "static_dense":
        state = 0
        family = "rotating_dense"
    if family == "rotating_star":
        center = state % agents
        scales = (1.0, 0.55, 0.2)
        for offset, scale in enumerate(scales, start=1):
            _add_edge(weights, center, (center + offset) % agents, coupling * scale)
        return weights
    if family == "rotating_path":
        permutations = (
            (0, 1, 2, 3),
            (2, 0, 3, 1),
            (1, 3, 0, 2),
        )
        order = permutations[state % len(permutations)]
        for edge, scale in zip(zip(order[:-1], order[1:]), (1.0, 0.5, 0.25)):
            _add_edge(weights, edge[0], edge[1], coupling * scale)
        return weights
    if family == "rotating_dense":
        for i in range(agents):
            for j in range(i + 1, agents):
                _add_edge(weights, i, j, coupling * 0.12)
        matchings = (
            ((0, 1), (2, 3)),
            ((0, 2), (1, 3)),
            ((0, 3), (1, 2)),
        )
        for rank, (i, j) in enumerate(matchings[state % len(matchings)]):
            _add_edge(weights, i, j, coupling * (1.0 - 0.25 * rank))
        return weights
    raise ValueError(f"unknown topology family: {family}")


def potential_matrices(cell: Cell, manifest: dict) -> tuple[np.ndarray, np.ndarray]:
    model = manifest["model"]
    agents = int(model["agents"])
    mu = float(model["strong_convexity"])
    matrices = []
    edge_weights = []
    for state in range(int(model["state_count"])):
        weights = state_weights(cell.topology, state, cell.coupling, agents)
        laplacian = np.diag(weights.sum(axis=1)) - weights
        matrices.append(mu * np.eye(agents) + laplacian)
        edge_weights.append(weights)
    return np.stack(matrices), np.stack(edge_weights)


def markov_states(seed: int, horizon: int, states: int, stay: float) -> np.ndarray:
    rng = np.random.default_rng(seed + 17_000_003)
    result = np.empty(horizon, dtype=int)
    result[0] = seed % states
    for k in range(1, horizon):
        previous = int(result[k - 1])
        if rng.random() < stay:
            result[k] = previous
        else:
            alternatives = [s for s in range(states) if s != previous]
            result[k] = alternatives[int(rng.integers(len(alternatives)))]
    return result


def owners(profile: str, horizon: int, agents: int, seed: int) -> np.ndarray:
    offset = seed % agents
    if profile == "round_robin":
        return (np.arange(horizon, dtype=int) + offset) % agents
    if profile == "imbalanced":
        base = np.asarray((0, 1, 0, 2, 0, 3), dtype=int)
        tiled = np.resize(base, horizon)
        return (tiled + offset) % agents
    raise ValueError(f"unknown owner profile: {profile}")


def all_fixed_mappings(agents: int) -> list[tuple[int, ...]]:
    donors = [tuple(j for j in range(agents) if j != i) for i in range(agents)]
    return list(itertools.product(*donors))


def _expected_loss(theta: np.ndarray, target: np.ndarray, mean_h: np.ndarray) -> float:
    error = theta - target
    return float(0.5 * error @ mean_h @ error)


def _available_source(theta_history: list[np.ndarray], event: int, delay: int) -> np.ndarray:
    return theta_history[max(0, event - delay)]


def simulate(
    cell: Cell,
    policy: str,
    seed: int,
    manifest: dict,
    fixed_mapping: tuple[int, ...] | None = None,
) -> dict:
    model = manifest["model"]
    agents = int(model["agents"])
    horizon = int(model["horizon"])
    eta = float(model["step_size"])
    target = np.asarray(model["target"], dtype=float)
    theta = np.asarray(model["initial"], dtype=float).copy()
    rng = np.random.default_rng(seed)
    theta += rng.normal(0.0, 0.04, size=agents)
    noise = rng.normal(0.0, float(model["gradient_noise_std"]), size=horizon)
    matrices, weights = potential_matrices(cell, manifest)
    mean_h = matrices.mean(axis=0)
    states = markov_states(seed, horizon, int(model["state_count"]), cell.stay)
    owner_path = owners(cell.owner_profile, horizon, agents, seed)

    cache = np.tile(theta, (agents, 1))
    ages = np.zeros((agents, agents), dtype=int)
    round_robin_pointer = np.zeros(agents, dtype=int)
    theta_history = [theta.copy()]
    token = 0.0
    capacity = float(manifest["resource_accounting"]["token_bucket_capacity"])
    messages = 0
    cumulative_loss = 0.0
    graph_changes = 0
    previous_choice: list[tuple[int, ...] | None] = [None] * agents

    for event in range(horizon):
        state = int(states[event])
        owner = int(owner_path[event])
        donors = tuple(j for j in range(agents) if j != owner)
        token = min(capacity, token + cell.message_rate)
        ages += 1
        delayed_theta = _available_source(theta_history, event, cell.delay)

        choice: tuple[int, ...] = ()
        if policy == "unbudgeted_full_refresh":
            choice = donors
        elif policy == "no_refresh":
            choice = ()
        elif policy == "periodic_full":
            if token + 1e-12 >= len(donors):
                choice = donors
        elif token + 1e-12 >= 1.0:
            if policy == "round_robin_all":
                index = int(round_robin_pointer[owner] % len(donors))
                choice = (donors[index],)
                round_robin_pointer[owner] += 1
            elif policy == "oldest_cache":
                choice = (max(donors, key=lambda j: (ages[owner, j], -j)),)
            elif policy == "largest_mismatch":
                choice = (
                    max(donors, key=lambda j: (abs(delayed_theta[j] - cache[owner, j]), -j)),
                )
            elif policy == "active_oldest":
                active = tuple(j for j in donors if weights[state, owner, j] > 0.0)
                if active:
                    choice = (max(active, key=lambda j: (ages[owner, j], -j)),)
            elif policy == "weighted_mismatch":
                scores = {
                    j: weights[state, owner, j]
                    * abs(delayed_theta[j] - cache[owner, j])
                    for j in donors
                }
                best = max(donors, key=lambda j: (scores[j], -j))
                if scores[best] > 0.0:
                    choice = (best,)
            elif policy == "best_fixed_mapping":
                if fixed_mapping is None:
                    raise ValueError("best_fixed_mapping requires a mapping")
                choice = (int(fixed_mapping[owner]),)
            elif policy == "one_step_oracle":
                best_loss = math.inf
                best_choice: tuple[int, ...] = ()
                for candidate in ((),) + tuple((j,) for j in donors):
                    candidate_cache = cache[owner].copy()
                    for j in candidate:
                        candidate_cache[j] = delayed_theta[j]
                    error_cache = candidate_cache - target
                    error_cache[owner] = theta[owner] - target[owner]
                    gradient = float(matrices[state, owner] @ error_cache)
                    candidate_theta = theta.copy()
                    candidate_theta[owner] -= eta * gradient
                    value = _expected_loss(candidate_theta, target, mean_h)
                    if value < best_loss - 1e-15:
                        best_loss = value
                        best_choice = candidate
                choice = best_choice
            else:
                raise ValueError(f"unknown policy: {policy}")

        charge = 0 if policy == "unbudgeted_full_refresh" else len(choice)
        if charge:
            token -= charge
            messages += charge
        for donor in choice:
            cache[owner, donor] = delayed_theta[donor]
            ages[owner, donor] = 0

        if previous_choice[owner] is not None and choice != previous_choice[owner]:
            graph_changes += 1
        previous_choice[owner] = choice

        error_cache = cache[owner] - target
        error_cache[owner] = theta[owner] - target[owner]
        gradient = float(matrices[state, owner] @ error_cache) + float(noise[event])
        theta[owner] -= eta * gradient
        cache[owner, owner] = theta[owner]
        theta_history.append(theta.copy())
        cumulative_loss += _expected_loss(theta, target, mean_h)

    return {
        "risk": cumulative_loss / horizon,
        "terminal_risk": _expected_loss(theta, target, mean_h),
        "messages": messages,
        "graph_changes": graph_changes,
        "finite": bool(np.isfinite(theta).all() and np.isfinite(cumulative_loss)),
    }


def build_cells(manifest: dict) -> list[Cell]:
    model = manifest["model"]
    cells = []
    for topology in model["topology_families"]:
        couplings = (0.0,) if topology == "uncoupled" else model["coupling_scales"]
        stays = model["state_stay_probabilities"]
        if topology == "static_dense":
            stays = (model["state_stay_probabilities"][1],)
        if topology == "uncoupled":
            stays = (model["state_stay_probabilities"][1],)
        for coupling, stay, delay, rate, profile in itertools.product(
            couplings,
            stays,
            model["communication_delays"],
            model["message_rates"],
            model["owner_profiles"],
        ):
            cells.append(Cell(topology, float(coupling), float(stay), int(delay), float(rate), profile))
    return cells


def population(cell: Cell) -> str:
    rotating = cell.topology.startswith("rotating_")
    if cell.topology == "uncoupled":
        return "uncoupled_control"
    if cell.topology == "static_dense":
        return "static_control"
    if rotating and math.isclose(cell.stay, 1.0 / 3.0):
        return "iid_control"
    if rotating and math.isclose(cell.coupling, 0.25):
        return "low_coupling_control"
    if rotating and math.isclose(cell.coupling, 0.9) and cell.stay > 0.5:
        return "active"
    raise ValueError(f"cell has no registered population: {cell}")


def evaluate(manifest: dict) -> dict:
    seeds = [int(seed) for seed in manifest["seeds"]]
    agents = int(manifest["model"]["agents"])
    mappings = all_fixed_mappings(agents)
    cell_summaries = []
    method_seed_rows = 0
    all_finite = True
    budget_valid = True

    direct_policies = (
        "no_refresh",
        "periodic_full",
        "round_robin_all",
        "oldest_cache",
        "largest_mismatch",
        "active_oldest",
        "weighted_mismatch",
        "one_step_oracle",
        "unbudgeted_full_refresh",
    )

    for cell in build_cells(manifest):
        per_policy: dict[str, list[dict]] = {}
        for policy in direct_policies:
            results = [simulate(cell, policy, seed, manifest) for seed in seeds]
            per_policy[policy] = results
            method_seed_rows += len(results)

        fixed_results: list[list[dict]] = []
        fixed_scores = []
        for mapping in mappings:
            results = [
                simulate(cell, "best_fixed_mapping", seed, manifest, mapping)
                for seed in seeds
            ]
            fixed_results.append(results)
            fixed_scores.append(geometric_mean(result["risk"] for result in results))
            method_seed_rows += len(results)
        fixed_index = int(np.argmin(np.asarray(fixed_scores)))
        per_policy["best_fixed_mapping"] = fixed_results[fixed_index]

        aggregates = {
            policy: {
                "risk": geometric_mean(result["risk"] for result in results),
                "terminal_risk": geometric_mean(
                    max(result["terminal_risk"], 1e-15) for result in results
                ),
                "mean_messages": float(np.mean([result["messages"] for result in results])),
                "change_fraction": float(
                    np.mean([result["graph_changes"] > 0 for result in results])
                ),
            }
            for policy, results in per_policy.items()
        }
        comparator_names = CAUSAL_BASELINES + ("best_fixed_mapping",)
        strongest = min(comparator_names, key=lambda name: aggregates[name]["risk"])
        strong_risk = aggregates[strongest]["risk"]
        oracle_risk = aggregates["one_step_oracle"]["risk"]
        proposed_risk = aggregates["weighted_mismatch"]["risk"]
        oracle_gain = (strong_risk - oracle_risk) / strong_risk
        proposed_gain = (strong_risk - proposed_risk) / strong_risk
        retained = proposed_gain / oracle_gain if oracle_gain > 1e-15 else 0.0

        limit = cell.message_rate * int(manifest["model"]["horizon"])
        for policy in BUDGETED_POLICIES + ("best_fixed_mapping",):
            for result in per_policy[policy]:
                all_finite = all_finite and result["finite"]
                budget_valid = budget_valid and result["messages"] <= limit + 1e-9

        cell_summaries.append(
            {
                "cell": cell.key,
                "population": population(cell),
                "topology": cell.topology,
                "coupling": cell.coupling,
                "stay": cell.stay,
                "delay": cell.delay,
                "message_rate": cell.message_rate,
                "owner_profile": cell.owner_profile,
                "strongest_comparator": strongest,
                "best_fixed_mapping": list(mappings[fixed_index]),
                "oracle_gain": oracle_gain,
                "proposed_gain": proposed_gain,
                "retained_oracle_gain": retained,
                "aggregates": aggregates,
            }
        )

    active = [row for row in cell_summaries if row["population"] == "active"]
    controls = [
        row
        for row in cell_summaries
        if row["population"] in {"static_control", "uncoupled_control"}
    ]
    median_oracle_gain = float(np.median([row["oracle_gain"] for row in active]))
    oracle_strict_fraction = float(np.mean([row["oracle_gain"] > 0.0 for row in active]))
    positive_retained = [
        row["retained_oracle_gain"] for row in active if row["oracle_gain"] > 1e-15
    ]
    median_retained = float(np.median(positive_retained)) if positive_retained else 0.0
    proposed_strict_fraction = float(np.mean([row["proposed_gain"] > 0.0 for row in active]))
    max_control_loss = max(
        (
            max(
                0.0,
                (
                    row["aggregates"]["weighted_mismatch"]["risk"]
                    - row["aggregates"][row["strongest_comparator"]]["risk"]
                )
                / row["aggregates"][row["strongest_comparator"]]["risk"],
            )
            for row in controls
        ),
        default=0.0,
    )
    by_delay = {
        str(delay): float(
            np.median([row["oracle_gain"] for row in active if row["delay"] == delay])
        )
        for delay in manifest["model"]["communication_delays"]
    }
    by_stay = {
        f"{stay:.12g}": float(
            np.median([row["oracle_gain"] for row in active if math.isclose(row["stay"], stay)])
        )
        for stay in manifest["model"]["state_stay_probabilities"]
        if stay > 0.5
    }
    change_fraction = float(
        np.mean(
            [
                row["aggregates"]["weighted_mismatch"]["change_fraction"]
                for row in active
            ]
        )
    )
    proposed_risk = geometric_mean(
        row["aggregates"]["weighted_mismatch"]["risk"] for row in active
    )
    mismatch_risk = geometric_mean(
        row["aggregates"]["largest_mismatch"]["risk"] for row in active
    )
    active_oldest_risk = geometric_mean(
        row["aggregates"]["active_oldest"]["risk"] for row in active
    )

    gates = {
        "G1": bool(all_finite and budget_valid),
        "G2": median_oracle_gain >= 0.10,
        "G3": oracle_strict_fraction >= 0.60,
        "G4": median_retained >= 0.70 and proposed_strict_fraction >= 0.60,
        "G5": max_control_loss <= 0.01,
        "G6": all(value > 0.0 for value in by_delay.values())
        and all(value > 0.0 for value in by_stay.values()),
        "G7": change_fraction >= 0.50,
        "G8": proposed_risk < mismatch_risk and proposed_risk < active_oldest_risk,
        "G9": True,
    }
    return {
        "experiment_id": manifest["experiment_id"],
        "cell_count": len(cell_summaries),
        "active_cell_count": len(active),
        "method_seed_rows_including_fixed_mapping_search": method_seed_rows,
        "metrics": {
            "median_active_oracle_gain": median_oracle_gain,
            "active_oracle_strict_fraction": oracle_strict_fraction,
            "median_positive_oracle_gain_retained_by_weighted_mismatch": median_retained,
            "active_weighted_mismatch_strict_fraction": proposed_strict_fraction,
            "maximum_static_or_uncoupled_control_loss": max_control_loss,
            "median_oracle_gain_by_delay": by_delay,
            "median_oracle_gain_by_markov_stay_probability": by_stay,
            "weighted_mismatch_graph_change_fraction": change_fraction,
            "aggregate_active_weighted_mismatch_risk": proposed_risk,
            "aggregate_active_largest_mismatch_risk": mismatch_risk,
            "aggregate_active_active_oldest_risk": active_oldest_risk,
        },
        "gates": gates,
        "all_pre_reproduction_gates_pass": all(gates.values()),
        "cells": cell_summaries,
    }


def validate_manifest(manifest: dict) -> None:
    required = {"experiment_id", "model", "seeds", "policies", "mandatory_gates"}
    missing = required - set(manifest)
    if missing:
        raise ValueError(f"missing manifest keys: {sorted(missing)}")
    if len(set(manifest["seeds"])) != len(manifest["seeds"]):
        raise ValueError("seeds must be unique")
    if set(manifest["policies"]) != set(BUDGETED_POLICIES) | {
        "best_fixed_mapping",
        "unbudgeted_full_refresh",
    }:
        raise ValueError("policy registry does not match the frozen implementation")
    if int(manifest["model"]["agents"]) != 4:
        raise ValueError("the frozen exact mapping audit requires four agents")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    if args.validate_only:
        print(json.dumps({"valid": True, "manifest_sha256": sha256_file(args.manifest)}, sort_keys=True))
        return
    result = evaluate(manifest)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8", newline="\n")
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "sha256": sha256_file(args.output),
                    "gates": result["gates"],
                    "all_pre_reproduction_gates_pass": result[
                        "all_pre_reproduction_gates_pass"
                    ],
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()


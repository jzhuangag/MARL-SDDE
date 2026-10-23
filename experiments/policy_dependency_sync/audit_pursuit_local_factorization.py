"""Outcome-free structural audit for a bounded local Pursuit factor graph."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


def bounded_local_graph(
    positions: Sequence[Sequence[int]], *, radius: int, max_neighbors: int
) -> tuple[tuple[tuple[int, int], ...], tuple[int, ...]]:
    """Return directed local edges and uncapped out-degrees.

    Distance is Chebyshev distance, matching the square ego-centric Pursuit
    observation. Ties are resolved by Manhattan distance and donor id, so the
    graph is a predictable deterministic function of the centralized state.
    """

    array = np.asarray(positions, dtype=int)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("positions must have shape (n_agents, 2)")
    if radius < 0 or max_neighbors < 0:
        raise ValueError("radius and max_neighbors must be nonnegative")
    edges: list[tuple[int, int]] = []
    raw_degrees: list[int] = []
    for owner, owner_position in enumerate(array):
        candidates: list[tuple[int, int, int]] = []
        for donor, donor_position in enumerate(array):
            if donor == owner:
                continue
            displacement = np.abs(owner_position - donor_position)
            chebyshev = int(np.max(displacement))
            if chebyshev <= radius:
                candidates.append((chebyshev, int(np.sum(displacement)), donor))
        candidates.sort()
        raw_degrees.append(len(candidates))
        edges.extend((owner, donor) for _, _, donor in candidates[:max_neighbors])
    return tuple(edges), tuple(raw_degrees)


def jaccard_distance(
    left: Iterable[tuple[int, int]], right: Iterable[tuple[int, int]]
) -> float:
    left_set, right_set = set(left), set(right)
    union = left_set | right_set
    if not union:
        return 0.0
    return 1.0 - len(left_set & right_set) / len(union)


def _pursuit_model(environment: Any) -> Any:
    current = environment
    for _ in range(12):
        if hasattr(current, "pursuer_layer"):
            return current
        if hasattr(current, "aec_env"):
            current = current.aec_env
        elif hasattr(current, "env"):
            current = current.env
        else:
            break
    raise RuntimeError("unable to locate the pinned Pursuit model")


def run_audit(
    *,
    seeds: Sequence[int],
    action_seed: int,
    cycles: int = 100,
    n_pursuers: int = 8,
    n_evaders: int = 30,
    observation_range: int = 7,
    max_neighbors: int = 4,
) -> dict[str, Any]:
    from pettingzoo import __version__ as pettingzoo_version
    from pettingzoo.sisl import pursuit_v4
    from pettingzoo.sisl.pursuit import pursuit, pursuit_base

    if observation_range % 2 != 1:
        raise ValueError("observation_range must be odd")
    radius = observation_range // 2
    action_rng = np.random.default_rng(int(action_seed))
    edge_counts: list[int] = []
    raw_degrees: list[int] = []
    reverse_calls: list[int] = []
    jaccard_changes: list[float] = []
    graph_turnover: list[bool] = []
    fixed_edge_misses: list[bool] = []
    unique_graph_counts: list[int] = []
    states_per_seed: list[int] = []

    for seed in seeds:
        environment = pursuit_v4.parallel_env(
            n_pursuers=n_pursuers,
            n_evaders=n_evaders,
            max_cycles=cycles,
            obs_range=observation_range,
            n_catch=2,
            shared_reward=True,
        )
        environment.reset(seed=int(seed))
        previous: tuple[tuple[int, int], ...] | None = None
        fixed: set[tuple[int, int]] | None = None
        episode_graphs: set[tuple[tuple[int, int], ...]] = set()
        episode_states = 0
        try:
            for _ in range(cycles):
                model = _pursuit_model(environment)
                positions = tuple(
                    tuple(int(value) for value in model.pursuer_layer.get_position(i))
                    for i in range(n_pursuers)
                )
                graph, degrees = bounded_local_graph(
                    positions, radius=radius, max_neighbors=max_neighbors
                )
                graph_set = set(graph)
                if fixed is None:
                    fixed = graph_set
                fixed_edge_misses.extend(edge not in fixed for edge in graph)
                if previous is not None:
                    graph_turnover.append(graph != previous)
                    jaccard_changes.append(jaccard_distance(previous, graph))
                previous = graph
                episode_graphs.add(graph)
                edge_counts.append(len(graph))
                raw_degrees.extend(degrees)
                reverse_calls.extend(2 + min(degree, max_neighbors) for degree in degrees)
                episode_states += 1

                actions = {
                    agent: int(action_rng.integers(0, 5))
                    for agent in environment.agents
                }
                _, _, _, _, _ = environment.step(actions)
                if not environment.agents:
                    break
        finally:
            environment.close()
        unique_graph_counts.append(len(episode_graphs))
        states_per_seed.append(episode_states)

    edge_array = np.asarray(edge_counts, dtype=float)
    degree_array = np.asarray(raw_degrees, dtype=float)
    reverse_array = np.asarray(reverse_calls, dtype=float)
    jaccard_array = np.asarray(jaccard_changes, dtype=float)
    if min(edge_array.size, degree_array.size, reverse_array.size) == 0:
        raise RuntimeError("audit produced no graph states")
    truncation_fraction = float(np.mean(degree_array > max_neighbors))
    turnover_fraction = float(np.mean(graph_turnover))
    fixed_miss_fraction = float(np.mean(fixed_edge_misses))
    active_fraction = float(np.mean(edge_array > 0))
    gates = {
        "F1_bounded_reverse_interface": float(np.max(reverse_array)) <= 6.0,
        "F2_rare_architectural_truncation": truncation_fraction <= 0.02,
        "F3_state_dependent_turnover": turnover_fraction >= 0.30,
        "F4_static_graph_misses_future_edges": fixed_miss_fraction >= 0.30,
        "F5_graph_active": active_fraction >= 0.95,
    }
    return {
        "audit": "PURSUIT-LOCAL-FACTOR-INTERFACE-QUALIFICATION",
        "scientific_efficacy_evidence": False,
        "reward_values_read": False,
        "passed": all(gates.values()),
        "gates": gates,
        "configuration": {
            "seeds": list(map(int, seeds)),
            "action_seed": int(action_seed),
            "cycles": int(cycles),
            "n_pursuers": int(n_pursuers),
            "n_evaders": int(n_evaders),
            "observation_range": int(observation_range),
            "radius": int(radius),
            "max_neighbors": int(max_neighbors),
            "n_catch": 2,
            "shared_reward": True,
        },
        "pettingzoo_version": pettingzoo_version,
        "source_sha256": {
            "pursuit.py": hashlib.sha256(
                Path(inspect.getfile(pursuit)).read_bytes()
            ).hexdigest(),
            "pursuit_base.py": hashlib.sha256(
                Path(inspect.getfile(pursuit_base)).read_bytes()
            ).hexdigest(),
        },
        "states": int(edge_array.size),
        "states_per_seed": states_per_seed,
        "mean_directed_edges": float(np.mean(edge_array)),
        "maximum_raw_out_degree": int(np.max(degree_array)),
        "architectural_truncation_fraction": truncation_fraction,
        "graph_turnover_fraction": turnover_fraction,
        "mean_consecutive_jaccard_distance": float(np.mean(jaccard_array)),
        "fixed_initial_graph_miss_fraction": fixed_miss_fraction,
        "active_graph_fraction": active_fraction,
        "median_unique_graphs_per_seed": float(np.median(unique_graph_counts)),
        "reverse_evaluations": {
            "minimum": int(np.min(reverse_array)),
            "median": float(np.median(reverse_array)),
            "p90": float(np.quantile(reverse_array, 0.9)),
            "maximum": int(np.max(reverse_array)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_audit(
        seeds=tuple(range(91000, 91032)),
        action_seed=8842,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

"""Exact finite-horizon separation for signed local cache scheduling."""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

from .joint_factor_lyapunov import projected_quadratic_minimizer


StatePath = tuple[int, ...]
Scheduler = Callable[[StatePath, int], int]


@dataclass(frozen=True)
class SeparationResult:
    expected_terminal_potential: float
    expected_cumulative_potential: float
    expected_refresh_cost: float


def binary_markov_paths(horizon: int, persistence: float) -> Iterable[tuple[StatePath, float]]:
    if horizon <= 0 or not 0.0 <= persistence <= 1.0:
        raise ValueError("invalid horizon or persistence")
    for path in itertools.product((0, 1), repeat=horizon):
        probability = 0.5
        for previous, current in zip(path, path[1:]):
            probability *= persistence if current == previous else 1.0 - persistence
        yield tuple(path), probability


def simulate_path(
    path: StatePath,
    *,
    scheduler: Scheduler,
    initial_parameter: float,
    packet_magnitude: float,
    maximum_packet_weight: float,
) -> tuple[float, float, float]:
    """Return terminal potential, cumulative pre-update potential and cost.

    Actions 0 and 1 refresh one of two equally old, equally displaced local
    cache edges. Action 2 is the zero-cost null. In relevance state ``z``, the
    matching refresh produces gradient ``+delta`` and the other produces
    ``-delta``. The joint controller uses the certified scalar minimizer, so a
    negatively aligned packet receives zero weight.
    """

    if initial_parameter < 0.0 or packet_magnitude <= 0.0:
        raise ValueError("the separating instance uses x>=0 and delta>0")
    parameter = float(initial_parameter)
    cumulative = 0.0
    cost = 0.0
    for event, state in enumerate(path):
        cumulative += 0.5 * parameter**2
        action = int(scheduler(path, event))
        if action not in (0, 1, 2):
            raise ValueError("scheduler action must be edge 0, edge 1 or null 2")
        if action == 2:
            gradient = 0.0
        else:
            cost += 1.0
            gradient = packet_magnitude if action == state else -packet_magnitude
        alignment = parameter * gradient
        packet_weight, _ = projected_quadratic_minimizer(
            linear_gain=alignment,
            quadratic_curvature=packet_magnitude**2,
            maximum_weight=maximum_packet_weight,
        )
        parameter -= packet_weight * gradient
    return 0.5 * parameter**2, cumulative, cost


def evaluate_scheduler(
    *,
    horizon: int,
    persistence: float,
    scheduler: Scheduler,
    initial_parameter: float = 1.0,
    packet_magnitude: float = 0.25,
    maximum_packet_weight: float = 0.5,
) -> SeparationResult:
    terminal = cumulative = cost = mass = 0.0
    for path, probability in binary_markov_paths(horizon, persistence):
        path_terminal, path_cumulative, path_cost = simulate_path(
            path,
            scheduler=scheduler,
            initial_parameter=initial_parameter,
            packet_magnitude=packet_magnitude,
            maximum_packet_weight=maximum_packet_weight,
        )
        terminal += probability * path_terminal
        cumulative += probability * path_cumulative
        cost += probability * path_cost
        mass += probability
    if abs(mass - 1.0) > 1e-12:
        raise RuntimeError("Markov path probabilities do not sum to one")
    return SeparationResult(terminal, cumulative, cost)


def separation_table(horizon: int = 4, persistence: float = 0.9) -> dict[str, object]:
    schedulers: dict[str, Scheduler] = {
        "joint_signed": lambda path, event: path[event],
        "fixed_edge_0": lambda path, event: 0,
        "fixed_edge_1": lambda path, event: 1,
        "fixed_initial_edge": lambda path, event: path[0],
        "round_robin_0": lambda path, event: event % 2,
        "round_robin_1": lambda path, event: 1 - event % 2,
        "no_refresh": lambda path, event: 2,
    }
    rows = {
        name: evaluate_scheduler(
            horizon=horizon, persistence=persistence, scheduler=scheduler
        )
        for name, scheduler in schedulers.items()
    }
    strong_baseline_names = (
        "fixed_edge_0",
        "fixed_edge_1",
        "fixed_initial_edge",
        "round_robin_0",
        "round_robin_1",
    )
    best_name = min(
        strong_baseline_names,
        key=lambda name: (rows[name].expected_terminal_potential, name),
    )
    proposed = rows["joint_signed"].expected_terminal_potential
    baseline = rows[best_name].expected_terminal_potential
    return {
        "horizon": horizon,
        "persistence": persistence,
        "rows": {
            name: {
                "expected_terminal_potential": row.expected_terminal_potential,
                "expected_cumulative_potential": row.expected_cumulative_potential,
                "expected_refresh_cost": row.expected_refresh_cost,
            }
            for name, row in rows.items()
        },
        "best_equal_cost_baseline": best_name,
        "terminal_potential_ratio": proposed / baseline,
        "terminal_potential_improvement": 1.0 - proposed / baseline,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--persistence", type=float, default=0.9)
    args = parser.parse_args()
    result = separation_table(args.horizon, args.persistence)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

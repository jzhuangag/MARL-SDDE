"""Frozen CPU pilot for Lyapunov scheduling of perishable MARL updates.

The environment is a cooperative quadratic potential game with distinct
scalar agent policies and Markovian asynchronous completion times.  Its
potential, block smoothness and cross-agent gradient sensitivities are
analytic.  The pilot tests the actual closed-form controller; it does not
reuse the earlier beam schedules.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import heapq
import json
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from experiments.policy_update_backpressure.phase_theory import (
    EventCertificate,
    closed_form_step,
    freshness_lyapunov_drift_bound,
    freshness_residual,
)


Array = np.ndarray
PILOT_SEEDS = tuple(range(910_001, 910_049))
HORIZON = 64
POTENTIAL_WEIGHT = 8.0


@dataclass(frozen=True)
class Cell:
    phase: str
    coupling: float
    slow_latency: int
    n_agents: int
    latency_family: str
    initial_name: str
    initial: tuple[float, ...]

    @property
    def cell_id(self) -> str:
        return (
            f"{self.phase}-c{self.coupling:.2f}-m{self.slow_latency}"
            f"-n{self.n_agents}-{self.latency_family}-{self.initial_name}"
        )


@dataclass(order=True, frozen=True)
class Event:
    time: int
    sequence: int
    agent: int


@dataclass
class ProposalState:
    gradient: float
    debt: float
    birth_event: int


class MarkovLatency:
    """Seeded two-state latency stream for one agent."""

    def __init__(
        self,
        rng: np.random.Generator,
        *,
        slow_latency: int,
        slow_probability: float,
        persistence: float,
    ) -> None:
        self.rng = rng
        self.slow_latency = slow_latency
        self.slow_probability = slow_probability
        self.persistence = persistence
        self.slow = bool(rng.random() < slow_probability)

    def next(self) -> int:
        if self.rng.random() >= self.persistence:
            self.slow = bool(self.rng.random() < self.slow_probability)
        jitter = int(self.rng.integers(0, 2)) if self.slow else 0
        return self.slow_latency+jitter if self.slow else 1


def declared_cells() -> tuple[Cell, ...]:
    """Outcome-independent analytic phase rays."""

    rays = (
        ("low", 0.05, 2),
        ("transition", 0.45, 5),
        ("high", 0.85, 10),
    )
    cells: list[Cell] = []
    for phase, coupling, slow_latency in rays:
        for n_agents in (3, 5):
            initials = {
                "sparse": tuple([1.0]+[0.0]*(n_agents-1)),
                "alternating": tuple(0.8*(-1.0)**i for i in range(n_agents)),
                "dense": tuple(np.linspace(1.0, -0.4, n_agents)),
            }
            for latency_family in ("persistent", "bursty"):
                for initial_name, initial in initials.items():
                    cells.append(Cell(
                        phase=phase,
                        coupling=coupling,
                        slow_latency=slow_latency,
                        n_agents=n_agents,
                        latency_family=latency_family,
                        initial_name=initial_name,
                        initial=tuple(float(x) for x in initial),
                    ))
    return tuple(cells)


def hessian(cell: Cell) -> Array:
    matrix = np.full((cell.n_agents, cell.n_agents), cell.coupling, dtype=float)
    np.fill_diagonal(matrix, 1.0)
    eigenvalues = np.linalg.eigvalsh(matrix)
    if eigenvalues.min() <= 0:
        raise AssertionError("declared quadratic potential is not strictly concave")
    return matrix


def potential(theta: Array, matrix: Array) -> float:
    return float(-0.5*np.asarray(theta, dtype=float)@matrix@np.asarray(theta, dtype=float))


def gradient(theta: Array, matrix: Array) -> Array:
    return -matrix@np.asarray(theta, dtype=float)


def latency_streams(cell: Cell, seed: int) -> tuple[MarkovLatency, ...]:
    streams: list[MarkovLatency] = []
    for agent in range(cell.n_agents):
        sequence = np.random.SeedSequence([seed, cell.n_agents, agent, cell.slow_latency])
        rng = np.random.default_rng(sequence)
        if cell.latency_family == "persistent":
            slow_probability = 0.92 if agent == cell.n_agents-1 else 0.04
            persistence = 0.96
        elif cell.latency_family == "bursty":
            slow_probability = 0.55 if agent == cell.n_agents-1 else 0.12
            persistence = 0.82
        else:
            raise ValueError("unknown latency family")
        streams.append(MarkovLatency(
            rng,
            slow_latency=cell.slow_latency,
            slow_probability=slow_probability,
            persistence=persistence,
        ))
    return tuple(streams)


def completion_schedule(cell: Cell, seed: int, horizon: int = HORIZON) -> tuple[Event, ...]:
    streams = latency_streams(cell, seed)
    heap: list[Event] = []
    sequence = 0
    for agent, stream in enumerate(streams):
        heapq.heappush(heap, Event(stream.next(), sequence, agent))
        sequence += 1
    events: list[Event] = []
    while heap:
        item = heapq.heappop(heap)
        if item.time > horizon:
            break
        events.append(item)
        heapq.heappush(
            heap,
            Event(item.time+streams[item.agent].next(), sequence, item.agent),
        )
        sequence += 1
    return tuple(events)


def integrate_regret(
    trajectory: list[tuple[int, float]], horizon: int = HORIZON
) -> float:
    if not trajectory or trajectory[0][0] != 0:
        raise ValueError("trajectory must start at zero")
    area = 0.0
    last_time, last_value = trajectory[0]
    for event_time, value in trajectory[1:]:
        if event_time < last_time or event_time > horizon:
            raise ValueError("invalid event time")
        area += (event_time-last_time)*(-last_value)
        last_time, last_value = event_time, value
    area += (horizon-last_time)*(-last_value)
    return float(max(area, 0.0))


def _initial_proposals(theta: Array, matrix: Array) -> list[ProposalState]:
    grad = gradient(theta, matrix)
    return [ProposalState(float(grad[i]), 0.0, 0) for i in range(theta.size)]


def _finish(
    theta: Array,
    matrix: Array,
    trajectory: list[tuple[int, float]],
    accepted: int,
    rejected: int,
    harmful: int,
    loads: list[float],
    steps: list[float],
    operations: int,
) -> dict[str, float | int]:
    initial_regret = -trajectory[0][1]*HORIZON
    return {
        "normalized_regret": integrate_regret(trajectory)/max(initial_regret, 1e-15),
        "final_gradient_norm": float(np.linalg.norm(gradient(theta, matrix))),
        "final_potential": potential(theta, matrix),
        "accepted": accepted,
        "rejected": rejected,
        "harmful": harmful,
        "acceptance_rate": accepted/max(accepted+rejected, 1),
        "median_load": float(np.median(loads)) if loads else 0.0,
        "high_load_fraction": float(np.mean(np.asarray(loads) >= 1.0)) if loads else 0.0,
        "mean_step": float(np.mean(steps)) if steps else 0.0,
        "controller_operations": operations,
    }


def simulate_pub(cell: Cell, events: tuple[Event, ...]) -> dict[str, float | int]:
    matrix = hessian(cell)
    theta = np.asarray(cell.initial, dtype=float)
    proposals = _initial_proposals(theta, matrix)
    trajectory = [(0, potential(theta, matrix))]
    accepted = rejected = harmful = operations = 0
    loads: list[float] = []
    steps: list[float] = []
    index = 0
    while index < len(events):
        event_time = events[index].time
        ready: list[Event] = []
        while index < len(events) and events[index].time == event_time:
            ready.append(events[index])
            index += 1
        while ready:
            candidates: list[tuple[float, int, Event, EventCertificate, float]] = []
            for item in ready:
                state = proposals[item.agent]
                cross = tuple(
                    cell.coupling for j in range(cell.n_agents) if j != item.agent
                )
                debts = tuple(
                    proposals[j].debt for j in range(cell.n_agents) if j != item.agent
                )
                cert = EventCertificate(
                    proposal_norm=abs(state.gradient),
                    own_debt=state.debt,
                    markov_radius=0.0,
                    smoothness=1.0,
                    cross_to_pending=cross,
                    pending_debts=debts,
                    max_step=1.0,
                    potential_weight=POTENTIAL_WEIGHT,
                )
                alpha = closed_form_step(cert)
                bound = freshness_lyapunov_drift_bound(cert, alpha)
                candidates.append((bound, item.agent, item, cert, alpha))
                operations += 2*cell.n_agents+8
            _, _, item, cert, alpha = min(candidates, key=lambda row: (row[0], row[1]))
            ready.remove(item)
            state = proposals[item.agent]
            s = abs(state.gradient)
            outgoing = sum(
                cell.coupling*proposals[j].debt
                for j in range(cell.n_agents) if j != item.agent
            )
            load = float("inf") if s == 0 else (
                state.debt/s+outgoing/(POTENTIAL_WEIGHT*s)
            )
            loads.append(load)
            steps.append(alpha)
            before = potential(theta, matrix)
            displacement = alpha*state.gradient
            theta[item.agent] += displacement
            after = potential(theta, matrix)
            accepted += int(alpha > 0)
            rejected += int(alpha == 0)
            harmful += int(alpha > 0 and after < before-1e-12)
            for other in range(cell.n_agents):
                if other != item.agent:
                    proposals[other].debt += cell.coupling*abs(displacement)
            current_gradient = gradient(theta, matrix)
            proposals[item.agent] = ProposalState(
                float(current_gradient[item.agent]), 0.0, index
            )
            trajectory.append((event_time, after))
            if freshness_residual(cert) == 0 and alpha != 0:
                raise AssertionError("zero residual produced a nonzero PUB action")
    return _finish(
        theta, matrix, trajectory, accepted, rejected, harmful, loads, steps,
        operations,
    )


def simulate_event_baseline(
    cell: Cell,
    events: tuple[Event, ...],
    *,
    eta: float,
    age_power: float = 0.0,
    rho_threshold: float = float("inf"),
) -> dict[str, float | int]:
    matrix = hessian(cell)
    theta = np.asarray(cell.initial, dtype=float)
    proposals = _initial_proposals(theta, matrix)
    trajectory = [(0, potential(theta, matrix))]
    accepted = rejected = harmful = 0
    steps: list[float] = []
    loads: list[float] = []
    for event_index, item in enumerate(events, start=1):
        state = proposals[item.agent]
        s = abs(state.gradient)
        rho = float("inf") if s == 0 else state.debt/s
        age = event_index-state.birth_event
        alpha = eta/(1.0+age)**age_power
        if rho > rho_threshold:
            alpha = 0.0
        before = potential(theta, matrix)
        displacement = alpha*state.gradient
        theta[item.agent] += displacement
        after = potential(theta, matrix)
        accepted += int(alpha > 0)
        rejected += int(alpha == 0)
        harmful += int(alpha > 0 and after < before-1e-12)
        for other in range(cell.n_agents):
            if other != item.agent:
                proposals[other].debt += cell.coupling*abs(displacement)
        current_gradient = gradient(theta, matrix)
        proposals[item.agent] = ProposalState(
            float(current_gradient[item.agent]), 0.0, event_index
        )
        trajectory.append((item.time, after))
        loads.append(rho)
        steps.append(alpha)
    return _finish(
        theta, matrix, trajectory, accepted, rejected, harmful, loads, steps,
        0,
    )


def simulate_fresh_serial(
    cell: Cell, seed: int, eta: float
) -> dict[str, float | int]:
    matrix = hessian(cell)
    theta = np.asarray(cell.initial, dtype=float)
    streams = latency_streams(cell, seed)
    trajectory = [(0, potential(theta, matrix))]
    now = accepted = 0
    while True:
        agent = accepted % cell.n_agents
        now += streams[agent].next()
        if now > HORIZON:
            break
        grad = gradient(theta, matrix)
        theta[agent] += eta*grad[agent]
        accepted += 1
        trajectory.append((now, potential(theta, matrix)))
    return _finish(theta, matrix, trajectory, accepted, 0, 0, [], [eta]*accepted, 0)


def simulate_barrier(cell: Cell, seed: int, eta: float) -> dict[str, float | int]:
    matrix = hessian(cell)
    theta = np.asarray(cell.initial, dtype=float)
    streams = latency_streams(cell, seed)
    trajectory = [(0, potential(theta, matrix))]
    now = accepted = harmful = 0
    while True:
        now += max(stream.next() for stream in streams)
        if now > HORIZON:
            break
        before = potential(theta, matrix)
        theta = theta+eta*gradient(theta, matrix)
        after = potential(theta, matrix)
        accepted += cell.n_agents
        harmful += int(after < before-1e-12)
        trajectory.append((now, after))
    return _finish(theta, matrix, trajectory, accepted, 0, harmful, [], [eta]*accepted, 0)


def baseline_specs() -> tuple[tuple[str, Callable[[Cell, tuple[Event, ...], int], dict]], ...]:
    specs: list[tuple[str, Callable[[Cell, tuple[Event, ...], int], dict]]] = []
    for eta in (0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0):
        specs.append((
            f"fixed_eta:{eta}",
            lambda c, e, s, eta=eta: simulate_event_baseline(c, e, eta=eta),
        ))
        specs.append((
            f"fresh_serial:{eta}",
            lambda c, e, s, eta=eta: simulate_fresh_serial(c, s, eta),
        ))
        specs.append((
            f"barrier:{eta}",
            lambda c, e, s, eta=eta: simulate_barrier(c, s, eta),
        ))
    for eta in (0.2, 0.5, 1.0):
        for power in (0.5, 1.0, 2.0):
            specs.append((
                f"age_decay:{eta}:{power}",
                lambda c, e, s, eta=eta, power=power: simulate_event_baseline(
                    c, e, eta=eta, age_power=power
                ),
            ))
        for threshold in (0.25, 0.5, 0.75, 1.0):
            specs.append((
                f"rho_gate:{eta}:{threshold}",
                lambda c, e, s, eta=eta, threshold=threshold: simulate_event_baseline(
                    c, e, eta=eta, rho_threshold=threshold
                ),
            ))
    return tuple(specs)


def validate_design() -> dict[str, object]:
    cells = declared_cells()
    specs = baseline_specs()
    if len(cells) != 36 or len(specs) != 42 or len(PILOT_SEEDS) != 48:
        raise AssertionError("frozen design cardinality changed")
    if len(set(PILOT_SEEDS)) != len(PILOT_SEEDS):
        raise AssertionError("pilot seeds are not unique")
    phase_counts = {phase: sum(c.phase == phase for c in cells) for phase in (
        "low", "transition", "high"
    )}
    if set(phase_counts.values()) != {12}:
        raise AssertionError("phase strata are unbalanced")
    for cell in cells:
        eig = np.linalg.eigvalsh(hessian(cell))
        if eig.min() <= 0 or not np.isfinite(eig).all():
            raise AssertionError("invalid cell Hessian")
        for seed in PILOT_SEEDS[:2]:
            events = completion_schedule(cell, seed)
            if not events or events[-1].time > HORIZON:
                raise AssertionError("invalid completion schedule")
    return {
        "cells": len(cells),
        "cells_per_phase": phase_counts,
        "pilot_seeds": len(PILOT_SEEDS),
        "policies": 1+len(specs),
        "trajectories": len(cells)*len(PILOT_SEEDS)*(1+len(specs)),
        "horizon": HORIZON,
        "potential_weight": POTENTIAL_WEIGHT,
        "status": "static_design_valid_no_population_outcomes",
    }


def run(output: Path) -> dict[str, object]:
    design = validate_design()
    specs = baseline_specs()
    rows: list[dict[str, object]] = []
    for cell in declared_cells():
        for seed in PILOT_SEEDS:
            events = completion_schedule(cell, seed)
            pub = simulate_pub(cell, events)
            rows.append({
                "cell_id": cell.cell_id,
                "phase": cell.phase,
                "coupling": cell.coupling,
                "slow_latency": cell.slow_latency,
                "n_agents": cell.n_agents,
                "latency_family": cell.latency_family,
                "initial_name": cell.initial_name,
                "seed": seed,
                "policy": "pub",
                **pub,
            })
            for name, runner in specs:
                result = runner(cell, events, seed)
                rows.append({
                    "cell_id": cell.cell_id,
                    "phase": cell.phase,
                    "coupling": cell.coupling,
                    "slow_latency": cell.slow_latency,
                    "n_agents": cell.n_agents,
                    "latency_family": cell.latency_family,
                    "initial_name": cell.initial_name,
                    "seed": seed,
                    "policy": name,
                    **result,
                })
    payload = {
        "design": design,
        "rows": rows,
    }
    output.mkdir(parents=True, exist_ok=False)
    (output/"trajectories.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return {"output": str(output), **design}


def main(argv: Iterable[str] | None = None) -> dict[str, object]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("validate", "run"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(tuple(argv) if argv is not None else None)
    if args.mode == "validate":
        return validate_design()
    if args.output is None:
        raise ValueError("run mode requires --output")
    return run(args.output)


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, sort_keys=True))

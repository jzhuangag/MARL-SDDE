"""Outcome-free wall-clock headroom scan for exact gradient transport.

The scan uses deterministic coupled quadratic games and deterministic service
patterns.  It is a design feasibility calculation, not sampled efficacy or
formal paper evidence.  ``run`` must only be called after its static plan is
committed.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import heapq
import json
from pathlib import Path
from typing import Callable, Iterable

import numpy as np


Array = np.ndarray
HORIZON = 96.0


@dataclass(frozen=True)
class Cell:
    n_agents: int
    coupling: float
    latency_pattern: str
    slow_latency: float
    initial_name: str
    initial: tuple[float, ...]
    hvp_overhead: float

    @property
    def phase(self) -> str:
        if self.coupling == 0.2 and self.slow_latency == 4.0:
            return "low"
        if self.coupling == 0.8 and self.slow_latency == 12.0:
            return "high"
        return "transition"

    @property
    def cell_id(self) -> str:
        return (
            f"n{self.n_agents}-c{self.coupling:.1f}-{self.latency_pattern}"
            f"-m{self.slow_latency:.0f}-{self.initial_name}-h{self.hvp_overhead:g}"
        )


@dataclass(frozen=True)
class Proposal:
    agent: int
    birth_time: float
    birth_theta: tuple[float, ...]
    birth_gradient: float


def declared_cells() -> tuple[Cell, ...]:
    cells: list[Cell] = []
    for n_agents in (3, 5, 8):
        initials = {
            "sparse": tuple([1.0]+[0.0]*(n_agents-1)),
            "alternating": tuple(0.8*(-1.0)**i for i in range(n_agents)),
            "ramp": tuple(float(x) for x in np.linspace(1.0, -0.3, n_agents)),
        }
        for coupling in (0.2, 0.5, 0.8):
            for latency_pattern in ("one_straggler", "two_tier"):
                for slow_latency in (4.0, 12.0):
                    for initial_name, initial in initials.items():
                        for hvp_overhead in (0.0, 0.25, 1.0):
                            cells.append(Cell(
                                n_agents=n_agents,
                                coupling=coupling,
                                latency_pattern=latency_pattern,
                                slow_latency=slow_latency,
                                initial_name=initial_name,
                                initial=initial,
                                hvp_overhead=hvp_overhead,
                            ))
    return tuple(cells)


def latencies(cell: Cell) -> Array:
    if cell.latency_pattern == "one_straggler":
        result = np.ones(cell.n_agents, dtype=float)
        result[-1] = cell.slow_latency
        return result
    if cell.latency_pattern == "two_tier":
        return np.asarray([
            2.0 if i % 2 == 0 else cell.slow_latency
            for i in range(cell.n_agents)
        ])
    raise ValueError("unknown latency pattern")


def matrix(cell: Cell) -> Array:
    result = np.full(
        (cell.n_agents, cell.n_agents), cell.coupling, dtype=float
    )
    np.fill_diagonal(result, 1.0)
    if np.linalg.eigvalsh(result).min() <= 0:
        raise AssertionError("quadratic game is not strictly concave")
    return result


def potential(theta: Array, hessian: Array) -> float:
    theta = np.asarray(theta, dtype=float)
    return float(-0.5*theta@hessian@theta)


def gradient(theta: Array, hessian: Array) -> Array:
    return -hessian@np.asarray(theta, dtype=float)


def integrated_regret(
    trajectory: list[tuple[float, float]], horizon: float = HORIZON
) -> float:
    if not trajectory or trajectory[0][0] != 0.0:
        raise ValueError("trajectory must begin at time zero")
    area = 0.0
    last_time, last_value = trajectory[0]
    for event_time, value in trajectory[1:]:
        if event_time < last_time or event_time > horizon:
            raise ValueError("invalid trajectory time")
        area += (event_time-last_time)*(-last_value)
        last_time, last_value = event_time, value
    area += (horizon-last_time)*(-last_value)
    return float(max(area, 0.0))


def _result(
    theta: Array,
    hessian: Array,
    trajectory: list[tuple[float, float]],
    accepted: int,
    harmful: int,
) -> dict[str, float | int]:
    initial_area = -trajectory[0][1]*HORIZON
    return {
        "normalized_regret": integrated_regret(trajectory)/max(initial_area, 1e-15),
        "final_gradient_norm": float(np.linalg.norm(gradient(theta, hessian))),
        "accepted": accepted,
        "harmful": harmful,
    }


def _new_proposal(
    agent: int, now: float, theta: Array, hessian: Array
) -> Proposal:
    return Proposal(
        agent=agent,
        birth_time=now,
        birth_theta=tuple(float(x) for x in theta),
        birth_gradient=float(gradient(theta, hessian)[agent]),
    )


def simulate_event_policy(
    cell: Cell,
    *,
    mode: str,
    eta: float,
    age_power: float = 0.0,
    delay_beta: float = 0.0,
    rho_threshold: float = float("inf"),
) -> dict[str, float | int]:
    hessian = matrix(cell)
    theta = np.asarray(cell.initial, dtype=float)
    worker_latency = latencies(cell)
    trajectory = [(0.0, potential(theta, hessian))]
    heap: list[tuple[float, int, Proposal]] = []
    sequence = 0
    for agent in range(cell.n_agents):
        proposal = _new_proposal(agent, 0.0, theta, hessian)
        heapq.heappush(heap, (worker_latency[agent], sequence, proposal))
        sequence += 1
    server_free = 0.0
    accepted = harmful = 0
    while heap:
        arrival, _, proposal = heapq.heappop(heap)
        service = cell.hvp_overhead if mode == "transport" else 0.0
        finish = max(arrival, server_free)+service
        if finish > HORIZON:
            break
        current_gradient = gradient(theta, hessian)
        if mode == "transport":
            direction = float(current_gradient[proposal.agent])
            step = 1.0
        else:
            direction = proposal.birth_gradient
            age = finish-proposal.birth_time
            birth_theta = np.asarray(proposal.birth_theta, dtype=float)
            debt = cell.coupling*float(np.sum(np.abs(theta-birth_theta)))
            debt -= cell.coupling*abs(float(
                theta[proposal.agent]-birth_theta[proposal.agent]
            ))
            rho = debt/max(abs(direction), 1e-15)
            step = eta
            if age_power > 0:
                step /= (1.0+age/cell.slow_latency)**age_power
            if delay_beta > 0:
                step /= 1.0+delay_beta*age
            if rho > rho_threshold:
                step = 0.0
        before = potential(theta, hessian)
        theta[proposal.agent] += step*direction
        after = potential(theta, hessian)
        accepted += int(step > 0)
        harmful += int(step > 0 and after < before-1e-12)
        trajectory.append((finish, after))
        server_free = finish
        replacement = _new_proposal(proposal.agent, finish, theta, hessian)
        heapq.heappush(
            heap,
            (finish+worker_latency[proposal.agent], sequence, replacement),
        )
        sequence += 1
    return _result(theta, hessian, trajectory, accepted, harmful)


def simulate_fresh_serial(cell: Cell, eta: float) -> dict[str, float | int]:
    hessian = matrix(cell)
    theta = np.asarray(cell.initial, dtype=float)
    worker_latency = latencies(cell)
    trajectory = [(0.0, potential(theta, hessian))]
    now = 0.0
    accepted = harmful = 0
    while True:
        agent = accepted % cell.n_agents
        now += worker_latency[agent]
        if now > HORIZON:
            break
        before = potential(theta, hessian)
        theta[agent] += eta*gradient(theta, hessian)[agent]
        after = potential(theta, hessian)
        accepted += 1
        harmful += int(after < before-1e-12)
        trajectory.append((now, after))
    return _result(theta, hessian, trajectory, accepted, harmful)


def simulate_barrier(cell: Cell, eta: float) -> dict[str, float | int]:
    hessian = matrix(cell)
    theta = np.asarray(cell.initial, dtype=float)
    round_time = float(np.max(latencies(cell)))
    trajectory = [(0.0, potential(theta, hessian))]
    now = 0.0
    accepted = harmful = 0
    while now+round_time <= HORIZON:
        now += round_time
        before = potential(theta, hessian)
        theta = theta+eta*gradient(theta, hessian)
        after = potential(theta, hessian)
        accepted += cell.n_agents
        harmful += int(after < before-1e-12)
        trajectory.append((now, after))
    return _result(theta, hessian, trajectory, accepted, harmful)


def baseline_specs() -> tuple[tuple[str, Callable[[Cell], dict]], ...]:
    specs: list[tuple[str, Callable[[Cell], dict]]] = []
    etas = (0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0)
    for eta in etas:
        specs.append((
            f"raw:{eta}",
            lambda c, eta=eta: simulate_event_policy(c, mode="raw", eta=eta),
        ))
        specs.append((
            f"fresh_serial:{eta}",
            lambda c, eta=eta: simulate_fresh_serial(c, eta),
        ))
        specs.append((
            f"barrier:{eta}",
            lambda c, eta=eta: simulate_barrier(c, eta),
        ))
    for eta in (0.4, 0.8, 1.0):
        for power in (0.5, 1.0, 2.0):
            specs.append((
                f"age_decay:{eta}:{power}",
                lambda c, eta=eta, power=power: simulate_event_policy(
                    c, mode="raw", eta=eta, age_power=power
                ),
            ))
        for beta in (0.05, 0.1, 0.2, 0.5):
            specs.append((
                f"delay_adaptive:{eta}:{beta}",
                lambda c, eta=eta, beta=beta: simulate_event_policy(
                    c, mode="raw", eta=eta, delay_beta=beta
                ),
            ))
        for threshold in (0.25, 0.5, 0.75, 1.0):
            specs.append((
                f"rho_gate:{eta}:{threshold}",
                lambda c, eta=eta, threshold=threshold: simulate_event_policy(
                    c, mode="raw", eta=eta, rho_threshold=threshold
                ),
            ))
    return tuple(specs)


def validate_design() -> dict[str, object]:
    cells = declared_cells()
    specs = baseline_specs()
    if len(cells) != 324 or len(specs) != 54:
        raise AssertionError("static scan cardinality changed")
    if len({cell.cell_id for cell in cells}) != len(cells):
        raise AssertionError("duplicate analytic cells")
    phases = {
        phase: sum(cell.phase == phase for cell in cells)
        for phase in ("low", "transition", "high")
    }
    if phases != {"low": 54, "transition": 216, "high": 54}:
        raise AssertionError("phase strata changed")
    return {
        "cells": len(cells),
        "phase_cells": phases,
        "comparators": len(specs),
        "policies": 1+len(specs),
        "trajectories": len(cells)*(1+len(specs)),
        "horizon": HORIZON,
        "status": "static_design_valid_no_scan_outcomes",
    }


def _geometric(values: list[float]) -> float:
    array = np.asarray(values, dtype=float)
    if array.size == 0 or not np.isfinite(array).all() or (array <= 0).any():
        raise ValueError("geometric mean needs positive finite values")
    return float(np.exp(np.mean(np.log(array))))


def run(output: Path) -> dict[str, object]:
    design = validate_design()
    specs = baseline_specs()
    rows: list[dict[str, object]] = []
    cell_results: list[dict[str, object]] = []
    for cell in declared_cells():
        transport = simulate_event_policy(cell, mode="transport", eta=1.0)
        policy_results = {name: runner(cell) for name, runner in specs}
        best_name = min(
            policy_results,
            key=lambda name: (
                float(policy_results[name]["normalized_regret"]), name
            ),
        )
        best = policy_results[best_name]
        ratio = (
            float(transport["normalized_regret"])+1e-15
        )/(
            float(best["normalized_regret"])+1e-15
        )
        cell_results.append({
            "cell_id": cell.cell_id,
            "phase": cell.phase,
            "n_agents": cell.n_agents,
            "coupling": cell.coupling,
            "latency_pattern": cell.latency_pattern,
            "slow_latency": cell.slow_latency,
            "initial_name": cell.initial_name,
            "hvp_overhead": cell.hvp_overhead,
            "best_comparator": best_name,
            "transport_regret": transport["normalized_regret"],
            "best_comparator_regret": best["normalized_regret"],
            "regret_ratio": ratio,
            "regret_gain": 1.0-ratio,
            "transport_final_gradient": transport["final_gradient_norm"],
            "best_comparator_final_gradient": best["final_gradient_norm"],
            "transport_harmful": transport["harmful"],
        })
        rows.append({
            "cell_id": cell.cell_id,
            "phase": cell.phase,
            "policy": "exact_transport",
            **transport,
        })
        for name, result in policy_results.items():
            rows.append({
                "cell_id": cell.cell_id,
                "phase": cell.phase,
                "policy": name,
                **result,
            })

    phase_summary = {}
    for phase in ("low", "transition", "high"):
        selected = [row for row in cell_results if row["phase"] == phase]
        ratios = [float(row["regret_ratio"]) for row in selected]
        phase_summary[phase] = {
            "cells": len(selected),
            "geometric_regret_ratio": _geometric(ratios),
            "median_regret_gain": float(np.median(1.0-np.asarray(ratios))),
            "fraction_ten_percent_gain": float(np.mean(np.asarray(ratios) <= 0.9)),
        }
    high_full_cost = [
        row for row in cell_results
        if row["phase"] == "high" and float(row["hvp_overhead"]) == 1.0
    ]
    finite = all(
        np.isfinite(float(row[key]))
        for row in rows
        for key in ("normalized_regret", "final_gradient_norm")
    )
    gates = {
        "complete_finite": len(rows) == int(design["trajectories"]) and finite,
        "high_geometric_ratio_le_0_90": (
            phase_summary["high"]["geometric_regret_ratio"] <= 0.90
        ),
        "high_ten_percent_cells_ge_0_60": (
            phase_summary["high"]["fraction_ten_percent_gain"] >= 0.60
        ),
        "transition_geometric_ratio_le_0_95": (
            phase_summary["transition"]["geometric_regret_ratio"] <= 0.95
        ),
        "low_geometric_ratio_le_1_05": (
            phase_summary["low"]["geometric_regret_ratio"] <= 1.05
        ),
        "high_full_hvp_cost_ratio_le_0_95": (
            _geometric([float(row["regret_ratio"]) for row in high_full_cost])
            <= 0.95
        ),
        "no_transport_harmful_steps": (
            sum(int(row["transport_harmful"]) for row in cell_results) == 0
        ),
    }
    payload = {
        "kind": "outcome_free_exact_transport_headroom_scan",
        "design": design,
        "phase_summary": phase_summary,
        "high_full_hvp_cost_geometric_ratio": _geometric([
            float(row["regret_ratio"]) for row in high_full_cost
        ]),
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "efficacy_pilot_authorized": False,
        "cell_results": cell_results,
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def main(argv: Iterable[str] | None = None) -> dict[str, object]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("validate", "run"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(tuple(argv) if argv is not None else None)
    if args.mode == "validate":
        return validate_design()
    if args.output is None:
        raise ValueError("run requires --output")
    return run(args.output)


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, sort_keys=True))

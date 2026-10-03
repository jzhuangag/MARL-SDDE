"""Outcome-free scheduling ceiling for compatible asynchronous updates."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from .compatible_update_theory import (
    greedy_maximal_weight_independent_set,
    is_independent,
    path_max_weight_independent_set,
)


HORIZON = 240
POTENTIAL_WEIGHT = 0.75
POLICIES = (
    "compatible_maxweight",
    "best_color",
    "cycle_color",
    "static_priority",
    "sequential_maxweight",
)


@dataclass(frozen=True)
class Scenario:
    n_agents: int
    graph: str
    workload: str

    @property
    def scenario_id(self) -> str:
        return f"n{self.n_agents}-{self.graph}-{self.workload}"


def declared_scenarios() -> tuple[Scenario, ...]:
    return tuple(
        Scenario(n, graph, workload)
        for n in (16, 36, 64)
        for graph in ("path", "tree", "grid", "clustered")
        for workload in (
            "homogeneous", "two_tier", "rotating_burst", "color_skew"
        )
    )


def _add_edge(adjacency: list[set[int]], left: int, right: int) -> None:
    if left == right:
        return
    adjacency[left].add(right)
    adjacency[right].add(left)


def graph_adjacency(scenario: Scenario) -> tuple[frozenset[int], ...]:
    n = scenario.n_agents
    adjacency = [set() for _ in range(n)]
    if scenario.graph == "path":
        for node in range(n-1):
            _add_edge(adjacency, node, node+1)
    elif scenario.graph == "tree":
        for node in range(1, n):
            _add_edge(adjacency, node, (node-1)//2)
    elif scenario.graph == "grid":
        width = int(round(math.sqrt(n)))
        if width*width != n:
            raise AssertionError("grid scenarios must have square agent count")
        for row in range(width):
            for column in range(width):
                node = row*width+column
                if row+1 < width:
                    _add_edge(adjacency, node, node+width)
                if column+1 < width:
                    _add_edge(adjacency, node, node+1)
    elif scenario.graph == "clustered":
        cluster_size = 4
        for start in range(0, n, cluster_size):
            nodes = list(range(start, min(start+cluster_size, n)))
            for offset, left in enumerate(nodes):
                for right in nodes[offset+1:]:
                    _add_edge(adjacency, left, right)
            if start > 0:
                _add_edge(adjacency, start-1, start)
    else:
        raise ValueError("unknown graph")
    return tuple(frozenset(neighbors) for neighbors in adjacency)


def greedy_coloring(adjacency: tuple[frozenset[int], ...]) -> tuple[int, ...]:
    order = sorted(range(len(adjacency)), key=lambda i: (-len(adjacency[i]), i))
    colors = [-1]*len(adjacency)
    for node in order:
        used = {colors[neighbor] for neighbor in adjacency[node] if colors[neighbor] >= 0}
        color = 0
        while color in used:
            color += 1
        colors[node] = color
    return tuple(colors)


def design_payload() -> dict[str, object]:
    scenarios = declared_scenarios()
    return {
        "graphs": ["path", "tree", "grid", "clustered"],
        "horizon": HORIZON,
        "n_agents": [16, 36, 64],
        "policies": list(POLICIES),
        "potential_weight": POTENTIAL_WEIGHT,
        "scenario_policy_runs": len(scenarios)*len(POLICIES),
        "scenarios": len(scenarios),
        "workloads": [
            "homogeneous", "two_tier", "rotating_burst", "color_skew"
        ],
    }


def design_hash() -> str:
    return hashlib.sha256(json.dumps(
        design_payload(), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def validate_design() -> dict[str, object]:
    scenarios = declared_scenarios()
    if len(scenarios) != 48 or len({x.scenario_id for x in scenarios}) != 48:
        raise AssertionError("compatible scheduling scenarios changed")
    for scenario in scenarios:
        adjacency = graph_adjacency(scenario)
        colors = greedy_coloring(adjacency)
        for color in set(colors):
            nodes = [i for i, value in enumerate(colors) if value == color]
            if not is_independent(nodes, adjacency):
                raise AssertionError("invalid declared coloring")
    return {
        "design": design_payload(),
        "design_hash": design_hash(),
        "status": "static_design_valid_no_outcomes",
    }


def arrivals(scenario: Scenario, epoch: int, colors: tuple[int, ...]) -> np.ndarray:
    n = scenario.n_agents
    result = np.zeros(n, dtype=int)
    color_count = max(colors)+1
    for node in range(n):
        phase = (7*node+3*colors[node]) % 11
        if scenario.workload == "homogeneous":
            active = (epoch+phase) % 5 == 0
        elif scenario.workload == "two_tier":
            period = 3 if node % 3 else 9
            active = (epoch+phase) % period == 0
        elif scenario.workload == "rotating_burst":
            active_group = (epoch//12) % 4
            period = 2 if node % 4 == active_group else 11
            active = (epoch+phase) % period == 0
        elif scenario.workload == "color_skew":
            favored = (epoch//10) % color_count
            period = 2 if colors[node] == favored else 8
            active = (epoch+phase) % period == 0
        else:
            raise ValueError("unknown workload")
        result[node] = int(active)
    return result


def progress_weights(scenario: Scenario, epoch: int) -> np.ndarray:
    node = np.arange(scenario.n_agents, dtype=float)
    return 0.5+0.5*(
        1.0+np.sin(2.0*np.pi*(epoch/31.0+node/(scenario.n_agents+3.0)))
    )


def _best_color_set(
    weights: np.ndarray, ready: np.ndarray, colors: tuple[int, ...]
) -> tuple[int, ...]:
    best: tuple[int, ...] = ()
    best_value = -1.0
    for color in range(max(colors)+1):
        chosen = tuple(
            node for node, value in enumerate(colors)
            if value == color and ready[node]
        )
        value = float(sum(weights[node] for node in chosen))
        if value > best_value+1e-15 or (abs(value-best_value) <= 1e-15 and chosen < best):
            best, best_value = chosen, value
    return best


def select(
    policy: str,
    scenario: Scenario,
    epoch: int,
    queues: np.ndarray,
    progress: np.ndarray,
    adjacency: tuple[frozenset[int], ...],
    colors: tuple[int, ...],
) -> tuple[int, ...]:
    ready = queues > 0
    weights = queues+POTENTIAL_WEIGHT*progress
    if not ready.any():
        return ()
    if policy == "best_color":
        return _best_color_set(weights, ready, colors)
    if policy == "cycle_color":
        color = epoch % (max(colors)+1)
        return tuple(
            node for node, value in enumerate(colors)
            if value == color and ready[node]
        )
    if policy == "static_priority":
        priority = np.asarray([
            2.0-(13*node % (scenario.n_agents+1))/scenario.n_agents
            for node in range(scenario.n_agents)
        ])
        return greedy_maximal_weight_independent_set(priority, adjacency, ready)
    if policy == "sequential_maxweight":
        node = min(
            np.flatnonzero(ready), key=lambda i: (-weights[i], int(i))
        )
        return (int(node),)
    if policy == "compatible_maxweight":
        if scenario.graph == "path":
            candidate = path_max_weight_independent_set(weights, ready)
        else:
            candidate = greedy_maximal_weight_independent_set(
                weights, adjacency, ready
            )
        color_candidate = _best_color_set(weights, ready, colors)
        candidate_value = float(sum(weights[node] for node in candidate))
        color_value = float(sum(weights[node] for node in color_candidate))
        return candidate if candidate_value >= color_value-1e-15 else color_candidate
    raise ValueError("unknown policy")


def simulate(scenario: Scenario, policy: str) -> dict[str, object]:
    adjacency = graph_adjacency(scenario)
    colors = greedy_coloring(adjacency)
    queues = np.zeros(scenario.n_agents, dtype=int)
    waiting = [deque() for _ in range(scenario.n_agents)]
    arrivals_total = np.zeros(scenario.n_agents, dtype=int)
    services_total = np.zeros(scenario.n_agents, dtype=int)
    waits: list[int] = []
    queue_area = 0.0
    compatible = True
    for epoch in range(HORIZON):
        new = arrivals(scenario, epoch, colors)
        queues += new
        arrivals_total += new
        for node, count in enumerate(new):
            if count:
                waiting[node].append(epoch)
        chosen = select(
            policy, scenario, epoch, queues,
            progress_weights(scenario, epoch), adjacency, colors
        )
        compatible &= is_independent(chosen, adjacency)
        for node in chosen:
            if queues[node] <= 0:
                raise AssertionError("scheduler served an empty queue")
            queues[node] -= 1
            services_total[node] += 1
            waits.append(epoch-waiting[node].popleft())
        queue_area += float(np.sum(queues))
    unfinished_waits = [
        HORIZON-born
        for node_queue in waiting
        for born in node_queue
    ]
    all_waits = waits+unfinished_waits
    terminal = int(np.sum(queues))
    cost = queue_area+HORIZON*terminal
    active = arrivals_total > 0
    coverage = float(np.mean(services_total[active] > 0))
    return {
        "arrivals": int(np.sum(arrivals_total)),
        "completed": int(np.sum(services_total)),
        "compatible": bool(compatible),
        "coverage": coverage,
        "cost": float(cost),
        "max_wait": int(max(all_waits, default=0)),
        "mean_wait": float(np.mean(all_waits)) if all_waits else 0.0,
        "queue_area": queue_area,
        "terminal_backlog": terminal,
    }


def _geometric(values: list[float]) -> float:
    array = np.asarray(values, dtype=float)
    if (array <= 0).any() or not np.isfinite(array).all():
        raise ValueError("geometric mean needs positive finite values")
    return float(np.exp(np.mean(np.log(array))))


def run(output: Path) -> dict[str, object]:
    validation = validate_design()
    rows: list[dict[str, object]] = []
    scenario_results: list[dict[str, object]] = []
    for scenario in declared_scenarios():
        results = {policy: simulate(scenario, policy) for policy in POLICIES}
        for policy, result in results.items():
            rows.append({
                "scenario_id": scenario.scenario_id,
                "policy": policy,
                **result,
            })
        strong_names = ("best_color", "cycle_color", "static_priority")
        best_name = min(
            strong_names,
            key=lambda name: (float(results[name]["cost"]), name),
        )
        dynamic = results["compatible_maxweight"]
        best = results[best_name]
        sequential = results["sequential_maxweight"]
        cost_ratio = (float(dynamic["cost"])+1.0)/(float(best["cost"])+1.0)
        throughput_ratio = (
            float(dynamic["completed"])/max(float(sequential["completed"]), 1.0)
        )
        scenario_results.append({
            **asdict(scenario),
            "scenario_id": scenario.scenario_id,
            "best_compatible_baseline": best_name,
            "cost_ratio": cost_ratio,
            "cost_reduction": 1.0-cost_ratio,
            "throughput_ratio_vs_sequential": throughput_ratio,
            "dynamic_completed": dynamic["completed"],
            "dynamic_coverage": dynamic["coverage"],
            "dynamic_max_wait": dynamic["max_wait"],
            "baseline_completed": best["completed"],
            "baseline_max_wait": best["max_wait"],
        })
    cost_ratios = [float(row["cost_ratio"]) for row in scenario_results]
    throughput_ratios = [
        float(row["throughput_ratio_vs_sequential"]) for row in scenario_results
    ]
    finite = all(
        np.isfinite(float(row[key]))
        for row in rows
        for key in ("cost", "mean_wait", "queue_area")
    )
    gates = {
        "complete_finite_compatible": (
            len(rows) == int(validation["design"]["scenario_policy_runs"])
            and finite and all(bool(row["compatible"]) for row in rows)
        ),
        "aggregate_cost_ratio_le_0_85": _geometric(cost_ratios) <= 0.85,
        "scenario_ten_percent_fraction_ge_0_60": float(
            np.mean(np.asarray(cost_ratios) <= 0.90)
        ) >= 0.60,
        "throughput_ratio_vs_sequential_ge_2": _geometric(throughput_ratios) >= 2.0,
        "full_actor_coverage": min(
            float(row["dynamic_coverage"]) for row in scenario_results
        ) >= 1.0,
        "dynamic_no_worse_than_strong_baseline": max(cost_ratios) <= 1.0+1e-12,
    }
    payload = {
        "kind": "outcome_free_compatible_schedule_headroom",
        **validation,
        "aggregate": {
            "geometric_cost_ratio": _geometric(cost_ratios),
            "scenario_ten_percent_fraction": float(
                np.mean(np.asarray(cost_ratios) <= 0.90)
            ),
            "geometric_throughput_ratio_vs_sequential": _geometric(throughput_ratios),
            "median_cost_reduction": float(np.median(1.0-np.asarray(cost_ratios))),
        },
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "stochastic_pilot_authorized": False,
        "scenario_results": scenario_results,
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


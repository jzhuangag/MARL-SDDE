"""Outcome-free CPU value screen for perishable MARL policy updates.

The scan uses exact finite cooperative Markov games and deterministic latency
traces.  It does not use T-083A outcomes, random/formal seeds, a neural policy,
or a claimed implementable controller.  A finite-width beam is a feasible
non-myopic schedule and therefore a lower bound on achievable dynamic value,
not an oracle ceiling.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import heapq
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from experiments.policy_backpressure.policy_backpressure_feasibility import (
    FiniteTeamGame,
    joint_tv_max,
)


Array = np.ndarray


@dataclass(order=True, frozen=True)
class Completion:
    """One completed unilateral proposal in the deterministic event trace."""

    time: int
    sequence: int
    agent: int


@dataclass
class TraceCursor:
    values: tuple[int, ...]
    index: int = 0

    def next(self) -> int:
        if not self.values or min(self.values) < 1:
            raise ValueError("latency trace must contain positive integers")
        value = self.values[self.index % len(self.values)]
        self.index += 1
        return value


@dataclass
class BeamState:
    current: Array
    references: tuple[Array, ...]
    birth_events: tuple[int, ...]
    area: float
    last_time: int
    accepted: int
    harmful: int
    actions: tuple[float, ...]


def make_role_switch_game(
    gamma: float,
    transition_focus: float,
    coupling: float,
) -> FiniteTeamGame:
    """Three-agent two-state team game with coordination-dependent roles.

    State 0 favors full coordination, while state 1 favors one-agent
    specialization.  Joint actions also change the next-state distribution.
    The family is fixed before the scan and is not fitted to outcomes.
    """

    if not 0.5 < transition_focus < 1.0:
        raise ValueError("transition_focus must lie in (0.5,1)")
    if not 0.0 <= coupling <= 1.0:
        raise ValueError("coupling must lie in [0,1]")
    n_agents = 3
    n_actions = 2**n_agents
    transition = np.zeros((2, n_actions, 2), dtype=float)
    reward = np.zeros((2, n_actions), dtype=float)
    targets = ((0, 0, 0), (1, 0, 0))
    for state in (0, 1):
        for idx, action in enumerate(product((0, 1), repeat=n_agents)):
            count = sum(action)
            coordinated = (count in (0, 3)) if state == 0 else (count == 1)
            local = sum(int(a == b) for a, b in zip(action, targets[state])) / n_agents
            reward[state, idx] = coupling * float(coordinated) + (1-coupling) * local
            next_state = count % 2
            transition[state, idx, next_state] = transition_focus
            transition[state, idx, 1-next_state] = 1.0-transition_focus
    return FiniteTeamGame(
        transition=transition,
        reward=reward,
        gamma=gamma,
        initial=np.array([0.5, 0.5], dtype=float),
        n_agents=n_agents,
    )


def deterministic_optimum(game: FiniteTeamGame) -> tuple[float, Array]:
    """Enumerate deterministic factorized stationary policies exactly."""

    best: tuple[float, Array] | None = None
    shape = (game.n_agents, game.n_states)
    for bits in product((0.0, 1.0), repeat=game.n_agents * game.n_states):
        policy = np.asarray(bits, dtype=float).reshape(shape)
        value = game.evaluate(policy)[0]
        if best is None or value > best[0]:
            best = (value, policy.copy())
    assert best is not None
    return best


def completion_schedule(
    latency_traces: tuple[tuple[int, ...], ...], horizon: int
) -> list[Completion]:
    """Generate completions when every agent immediately restarts its work."""

    if horizon < 1:
        raise ValueError("horizon must be positive")
    cursors = [TraceCursor(tuple(trace)) for trace in latency_traces]
    events: list[Completion] = []
    heap: list[Completion] = []
    seq = 0
    for agent, cursor in enumerate(cursors):
        heapq.heappush(heap, Completion(cursor.next(), seq, agent))
        seq += 1
    while heap:
        item = heapq.heappop(heap)
        if item.time > horizon:
            break
        events.append(item)
        ready = item.time + cursors[item.agent].next()
        heapq.heappush(heap, Completion(ready, seq, item.agent))
        seq += 1
    return events


def integrate_regret(
    trajectory: list[tuple[int, float]], horizon: int, optimum: float
) -> tuple[float, float]:
    """Return wall-clock reward area and regret area to the exact optimum."""

    if not trajectory or trajectory[0][0] != 0:
        raise ValueError("trajectory must begin at time zero")
    area = 0.0
    previous_time, previous_value = trajectory[0]
    for time, value in trajectory[1:]:
        if time < previous_time or time > horizon:
            raise ValueError("invalid trajectory time")
        area += (time-previous_time) * previous_value
        previous_time, previous_value = time, value
    area += (horizon-previous_time) * previous_value
    regret = horizon*optimum-area
    if regret < -1e-8:
        raise AssertionError("policy exceeded enumerated optimum")
    return float(area), float(max(regret, 0.0))


def simulate_event_rule(
    game: FiniteTeamGame,
    initial: Array,
    events: list[Completion],
    horizon: int,
    optimum: float,
    *,
    eta: float,
    age_power: float = 0.0,
    tv_threshold: float = float("inf"),
) -> dict[str, object]:
    """Apply a fixed, age-decayed or direct-TV-gated asynchronous rule."""

    if not 0.0 <= eta <= 1.0 or age_power < 0 or tv_threshold < 0:
        raise ValueError("invalid rule parameter")
    current = initial.copy()
    references = [initial.copy() for _ in range(game.n_agents)]
    births = [0 for _ in range(game.n_agents)]
    trajectory = [(0, game.evaluate(current)[0])]
    accepted = harmful = 0
    for event_index, item in enumerate(events, start=1):
        reference = references[item.agent]
        candidate = game.greedy_candidate(reference, item.agent)
        age = event_index-births[item.agent]
        direct_tv = joint_tv_max(game, current, reference)
        scale = eta/(1.0+age)**age_power
        if direct_tv > tv_threshold:
            scale = 0.0
        j0 = game.evaluate(current)[0]
        updated = game.mix_agent(current, candidate, item.agent, scale)
        j1 = game.evaluate(updated)[0]
        accepted += int(scale > 0)
        harmful += int(scale > 0 and j1 < j0-1e-12)
        current = updated
        references[item.agent] = current.copy()
        births[item.agent] = event_index
        trajectory.append((item.time, j1))
    area, regret = integrate_regret(trajectory, horizon, optimum)
    return {
        "area": area,
        "regret": regret,
        "final_return": game.evaluate(current)[0],
        "accepted": accepted,
        "harmful": harmful,
    }


def simulate_fresh_serial(
    game: FiniteTeamGame,
    initial: Array,
    latency_traces: tuple[tuple[int, ...], ...],
    horizon: int,
    optimum: float,
    eta: float,
) -> dict[str, object]:
    """Round-robin fresh unilateral updates with no stale proposal."""

    cursors = [TraceCursor(tuple(trace)) for trace in latency_traces]
    current = initial.copy()
    trajectory = [(0, game.evaluate(current)[0])]
    now = updates = 0
    while True:
        agent = updates % game.n_agents
        ready = now+cursors[agent].next()
        if ready > horizon:
            break
        candidate = game.greedy_candidate(current, agent)
        current = game.mix_agent(current, candidate, agent, eta)
        now = ready
        updates += 1
        trajectory.append((now, game.evaluate(current)[0]))
    area, regret = integrate_regret(trajectory, horizon, optimum)
    return {"area": area, "regret": regret, "final_return": game.evaluate(current)[0]}


def simulate_barrier_batch(
    game: FiniteTeamGame,
    initial: Array,
    latency_traces: tuple[tuple[int, ...], ...],
    horizon: int,
    optimum: float,
    eta: float,
) -> dict[str, object]:
    """Wait for all fresh proposals, then apply their agent rows together."""

    cursors = [TraceCursor(tuple(trace)) for trace in latency_traces]
    current = initial.copy()
    trajectory = [(0, game.evaluate(current)[0])]
    now = 0
    while True:
        duration = max(cursor.next() for cursor in cursors)
        ready = now+duration
        if ready > horizon:
            break
        reference = current.copy()
        updated = current.copy()
        for agent in range(game.n_agents):
            candidate = game.greedy_candidate(reference, agent)
            updated[agent] = (1.0-eta)*reference[agent] + eta*candidate[agent]
        current = updated
        now = ready
        trajectory.append((now, game.evaluate(current)[0]))
    area, regret = integrate_regret(trajectory, horizon, optimum)
    return {"area": area, "regret": regret, "final_return": game.evaluate(current)[0]}


def nonmyopic_beam_schedule(
    game: FiniteTeamGame,
    initial: Array,
    events: list[Completion],
    horizon: int,
    optimum: float,
    eta_grid: Iterable[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    beam_width: int = 64,
) -> dict[str, object]:
    """Search feasible event actions; use accumulated reward for pruning."""

    choices = tuple(float(x) for x in eta_grid)
    if not choices or min(choices) < 0 or max(choices) > 1 or beam_width < 1:
        raise ValueError("invalid beam configuration")
    value_cache: dict[bytes, float] = {}
    candidate_cache: dict[tuple[int, bytes], Array] = {}

    def value(policy: Array) -> float:
        key = policy.tobytes()
        if key not in value_cache:
            value_cache[key] = game.evaluate(policy)[0]
        return value_cache[key]

    def candidate(reference: Array, agent: int) -> Array:
        key = (agent, reference.tobytes())
        if key not in candidate_cache:
            candidate_cache[key] = game.greedy_candidate(reference, agent)
        return candidate_cache[key]

    beam = [BeamState(
        initial.copy(),
        tuple(initial.copy() for _ in range(game.n_agents)),
        tuple(0 for _ in range(game.n_agents)),
        0.0,
        0,
        0,
        0,
        tuple(),
    )]
    for event_index, item in enumerate(events, start=1):
        deduplicated: dict[tuple[bytes, ...], tuple[float, float, BeamState]] = {}
        for state in beam:
            j0 = value(state.current)
            area = state.area+(item.time-state.last_time)*j0
            proposal = candidate(state.references[item.agent], item.agent)
            for eta in choices:
                updated = game.mix_agent(state.current, proposal, item.agent, eta)
                j1 = value(updated)
                references = list(state.references)
                references[item.agent] = updated.copy()
                births = list(state.birth_events)
                births[item.agent] = event_index
                child = BeamState(
                    updated,
                    tuple(references),
                    tuple(births),
                    area,
                    item.time,
                    state.accepted+int(eta > 0),
                    state.harmful+int(eta > 0 and j1 < j0-1e-12),
                    state.actions+(eta,),
                )
                optimistic_score = area+(horizon-item.time)*j1
                state_key = (updated.tobytes(), *(ref.tobytes() for ref in references))
                record = (optimistic_score, j1, child)
                previous = deduplicated.get(state_key)
                if previous is None or (record[0], record[1]) > (previous[0], previous[1]):
                    deduplicated[state_key] = record
        expanded = list(deduplicated.values())
        expanded.sort(key=lambda item: (item[0], item[1]), reverse=True)
        beam = [item[2] for item in expanded[:beam_width]]
    finals: list[tuple[float, float, BeamState]] = []
    for state in beam:
        final_value = value(state.current)
        area = state.area+(horizon-state.last_time)*final_value
        finals.append((area, final_value, state))
    area, final_value, best = max(finals, key=lambda x: (x[0], x[1]))
    regret = max(horizon*optimum-area, 0.0)
    return {
        "area": float(area),
        "regret": float(regret),
        "final_return": float(final_value),
        "accepted": best.accepted,
        "harmful": best.harmful,
        "actions": best.actions,
        "beam_width": beam_width,
        "status": "feasible_nonmyopic_lower_bound_not_oracle_ceiling",
    }


def declared_initial_policies() -> tuple[Array, ...]:
    raw = (
        (0.35, 0.55, 0.65, 0.45, 0.50, 0.60),
        (0.60, 0.40, 0.45, 0.65, 0.55, 0.35),
        (0.25, 0.75, 0.70, 0.30, 0.40, 0.60),
        (0.48, 0.52, 0.52, 0.48, 0.42, 0.58),
    )
    return tuple(np.asarray(values, dtype=float).reshape(3, 2) for values in raw)


def declared_latency_families() -> dict[str, tuple[tuple[int, ...], ...]]:
    return {
        "persistent_heterogeneity": ((1,), (3,), (7,)),
        "bursty_slow_agent": ((1, 1, 2), (2, 4, 2), (8, 2, 8, 3)),
        "rotating_straggler": ((1, 1, 6, 6), (3, 2, 3, 2), (6, 6, 1, 1)),
    }


def scan() -> dict[str, object]:
    """Run the complete declared feasibility grid."""

    eta_grid = tuple(float(x) for x in np.linspace(0.0, 1.0, 11))
    age_powers = (0.5, 1.0, 2.0)
    tv_thresholds = (0.0, 0.025, 0.05, 0.1, 0.2, 0.4, 1.0)
    horizon = 24
    initials = declared_initial_policies()
    rows: list[dict[str, object]] = []
    for gamma, focus, coupling, latency_item, initial_item in product(
        (0.6, 0.85),
        (0.65, 0.9),
        (0.55, 0.75, 1.0),
        declared_latency_families().items(),
        enumerate(initials),
    ):
        latency_name, latency_traces = latency_item
        initial_index, initial = initial_item
        game = make_role_switch_game(gamma, focus, coupling)
        optimum, _ = deterministic_optimum(game)
        events = completion_schedule(latency_traces, horizon)
        candidates: list[tuple[str, dict[str, object], dict[str, float]]] = []
        for eta in eta_grid:
            run = simulate_event_rule(
                game, initial, events, horizon, optimum, eta=eta
            )
            candidates.append(("fixed_eta", run, {"eta": eta}))
            for power in age_powers:
                run = simulate_event_rule(
                    game, initial, events, horizon, optimum,
                    eta=eta, age_power=power,
                )
                candidates.append(("fixed_age_decay", run, {"eta": eta, "power": power}))
            for threshold in tv_thresholds:
                run = simulate_event_rule(
                    game, initial, events, horizon, optimum,
                    eta=eta, tv_threshold=threshold,
                )
                candidates.append(("fixed_tv_gate", run, {"eta": eta, "threshold": threshold}))
            serial = simulate_fresh_serial(
                game, initial, latency_traces, horizon, optimum, eta
            )
            candidates.append(("fresh_serial", serial, {"eta": eta}))
            barrier = simulate_barrier_batch(
                game, initial, latency_traces, horizon, optimum, eta
            )
            candidates.append(("barrier_batch", barrier, {"eta": eta}))
        baseline_name, baseline, baseline_params = min(
            candidates, key=lambda item: (float(item[1]["regret"]), -float(item[1]["final_return"]))
        )
        dynamic = nonmyopic_beam_schedule(
            game, initial, events, horizon, optimum
        )
        baseline_regret = float(baseline["regret"])
        reduction = (
            (baseline_regret-float(dynamic["regret"]))/baseline_regret
            if baseline_regret > 1e-12 else 0.0
        )
        rows.append({
            "gamma": gamma,
            "transition_focus": focus,
            "coupling": coupling,
            "latency_family": latency_name,
            "initial_index": initial_index,
            "events": len(events),
            "initial_return": game.evaluate(initial)[0],
            "optimum_return": optimum,
            "strong_baseline_name": baseline_name,
            "strong_baseline_params": baseline_params,
            "strong_baseline_regret": baseline_regret,
            "strong_baseline_final_return": baseline["final_return"],
            "dynamic_regret": dynamic["regret"],
            "dynamic_final_return": dynamic["final_return"],
            "dynamic_accepted": dynamic["accepted"],
            "dynamic_harmful": dynamic["harmful"],
            "dynamic_regret_reduction": reduction,
        })
    reductions = np.asarray([row["dynamic_regret_reduction"] for row in rows], dtype=float)
    result = {
        "kind": "outcome_free_exact_cpu_problem_value_screen",
        "cells": len(rows),
        "horizon": horizon,
        "static_baseline_candidates_per_cell": len(candidates),
        "dynamic_scope": "Feasible finite-width beam lower bound, not an oracle ceiling or executable controller.",
        "median_dynamic_regret_reduction": float(np.median(reductions)),
        "fraction_cells_regret_reduction_ge_0_05": float(np.mean(reductions >= 0.05)),
        "maximum_dynamic_regret_reduction": float(np.max(reductions)),
        "minimum_dynamic_regret_reduction": float(np.min(reductions)),
        "gate_median_reduction_ge_0_10": bool(np.median(reductions) >= 0.10),
        "gate_fraction_cells_ge_0_05_ge_0_60": bool(np.mean(reductions >= 0.05) >= 0.60),
        "rows": rows,
    }
    result["authorized_next_step"] = bool(
        result["gate_median_reduction_ge_0_10"]
        and result["gate_fraction_cells_ge_0_05_ge_0_60"]
    )
    return result


def beam_saturation_scan(
    baseline_result: Path,
    widths: Iterable[int] = (128, 256),
    output: Path | None = None,
) -> dict[str, object]:
    """Re-evaluate only the feasible dynamic search at larger beam widths.

    The original strong baselines, cells, horizon, metrics and gates are read
    from the immutable first-pass result.  This diagnostic can determine
    whether a failed value gate was merely caused by beam-search truncation;
    it cannot alter or erase the first-pass decision.
    """

    baseline = json.loads(baseline_result.read_text(encoding="utf-8"))
    if baseline.get("kind") != "outcome_free_exact_cpu_problem_value_screen":
        raise ValueError("unexpected baseline result kind")
    if int(baseline.get("cells", -1)) != len(baseline.get("rows", [])):
        raise ValueError("baseline row count mismatch")
    horizon = int(baseline["horizon"])
    initials = declared_initial_policies()
    latency_families = declared_latency_families()
    summaries: list[dict[str, object]] = []
    for width in tuple(int(w) for w in widths):
        if width < 1:
            raise ValueError("beam widths must be positive")
        reductions: list[float] = []
        dynamic_regrets: list[float] = []
        for row in baseline["rows"]:
            game = make_role_switch_game(
                float(row["gamma"]),
                float(row["transition_focus"]),
                float(row["coupling"]),
            )
            initial = initials[int(row["initial_index"])]
            optimum, _ = deterministic_optimum(game)
            events = completion_schedule(
                latency_families[str(row["latency_family"])], horizon
            )
            dynamic = nonmyopic_beam_schedule(
                game, initial, events, horizon, optimum, beam_width=width
            )
            baseline_regret = float(row["strong_baseline_regret"])
            dynamic_regret = float(dynamic["regret"])
            reduction = (
                (baseline_regret-dynamic_regret)/baseline_regret
                if baseline_regret > 1e-12 else 0.0
            )
            reductions.append(reduction)
            dynamic_regrets.append(dynamic_regret)
        values = np.asarray(reductions, dtype=float)
        summaries.append({
            "beam_width": width,
            "median_dynamic_regret_reduction": float(np.median(values)),
            "fraction_cells_regret_reduction_ge_0_05": float(np.mean(values >= 0.05)),
            "maximum_dynamic_regret_reduction": float(np.max(values)),
            "minimum_dynamic_regret_reduction": float(np.min(values)),
            "gate_median_reduction_ge_0_10": bool(np.median(values) >= 0.10),
            "gate_fraction_cells_ge_0_05_ge_0_60": bool(np.mean(values >= 0.05) >= 0.60),
            "dynamic_regrets": dynamic_regrets,
        })
    result = {
        "kind": "beam_saturation_audit_preserving_first_pass_failure",
        "baseline_result": str(baseline_result),
        "cells": len(baseline["rows"]),
        "horizon": horizon,
        "first_pass_median": baseline["median_dynamic_regret_reduction"],
        "first_pass_fraction_ge_0_05": baseline["fraction_cells_regret_reduction_ge_0_05"],
        "first_pass_authorized_next_step": baseline["authorized_next_step"],
        "summaries": summaries,
        "decision_rule": "Both unchanged gates must pass at a saturated width; the first-pass failure remains recorded.",
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def main(output: Path | None = None) -> dict[str, object]:
    result = scan()
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, sort_keys=True))

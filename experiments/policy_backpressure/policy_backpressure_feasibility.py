"""Exact development audit for asynchronous cooperative policy improvement.

This module is deliberately small and model based.  It does not create a new
experiment identifier, use frozen data, or claim benchmark efficacy.  Its two
purposes are to (i) check a stale unilateral-policy performance lower bound and
(ii) measure whether an exact dynamic admission oracle has any headroom over a
per-scenario best fixed trust radius and a fresh serial learner.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import heapq
import json
from pathlib import Path
from typing import Iterable

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class FiniteTeamGame:
    """Finite discounted common-payoff Markov game with binary actions."""

    transition: Array  # [state, joint-action-index, next-state]
    reward: Array  # [state, joint-action-index]
    gamma: float
    initial: Array
    n_agents: int = 2

    def __post_init__(self) -> None:
        s = self.reward.shape[0]
        a = 2 ** self.n_agents
        if self.transition.shape != (s, a, s):
            raise ValueError("transition has the wrong shape")
        if self.reward.shape != (s, a):
            raise ValueError("reward has the wrong shape")
        if self.initial.shape != (s,) or not np.isclose(self.initial.sum(), 1):
            raise ValueError("invalid initial distribution")
        if not 0 < self.gamma < 1:
            raise ValueError("gamma must be in (0,1)")
        if np.min(self.reward) < 0:
            raise ValueError("the audit assumes nonnegative bounded rewards")
        if not np.allclose(self.transition.sum(axis=-1), 1):
            raise ValueError("transition rows must sum to one")

    @property
    def n_states(self) -> int:
        return self.reward.shape[0]

    @property
    def reward_max(self) -> float:
        return float(np.max(self.reward))

    def joint_probabilities(self, policy: Array) -> Array:
        self._check_policy(policy)
        ans = np.empty((self.n_states, 2 ** self.n_agents), dtype=float)
        for s in range(self.n_states):
            for idx, action in enumerate(product((0, 1), repeat=self.n_agents)):
                prob = 1.0
                for i, ai in enumerate(action):
                    p = policy[i, s]
                    prob *= p if ai else 1.0 - p
                ans[s, idx] = prob
        return ans

    def evaluate(self, policy: Array) -> tuple[float, Array, Array, Array]:
        """Return J, V, Q and normalized discounted occupancy."""
        joint = self.joint_probabilities(policy)
        p_pi = np.einsum("sa,san->sn", joint, self.transition)
        r_pi = np.einsum("sa,sa->s", joint, self.reward)
        resolvent = np.linalg.inv(np.eye(self.n_states) - self.gamma * p_pi)
        value = resolvent @ r_pi
        q = self.reward + self.gamma * np.einsum(
            "san,n->sa", self.transition, value
        )
        occupancy = (1.0 - self.gamma) * (self.initial @ resolvent)
        return float(self.initial @ value), value, q, occupancy

    def local_direction_surrogate(
        self, reference: Array, current: Array, candidate: Array, agent: int
    ) -> float:
        """Stale normalized-occupancy surrogate for candidate-current."""
        self._check_policy(reference)
        self._check_policy(current)
        self._check_policy(candidate)
        _, _, q, occupancy = self.evaluate(reference)
        total = 0.0
        others = [j for j in range(self.n_agents) if j != agent]
        for s in range(self.n_states):
            for other_action in product((0, 1), repeat=len(others)):
                prob_other = 1.0
                base_action = [0] * self.n_agents
                for j, aj in zip(others, other_action):
                    base_action[j] = aj
                    p = reference[j, s]
                    prob_other *= p if aj else 1.0 - p
                signed_q = 0.0
                for ai in (0, 1):
                    base_action[agent] = ai
                    idx = sum(a << (self.n_agents - 1 - k)
                              for k, a in enumerate(base_action))
                    new_prob = candidate[agent, s] if ai else 1-candidate[agent, s]
                    old_prob = current[agent, s] if ai else 1-current[agent, s]
                    signed_q += (new_prob-old_prob) * q[s, idx]
                total += occupancy[s] * prob_other * signed_q
        return float(total)

    def greedy_candidate(self, reference: Array, agent: int) -> Array:
        """Exact stale best-response direction, holding other agents fixed."""
        _, _, q, occupancy = self.evaluate(reference)
        candidate = reference.copy()
        others = [j for j in range(self.n_agents) if j != agent]
        for s in range(self.n_states):
            action_values = np.zeros(2)
            for ai in (0, 1):
                for other_action in product((0, 1), repeat=len(others)):
                    prob_other = 1.0
                    action = [0] * self.n_agents
                    action[agent] = ai
                    for j, aj in zip(others, other_action):
                        action[j] = aj
                        p = reference[j, s]
                        prob_other *= p if aj else 1.0-p
                    idx = sum(a << (self.n_agents - 1 - k)
                              for k, a in enumerate(action))
                    action_values[ai] += prob_other * q[s, idx]
            if occupancy[s] > 0:
                candidate[agent, s] = float(action_values[1] >= action_values[0])
        return candidate

    def mix_agent(self, current: Array, candidate: Array, agent: int, eta: float) -> Array:
        if not 0 <= eta <= 1:
            raise ValueError("eta must be in [0,1]")
        ans = current.copy()
        ans[agent] = (1.0-eta)*current[agent] + eta*candidate[agent]
        return ans

    def performance_lower_bound(
        self, reference: Array, current: Array, candidate: Array, agent: int, eta: float
    ) -> dict[str, float]:
        """A conservative stale-update lower bound with explicit constants.

        The proof uses the performance-difference identity, total-variation
        perturbation of discounted occupancy and a telescoping bound between
        the reference and current joint policies.  Constants are not optimized.
        """
        proposed = self.mix_agent(current, candidate, agent, eta)
        j0 = self.evaluate(current)[0]
        j1 = self.evaluate(proposed)[0]
        d_i = eta * agent_tv_max(current, candidate, agent)
        delta = joint_tv_max(self, current, reference)
        surrogate = eta * self.local_direction_surrogate(
            reference, current, candidate, agent
        )
        rmax = self.reward_max
        one_minus = 1.0-self.gamma
        stale_constant = 4.0*rmax*(1.0+self.gamma)/(one_minus**3)
        trust_constant = 8.0*self.gamma*rmax/(one_minus**3)
        lower = (surrogate/one_minus
                 - stale_constant*d_i*delta
                 - trust_constant*d_i*d_i)
        return {
            "actual_improvement": j1-j0,
            "lower_bound": lower,
            "surrogate": surrogate,
            "agent_tv": d_i,
            "reference_current_tv": delta,
            "stale_penalty": stale_constant*d_i*delta,
            "trust_penalty": trust_constant*d_i*d_i,
        }

    def _check_policy(self, policy: Array) -> None:
        if policy.shape != (self.n_agents, self.n_states):
            raise ValueError("policy has the wrong shape")
        if np.min(policy) < 0 or np.max(policy) > 1:
            raise ValueError("policy probabilities must lie in [0,1]")


def make_interference_game(gamma: float, transition_focus: float) -> FiniteTeamGame:
    """Two-state team game: state selects coordination versus anti-coordination."""
    if not 0.5 < transition_focus < 1:
        raise ValueError("transition_focus must be in (0.5,1)")
    transition = np.empty((2, 4, 2), dtype=float)
    reward = np.empty((2, 4), dtype=float)
    for s in (0, 1):
        for idx, (a0, a1) in enumerate(product((0, 1), repeat=2)):
            parity = a0 ^ a1
            reward[s, idx] = float(parity == s)
            transition[s, idx, parity] = transition_focus
            transition[s, idx, 1-parity] = 1.0-transition_focus
    return FiniteTeamGame(
        transition=transition,
        reward=reward,
        gamma=gamma,
        initial=np.array([0.5, 0.5], dtype=float),
    )


def agent_tv_max(left: Array, right: Array, agent: int) -> float:
    return float(np.max(np.abs(left[agent]-right[agent])))


def joint_tv_max(game: FiniteTeamGame, left: Array, right: Array) -> float:
    p = game.joint_probabilities(left)
    q = game.joint_probabilities(right)
    return float(np.max(0.5*np.sum(np.abs(p-q), axis=1)))


def policy_path_length(game: FiniteTeamGame, history: list[Array], birth: int) -> float:
    return float(sum(
        joint_tv_max(game, history[k], history[k+1])
        for k in range(birth, len(history)-1)
    ))


def wallclock_mean_return(trajectory: list[tuple[int, float]], horizon: int) -> float:
    """Piecewise-constant return averaged over the declared wall-clock horizon."""
    if horizon <= 0 or not trajectory or trajectory[0][0] != 0:
        raise ValueError("invalid trajectory or horizon")
    area = 0.0
    last_time, last_value = trajectory[0]
    for time, value in trajectory[1:]:
        if time < last_time or time > horizon:
            raise ValueError("trajectory times are not ordered within the horizon")
        area += (time-last_time)*last_value
        last_time, last_value = time, value
    area += (horizon-last_time)*last_value
    return area/horizon


@dataclass(order=True)
class PendingProposal:
    ready_time: int
    sequence: int
    agent: int
    birth_version: int
    reference: Array


@dataclass
class BeamState:
    current: Array
    references: tuple[Array, ...]
    area: float
    last_time: int
    harmful: int
    accepted: int


def simulate_async(
    game: FiniteTeamGame,
    initial_policy: Array,
    latencies: tuple[int, ...],
    wall_horizon: int,
    mode: str,
    fixed_eta: float | None = None,
    eta_grid: Iterable[float] = tuple(np.linspace(0, 1, 21)),
) -> dict[str, object]:
    """Run parallel proposal generation under a declared admission rule."""
    if mode not in {"oracle", "fixed", "age", "path_bound"}:
        raise ValueError("unknown mode")
    if mode in {"fixed", "age"} and fixed_eta is None:
        raise ValueError("fixed_eta required")
    if len(latencies) != game.n_agents or min(latencies) < 1:
        raise ValueError("invalid latencies")
    current = initial_policy.copy()
    history = [current.copy()]
    events: list[PendingProposal] = []
    sequence = 0
    for i, latency in enumerate(latencies):
        heapq.heappush(events, PendingProposal(latency, sequence, i, 0, current.copy()))
        sequence += 1
    accepted = 0
    harmful = 0
    proposals = 0
    path_ratios = []
    trajectory = [(0, game.evaluate(current)[0])]
    while events and events[0].ready_time <= wall_horizon:
        item = heapq.heappop(events)
        proposals += 1
        candidate = game.greedy_candidate(item.reference, item.agent)
        age = len(history)-1-item.birth_version
        path = policy_path_length(game, history, item.birth_version)
        direct = joint_tv_max(game, current, item.reference)
        if path > 0:
            path_ratios.append(direct/path)
        if mode == "oracle":
            choices = []
            j0 = game.evaluate(current)[0]
            for eta in eta_grid:
                trial = game.mix_agent(current, candidate, item.agent, float(eta))
                choices.append((game.evaluate(trial)[0]-j0, -float(eta), float(eta)))
            eta = max(choices)[2]
        elif mode == "fixed":
            eta = float(fixed_eta)
        elif mode == "age":
            eta = float(fixed_eta)/(1.0+age)
        else:
            d = agent_tv_max(current, candidate, item.agent)
            g = game.local_direction_surrogate(item.reference, current, candidate, item.agent)
            om = 1.0-game.gamma
            cs = 4.0*game.reward_max*(1.0+game.gamma)/(om**3)
            cq = 8.0*game.gamma*game.reward_max/(om**3)
            linear = g/om-cs*d*path
            eta = float(np.clip(linear/(2*cq*d*d), 0, 1)) if d > 0 else 0.0
        j0 = game.evaluate(current)[0]
        updated = game.mix_agent(current, candidate, item.agent, eta)
        j1 = game.evaluate(updated)[0]
        if eta > 0:
            accepted += 1
            harmful += int(j1 < j0-1e-12)
        current = updated
        history.append(current.copy())
        trajectory.append((item.ready_time, j1))
        heapq.heappush(
            events,
            PendingProposal(
                item.ready_time+latencies[item.agent], sequence, item.agent,
                len(history)-1, current.copy()
            ),
        )
        sequence += 1
    result = {
        "final_return": game.evaluate(current)[0],
        "initial_return": game.evaluate(initial_policy)[0],
        "proposals": proposals,
        "accepted": accepted,
        "harmful": harmful,
        "max_direct_over_path": max(path_ratios, default=0.0),
        "trajectory": trajectory,
    }
    result["wallclock_mean_return"] = wallclock_mean_return(trajectory, wall_horizon)
    return result


def simulate_fresh_serial(
    game: FiniteTeamGame,
    initial_policy: Array,
    latencies: tuple[int, ...],
    wall_horizon: int,
    eta: float,
) -> dict[str, object]:
    current = initial_policy.copy()
    now = 0
    updates = 0
    trajectory = [(0, game.evaluate(current)[0])]
    while True:
        agent = updates % game.n_agents
        ready = now+latencies[agent]
        if ready > wall_horizon:
            break
        candidate = game.greedy_candidate(current, agent)
        current = game.mix_agent(current, candidate, agent, eta)
        now = ready
        updates += 1
        trajectory.append((now, game.evaluate(current)[0]))
    result = {
        "final_return": game.evaluate(current)[0],
        "initial_return": game.evaluate(initial_policy)[0],
        "updates": updates,
        "trajectory": trajectory,
    }
    result["wallclock_mean_return"] = wallclock_mean_return(trajectory, wall_horizon)
    return result


def lookahead_beam_oracle(
    game: FiniteTeamGame,
    initial_policy: Array,
    latencies: tuple[int, ...],
    wall_horizon: int,
    eta_grid: Iterable[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    beam_width: int = 64,
) -> dict[str, object]:
    """Non-myopic feasible schedule search; a lower bound, not an oracle ceiling."""
    if beam_width < 1 or len(latencies) != game.n_agents:
        raise ValueError("invalid beam configuration")
    schedule = sorted(
        (time, agent)
        for agent, latency in enumerate(latencies)
        for time in range(latency, wall_horizon+1, latency)
    )
    refs = tuple(initial_policy.copy() for _ in range(game.n_agents))
    beam = [BeamState(initial_policy.copy(), refs, 0.0, 0, 0, 0)]
    choices = tuple(float(x) for x in eta_grid)
    for time, agent in schedule:
        expanded: list[tuple[float, BeamState]] = []
        for state in beam:
            j0 = game.evaluate(state.current)[0]
            area = state.area+(time-state.last_time)*j0
            candidate = game.greedy_candidate(state.references[agent], agent)
            for eta in choices:
                updated = game.mix_agent(state.current, candidate, agent, eta)
                j1 = game.evaluate(updated)[0]
                new_refs = list(state.references)
                new_refs[agent] = updated.copy()
                new_state = BeamState(
                    updated,
                    tuple(new_refs),
                    area,
                    time,
                    state.harmful+int(eta > 0 and j1 < j0-1e-12),
                    state.accepted+int(eta > 0),
                )
                # Feasible continuation score used only for beam pruning.
                score = area+(wall_horizon-time)*j1
                expanded.append((score, new_state))
        expanded.sort(key=lambda x: x[0], reverse=True)
        beam = [x[1] for x in expanded[:beam_width]]
    finals = []
    for state in beam:
        j = game.evaluate(state.current)[0]
        area = state.area+(wall_horizon-state.last_time)*j
        finals.append((area, j, -state.harmful, state))
    area, final_return, _, best = max(finals, key=lambda x: (x[0], x[1], x[2]))
    return {
        "wallclock_mean_return": area/wall_horizon,
        "final_return": final_return,
        "accepted": best.accepted,
        "harmful": best.harmful,
        "beam_width": beam_width,
        "eta_choices": len(choices),
        "status": "feasible_nonmyopic_lower_bound_not_oracle_ceiling",
    }


def bound_audit() -> dict[str, object]:
    checks = 0
    minimum_slack = float("inf")
    maximum_path_ratio = 0.0
    for gamma, focus in product((0.35, 0.6, 0.85), (0.6, 0.8, 0.95)):
        game = make_interference_game(gamma, focus)
        levels = (0.15, 0.4, 0.65, 0.9)
        policies = [np.array(x, dtype=float).reshape(2, 2)
                    for x in product(levels, repeat=4)]
        for reference, current in zip(policies[::17], policies[5::17]):
            for agent, eta in product((0, 1), (0.1, 0.35, 0.7, 1.0)):
                candidate = game.greedy_candidate(reference, agent)
                record = game.performance_lower_bound(
                    reference, current, candidate, agent, eta
                )
                slack = record["actual_improvement"]-record["lower_bound"]
                if slack < -1e-10:
                    raise AssertionError(f"performance bound violated: {record}")
                minimum_slack = min(minimum_slack, slack)
                checks += 1
            path = [reference, 0.5*(reference+current), current]
            direct = joint_tv_max(game, current, reference)
            plen = policy_path_length(game, path, 0)
            if direct > plen+1e-12:
                raise AssertionError("policy path did not upper bound direct TV")
            maximum_path_ratio = max(maximum_path_ratio, direct/plen if plen else 0.0)
    return {
        "bound_checks": checks,
        "minimum_actual_minus_lower_bound": minimum_slack,
        "maximum_direct_tv_over_path_tv": maximum_path_ratio,
    }


def oracle_headroom_scan() -> dict[str, object]:
    eta_grid = tuple(float(x) for x in np.linspace(0.0, 1.0, 21))
    initials = [np.array(x, dtype=float).reshape(2, 2) for x in (
        (0.2, 0.8, 0.8, 0.2),
        (0.35, 0.65, 0.65, 0.35),
        (0.2, 0.2, 0.8, 0.8),
        (0.8, 0.8, 0.2, 0.2),
        (0.45, 0.55, 0.55, 0.45),
        (0.1, 0.9, 0.55, 0.45),
    )]
    rows = []
    for gamma, focus, latency, horizon, initial in product(
        (0.6, 0.85), (0.65, 0.9), ((1, 3), (1, 6), (2, 7)), (18, 30), initials
    ):
        game = make_interference_game(gamma, focus)
        fixed_runs = [simulate_async(
            game, initial, latency, horizon, "fixed", eta, eta_grid
        ) for eta in eta_grid]
        best_fixed = max(float(r["final_return"]) for r in fixed_runs)
        best_fixed_mean = max(float(r["wallclock_mean_return"]) for r in fixed_runs)
        serial_runs = [simulate_fresh_serial(
            game, initial, latency, horizon, eta
        ) for eta in eta_grid]
        best_serial = max(float(r["final_return"]) for r in serial_runs)
        best_serial_mean = max(float(r["wallclock_mean_return"]) for r in serial_runs)
        strong = max(best_fixed, best_serial)
        strong_mean = max(best_fixed_mean, best_serial_mean)
        oracle = simulate_async(game, initial, latency, horizon, "oracle", eta_grid=eta_grid)
        lookahead = lookahead_beam_oracle(game, initial, latency, horizon)
        path_bound = simulate_async(game, initial, latency, horizon, "path_bound")
        rows.append({
            "gamma": gamma,
            "focus": focus,
            "latencies": latency,
            "horizon": horizon,
            "initial_return": game.evaluate(initial)[0],
            "best_fixed": best_fixed,
            "best_fresh_serial": best_serial,
            "strong_baseline": strong,
            "strong_wallclock_mean_return": strong_mean,
            "oracle": oracle["final_return"],
            "oracle_wallclock_mean_return": oracle["wallclock_mean_return"],
            "lookahead_wallclock_mean_return": lookahead["wallclock_mean_return"],
            "path_bound": path_bound["final_return"],
            "oracle_relative_headroom": (float(oracle["final_return"])-strong)
                                         / max(abs(strong), 1e-12),
            "oracle_wallclock_relative_headroom": (
                float(lookahead["wallclock_mean_return"])-strong_mean
            )/max(abs(strong_mean), 1e-12),
            "oracle_harmful_updates": oracle["harmful"],
            "lookahead_harmful_updates": lookahead["harmful"],
            "path_bound_acceptance": path_bound["accepted"],
            "path_bound_harmful_updates": path_bound["harmful"],
        })
    endpoint_headroom = np.array([r["oracle_relative_headroom"] for r in rows])
    headroom = np.array([r["oracle_wallclock_relative_headroom"] for r in rows])
    return {
        "kind": "development_oracle_feasibility_not_scientific_evidence",
        "oracle_scope": "Lookahead beam is a feasible dynamic lower bound, not an oracle ceiling.",
        "cells": len(rows),
        "gate_oracle_median_relative_headroom_ge_0_10": bool(np.median(headroom) >= 0.10),
        "gate_oracle_fraction_cells_ge_0_05_ge_0_60": bool(np.mean(headroom >= 0.05) >= 0.60),
        "median_oracle_relative_headroom": float(np.median(headroom)),
        "fraction_oracle_headroom_ge_0_05": float(np.mean(headroom >= 0.05)),
        "maximum_oracle_relative_headroom": float(np.max(headroom)),
        "median_oracle_endpoint_relative_headroom": float(np.median(endpoint_headroom)),
        "maximum_oracle_endpoint_relative_headroom": float(np.max(endpoint_headroom)),
        "path_bound_active_fraction": float(np.mean([
            r["path_bound_acceptance"] > 0 for r in rows
        ])),
        "rows": rows,
    }


def audit(output: Path | None = None) -> dict[str, object]:
    result = {
        "bound": bound_audit(),
        "headroom": oracle_headroom_scan(),
        "decision_rule": (
            "Both oracle gates must pass before an implementable sampled controller or "
            "standard MARL benchmark is authorized."
        ),
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2, sort_keys=True))

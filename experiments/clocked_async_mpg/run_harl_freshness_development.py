"""CPU development runner for commit-barrier LSFF on HARL's MPE interface.

The runner is intentionally development-only.  It uses paired, fully charged
birth trajectories, optionally buys a paired current-policy measurement while
parameter commits are frozen, and applies the closed-form fused owner update.
"""

from __future__ import annotations

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import heapq
import json
import math
from pathlib import Path
import random
import subprocess
import sys
from typing import Any

import numpy as np

from .freshness_sensing import (
    choose_budgeted_freshness_refresh,
    fuse_gradient_estimates,
    optimal_fusion_certificate,
)
from .harl_packet_overlay import (
    SingleFlightRegistry,
    diagonal_gaussian_mean_kl,
    teammate_tv_drift_upper,
)
from .run_harl_layer0_development_comparison import (
    _collect_on_reused_environment,
    _evaluate_on_reused_environment,
    _policy_gradient_with_frozen_baseline,
    _service_duration,
)
from .run_harl_layer0_packet_smoke import (
    _add_flat_step,
    _discounted_returns,
    _distribution_parameters,
    _flat_parameters,
)


MODES = (
    "lsff",
    "lsff_transition",
    "never_refresh",
    "always_refresh",
    "periodic_phase_0",
    "periodic_phase_1",
    "periodic_phase_2",
    "periodic_phase_3",
)
PROFILE_NAMES = ("balanced", "heterogeneous")


@dataclass
class _FreshPacket:
    ticket: int
    agent_id: int
    birth_event: int
    completion_time: float
    birth_policies: list[Any]
    reference_observations: list[np.ndarray]
    birth_step: np.ndarray
    birth_variance: float
    charged_environment_steps: int
    charged_actor_transitions: int


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checked_harl_root(harl_root: str, expected_commit: str) -> Path:
    root = Path(harl_root).resolve()
    actual = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual != expected_commit:
        raise RuntimeError(f"unexpected HARL commit {actual}")
    return root


def _clip(vector: np.ndarray, maximum_norm: float) -> np.ndarray:
    result = np.asarray(vector, dtype=float).copy()
    norm = float(np.linalg.norm(result))
    if norm > maximum_norm:
        result *= maximum_norm / norm
    return result


def _paired_step(
    *,
    environment: Any,
    policies: list[Any],
    target_agent: int,
    seeds: tuple[int, int],
    episode_length: int,
    gamma: float,
    learning_rate: float,
    maximum_step_norm: float,
    frozen_baseline: np.ndarray,
) -> tuple[np.ndarray, float, list[np.ndarray], int]:
    trajectories = [
        _collect_on_reused_environment(
            environment=environment,
            policies=policies,
            target_agent=target_agent,
            seed=seed,
            episode_length=episode_length,
        )
        for seed in seeds
    ]
    steps = []
    for trajectory in trajectories:
        gradient = _policy_gradient_with_frozen_baseline(
            policies[target_agent],
            trajectory,
            gamma=gamma,
            frozen_baseline=frozen_baseline,
        )
        steps.append(_clip(learning_rate * gradient, maximum_step_norm))
    mean_step = 0.5 * (steps[0] + steps[1])
    variance = 0.25 * float(np.sum((steps[0] - steps[1]) ** 2))
    environment_steps = sum(trajectory.steps for trajectory in trajectories)
    return mean_step, variance, trajectories[1].observations, environment_steps


def _periodic_phase(mode: str) -> int | None:
    if mode.startswith("periodic_phase_"):
        return int(mode.rsplit("_", 1)[1])
    return None


def _service_bases(profile: str, num_agents: int) -> tuple[float, ...]:
    if num_agents <= 0 or profile not in PROFILE_NAMES:
        raise ValueError("invalid service profile or agent count")
    if profile == "balanced":
        return tuple(1.0 for _ in range(num_agents))
    return tuple(float(value) for value in np.geomspace(1.0, 4.0, num_agents))


def _run_one(specification: dict[str, Any]) -> dict[str, Any]:
    harl_root = _checked_harl_root(
        specification["harl_root"], specification["harl_commit"]
    )
    if str(harl_root) not in sys.path:
        sys.path.insert(0, str(harl_root))
    import torch
    import yaml
    from harl.envs.pettingzoo_mpe.pettingzoo_mpe_env import PettingZooMPEEnv
    from harl.models.policy_models.stochastic_policy import StochasticPolicy

    seed = int(specification["seed"])
    mode = str(specification["mode"])
    profile = str(specification["profile"])
    if mode not in MODES or profile not in PROFILE_NAMES:
        raise ValueError("unknown mode or service profile")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    environment = PettingZooMPEEnv(
        {
            "scenario": "simple_spread_v2",
            "N": specification["num_agents"],
            "continuous_actions": True,
            "max_cycles": specification["episode_length"],
        }
    )
    num_agents = environment.n_agents
    if num_agents != specification["num_agents"]:
        raise RuntimeError("environment returned the wrong number of agents")
    with (harl_root / "harl" / "configs" / "algos_cfgs" / "haa2c.yaml").open(
        "r", encoding="utf-8"
    ) as handle:
        configuration = yaml.safe_load(handle)
    policy_args: dict[str, Any] = {}
    for section in ("model", "algo"):
        policy_args.update(configuration[section])
    policy_args["hidden_sizes"] = [32, 32]
    policies = [
        StochasticPolicy(
            policy_args,
            environment.observation_space[index],
            environment.action_space[index],
        )
        for index in range(num_agents)
    ]
    evaluation_seeds = [seed + 900_000 + index for index in range(8)]
    initial_return = _evaluate_on_reused_environment(
        environment=environment,
        policies=policies,
        seeds=evaluation_seeds,
        episode_length=specification["episode_length"],
    )
    baseline_returns = []
    for index in range(specification["baseline_episodes"]):
        trajectory = _collect_on_reused_environment(
            environment=environment,
            policies=policies,
            target_agent=0,
            seed=seed + 500_000 + index,
            episode_length=specification["episode_length"],
        )
        baseline_returns.append(
            _discounted_returns(trajectory.target_rewards, specification["gamma"])
        )
    frozen_baseline = np.mean(np.asarray(baseline_returns), axis=0)

    service_bases = _service_bases(profile, num_agents)
    mean_service = float(np.mean(service_bases))
    registry = SingleFlightRegistry(num_agents)
    resource_debts = [np.zeros(2, dtype=float) for _ in range(num_agents)]
    fresh_variances: list[float | None] = [None] * num_agents
    per_agent_completions = [0] * num_agents
    queue: list[tuple[float, int, _FreshPacket]] = []
    ticket = 0
    event = 0
    wall_time = 0.0
    baseline_environment_steps = (
        specification["baseline_episodes"] * specification["episode_length"]
    )
    charged_environment_steps = baseline_environment_steps
    charged_actor_transitions = baseline_environment_steps * num_agents
    birth_actor_cost = 2 * specification["episode_length"] * num_agents
    refresh_count = 0
    applied_packets = 0
    fused_weight_sum = 0.0
    risk_sum = 0.0
    positive_drift = 0
    discrepancy_sum = 0.0
    discrepancy_covered = 0
    maximum_self_fresh_error = 0.0
    maximum_transition_overshoot = 0
    birth_variance_upper_sum = 0.0
    bias_square_sum = 0.0
    refresh_value_sum = 0.0
    selected_bias_square_sum = 0.0
    selected_birth_variance_upper_sum = 0.0
    per_agent_refresh_counts = [0] * num_agents

    def can_launch() -> bool:
        return (
            charged_actor_transitions + birth_actor_cost
            <= specification["actor_transition_budget"]
        )

    def launch(agent_id: int) -> None:
        nonlocal ticket, charged_environment_steps, charged_actor_transitions
        birth_policies = [copy.deepcopy(policy) for policy in policies]
        birth_step, birth_variance, observations, environment_steps = _paired_step(
            environment=environment,
            policies=birth_policies,
            target_agent=agent_id,
            seeds=(seed + 100_000 + 2 * ticket, seed + 100_001 + 2 * ticket),
            episode_length=specification["episode_length"],
            gamma=specification["gamma"],
            learning_rate=specification["learning_rate"],
            maximum_step_norm=specification["maximum_step_norm"],
            frozen_baseline=frozen_baseline,
        )
        actor_cost = environment_steps * num_agents
        if charged_actor_transitions + actor_cost > specification["actor_transition_budget"]:
            raise RuntimeError("launch exceeded the hard actor-transition budget")
        registry.launch(
            agent_id,
            birth_event=event,
            charged_transitions=actor_cost,
        )
        duration = _service_duration(service_bases, agent_id, ticket)
        packet = _FreshPacket(
            ticket=ticket,
            agent_id=agent_id,
            birth_event=event,
            completion_time=wall_time + duration,
            birth_policies=birth_policies,
            reference_observations=observations,
            birth_step=birth_step,
            birth_variance=birth_variance,
            charged_environment_steps=environment_steps,
            charged_actor_transitions=actor_cost,
        )
        heapq.heappush(queue, (packet.completion_time, ticket, packet))
        charged_environment_steps += environment_steps
        charged_actor_transitions += actor_cost
        ticket += 1

    for agent_id in range(num_agents):
        if can_launch():
            launch(agent_id)

    while queue:
        completion_time, _, packet = heapq.heappop(queue)
        wall_time = max(wall_time, completion_time)
        event += 1
        registry.complete(packet.agent_id, completion_event=event)
        self_fresh_error = float(
            np.max(
                np.abs(
                    _flat_parameters(policies[packet.agent_id])
                    - _flat_parameters(packet.birth_policies[packet.agent_id])
                )
            )
        )
        maximum_self_fresh_error = max(maximum_self_fresh_error, self_fresh_error)
        if self_fresh_error > 1e-10:
            raise RuntimeError("single-flight owner freshness failed")

        teammate_kls = []
        for teammate in range(num_agents):
            if teammate == packet.agent_id:
                continue
            birth_mean, birth_log_std = _distribution_parameters(
                packet.birth_policies[teammate],
                packet.reference_observations[teammate],
            )
            current_mean, current_log_std = _distribution_parameters(
                policies[teammate],
                packet.reference_observations[teammate],
            )
            teammate_kls.append(
                diagonal_gaussian_mean_kl(
                    birth_mean,
                    birth_log_std,
                    current_mean,
                    current_log_std,
                )
            )
        tv_drift = teammate_tv_drift_upper(np.asarray(teammate_kls))
        positive_drift += int(tv_drift > 1e-12)
        bias_upper = specification["cross_step_lipschitz"] * tv_drift
        birth_variance_upper = max(
            specification["variance_floor"],
            specification["variance_inflation"] * packet.birth_variance,
        )
        predicted_fresh_variance = (
            birth_variance_upper
            if fresh_variances[packet.agent_id] is None
            else max(birth_variance_upper, fresh_variances[packet.agent_id])
        )
        certificate = optimal_fusion_certificate(
            birth_variance=birth_variance_upper,
            fresh_variance=predicted_fresh_variance,
            birth_bias_upper=bias_upper,
        )
        birth_variance_upper_sum += birth_variance_upper
        bias_square_sum += bias_upper * bias_upper
        refresh_value_sum += certificate.refresh_value
        refresh_duration = _service_duration(
            service_bases, packet.agent_id, packet.ticket
        )
        refresh_actor_cost = birth_actor_cost
        hard_feasible = (
            charged_actor_transitions + refresh_actor_cost
            <= specification["actor_transition_budget"]
        )
        should_refresh = False
        if mode in {"lsff", "lsff_transition"}:
            if mode == "lsff_transition":
                decision_debts = resource_debts[packet.agent_id][:1]
                decision_costs = np.asarray([1.0], dtype=float)
                decision_budgets = np.asarray(
                    [specification["refresh_fraction"]], dtype=float
                )
            else:
                decision_debts = resource_debts[packet.agent_id]
                decision_costs = np.asarray(
                    [1.0, refresh_duration / mean_service], dtype=float
                )
                decision_budgets = np.asarray(
                    [
                        specification["refresh_fraction"],
                        specification["refresh_fraction"],
                    ],
                    dtype=float,
                )
            decision = choose_budgeted_freshness_refresh(
                certificate,
                resource_debts=decision_debts,
                refresh_costs=decision_costs,
                average_budgets=decision_budgets,
                risk_tradeoff=specification["risk_tradeoff_normalized"]
                / specification["risk_normalizer"],
                hard_budget_feasible=hard_feasible,
            )
            if mode == "lsff_transition":
                resource_debts[packet.agent_id][0] = (
                    decision.resource_debts_after[0]
                )
            else:
                resource_debts[packet.agent_id] = np.asarray(
                    decision.resource_debts_after
                )
            should_refresh = decision.refresh
        elif mode == "always_refresh":
            should_refresh = hard_feasible
        elif mode == "never_refresh":
            should_refresh = False
        else:
            phase = _periodic_phase(mode)
            should_refresh = bool(
                hard_feasible
                and per_agent_completions[packet.agent_id]
                % specification["period"]
                == phase
            )

        update_step = packet.birth_step
        incurred_risk = certificate.no_refresh_mse_upper
        if should_refresh:
            fresh_step, fresh_variance, _, environment_steps = _paired_step(
                environment=environment,
                policies=policies,
                target_agent=packet.agent_id,
                seeds=(
                    seed + 300_000 + 2 * packet.ticket,
                    seed + 300_001 + 2 * packet.ticket,
                ),
                episode_length=specification["episode_length"],
                gamma=specification["gamma"],
                learning_rate=specification["learning_rate"],
                maximum_step_norm=specification["maximum_step_norm"],
                frozen_baseline=frozen_baseline,
            )
            actual_actor_cost = environment_steps * num_agents
            charged_environment_steps += environment_steps
            charged_actor_transitions += actual_actor_cost
            maximum_transition_overshoot = max(
                maximum_transition_overshoot,
                charged_actor_transitions - specification["actor_transition_budget"],
            )
            wall_time += refresh_duration
            update_step = fuse_gradient_estimates(
                packet.birth_step, fresh_step, certificate
            )
            discrepancy = float(np.sum((packet.birth_step - fresh_step) ** 2))
            discrepancy_sum += discrepancy
            discrepancy_covered += int(
                discrepancy
                <= certificate.no_refresh_mse_upper
                + max(specification["variance_floor"], fresh_variance)
            )
            observed_fresh_upper = max(
                specification["variance_floor"],
                specification["variance_inflation"] * fresh_variance,
            )
            previous_fresh_variance = fresh_variances[packet.agent_id]
            fresh_variances[packet.agent_id] = (
                observed_fresh_upper
                if previous_fresh_variance is None
                else specification["variance_ewma"] * previous_fresh_variance
                + (1.0 - specification["variance_ewma"])
                * observed_fresh_upper
            )
            refresh_count += 1
            per_agent_refresh_counts[packet.agent_id] += 1
            fused_weight_sum += certificate.fresh_weight
            selected_bias_square_sum += bias_upper * bias_upper
            selected_birth_variance_upper_sum += birth_variance_upper
            incurred_risk = certificate.refresh_mse_upper
        _add_flat_step(policies[packet.agent_id], update_step)
        risk_sum += incurred_risk
        applied_packets += 1
        per_agent_completions[packet.agent_id] += 1
        if can_launch():
            launch(packet.agent_id)

    if registry.active_agents or maximum_transition_overshoot > 0:
        raise RuntimeError("final registry or actor-transition budget invariant failed")
    final_return = _evaluate_on_reused_environment(
        environment=environment,
        policies=policies,
        seeds=evaluation_seeds,
        episode_length=specification["episode_length"],
    )
    environment.close()
    return {
        "seed": seed,
        "num_agents": num_agents,
        "mode": mode,
        "service_profile": profile,
        "initial_return": initial_return,
        "final_return": final_return,
        "return_change": final_return - initial_return,
        "applied_packets": applied_packets,
        "refresh_count": refresh_count,
        "refresh_fraction": refresh_count / max(applied_packets, 1),
        "mean_fresh_weight_when_refreshed": fused_weight_sum
        / max(refresh_count, 1),
        "mean_certified_step_mse": risk_sum / max(applied_packets, 1),
        "mean_refreshed_birth_fresh_discrepancy": discrepancy_sum
        / max(refresh_count, 1),
        "descriptive_discrepancy_coverage": discrepancy_covered
        / max(refresh_count, 1),
        "positive_teammate_drift_fraction": positive_drift
        / max(applied_packets, 1),
        "charged_environment_steps": charged_environment_steps,
        "charged_actor_transitions": charged_actor_transitions,
        "actor_transition_budget": specification["actor_transition_budget"],
        "budget_utilization": charged_actor_transitions
        / specification["actor_transition_budget"],
        "wall_clock_service_units": wall_time,
        "maximum_self_fresh_error": maximum_self_fresh_error,
        "maximum_transition_overshoot": maximum_transition_overshoot,
        "mean_birth_variance_upper": birth_variance_upper_sum
        / max(applied_packets, 1),
        "mean_bias_square_upper": bias_square_sum / max(applied_packets, 1),
        "mean_refresh_value": refresh_value_sum / max(applied_packets, 1),
        "mean_selected_bias_square_upper": selected_bias_square_sum
        / max(refresh_count, 1),
        "mean_selected_birth_variance_upper": selected_birth_variance_upper_sum
        / max(refresh_count, 1),
        "per_agent_refresh_counts": per_agent_refresh_counts,
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cells: dict[str, Any] = {}
    for profile in sorted({row["service_profile"] for row in rows}):
        selected = [row for row in rows if row["service_profile"] == profile]
        present_modes = sorted({row["mode"] for row in selected})
        modes: dict[str, Any] = {}
        for mode in present_modes:
            mode_rows = [row for row in selected if row["mode"] == mode]
            modes[mode] = {
                "mean_final_return": float(
                    np.mean([row["final_return"] for row in mode_rows])
                ),
                "mean_return_change": float(
                    np.mean([row["return_change"] for row in mode_rows])
                ),
                "mean_budget_utilization": float(
                    np.mean([row["budget_utilization"] for row in mode_rows])
                ),
                "mean_wall_clock_service_units": float(
                    np.mean([row["wall_clock_service_units"] for row in mode_rows])
                ),
                "mean_refresh_fraction": float(
                    np.mean([row["refresh_fraction"] for row in mode_rows])
                ),
                "mean_certified_step_mse": float(
                    np.mean([row["mean_certified_step_mse"] for row in mode_rows])
                ),
                "mean_birth_variance_upper": float(
                    np.mean([row["mean_birth_variance_upper"] for row in mode_rows])
                ),
                "mean_bias_square_upper": float(
                    np.mean([row["mean_bias_square_upper"] for row in mode_rows])
                ),
            }
        cell: dict[str, Any] = {"modes": modes}
        periodic_modes = [
            mode for mode in present_modes if mode.startswith("periodic_phase_")
        ]
        if periodic_modes:
            best_periodic = max(
                periodic_modes, key=lambda key: modes[key]["mean_final_return"]
            )
            cell["development_best_periodic_mode"] = best_periodic
        for controller in ("lsff", "lsff_transition"):
            if controller not in modes:
                continue
            for baseline in ("never_refresh", "always_refresh"):
                if baseline in modes:
                    cell[f"{controller}_minus_{baseline}_mean_return"] = (
                        modes[controller]["mean_final_return"]
                        - modes[baseline]["mean_final_return"]
                    )
            if periodic_modes:
                cell[f"{controller}_minus_best_periodic_mean_return"] = (
                    modes[controller]["mean_final_return"]
                    - modes[best_periodic]["mean_final_return"]
                )
        cells[profile] = cell
    return cells


def run(specifications: list[dict[str, Any]], workers: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_run_one, spec) for spec in specifications]
        for index, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            rows.append(row)
            print(
                f"[{index}/{len(futures)}] {row['service_profile']} {row['mode']} "
                f"seed={row['seed']} final={row['final_return']:.6f}",
                flush=True,
            )
    rows.sort(key=lambda row: (row["seed"], row["service_profile"], row["mode"]))
    return {
        "scope": "arrival-fresh commit-barrier MPE development only",
        "rows": rows,
        "cells": _summarize(rows),
        "gpu_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--harl-root", required=True)
    parser.add_argument(
        "--harl-commit", default="b1af98b0dbab72a2eee9d160751cd09aedbb8ce2"
    )
    parser.add_argument("--seed-start", type=int, default=4701)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--episode-length", type=int, default=15)
    parser.add_argument("--baseline-episodes", type=int, default=6)
    parser.add_argument("--actor-transition-budget", type=int, default=30000)
    parser.add_argument("--num-agents", type=int, default=3)
    parser.add_argument("--profiles", nargs="+", default=list(PROFILE_NAMES))
    parser.add_argument("--modes", nargs="+", default=list(MODES))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    _checked_harl_root(args.harl_root, args.harl_commit)
    common = {
        "harl_root": str(Path(args.harl_root).resolve()),
        "harl_commit": args.harl_commit,
        "episode_length": args.episode_length,
        "baseline_episodes": args.baseline_episodes,
        "actor_transition_budget": args.actor_transition_budget,
        "num_agents": args.num_agents,
        "gamma": 0.99,
        "learning_rate": 5e-4,
        "maximum_step_norm": 0.05,
        "risk_normalizer": 0.01,
        "variance_floor": 1e-8,
        "variance_inflation": 8.0,
        "variance_ewma": 0.9,
        "cross_step_lipschitz": 0.05,
        "refresh_fraction": 0.25,
        "risk_tradeoff_normalized": 4.0,
        "period": 4,
    }
    specifications = [
        {**common, "seed": seed, "profile": profile, "mode": mode}
        for seed in range(args.seed_start, args.seed_start + args.seeds)
        for profile in args.profiles
        for mode in args.modes
    ]
    payload = run(specifications, args.workers)
    payload["configuration"] = {
        key: value for key, value in common.items() if key != "harl_root"
    }
    payload["seeds"] = list(range(args.seed_start, args.seed_start + args.seeds))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "summary.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["cells"], indent=2, sort_keys=True))
    print(f"summary_sha256={_sha256(path)}")


if __name__ == "__main__":
    main()

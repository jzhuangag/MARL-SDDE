"""CPU development comparison for the Layer-0 clocked packet mechanism.

The output is architecture-development evidence only.  It uses common random
numbers and equal two-trajectory packet charges to compare the sample-split
strategic controller with two raw asynchronous baselines on HARL's continuous
MPE interface.  No threshold in this file authorizes a formal or GPU run.
"""

from __future__ import annotations

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
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

from .harl_packet_overlay import (
    SingleFlightRegistry,
    decide_harl_packet_scale,
    diagonal_gaussian_mean_kl,
    sample_split_directional_value,
)
from .run_harl_layer0_packet_smoke import (
    _Packet,
    _Trajectory,
    _add_flat_step,
    _distribution_parameters,
    _flat_parameters,
    _discounted_returns,
)


MODES = ("strategic_split", "raw_full_data", "raw_half_data")
SERVICE_PROFILES = {
    "balanced": (1.0, 1.0, 1.0),
    "heterogeneous": (1.0, 1.55, 4.0),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checked_harl_root(harl_root: str, expected_commit: str) -> Path:
    root = Path(harl_root).resolve()
    if not (root / "harl").is_dir():
        raise FileNotFoundError("HARL root does not contain the harl package")
    actual = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual != expected_commit:
        raise RuntimeError(f"unexpected HARL commit {actual}")
    return root


def _service_duration(
    service_bases: tuple[float, float, float], agent_id: int, ticket: int
) -> float:
    modulation = 1.0 + 0.04 * ((3 * ticket + agent_id) % 4)
    return float(service_bases[agent_id] * modulation)


def _policy_gradient_with_frozen_baseline(
    policy: Any,
    trajectory: Any,
    *,
    gamma: float,
    frozen_baseline: np.ndarray,
) -> np.ndarray:
    """MC ascent direction with a past-data, action-independent baseline."""

    import torch

    observations = torch.as_tensor(trajectory.target_observations, dtype=torch.float32)
    actions = torch.as_tensor(trajectory.target_actions, dtype=torch.float32)
    batch = observations.shape[0]
    rnn_states = torch.zeros((batch, 1, policy.hidden_sizes[-1]), dtype=torch.float32)
    masks = torch.ones((batch, 1), dtype=torch.float32)
    advantages = _discounted_returns(trajectory.target_rewards, gamma) - frozen_baseline[:batch]
    advantages_tensor = torch.as_tensor(advantages, dtype=torch.float32)
    action_log_probs, _, _ = policy.evaluate_actions(
        observations,
        rnn_states,
        actions,
        masks,
        None,
        None,
    )
    loss = -(action_log_probs.sum(dim=-1) * advantages_tensor).mean()
    gradients = torch.autograd.grad(loss, tuple(policy.parameters()))
    return np.concatenate(
        [(-gradient).detach().cpu().numpy().ravel() for gradient in gradients]
    ).astype(np.float64, copy=False)


def _collect_on_reused_environment(
    *,
    environment: Any,
    policies: list[Any],
    target_agent: int,
    seed: int,
    episode_length: int,
) -> _Trajectory:
    """Collect one trajectory without allocating another pygame surface."""

    import torch

    environment.seed(seed - 1)
    torch.manual_seed(seed)
    observations, _, _ = environment.reset()
    all_observations: list[list[np.ndarray]] = [[] for _ in policies]
    target_actions: list[np.ndarray] = []
    target_rewards: list[float] = []
    for _ in range(episode_length):
        actions: list[np.ndarray] = []
        for agent_id, policy in enumerate(policies):
            observation = np.asarray(observations[agent_id], dtype=np.float32)
            all_observations[agent_id].append(observation.copy())
            with torch.no_grad():
                action, _, _ = policy(
                    observation[None, :],
                    np.zeros((1, 1, policy.hidden_sizes[-1]), dtype=np.float32),
                    np.ones((1, 1), dtype=np.float32),
                    None,
                    False,
                )
            action_array = action.detach().cpu().numpy()[0]
            actions.append(action_array)
            if agent_id == target_agent:
                target_actions.append(action_array.copy())
        observations, _, rewards, dones, _, _ = environment.step(actions)
        target_rewards.append(float(rewards[target_agent][0]))
        if all(bool(value) for value in dones):
            break
    return _Trajectory(
        observations=[np.asarray(values, dtype=np.float32) for values in all_observations],
        target_observations=np.asarray(all_observations[target_agent], dtype=np.float32),
        target_actions=np.asarray(target_actions, dtype=np.float32),
        target_rewards=np.asarray(target_rewards, dtype=np.float64),
        steps=len(target_rewards),
    )


def _evaluate_on_reused_environment(
    *,
    environment: Any,
    policies: list[Any],
    seeds: list[int],
    episode_length: int,
) -> float:
    import torch

    returns: list[float] = []
    for seed in seeds:
        environment.seed(seed - 1)
        observations, _, _ = environment.reset()
        episode_return = 0.0
        for _ in range(episode_length):
            actions = []
            for agent_id, policy in enumerate(policies):
                observation = np.asarray(observations[agent_id], dtype=np.float32)
                with torch.no_grad():
                    action, _, _ = policy(
                        observation[None, :],
                        np.zeros((1, 1, policy.hidden_sizes[-1]), dtype=np.float32),
                        np.ones((1, 1), dtype=np.float32),
                        None,
                        True,
                    )
                actions.append(action.detach().cpu().numpy()[0])
            observations, _, rewards, dones, _, _ = environment.step(actions)
            episode_return += float(rewards[0][0])
            if all(bool(value) for value in dones):
                break
        returns.append(episode_return)
    return float(np.mean(returns))


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
    if mode not in MODES or profile not in SERVICE_PROFILES:
        raise ValueError("unknown development mode or service profile")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    environment = PettingZooMPEEnv(
        {
            "scenario": "simple_spread_v2",
            "continuous_actions": True,
            "max_cycles": specification["episode_length"],
        }
    )
    observation_spaces = environment.observation_space
    action_spaces = environment.action_space
    num_agents = environment.n_agents
    if num_agents != 3:
        raise RuntimeError("development comparison expects three agents")
    with (harl_root / "harl" / "configs" / "algos_cfgs" / "haa2c.yaml").open(
        "r", encoding="utf-8"
    ) as handle:
        configuration = yaml.safe_load(handle)
    policy_args: dict[str, Any] = {}
    for section in ("model", "algo"):
        policy_args.update(configuration[section])
    policy_args["hidden_sizes"] = [32, 32]
    policies = [
        StochasticPolicy(policy_args, observation_spaces[index], action_spaces[index])
        for index in range(num_agents)
    ]
    evaluation_seeds = [seed + 90_000 + index for index in range(5)]
    initial_return = _evaluate_on_reused_environment(
        environment=environment,
        policies=policies,
        seeds=evaluation_seeds,
        episode_length=specification["episode_length"],
    )

    # These past-policy episodes are collected before any learning action.  The
    # resulting time-indexed baseline is then frozen for the entire run, so it
    # is action-independent and predictable for every packet.  Its cost is
    # charged equally to every method.
    baseline_returns = []
    for baseline_index in range(specification["baseline_episodes"]):
        trajectory = _collect_on_reused_environment(
            environment=environment,
            policies=policies,
            target_agent=0,
            seed=seed + 50_000 + baseline_index,
            episode_length=specification["episode_length"],
        )
        baseline_returns.append(
            _discounted_returns(trajectory.target_rewards, specification["gamma"])
        )
    frozen_baseline = np.mean(np.asarray(baseline_returns), axis=0)

    registry = SingleFlightRegistry(num_agents)
    debts = [0.0] * num_agents
    service_bases = SERVICE_PROFILES[profile]
    queue: list[tuple[float, int, _Packet]] = []
    launched = 0
    event = 0
    wall_time = 0.0
    baseline_environment_steps = (
        specification["baseline_episodes"] * specification["episode_length"]
    )
    baseline_actor_transitions = baseline_environment_steps * num_agents
    charged_environment_steps = baseline_environment_steps
    charged_actor_transitions = baseline_actor_transitions
    positive_scales = 0
    positive_teammate_drift = 0
    maximum_self_fresh_error = 0.0
    scale_sum = 0.0
    intermediate_scales = 0
    maximum_debt = 0.0
    strategic_directional_sum = 0.0
    strategic_certificate_sum = 0.0
    preclip_step_norm_sum = 0.0
    clipped_packets = 0

    def launch(agent_id: int) -> None:
        nonlocal launched, charged_environment_steps, charged_actor_transitions
        nonlocal preclip_step_norm_sum, clipped_packets
        ticket = launched
        birth_policies = [copy.deepcopy(policy) for policy in policies]
        proposal = _collect_on_reused_environment(
            environment=environment,
            policies=birth_policies,
            target_agent=agent_id,
            seed=seed + 1000 + 2 * ticket,
            episode_length=specification["episode_length"],
        )
        validation = _collect_on_reused_environment(
            environment=environment,
            policies=birth_policies,
            target_agent=agent_id,
            seed=seed + 1001 + 2 * ticket,
            episode_length=specification["episode_length"],
        )
        proposal_gradient = _policy_gradient_with_frozen_baseline(
            birth_policies[agent_id],
            proposal,
            gamma=specification["gamma"],
            frozen_baseline=frozen_baseline,
        )
        validation_gradient = _policy_gradient_with_frozen_baseline(
            birth_policies[agent_id],
            validation,
            gamma=specification["gamma"],
            frozen_baseline=frozen_baseline,
        )
        if mode == "raw_full_data":
            update_gradient = 0.5 * (proposal_gradient + validation_gradient)
        else:
            update_gradient = proposal_gradient
        proposal_step = specification["learning_rate"] * update_gradient
        norm = float(np.linalg.norm(proposal_step))
        preclip_step_norm_sum += norm
        if norm > specification["maximum_step_norm"]:
            proposal_step *= specification["maximum_step_norm"] / norm
            clipped_packets += 1
        environment_steps = proposal.steps + validation.steps
        actor_transitions = environment_steps * num_agents
        registry.launch(
            agent_id,
            birth_event=event,
            charged_transitions=actor_transitions,
        )
        duration = _service_duration(service_bases, agent_id, ticket)
        packet = _Packet(
            ticket=ticket,
            agent_id=agent_id,
            birth_event=event,
            birth_time=wall_time,
            completion_time=wall_time + duration,
            charged_environment_steps=environment_steps,
            charged_actor_transitions=actor_transitions,
            birth_policies=birth_policies,
            reference_observations=validation.observations,
            proposal_step=proposal_step,
            validation_gradient=validation_gradient,
        )
        heapq.heappush(queue, (packet.completion_time, ticket, packet))
        charged_environment_steps += environment_steps
        charged_actor_transitions += actor_transitions
        launched += 1

    for agent_id in range(num_agents):
        launch(agent_id)

    while queue:
        completion_time, _, packet = heapq.heappop(queue)
        wall_time = completion_time
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
            raise RuntimeError("single-flight ownership invariant failed")
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
        if sum(teammate_kls) > 1e-15:
            positive_teammate_drift += 1
        if mode == "strategic_split":
            decision = decide_harl_packet_scale(
                proposal_step=packet.proposal_step,
                validation_gradient=packet.validation_gradient,
                teammate_mean_kls=np.asarray(teammate_kls),
                curvature_upper=specification["curvature_upper"],
                mixed_drift_coefficient=specification["mixed_drift_coefficient"],
                debt=debts[packet.agent_id],
                risk_budget=specification["risk_budget"],
                tradeoff=specification["tradeoff"],
                maximum_scale=1.0,
            )
            scale = decision.scale
            debts[packet.agent_id] = decision.debt_after
            intermediate_scales += int(0.0 < scale < 1.0)
            maximum_debt = max(maximum_debt, decision.debt_after)
            strategic_directional_sum += sample_split_directional_value(
                packet.proposal_step, packet.validation_gradient
            )
            strategic_certificate_sum += decision.certificate_penalty
        else:
            scale = 1.0
        _add_flat_step(policies[packet.agent_id], scale * packet.proposal_step)
        positive_scales += int(scale > 0.0)
        scale_sum += scale
        if launched < specification["packets"]:
            launch(packet.agent_id)

    if (
        registry.active_agents
        or registry.completed_transitions
        != charged_actor_transitions - baseline_actor_transitions
    ):
        raise RuntimeError("final transition-accounting invariant failed")
    final_return = _evaluate_on_reused_environment(
        environment=environment,
        policies=policies,
        seeds=evaluation_seeds,
        episode_length=specification["episode_length"],
    )
    environment.close()
    return {
        "seed": seed,
        "mode": mode,
        "service_profile": profile,
        "initial_return": initial_return,
        "final_return": final_return,
        "return_change": final_return - initial_return,
        "packets": specification["packets"],
        "baseline_episodes": specification["baseline_episodes"],
        "charged_environment_steps": charged_environment_steps,
        "charged_actor_transitions": charged_actor_transitions,
        "completed_actor_transitions": registry.completed_transitions
        + baseline_actor_transitions,
        "wall_clock_service_units": wall_time,
        "positive_scale_fraction": positive_scales / specification["packets"],
        "mean_scale": scale_sum / specification["packets"],
        "intermediate_scale_fraction": intermediate_scales
        / specification["packets"],
        "maximum_debt": maximum_debt,
        "mean_strategic_directional_value": strategic_directional_sum
        / specification["packets"],
        "mean_strategic_certificate_penalty": strategic_certificate_sum
        / specification["packets"],
        "mean_preclip_step_norm": preclip_step_norm_sum / specification["packets"],
        "clipped_packet_fraction": clipped_packets / specification["packets"],
        "positive_teammate_drift_fraction": positive_teammate_drift
        / specification["packets"],
        "maximum_self_fresh_error": maximum_self_fresh_error,
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def lower_quartile_mean(values: list[float]) -> float:
        count = max(1, math.ceil(0.25 * len(values)))
        return float(np.mean(np.sort(np.asarray(values, dtype=float))[:count]))

    cells: dict[str, Any] = {}
    for profile in SERVICE_PROFILES:
        selected = [row for row in rows if row["service_profile"] == profile]
        by_mode: dict[str, Any] = {}
        for mode in MODES:
            mode_rows = [row for row in selected if row["mode"] == mode]
            final_returns = [row["final_return"] for row in mode_rows]
            return_changes = [row["return_change"] for row in mode_rows]
            by_mode[mode] = {
                "mean_initial_return": float(
                    np.mean([row["initial_return"] for row in mode_rows])
                ),
                "mean_final_return": float(
                    np.mean(final_returns)
                ),
                "standard_deviation_final_return": float(
                    np.std(final_returns, ddof=1) if len(final_returns) > 1 else 0.0
                ),
                "lower_quartile_mean_final_return": lower_quartile_mean(
                    final_returns
                ),
                "mean_return_change": float(
                    np.mean(return_changes)
                ),
                "positive_return_change_fraction": float(
                    np.mean(np.asarray(return_changes) > 0.0)
                ),
                "mean_scale": float(np.mean([row["mean_scale"] for row in mode_rows])),
            }
        strategic = {row["seed"]: row for row in selected if row["mode"] == "strategic_split"}
        contrasts: dict[str, Any] = {}
        for baseline in ("raw_full_data", "raw_half_data"):
            baseline_rows = {row["seed"]: row for row in selected if row["mode"] == baseline}
            differences = np.asarray(
                [
                    strategic[seed]["final_return"] - baseline_rows[seed]["final_return"]
                    for seed in sorted(strategic)
                ],
                dtype=float,
            )
            contrasts[baseline] = {
                "mean_paired_final_return_difference": float(np.mean(differences)),
                "median_paired_final_return_difference": float(np.median(differences)),
                "strategic_strictly_better_fraction": float(np.mean(differences > 0.0)),
                "lower_quartile_mean_return_difference": (
                    by_mode["strategic_split"]["lower_quartile_mean_final_return"]
                    - by_mode[baseline]["lower_quartile_mean_final_return"]
                ),
                "relative_mean_shortfall": float(
                    max(0.0, -np.mean(differences))
                    / max(abs(by_mode[baseline]["mean_final_return"]), 1e-12)
                ),
            }
        cells[profile] = {"modes": by_mode, "strategic_contrasts": contrasts}
    return cells


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--harl-root", required=True)
    parser.add_argument(
        "--harl-commit", default="b1af98b0dbab72a2eee9d160751cd09aedbb8ce2"
    )
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--seed-start", type=int, default=2701)
    parser.add_argument("--packets", type=int, default=120)
    parser.add_argument("--episode-length", type=int, default=8)
    parser.add_argument("--baseline-episodes", type=int, default=8)
    parser.add_argument("--risk-budget", type=float, default=1e-4)
    parser.add_argument("--tradeoff", type=float, default=10.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    _checked_harl_root(args.harl_root, args.harl_commit)
    common = {
        "harl_root": str(Path(args.harl_root).resolve()),
        "harl_commit": args.harl_commit,
        "episode_length": args.episode_length,
        "packets": args.packets,
        "gamma": 0.99,
        "learning_rate": 5e-4,
        "maximum_step_norm": 0.05,
        "curvature_upper": 5.0,
        "mixed_drift_coefficient": 1.0,
        "risk_budget": args.risk_budget,
        "tradeoff": args.tradeoff,
        "baseline_episodes": args.baseline_episodes,
    }
    specifications = [
        {
            **common,
            "seed": args.seed_start + seed_offset,
            "mode": mode,
            "profile": profile,
        }
        for seed_offset in range(args.seeds)
        for profile in SERVICE_PROFILES
        for mode in MODES
    ]
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_run_one, specification) for specification in specifications]
        for index, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            rows.append(row)
            print(
                f"[{index}/{len(futures)}] {row['service_profile']} "
                f"{row['mode']} seed={row['seed']} final={row['final_return']:.6f}",
                flush=True,
            )
    rows.sort(key=lambda row: (row["seed"], row["service_profile"], row["mode"]))
    expected_charge = (args.packets * 2 + common["baseline_episodes"]) * args.episode_length * 3
    if any(
        row["charged_actor_transitions"] != expected_charge
        or row["completed_actor_transitions"] != expected_charge
        or row["maximum_self_fresh_error"] != 0.0
        for row in rows
    ):
        raise RuntimeError("comparison accounting or ownership check failed")
    summary = {
        "scope": "architecture-development evidence only",
        "harl_commit": args.harl_commit,
        "seeds": list(range(args.seed_start, args.seed_start + args.seeds)),
        "modes": list(MODES),
        "service_profiles": {key: list(value) for key, value in SERVICE_PROFILES.items()},
        "packets_per_run": args.packets,
        "episode_length": args.episode_length,
        "baseline_episodes_per_run": common["baseline_episodes"],
        "actor_transitions_per_run": expected_charge,
        "cells": _summarize(rows),
        "rows": rows,
        "gpu_or_formal_authorized": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "summary.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary["cells"], indent=2, sort_keys=True))
    print(f"output_sha256={_sha256(output)}")


if __name__ == "__main__":
    main()

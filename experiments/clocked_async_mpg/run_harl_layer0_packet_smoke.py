"""Run a non-scientific end-to-end packet smoke on HARL's MPE interface.

This script deliberately lives outside the pinned HARL checkout.  It uses the
upstream PettingZoo MPE wrapper and ``StochasticPolicy`` architecture, but it
implements the theorem-facing execution semantics explicitly:

* each agent owns a distinct policy block;
* at most one packet per agent is in flight;
* proposal and validation trajectories are independent and fully charged;
* a packet is self-fresh in its owner's block and may be stale only in the
  teammate blocks;
* the arrival-time update is the closed-form strategic-drift scale.

The run is an integration smoke, not benchmark evidence and not a formal seed.
"""

from __future__ import annotations

import argparse
import copy
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

from .harl_packet_overlay import (
    SingleFlightRegistry,
    decide_harl_packet_scale,
    diagonal_gaussian_mean_kl,
)


@dataclass
class _Trajectory:
    observations: list[np.ndarray]
    target_observations: np.ndarray
    target_actions: np.ndarray
    target_rewards: np.ndarray
    steps: int


@dataclass
class _Packet:
    ticket: int
    agent_id: int
    birth_event: int
    birth_time: float
    completion_time: float
    charged_environment_steps: int
    charged_actor_transitions: int
    birth_policies: list[Any]
    reference_observations: list[np.ndarray]
    proposal_step: np.ndarray
    validation_gradient: np.ndarray


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _discounted_returns(rewards: np.ndarray, gamma: float) -> np.ndarray:
    values = np.empty_like(rewards, dtype=np.float64)
    accumulator = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        accumulator = float(rewards[index]) + gamma * accumulator
        values[index] = accumulator
    return values


def _flat_parameters(policy: Any) -> np.ndarray:
    import torch

    with torch.no_grad():
        return np.concatenate(
            [parameter.detach().cpu().numpy().ravel() for parameter in policy.parameters()]
        ).astype(np.float64, copy=False)


def _add_flat_step(policy: Any, step: np.ndarray) -> None:
    import torch

    offset = 0
    with torch.no_grad():
        for parameter in policy.parameters():
            count = parameter.numel()
            value = torch.as_tensor(
                step[offset : offset + count].reshape(parameter.shape),
                dtype=parameter.dtype,
                device=parameter.device,
            )
            parameter.add_(value)
            offset += count
    if offset != step.size:
        raise RuntimeError("flat policy step has the wrong dimension")


def _policy_gradient(
    policy: Any,
    trajectory: _Trajectory,
    *,
    gamma: float,
) -> np.ndarray:
    """Return an ascent direction from a zero-baseline MC policy gradient."""

    import torch

    observations = torch.as_tensor(trajectory.target_observations, dtype=torch.float32)
    actions = torch.as_tensor(trajectory.target_actions, dtype=torch.float32)
    batch = observations.shape[0]
    rnn_states = torch.zeros((batch, 1, policy.hidden_sizes[-1]), dtype=torch.float32)
    masks = torch.ones((batch, 1), dtype=torch.float32)
    returns = _discounted_returns(trajectory.target_rewards, gamma)
    returns_tensor = torch.as_tensor(returns, dtype=torch.float32)
    action_log_probs, _, _ = policy.evaluate_actions(
        observations,
        rnn_states,
        actions,
        masks,
        None,
        None,
    )
    joint_coordinate_log_prob = action_log_probs.sum(dim=-1)
    loss = -(joint_coordinate_log_prob * returns_tensor).mean()
    gradients = torch.autograd.grad(loss, tuple(policy.parameters()))
    # Gradient descent on ``loss`` is gradient ascent on the MC return.
    return np.concatenate(
        [(-gradient).detach().cpu().numpy().ravel() for gradient in gradients]
    ).astype(np.float64, copy=False)


def _distribution_parameters(policy: Any, observations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import torch

    with torch.no_grad():
        tensor = torch.as_tensor(observations, dtype=torch.float32)
        features = policy.base(tensor)
        distribution = policy.act.action_out(features)
        mean = distribution.mean.detach().cpu().numpy().astype(np.float64)
        standard_deviation = (
            distribution.stddev.detach().cpu().numpy().astype(np.float64)
        )
    return mean, np.log(np.broadcast_to(standard_deviation, mean.shape))


def _collect_trajectory(
    *,
    environment_class: Any,
    policies: list[Any],
    target_agent: int,
    seed: int,
    episode_length: int,
) -> _Trajectory:
    import torch

    environment = environment_class(
        {
            "scenario": "simple_spread_v2",
            "continuous_actions": True,
            "max_cycles": episode_length,
        }
    )
    environment.seed(seed - 1)
    torch.manual_seed(seed)
    observations, _, _ = environment.reset()
    all_observations: list[list[np.ndarray]] = [[] for _ in policies]
    target_actions: list[np.ndarray] = []
    target_rewards: list[float] = []
    try:
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
    finally:
        environment.close()
    return _Trajectory(
        observations=[np.asarray(values, dtype=np.float32) for values in all_observations],
        target_observations=np.asarray(
            all_observations[target_agent], dtype=np.float32
        ),
        target_actions=np.asarray(target_actions, dtype=np.float32),
        target_rewards=np.asarray(target_rewards, dtype=np.float64),
        steps=len(target_rewards),
    )


def _service_duration(agent_id: int, launch_index: int) -> float:
    bases = (1.0, 1.55, 2.35)
    modulation = 1.0 + 0.04 * ((3 * launch_index + agent_id) % 4)
    return float(bases[agent_id] * modulation)


def _evaluate_joint_policy(
    *,
    environment_class: Any,
    policies: list[Any],
    seeds: list[int],
    episode_length: int,
) -> float:
    import torch

    returns: list[float] = []
    for seed in seeds:
        environment = environment_class(
            {
                "scenario": "simple_spread_v2",
                "continuous_actions": True,
                "max_cycles": episode_length,
            }
        )
        environment.seed(seed - 1)
        observations, _, _ = environment.reset()
        episode_return = 0.0
        try:
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
        finally:
            environment.close()
        returns.append(episode_return)
    return float(np.mean(returns))


def run(args: argparse.Namespace) -> dict[str, Any]:
    harl_root = Path(args.harl_root).resolve()
    if not (harl_root / "harl").is_dir():
        raise FileNotFoundError("--harl-root does not contain the HARL package")
    actual_harl_commit = subprocess.run(
        ["git", "-C", str(harl_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual_harl_commit != args.harl_commit:
        raise RuntimeError(
            "HARL checkout does not match --harl-commit: "
            f"{actual_harl_commit} != {args.harl_commit}"
        )
    sys.path.insert(0, str(harl_root))
    import torch
    import yaml
    import pettingzoo
    import supersuit
    from harl.envs.pettingzoo_mpe.pettingzoo_mpe_env import PettingZooMPEEnv
    from harl.models.policy_models.stochastic_policy import StochasticPolicy

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    environment = PettingZooMPEEnv(
        {
            "scenario": "simple_spread_v2",
            "continuous_actions": True,
            "max_cycles": args.episode_length,
        }
    )
    observation_spaces = environment.observation_space
    action_spaces = environment.action_space
    num_agents = environment.n_agents
    environment.close()
    if num_agents != 3:
        raise RuntimeError("Layer-0 smoke expects three simple-spread agents")

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
    initial_parameter_vectors = [_flat_parameters(policy) for policy in policies]
    evaluation_seeds = [args.seed + 90_000 + index for index in range(3)]
    initial_return = _evaluate_joint_policy(
        environment_class=PettingZooMPEEnv,
        policies=policies,
        seeds=evaluation_seeds,
        episode_length=args.episode_length,
    )

    registry = SingleFlightRegistry(num_agents)
    debts = [0.0] * num_agents
    queue: list[tuple[float, int, _Packet]] = []
    launched = 0
    completed = 0
    event = 0
    wall_time = 0.0
    charged_environment_steps = 0
    charged_actor_transitions = 0
    records: list[dict[str, Any]] = []

    def launch(agent_id: int) -> None:
        nonlocal launched, charged_environment_steps, charged_actor_transitions
        ticket = launched
        birth_policies = [copy.deepcopy(policy) for policy in policies]
        proposal = _collect_trajectory(
            environment_class=PettingZooMPEEnv,
            policies=birth_policies,
            target_agent=agent_id,
            seed=args.seed + 1000 + 2 * ticket,
            episode_length=args.episode_length,
        )
        validation = _collect_trajectory(
            environment_class=PettingZooMPEEnv,
            policies=birth_policies,
            target_agent=agent_id,
            seed=args.seed + 1001 + 2 * ticket,
            episode_length=args.episode_length,
        )
        proposal_gradient = _policy_gradient(
            birth_policies[agent_id], proposal, gamma=args.gamma
        )
        validation_gradient = _policy_gradient(
            birth_policies[agent_id], validation, gamma=args.gamma
        )
        proposal_step = args.learning_rate * proposal_gradient
        norm = float(np.linalg.norm(proposal_step))
        if norm > args.maximum_step_norm:
            proposal_step *= args.maximum_step_norm / norm
        environment_steps = proposal.steps + validation.steps
        actor_transitions = environment_steps * num_agents
        registry.launch(
            agent_id,
            birth_event=event,
            charged_transitions=actor_transitions,
        )
        duration = _service_duration(agent_id, ticket)
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

    for initial_agent in range(num_agents):
        launch(initial_agent)

    while queue:
        completion_time, _, packet = heapq.heappop(queue)
        wall_time = completion_time
        event += 1
        completion = registry.complete(packet.agent_id, completion_event=event)
        self_fresh_error = float(
            np.max(
                np.abs(
                    _flat_parameters(policies[packet.agent_id])
                    - _flat_parameters(packet.birth_policies[packet.agent_id])
                )
            )
        )
        if self_fresh_error > 1e-10:
            raise RuntimeError("single-flight owner block is not self-fresh")
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
        decision = decide_harl_packet_scale(
            proposal_step=packet.proposal_step,
            validation_gradient=packet.validation_gradient,
            teammate_mean_kls=np.asarray(teammate_kls),
            curvature_upper=args.curvature_upper,
            mixed_drift_coefficient=args.mixed_drift_coefficient,
            debt=debts[packet.agent_id],
            risk_budget=args.risk_budget,
            tradeoff=args.tradeoff,
            maximum_scale=1.0,
        )
        _add_flat_step(
            policies[packet.agent_id], decision.scale * packet.proposal_step
        )
        debts[packet.agent_id] = decision.debt_after
        records.append(
            {
                "ticket": packet.ticket,
                "agent_id": packet.agent_id,
                "birth_event": packet.birth_event,
                "completion_event": event,
                "event_delay": completion.event_delay,
                "birth_time": packet.birth_time,
                "completion_time": packet.completion_time,
                "charged_environment_steps": packet.charged_environment_steps,
                "charged_actor_transitions": packet.charged_actor_transitions,
                "self_fresh_error": self_fresh_error,
                "teammate_kl_sum": float(sum(teammate_kls)),
                "scale": decision.scale,
                "directional_gain": decision.predicted_gain
                / max(decision.scale, 1e-300),
                "certificate_penalty": decision.certificate_penalty,
                "debt_after": decision.debt_after,
            }
        )
        completed += 1
        if launched < args.packets:
            launch(packet.agent_id)

    if registry.active_agents:
        raise RuntimeError("packet queue ended with active owners")
    if registry.completed_transitions != charged_actor_transitions:
        raise RuntimeError("completed and launched transition charges disagree")
    final_return = _evaluate_joint_policy(
        environment_class=PettingZooMPEEnv,
        policies=policies,
        seeds=evaluation_seeds,
        episode_length=args.episode_length,
    )
    parameter_movements = [
        float(np.linalg.norm(_flat_parameters(policy) - initial))
        for policy, initial in zip(policies, initial_parameter_vectors)
    ]
    summary: dict[str, Any] = {
        "scope": "non-scientific Layer-0 integration smoke",
        "seed": args.seed,
        "harl_commit": actual_harl_commit,
        "runtime": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "torch": torch.__version__,
            "pettingzoo": getattr(pettingzoo, "__version__", "unknown"),
            "supersuit": getattr(supersuit, "__version__", "unknown"),
        },
        "scenario": "simple_spread_v2-continuous",
        "algorithmic_interface": "HARL StochasticPolicy plus owned packet overlay",
        "num_agents": num_agents,
        "packets_launched": launched,
        "packets_completed": completed,
        "proposal_trajectories_per_packet": 1,
        "validation_trajectories_per_packet": 1,
        "charged_environment_steps": charged_environment_steps,
        "charged_actor_transitions": charged_actor_transitions,
        "completed_actor_transitions": registry.completed_transitions,
        "wall_clock_service_units": wall_time,
        "maximum_self_fresh_error": max(record["self_fresh_error"] for record in records),
        "positive_teammate_drift_packets": sum(
            record["teammate_kl_sum"] > 1e-15 for record in records
        ),
        "positive_scale_packets": sum(record["scale"] > 0.0 for record in records),
        "scales": [record["scale"] for record in records],
        "parameter_movement_norms": parameter_movements,
        "initial_deterministic_return": initial_return,
        "final_deterministic_return": final_return,
        "return_is_scientific_evidence": False,
        "records": records,
    }
    if not all(
        math.isfinite(value)
        for value in (
            initial_return,
            final_return,
            wall_time,
            *parameter_movements,
        )
    ):
        raise RuntimeError("non-finite Layer-0 smoke output")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--harl-root", required=True)
    parser.add_argument(
        "--harl-commit", default="b1af98b0dbab72a2eee9d160751cd09aedbb8ce2"
    )
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--packets", type=int, default=12)
    parser.add_argument("--episode-length", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--maximum-step-norm", type=float, default=0.05)
    parser.add_argument("--curvature-upper", type=float, default=5.0)
    parser.add_argument("--mixed-drift-coefficient", type=float, default=1.0)
    parser.add_argument("--risk-budget", type=float, default=1e-4)
    parser.add_argument("--tradeoff", type=float, default=10.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.packets < 3 or args.episode_length <= 0:
        raise ValueError("packets and episode length are invalid")
    summary = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}, indent=2))
    print(f"output_sha256={_sha256(args.output)}")


if __name__ == "__main__":
    main()

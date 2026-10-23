"""Frozen standard-environment pilot for the Two Clocks phase claim.

The implementation uses distinct HARL actors and a single-flight packet per
actor.  Service times are deterministic and outcome-independent.  The
``frozen_barrier`` comparator keeps every actor busy, but all packets collected
within a barrier round use the same birth policy.  Thus all methods receive the
same packet opportunity set over a fixed service horizon; they differ in
adaptive query depth, not in uncharged data.

This neural experiment is an empirical extension.  It does not claim that the
unconstrained HARL network satisfies the finite-policy Lyapunov certificate.
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

from .run_harl_layer0_packet_smoke import (
    _add_flat_step,
    _flat_parameters,
)
from .run_two_clocks_hpc4_g0 import _wait_for_children_to_exit


METHODS = ("two_clocks_async", "delay_scaled_async", "frozen_barrier")
PROFILES = ("balanced", "heterogeneous")


@dataclass(frozen=True)
class _Trajectory:
    observations: np.ndarray
    actions: np.ndarray
    available_actions: np.ndarray | None
    rewards: np.ndarray
    episode_ends: np.ndarray
    environment_steps: int


@dataclass(frozen=True)
class _Packet:
    ticket: int
    owner: int
    birth_time: float
    completion_time: float
    birth_versions: tuple[int, ...]
    owner_birth_parameters: np.ndarray
    step: np.ndarray
    environment_steps: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checked_checkout(path: str, expected_commit: str, label: str) -> Path:
    root = Path(path).resolve()
    actual = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(root), "status", "--short"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual != expected_commit or dirty:
        raise RuntimeError(f"{label} checkout is not the frozen clean commit")
    return root


def packet_opportunities(service_times: list[float], horizon: float) -> int:
    """Number of complete single-flight packets available by ``horizon``."""

    if horizon <= 0.0 or not service_times or any(value <= 0.0 for value in service_times):
        raise ValueError("service times and horizon must be positive")
    return sum(int(math.floor((horizon + 1e-12) / value)) for value in service_times)


def barrier_update_count(service_times: list[float], horizon: float) -> int:
    round_length = max(service_times)
    rounds = int(math.floor((horizon + 1e-12) / round_length))
    return rounds * len(service_times)


def _evaluation_grid(horizon: float, fractions: list[float]) -> list[float]:
    values = [float(fraction) * horizon for fraction in fractions]
    if not values or values[0] != 0.0 or values[-1] != horizon:
        raise ValueError("evaluation fractions must include exact endpoints")
    if any(right <= left for left, right in zip(values, values[1:])):
        raise ValueError("evaluation fractions must be strictly increasing")
    return values


def _policy_args(harl_root: Path, hidden_sizes: list[int]) -> dict[str, Any]:
    import yaml

    path = harl_root / "harl" / "configs" / "algos_cfgs" / "haa2c.yaml"
    with path.open("r", encoding="utf-8") as handle:
        configuration = yaml.safe_load(handle)
    args: dict[str, Any] = {}
    for section in ("model", "algo"):
        args.update(configuration[section])
    args["hidden_sizes"] = list(hidden_sizes)
    args["use_naive_recurrent_policy"] = False
    args["use_recurrent_policy"] = False
    return args


def _make_environment(task: str, task_config: dict[str, Any]) -> Any:
    if task == "mamujoco_ant_4x2":
        from harl.envs.mamujoco.multiagent_mujoco.mujoco_multi import MujocoMulti

        return MujocoMulti(
            env_args={
                "scenario": "Ant-v2",
                "agent_conf": "4x2",
                "agent_obsk": 0,
                "episode_limit": int(task_config["episode_length"]),
            }
        )
    if task == "smacv2_terran_5v5":
        from harl.envs.smacv2.smacv2_env import SMACv2Env

        return SMACv2Env({"map_name": "terran_5_vs_5"})
    raise ValueError(f"unsupported frozen task {task}")


def _reset(environment: Any) -> tuple[list[np.ndarray], Any]:
    observations, _, available = environment.reset()
    return [np.asarray(value, dtype=np.float32) for value in observations], available


def _policy_action(
    policy: Any,
    observation: np.ndarray,
    available: np.ndarray | None,
    *,
    deterministic: bool,
) -> np.ndarray:
    import torch

    available_batch = None
    if available is not None:
        available_batch = np.asarray(available, dtype=np.float32)[None, :]
    with torch.no_grad():
        action, _, _ = policy(
            observation[None, :],
            np.zeros((1, 1, policy.hidden_sizes[-1]), dtype=np.float32),
            np.ones((1, 1), dtype=np.float32),
            available_batch,
            deterministic,
        )
    return action.detach().cpu().numpy()[0]


def _environment_actions(actions: list[np.ndarray], discrete: bool) -> Any:
    if discrete:
        return np.asarray(
            [[int(np.asarray(action).reshape(-1)[0])] for action in actions],
            dtype=np.int64,
        )
    return actions


def _collect_trajectory(
    *,
    environment: Any,
    policies: list[Any],
    owner: int,
    episode_length: int,
    discrete: bool,
) -> _Trajectory:
    observations, available = _reset(environment)
    owner_observations: list[np.ndarray] = []
    owner_actions: list[np.ndarray] = []
    owner_available: list[np.ndarray] = []
    owner_rewards: list[float] = []
    episode_ends: list[bool] = []
    for _ in range(episode_length):
        actions: list[np.ndarray] = []
        for agent, policy in enumerate(policies):
            available_agent = None if available is None else np.asarray(available[agent])
            action = _policy_action(
                policy,
                observations[agent],
                available_agent,
                deterministic=False,
            )
            actions.append(action)
            if agent == owner:
                owner_observations.append(observations[agent].copy())
                owner_actions.append(action.copy())
                if available_agent is not None:
                    owner_available.append(available_agent.astype(np.float32, copy=True))
        result = environment.step(_environment_actions(actions, discrete))
        observations = [np.asarray(value, dtype=np.float32) for value in result[0]]
        rewards = result[2]
        dones = result[3]
        available = result[5]
        owner_rewards.append(float(rewards[owner][0]))
        ended = all(bool(value) for value in dones)
        episode_ends.append(ended)
        if ended and len(owner_rewards) < episode_length:
            observations, available = _reset(environment)
    available_array = None
    if owner_available:
        available_array = np.asarray(owner_available, dtype=np.float32)
    action_dtype = np.int64 if discrete else np.float32
    return _Trajectory(
        observations=np.asarray(owner_observations, dtype=np.float32),
        actions=np.asarray(owner_actions, dtype=action_dtype),
        available_actions=available_array,
        rewards=np.asarray(owner_rewards, dtype=np.float64),
        episode_ends=np.asarray(episode_ends, dtype=bool),
        environment_steps=len(owner_rewards),
    )


def _segmented_discounted_returns(
    rewards: np.ndarray, episode_ends: np.ndarray, gamma: float
) -> np.ndarray:
    """Return-to-go without propagating value across environment resets."""

    if rewards.shape != episode_ends.shape:
        raise ValueError("rewards and episode boundaries must have equal shape")
    values = np.empty_like(rewards, dtype=np.float64)
    accumulator = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        if bool(episode_ends[index]):
            accumulator = 0.0
        accumulator = float(rewards[index]) + gamma * accumulator
        values[index] = accumulator
    return values


def _evaluate(
    *,
    environment: Any,
    policies: list[Any],
    episode_length: int,
    episodes: int,
    discrete: bool,
) -> tuple[float, int]:
    returns: list[float] = []
    steps = 0
    for _ in range(episodes):
        observations, available = _reset(environment)
        total = 0.0
        for _ in range(episode_length):
            actions: list[np.ndarray] = []
            for agent, policy in enumerate(policies):
                available_agent = None if available is None else np.asarray(available[agent])
                actions.append(
                    _policy_action(
                        policy,
                        observations[agent],
                        available_agent,
                        deterministic=True,
                    )
                )
            result = environment.step(_environment_actions(actions, discrete))
            observations = [np.asarray(value, dtype=np.float32) for value in result[0]]
            total += float(result[2][0][0])
            steps += 1
            if all(bool(value) for value in result[3]):
                break
            available = result[5]
        returns.append(total)
    return float(np.mean(returns)), steps


def _build_frozen_baseline(
    *,
    environment: Any,
    policies: list[Any],
    episode_length: int,
    episodes: int,
    gamma: float,
    discrete: bool,
) -> tuple[np.ndarray, int]:
    values: list[np.ndarray] = []
    charged_steps = 0
    for _ in range(episodes):
        trajectory = _collect_trajectory(
            environment=environment,
            policies=policies,
            owner=0,
            episode_length=episode_length,
            discrete=discrete,
        )
        values.append(
            _segmented_discounted_returns(
                trajectory.rewards, trajectory.episode_ends, gamma
            )
        )
        charged_steps += trajectory.environment_steps
    baseline = np.zeros(episode_length, dtype=np.float64)
    counts = np.zeros(episode_length, dtype=np.int64)
    for value in values:
        baseline[: len(value)] += value
        counts[: len(value)] += 1
    positive = counts > 0
    baseline[positive] /= counts[positive]
    return baseline, charged_steps


def _policy_gradient(
    policy: Any,
    trajectory: _Trajectory,
    *,
    gamma: float,
    frozen_baseline: np.ndarray,
) -> np.ndarray:
    import torch

    batch = trajectory.environment_steps
    observations = torch.as_tensor(
        trajectory.observations, dtype=torch.float32, device=policy.tpdv["device"]
    )
    actions = torch.as_tensor(trajectory.actions, device=policy.tpdv["device"])
    available = None
    if trajectory.available_actions is not None:
        available = torch.as_tensor(
            trajectory.available_actions,
            dtype=torch.float32,
            device=policy.tpdv["device"],
        )
    advantages = (
        _segmented_discounted_returns(
            trajectory.rewards, trajectory.episode_ends, gamma
        )
        - frozen_baseline[:batch]
    )
    advantage_tensor = torch.as_tensor(
        advantages, dtype=torch.float32, device=policy.tpdv["device"]
    )
    action_log_probabilities, _, _ = policy.evaluate_actions(
        observations,
        torch.zeros(
            (batch, 1, policy.hidden_sizes[-1]),
            dtype=torch.float32,
            device=policy.tpdv["device"],
        ),
        actions,
        torch.ones((batch, 1), dtype=torch.float32, device=policy.tpdv["device"]),
        available,
        None,
    )
    loss = -(action_log_probabilities.sum(dim=-1) * advantage_tensor).mean()
    gradients = torch.autograd.grad(loss, tuple(policy.parameters()))
    return np.concatenate(
        [(-gradient).detach().cpu().numpy().ravel() for gradient in gradients]
    ).astype(np.float64, copy=False)


def _bounded_step(
    gradient: np.ndarray, learning_rate: float, maximum_norm: float
) -> tuple[np.ndarray, bool]:
    step = learning_rate * np.asarray(gradient, dtype=np.float64)
    norm = float(np.linalg.norm(step))
    clipped = norm > maximum_norm
    if clipped:
        step *= maximum_norm / norm
    return step, clipped


def _curve_auc(curve: list[dict[str, float]], horizon: float) -> float:
    times = np.asarray([row["logical_time"] for row in curve], dtype=np.float64)
    values = np.asarray([row["mean_return"] for row in curve], dtype=np.float64)
    return float(np.trapz(values, times) / horizon)


def _make_policies(
    environment: Any,
    harl_root: Path,
    hidden_sizes: list[int],
    device: Any,
) -> list[Any]:
    from harl.models.policy_models.stochastic_policy import StochasticPolicy

    args = _policy_args(harl_root, hidden_sizes)
    return [
        StochasticPolicy(
            args,
            environment.observation_space[index],
            environment.action_space[index],
            device,
        )
        for index in range(environment.n_agents)
    ]


def _run_one(
    *,
    task: str,
    method: str,
    profile: str,
    seed: int,
    config: dict[str, Any],
    harl_root: Path,
    device: Any,
) -> dict[str, Any]:
    import psutil
    import torch

    if method not in METHODS or profile not in PROFILES:
        raise ValueError("unregistered method or service profile")
    task_config = config["tasks"][task]
    service_times = [float(value) for value in task_config["service_profiles"][profile]]
    horizon = float(task_config["logical_horizon"])
    episode_length = int(task_config["episode_length"])
    discrete = bool(task_config["discrete"])
    expected_agents = len(service_times)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    process = psutil.Process()
    inherited_children = {child.pid for child in process.children(recursive=True)}
    environment = _make_environment(task, task_config)
    environment.seed(seed + 700_000)
    try:
        policies = _make_policies(
            environment,
            harl_root,
            list(config["hidden_sizes"]),
            device,
        )
        if environment.n_agents != expected_agents:
            raise RuntimeError("task agent count differs from registered services")
        initial_parameters = [_flat_parameters(policy) for policy in policies]
        frozen_baseline, baseline_steps = _build_frozen_baseline(
            environment=environment,
            policies=policies,
            episode_length=episode_length,
            episodes=int(config["baseline_episodes"]),
            gamma=float(config["gamma"]),
            discrete=discrete,
        )
        versions = [0] * expected_agents
        charged_environment_steps = baseline_steps
        completed_packets = 0
        optimizer_updates = 0
        maximum_self_fresh_error = 0.0
        clipped_packets = 0
        event_delays: list[int] = []
        evaluation_steps = 0
        grid = _evaluation_grid(horizon, list(config["evaluation_fractions"]))
        initial_return, used = _evaluate(
            environment=environment,
            policies=policies,
            episode_length=episode_length,
            episodes=int(config["evaluation_episodes"]),
            discrete=discrete,
        )
        evaluation_steps += used
        curve: list[dict[str, float]] = [
            {
                "logical_time": 0.0,
                "mean_return": initial_return,
                "completed_packets": 0.0,
                "optimizer_updates": 0.0,
            }
        ]
        next_grid = 1

        def evaluate_through(logical_time: float) -> None:
            nonlocal next_grid, evaluation_steps
            while next_grid < len(grid) and logical_time + 1e-12 >= grid[next_grid]:
                mean_return, used_steps = _evaluate(
                    environment=environment,
                    policies=policies,
                    episode_length=episode_length,
                    episodes=int(config["evaluation_episodes"]),
                    discrete=discrete,
                )
                evaluation_steps += used_steps
                curve.append(
                    {
                        "logical_time": grid[next_grid],
                        "mean_return": mean_return,
                        "completed_packets": float(completed_packets),
                        "optimizer_updates": float(optimizer_updates),
                    }
                )
                next_grid += 1

        if method in {"two_clocks_async", "delay_scaled_async"}:
            queue: list[tuple[float, int, _Packet]] = []
            ticket = 0

            def launch(owner: int, birth_time: float) -> None:
                nonlocal ticket, charged_environment_steps
                completion = birth_time + service_times[owner]
                if completion > horizon + 1e-12:
                    return
                birth_policies = [copy.deepcopy(policy) for policy in policies]
                trajectory = _collect_trajectory(
                    environment=environment,
                    policies=birth_policies,
                    owner=owner,
                    episode_length=episode_length,
                    discrete=discrete,
                )
                gradient = _policy_gradient(
                    birth_policies[owner],
                    trajectory,
                    gamma=float(config["gamma"]),
                    frozen_baseline=frozen_baseline,
                )
                step, clipped = _bounded_step(
                    gradient,
                    float(task_config["learning_rate"]),
                    float(task_config["maximum_step_norm"]),
                )
                packet = _Packet(
                    ticket=ticket,
                    owner=owner,
                    birth_time=birth_time,
                    completion_time=completion,
                    birth_versions=tuple(versions),
                    owner_birth_parameters=_flat_parameters(birth_policies[owner]),
                    step=step,
                    environment_steps=trajectory.environment_steps,
                )
                heapq.heappush(queue, (completion, ticket, packet))
                charged_environment_steps += trajectory.environment_steps
                ticket += 1
                if clipped:
                    nonlocal_clipped[0] += 1

            nonlocal_clipped = [0]
            for owner in range(expected_agents):
                launch(owner, 0.0)
            while queue:
                current_time = queue[0][0]
                group: list[_Packet] = []
                while queue and abs(queue[0][0] - current_time) <= 1e-12:
                    group.append(heapq.heappop(queue)[2])
                group.sort(key=lambda packet: (packet.owner, packet.ticket))
                for packet in group:
                    self_error = float(
                        np.max(
                            np.abs(
                                _flat_parameters(policies[packet.owner])
                                - packet.owner_birth_parameters
                            )
                        )
                    )
                    maximum_self_fresh_error = max(maximum_self_fresh_error, self_error)
                    if self_error > 1e-10:
                        raise RuntimeError("single-flight owner ceased to be self-fresh")
                    event_delay = sum(
                        current - birth
                        for current, birth in zip(versions, packet.birth_versions)
                    )
                    event_delays.append(event_delay)
                    scale = 1.0
                    if method == "delay_scaled_async":
                        scale = 1.0 / (
                            1.0 + float(config["delay_scale_coefficient"]) * event_delay
                        )
                    _add_flat_step(policies[packet.owner], scale * packet.step)
                    versions[packet.owner] += 1
                    completed_packets += 1
                    optimizer_updates += 1
                    launch(packet.owner, current_time)
                evaluate_through(current_time)
            clipped_packets = nonlocal_clipped[0]
        else:
            round_length = max(service_times)
            round_start = 0.0
            while round_start + round_length <= horizon + 1e-12:
                birth_policies = [copy.deepcopy(policy) for policy in policies]
                owner_birth_parameters = [
                    _flat_parameters(policy) for policy in birth_policies
                ]
                owner_steps: list[list[np.ndarray]] = [[] for _ in policies]
                owner_environment_steps = [0] * expected_agents
                for owner in range(expected_agents):
                    packet_count = int(math.floor((round_length + 1e-12) / service_times[owner]))
                    for _ in range(packet_count):
                        trajectory = _collect_trajectory(
                            environment=environment,
                            policies=birth_policies,
                            owner=owner,
                            episode_length=episode_length,
                            discrete=discrete,
                        )
                        gradient = _policy_gradient(
                            birth_policies[owner],
                            trajectory,
                            gamma=float(config["gamma"]),
                            frozen_baseline=frozen_baseline,
                        )
                        step, clipped = _bounded_step(
                            gradient,
                            float(task_config["learning_rate"]),
                            float(task_config["maximum_step_norm"]),
                        )
                        owner_steps[owner].append(step)
                        owner_environment_steps[owner] += trajectory.environment_steps
                        clipped_packets += int(clipped)
                        completed_packets += 1
                for owner in range(expected_agents):
                    self_error = float(
                        np.max(
                            np.abs(
                                _flat_parameters(policies[owner])
                                - owner_birth_parameters[owner]
                            )
                        )
                    )
                    maximum_self_fresh_error = max(maximum_self_fresh_error, self_error)
                    if self_error > 1e-10:
                        raise RuntimeError("barrier birth policy changed within a round")
                    _add_flat_step(policies[owner], np.mean(owner_steps[owner], axis=0))
                    versions[owner] += 1
                    optimizer_updates += 1
                charged_environment_steps += sum(owner_environment_steps)
                round_start += round_length
                evaluate_through(round_start)

        if next_grid != len(grid):
            raise RuntimeError("training ended before every evaluation grid point")
        expected_packets = packet_opportunities(service_times, horizon)
        if completed_packets != expected_packets:
            raise RuntimeError("completed packet count differs from the service schedule")
        expected_updates = (
            expected_packets
            if method != "frozen_barrier"
            else barrier_update_count(service_times, horizon)
        )
        if optimizer_updates != expected_updates:
            raise RuntimeError("optimizer update count differs from the method contract")
        terminal_return = curve[-1]["mean_return"]
        parameter_movements = [
            float(np.linalg.norm(_flat_parameters(policy) - initial))
            for policy, initial in zip(policies, initial_parameters)
        ]
        row = {
            "task": task,
            "method": method,
            "service_profile": profile,
            "seed": seed,
            "agents": expected_agents,
            "logical_horizon": horizon,
            "completed_packets": completed_packets,
            "optimizer_updates": optimizer_updates,
            "charged_environment_steps": charged_environment_steps,
            "charged_actor_transitions": charged_environment_steps * expected_agents,
            "evaluation_environment_steps": evaluation_steps,
            "initial_return": initial_return,
            "terminal_return": terminal_return,
            "return_change": terminal_return - initial_return,
            "logical_time_auc": _curve_auc(curve, horizon),
            "maximum_self_fresh_error": maximum_self_fresh_error,
            "mean_event_delay": float(np.mean(event_delays)) if event_delays else 0.0,
            "maximum_event_delay": max(event_delays, default=0),
            "clipped_packet_fraction": clipped_packets / completed_packets,
            "parameter_movement_norms": parameter_movements,
            "curve": curve,
        }
        if not all(
            math.isfinite(float(row[key]))
            for key in (
                "initial_return",
                "terminal_return",
                "return_change",
                "logical_time_auc",
                "maximum_self_fresh_error",
                "mean_event_delay",
                "clipped_packet_fraction",
            )
        ):
            raise RuntimeError("non-finite pilot row")
        return row
    finally:
        environment.close()
        leaked = _wait_for_children_to_exit(process, inherited_children)
        if leaked:
            raise RuntimeError(f"environment teardown leaked descendants: {sorted(leaked)}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["experiment_id"] != "two_clocks_standard_pilot_v1":
        raise RuntimeError("unexpected pilot configuration")
    if args.task not in config["tasks"]:
        raise ValueError("task is not registered")
    harl_root = _checked_checkout(args.harl_root, config["harl_commit"], "HARL")
    smacv2_root = _checked_checkout(
        args.smacv2_root, config["smacv2_commit"], "SMACv2"
    )
    for root in (str(harl_root), str(smacv2_root)):
        if root not in sys.path:
            sys.path.insert(0, root)
    if not torch.cuda.is_available():
        raise RuntimeError("the frozen standard pilot requires CUDA")
    torch.set_num_threads(int(config["torch_threads"]))
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda:0")
    rows = []
    for seed in config["pilot_seeds"][args.task]:
        for profile in PROFILES:
            for method in METHODS:
                row = _run_one(
                    task=args.task,
                    method=method,
                    profile=profile,
                    seed=int(seed),
                    config=config,
                    harl_root=harl_root,
                    device=device,
                )
                rows.append(row)
                print(
                    f"task={args.task} seed={seed} profile={profile} method={method} "
                    f"terminal={row['terminal_return']:.6f}",
                    flush=True,
                )
    rows.sort(key=lambda row: (row["seed"], row["service_profile"], row["method"]))
    summary = {
        "experiment_id": config["experiment_id"],
        "scope": "fresh-seed standard neural pilot; not formal evidence",
        "task": args.task,
        "config_sha256": _sha256(args.config),
        "code_commit": args.code_commit,
        "harl_commit": config["harl_commit"],
        "smacv2_commit": config["smacv2_commit"],
        "methods": list(METHODS),
        "profiles": list(PROFILES),
        "seeds": list(config["pilot_seeds"][args.task]),
        "rows": rows,
        "formal_authorized": False,
    }
    expected_rows = len(config["pilot_seeds"][args.task]) * len(METHODS) * len(PROFILES)
    if len(rows) != expected_rows:
        raise RuntimeError("pilot output row count is incomplete")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--harl-root", required=True)
    parser.add_argument("--smacv2-root", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    summary = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"output_sha256={_sha256(args.output)}")


if __name__ == "__main__":
    main()

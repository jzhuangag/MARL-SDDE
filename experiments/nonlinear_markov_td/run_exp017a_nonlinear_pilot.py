"""Frozen EXP-017A standard-task nonlinear neural-TD GPU pilot runner.

The pilot evaluates policy prediction, not control.  The behavior policies are
fixed, every algorithm sees paired trajectory banks, and the online controller
cannot access the true correlation, source mask, held-out errors, or outcomes
from another policy.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from torch import nn

from exp017a_nonlinear_config import (
    B_CANDIDATES,
    BLOCK_SERVER_TICKS,
    BUDGETS,
    CERTIFICATE_ALPHA,
    CORRELATIONS,
    DELAY_TRACES,
    EVALUATION_TRANSITIONS,
    EXPERIMENT,
    FLOAT_BYTES,
    HIDDEN_WIDTH,
    LEARNING_RATE,
    MAXIMUM_AGENTS,
    MIXING_PROFILES,
    PILOT_SEEDS,
    POLICIES,
    Q_CANDIDATES,
    SERVER_OVERHEAD_BYTES,
    TASKS,
    TRAIN_BANK_LENGTH,
    delay_value,
    repository_root,
    sha256_json,
    trace_summary,
)


@dataclass(frozen=True)
class Action:
    q: int
    b: int


@dataclass
class ControllerState:
    collision_count: int = 0
    collision_trials: int = 0
    loss_ema: float = 1.0
    gradient_second_moment: float = 1.0
    progress_ema: float = 0.0


@dataclass(frozen=True)
class TransitionBank:
    states: np.ndarray
    following: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    task_name: str
    mixing_name: str
    seed: int


class ValueNetwork(nn.Module):
    def __init__(self, input_dimension: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dimension, HIDDEN_WIDTH),
            nn.ReLU(),
            nn.Linear(HIDDEN_WIDTH, HIDDEN_WIDTH),
            nn.ReLU(),
            nn.Linear(HIDDEN_WIDTH, 1),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.layers(states).squeeze(-1)


def task_seed_offset(task_name: str, mixing_name: str) -> int:
    digest = sha256_json({"task": task_name, "mixing": mixing_name})
    return int(digest[:8], 16) % 10_000_000


def normalize_states(task_name: str, states: np.ndarray) -> np.ndarray:
    if task_name == "cartpole":
        scale = np.asarray([2.4, 3.0, 0.2095, 3.5], dtype=np.float32)
    elif task_name == "acrobot":
        scale = np.asarray([1.0, 1.0, 1.0, 1.0, 4.0 * np.pi, 9.0 * np.pi], dtype=np.float32)
    else:
        raise ValueError(task_name)
    return np.clip(states.astype(np.float32) / scale, -5.0, 5.0)


def behavior_action(task_name: str, state: np.ndarray, uniform: float) -> int:
    """Frozen stochastic behavior policy; it never reads a learned model."""

    epsilon = 0.20
    if task_name == "cartpole":
        preferred = int(1.8 * float(state[2]) + 0.35 * float(state[3]) >= 0.0)
        if uniform < epsilon:
            return int((uniform / epsilon) * 2) % 2
        return preferred
    if task_name == "acrobot":
        energy_signal = (
            float(state[1])
            + 0.5 * (float(state[1]) * float(state[2]) + float(state[0]) * float(state[3]))
            + 0.08 * float(state[4])
            + 0.04 * float(state[5])
        )
        preferred = 2 if energy_signal >= 0.0 else 0
        if uniform < epsilon:
            return int((uniform / epsilon) * 3) % 3
        return preferred
    raise ValueError(task_name)


def generate_transition_bank(
    task_name: str,
    mixing_name: str,
    seed: int,
    length: int = TRAIN_BANK_LENGTH,
    source_count: int = MAXIMUM_AGENTS + 1,
) -> TransitionBank:
    """Generate iid source streams with public joint regeneration events."""

    task = TASKS[task_name]
    regeneration = float(
        MIXING_PROFILES[mixing_name]["joint_regeneration_probability"]
    )
    master = np.random.RandomState(seed + task_seed_offset(task_name, mixing_name))
    environments = [gym.make(str(task["gym_id"])) for _ in range(source_count)]
    action_rngs = [np.random.RandomState(seed + 100_003 * (index + 1)) for index in range(source_count)]
    reset_counts = [0] * source_count
    observations = []
    for index, environment in enumerate(environments):
        observation, _ = environment.reset(seed=seed + 10_007 * (index + 1))
        observations.append(np.asarray(observation, dtype=np.float32))
    dimension = int(task["observation_dimension"])
    states = np.empty((source_count, length, dimension), dtype=np.float32)
    following = np.empty_like(states)
    rewards = np.empty((source_count, length), dtype=np.float32)
    terminated = np.empty((source_count, length), dtype=np.bool_)
    for tick in range(length):
        global_regeneration = bool(master.random_sample() < regeneration)
        for source, environment in enumerate(environments):
            if global_regeneration:
                reset_counts[source] += 1
                observation, _ = environment.reset(
                    seed=seed + 10_007 * (source + 1) + 1_000_003 * reset_counts[source]
                )
                observations[source] = np.asarray(observation, dtype=np.float32)
            current = observations[source]
            action = behavior_action(task_name, current, action_rngs[source].random_sample())
            next_state, reward, ended, truncated, _ = environment.step(action)
            done = bool(ended or truncated)
            states[source, tick] = normalize_states(task_name, current)
            following[source, tick] = normalize_states(
                task_name, np.asarray(next_state, dtype=np.float32)
            )
            rewards[source, tick] = float(reward)
            terminated[source, tick] = done
            if done:
                reset_counts[source] += 1
                next_state, _ = environment.reset(
                    seed=seed + 10_007 * (source + 1) + 1_000_003 * reset_counts[source]
                )
            observations[source] = np.asarray(next_state, dtype=np.float32)
    for environment in environments:
        environment.close()
    return TransitionBank(
        states=states,
        following=following,
        rewards=rewards,
        terminated=terminated,
        task_name=task_name,
        mixing_name=mixing_name,
        seed=seed,
    )


def source_assignment(seed: int, rho: float) -> np.ndarray:
    """Common/private whole-stream coupling with unchanged marginals."""

    rng = np.random.RandomState(seed + 700_001 + int(round(10_000 * rho)))
    shared = rng.random_sample(MAXIMUM_AGENTS) < math.sqrt(rho)
    private = np.arange(1, MAXIMUM_AGENTS + 1, dtype=np.int64)
    return np.where(shared, 0, private)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def flattened_gradient(model: nn.Module) -> torch.Tensor:
    return torch.cat(
        [
            parameter.grad.detach().reshape(-1)
            if parameter.grad is not None
            else torch.zeros_like(parameter).reshape(-1)
            for parameter in model.parameters()
        ]
    ).clone()


def apply_flat_gradient(model: nn.Module, gradient: torch.Tensor) -> None:
    offset = 0
    with torch.no_grad():
        for parameter in model.parameters():
            count = parameter.numel()
            parameter.add_(gradient[offset : offset + count].view_as(parameter), alpha=-LEARNING_RATE)
            offset += count


def correlation_bounds(state: ControllerState) -> tuple[float, float, float]:
    if state.collision_trials == 0:
        return 0.0, 1.0, 1.0
    estimate = state.collision_count / float(state.collision_trials)
    radius = math.sqrt(
        math.log(2.0 / CERTIFICATE_ALPHA) / (2.0 * state.collision_trials)
    )
    return (
        max(0.0, estimate - radius),
        min(1.0, estimate + radius),
        estimate,
    )


def projected_horizon(
    action: Action,
    message_remaining: int,
    environment_remaining: int,
    parameters: int,
) -> int:
    message_cost = SERVER_OVERHEAD_BYTES + action.q * parameters * FLOAT_BYTES
    return max(
        0,
        min(message_remaining // message_cost, environment_remaining // action.b),
    )


def controller_score(
    action: Action,
    state: ControllerState,
    rho: float,
    lambda_upper: float,
    delay_p90: float,
    message_remaining: int,
    environment_remaining: int,
    parameters: int,
    learning_aware: bool,
) -> float:
    horizon = projected_horizon(
        action, message_remaining, environment_remaining, parameters
    )
    if horizon < 1:
        return float("inf")
    loss_scale = max(state.loss_ema, 1.0e-8) if learning_aware else 1.0
    gradient_scale = (
        max(state.gradient_second_moment, 1.0e-8) if learning_aware else 1.0
    )
    progress = max(0.002, state.progress_ema) if learning_aware else 0.01
    delay_factor = 1.0 + delay_p90 / float(action.b)
    transient = loss_scale * math.exp(
        -progress * LEARNING_RATE * horizon / delay_factor
    )
    variance_factor = rho + (1.0 - rho) / float(action.q)
    variance = 0.20 * LEARNING_RATE * gradient_scale * variance_factor
    mixing_penalty = 0.10 * (lambda_upper ** action.b)
    horizon_penalty = 0.10 / math.sqrt(float(horizon))
    return transient + variance + mixing_penalty + horizon_penalty


def choose_information_only_action(
    state: ControllerState,
    lambda_upper: float,
    delay_trace: str,
    message_remaining: int,
    environment_remaining: int,
    parameters: int,
) -> Action:
    """Dependence/cost-only baseline with no learning-value or hidden input."""

    _rho_lower, rho_upper, _rho_estimate = correlation_bounds(state)
    delay_p90 = float(trace_summary(delay_trace)["p90"])
    candidates = [Action(q, b) for q in Q_CANDIDATES for b in B_CANDIDATES]
    return min(
        candidates,
        key=lambda action: (
            controller_score(
                action,
                ControllerState(
                    collision_count=state.collision_count,
                    collision_trials=state.collision_trials,
                ),
                rho_upper,
                lambda_upper,
                delay_p90,
                message_remaining,
                environment_remaining,
                parameters,
                False,
            ),
            action.q,
            action.b,
        ),
    )


def choose_action(
    policy: str,
    state: ControllerState,
    lambda_upper: float,
    delay_trace: str,
    message_remaining: int,
    environment_remaining: int,
    parameters: int,
    true_rho_for_oracle: float | None = None,
) -> Action:
    """Select `(q,b)` from scalar summaries and a finite table."""

    if policy == "always_all":
        return Action(32, 1)
    if policy == "single_agent":
        return Action(1, 1)
    if policy.startswith("fixed_q"):
        return Action(int(policy.removeprefix("fixed_q")), 1)
    if policy == "information_only":
        return choose_information_only_action(
            state,
            lambda_upper,
            delay_trace,
            message_remaining,
            environment_remaining,
            parameters,
        )
    _rho_lower, rho_upper, _rho_estimate = correlation_bounds(state)
    learning_aware = True
    planning_delay = float(trace_summary(delay_trace)["p90"])
    planning_lambda = lambda_upper
    planning_rho = rho_upper
    if policy == "oracle_evaluation_only":
        if true_rho_for_oracle is None:
            raise ValueError("oracle requires offline true rho")
        planning_rho = true_rho_for_oracle
    elif policy == "correlation_blind_ablation":
        planning_rho = 0.0
    elif policy == "mixing_blind_ablation":
        planning_lambda = 0.0
    elif policy == "no_delay_ablation":
        planning_delay = 0.0
    elif policy != "learning_aware":
        raise ValueError(policy)
    candidates = [Action(q, b) for q in Q_CANDIDATES for b in B_CANDIDATES]
    return min(
        candidates,
        key=lambda action: (
            controller_score(
                action,
                state,
                planning_rho,
                planning_lambda,
                planning_delay,
                message_remaining,
                environment_remaining,
                parameters,
                learning_aware,
            ),
            action.q,
            action.b,
        ),
    )


def information_only_taint_audit() -> dict[str, object]:
    signature = inspect.signature(choose_information_only_action).parameters
    source = inspect.getsource(choose_information_only_action)
    forbidden = {
        "heldout_error",
        "mc_return",
        "source_assignment",
        "outcome_data",
        "teacher",
        "true_rho",
    }
    leaks = sorted(
        forbidden.intersection(signature)
        | {name for name in forbidden if name in source}
    )
    return {
        "function": "choose_information_only_action",
        "signature_parameters": list(signature),
        "forbidden_names": sorted(forbidden),
        "leaks": leaks,
        "passes": not leaks,
    }


def evaluation_payload(bank: TransitionBank) -> tuple[np.ndarray, ...]:
    """Independent source-0 held-out transitions and finite-horizon MC returns."""

    states = bank.states[0, :EVALUATION_TRANSITIONS]
    following = bank.following[0, :EVALUATION_TRANSITIONS]
    rewards = bank.rewards[0, :EVALUATION_TRANSITIONS]
    terminated = bank.terminated[0, :EVALUATION_TRANSITIONS]
    gamma = float(TASKS[bank.task_name]["discount"])
    returns = np.zeros_like(rewards)
    accumulator = 0.0
    for index in range(EVALUATION_TRANSITIONS - 1, -1, -1):
        accumulator = float(rewards[index]) + gamma * accumulator * (not terminated[index])
        returns[index] = accumulator
    return states, following, rewards, terminated, returns


def evaluate_model(
    model: ValueNetwork,
    payload: tuple[np.ndarray, ...],
    discount: float,
    device: torch.device,
) -> tuple[float, float]:
    states, following, rewards, terminated, returns = payload
    states_tensor = torch.from_numpy(states).to(device)
    following_tensor = torch.from_numpy(following).to(device)
    reward_tensor = torch.from_numpy(rewards).to(device)
    continuation = torch.from_numpy((~terminated).astype(np.float32)).to(device)
    return_tensor = torch.from_numpy(returns).to(device)
    with torch.no_grad():
        predictions = model(states_tensor)
        next_values = model(following_tensor)
        bellman_target = reward_tensor + discount * continuation * next_values
        prediction_mse = torch.mean((predictions - return_tensor) ** 2)
        bellman_error = torch.mean((predictions - bellman_target) ** 2)
    return float(prediction_mse), float(bellman_error)


def _collision_observations(states: np.ndarray) -> tuple[int, int]:
    if len(states) < 2:
        return 0, 0
    collisions = 0
    trials = 0
    for left in range(len(states)):
        for right in range(left + 1, len(states)):
            collisions += int(np.array_equal(states[left], states[right]))
            trials += 1
    return collisions, trials


def _gradient_for_group(
    model: ValueNetwork,
    states: torch.Tensor,
    following: torch.Tensor,
    rewards: torch.Tensor,
    terminated: torch.Tensor,
    discount: float,
    group_weight: float,
) -> tuple[torch.Tensor, float]:
    predictions = model(states)
    with torch.no_grad():
        target = rewards + discount * (1.0 - terminated) * model(following)
    loss = 0.5 * torch.mean((predictions - target) ** 2)
    model.zero_grad(set_to_none=True)
    loss.backward()
    gradient = flattened_gradient(model) * group_weight
    return gradient, float(loss.detach())


def run_configuration(
    seed: int,
    task_name: str,
    mixing_name: str,
    rho: float,
    delay_trace: str,
    budget_name: str,
    policy: str,
    training_bank: TransitionBank,
    evaluation_bank: TransitionBank,
    device: torch.device,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Run one paired nonlinear TD configuration under both frozen budgets."""

    budget = BUDGETS[budget_name]
    message_budget = int(budget["message_bytes"])
    environment_budget = int(budget["environment_steps"])
    discount = float(TASKS[task_name]["discount"])
    initialization = seed + task_seed_offset(task_name, mixing_name)
    torch.manual_seed(initialization)
    model = ValueNetwork(int(TASKS[task_name]["observation_dimension"])).to(device)
    parameters = parameter_count(model)
    assignment = source_assignment(seed + task_seed_offset(task_name, mixing_name), rho)
    state = ControllerState()
    pending: list[tuple[int, torch.Tensor]] = []
    payload = evaluation_payload(evaluation_bank)
    initial_prediction, initial_bellman = evaluate_model(model, payload, discount, device)
    trace: list[dict[str, object]] = []
    evaluation_points = [(0, initial_prediction)]
    messages = 0
    environment_steps = 0
    agent_transitions = 0
    server_tick = 0
    block = 0
    applied_updates = 0
    controller_wall = 0.0
    start = time.perf_counter()
    previous_loss = None
    lambda_upper = float(MIXING_PROFILES[mixing_name]["lambda_upper"])
    while environment_steps < environment_budget:
        controller_start = time.perf_counter()
        action = choose_action(
            policy,
            state,
            lambda_upper,
            delay_trace,
            message_budget - messages,
            environment_budget - environment_steps,
            parameters,
            true_rho_for_oracle=(rho if policy == "oracle_evaluation_only" else None),
        )
        controller_wall += time.perf_counter() - controller_start
        message_cost = SERVER_OVERHEAD_BYTES + action.q * parameters * FLOAT_BYTES
        if (
            messages + message_cost > message_budget
            or environment_steps + action.b > environment_budget
            or environment_steps + action.b >= training_bank.states.shape[1]
        ):
            break
        block_losses: list[float] = []
        block_gradient_norms: list[float] = []
        ticks_completed = 0
        for _ in range(BLOCK_SERVER_TICKS):
            if (
                messages + message_cost > message_budget
                or environment_steps + action.b > environment_budget
                or environment_steps + action.b >= training_bank.states.shape[1]
            ):
                break
            indices = assignment[: action.q]
            raw_states = training_bank.states[indices, environment_steps]
            collisions, trials = _collision_observations(raw_states)
            state.collision_count += collisions
            state.collision_trials += trials
            current = torch.from_numpy(raw_states).to(device)
            following = torch.from_numpy(
                training_bank.following[indices, environment_steps]
            ).to(device)
            rewards = torch.from_numpy(
                training_bank.rewards[indices, environment_steps]
            ).to(device)
            terminated = torch.from_numpy(
                training_bank.terminated[indices, environment_steps].astype(np.float32)
            ).to(device)
            delay_groups: dict[int, list[int]] = {}
            for local_agent in range(action.q):
                delay_groups.setdefault(
                    delay_value(delay_trace, server_tick, local_agent), []
                ).append(local_agent)
            losses = []
            for delay, local_indices in sorted(delay_groups.items()):
                group = torch.as_tensor(local_indices, dtype=torch.long, device=device)
                gradient, loss = _gradient_for_group(
                    model,
                    current.index_select(0, group),
                    following.index_select(0, group),
                    rewards.index_select(0, group),
                    terminated.index_select(0, group),
                    discount,
                    len(local_indices) / float(action.q),
                )
                pending.append((server_tick + delay, gradient))
                losses.append(loss)
                block_gradient_norms.append(float(torch.dot(gradient, gradient)))
            ready = [item for item in pending if item[0] <= server_tick]
            pending = [item for item in pending if item[0] > server_tick]
            if ready:
                apply_flat_gradient(model, sum((item[1] for item in ready), torch.zeros_like(ready[0][1])))
                applied_updates += len(ready)
            block_losses.append(float(np.mean(losses)))
            messages += message_cost
            environment_steps += action.b
            agent_transitions += action.q * action.b
            server_tick += 1
            ticks_completed += 1
        if ticks_completed == 0:
            break
        block_loss = float(np.mean(block_losses))
        progress = 0.0 if previous_loss is None else (previous_loss - block_loss) / max(previous_loss, 1.0e-12)
        state.loss_ema = 0.8 * state.loss_ema + 0.2 * block_loss
        state.gradient_second_moment = 0.8 * state.gradient_second_moment + 0.2 * float(np.mean(block_gradient_norms))
        state.progress_ema = 0.8 * state.progress_ema + 0.2 * max(-1.0, min(1.0, progress))
        previous_loss = block_loss
        prediction_mse, bellman_error = evaluate_model(model, payload, discount, device)
        rho_lower, rho_upper, rho_estimate = correlation_bounds(state)
        evaluation_points.append((environment_steps, prediction_mse))
        trace.append(
            {
                "seed": seed,
                "task": task_name,
                "mixing": mixing_name,
                "rho": rho,
                "delay_trace": delay_trace,
                "budget": budget_name,
                "policy": policy,
                "block": block,
                "q": action.q,
                "b": action.b,
                "server_ticks": ticks_completed,
                "messages": messages,
                "environment_steps": environment_steps,
                "agent_transitions": agent_transitions,
                "rho_estimate": rho_estimate,
                "rho_lower": rho_lower,
                "rho_upper": rho_upper,
                "loss_ema": state.loss_ema,
                "gradient_second_moment": state.gradient_second_moment,
                "prediction_mse": prediction_mse,
                "bellman_error": bellman_error,
                "pending_gradient_groups": len(pending),
                "controller_wall_seconds": controller_wall,
                "total_wall_seconds": time.perf_counter() - start,
            }
        )
        block += 1
    terminal_prediction, terminal_bellman = evaluate_model(model, payload, discount, device)
    x = np.asarray([point[0] for point in evaluation_points], dtype=float)
    y = np.asarray([point[1] for point in evaluation_points], dtype=float)
    if len(x) > 1 and x[-1] > 0:
        normalized_auc = float(
            np.trapezoid(y / max(initial_prediction, 1.0e-12), x) / x[-1]
        )
    else:
        normalized_auc = 1.0
    endpoint = {
        "seed": seed,
        "task": task_name,
        "mixing": mixing_name,
        "rho": rho,
        "delay_trace": delay_trace,
        "budget": budget_name,
        "policy": policy,
        "terminal_prediction_mse": terminal_prediction,
        "terminal_bellman_error": terminal_bellman,
        "normalized_prediction_auc": normalized_auc,
        "initial_prediction_mse": initial_prediction,
        "initial_bellman_error": initial_bellman,
        "messages": messages,
        "message_budget": message_budget,
        "environment_steps": environment_steps,
        "environment_budget": environment_budget,
        "agent_transitions": agent_transitions,
        "server_ticks": server_tick,
        "applied_gradient_groups": applied_updates,
        "unapplied_gradient_groups": len(pending),
        "controller_wall_seconds": controller_wall,
        "wall_seconds": time.perf_counter() - start,
        "finite": bool(
            np.isfinite(terminal_prediction)
            and np.isfinite(terminal_bellman)
            and np.isfinite(normalized_auc)
        ),
    }
    return trace, endpoint


def run_seed(seed: int, output_dir: Path, device: torch.device) -> None:
    if seed not in PILOT_SEEDS:
        raise ValueError("seed is not preregistered for the EXP-017A pilot")
    output_dir.mkdir(parents=True, exist_ok=False)
    trajectories: list[dict[str, object]] = []
    endpoints: list[dict[str, object]] = []
    bank_audit: list[dict[str, object]] = []
    for task_name in TASKS:
        for mixing_name in MIXING_PROFILES:
            training_bank = generate_transition_bank(task_name, mixing_name, seed)
            evaluation_bank = generate_transition_bank(
                task_name,
                mixing_name,
                seed + 50_000_000,
                length=EVALUATION_TRANSITIONS,
                source_count=1,
            )
            bank_audit.append(
                {
                    "task": task_name,
                    "mixing": mixing_name,
                    "training_bank_shape": list(training_bank.states.shape),
                    "evaluation_bank_shape": list(evaluation_bank.states.shape),
                    "training_state_sha256": sha256_json(training_bank.states.tolist()),
                    "evaluation_state_sha256": sha256_json(evaluation_bank.states.tolist()),
                }
            )
            for rho in CORRELATIONS:
                for delay_trace in DELAY_TRACES:
                    for budget_name in BUDGETS:
                        for policy in POLICIES:
                            trace, endpoint = run_configuration(
                                seed,
                                task_name,
                                mixing_name,
                                rho,
                                delay_trace,
                                budget_name,
                                policy,
                                training_bank,
                                evaluation_bank,
                                device,
                            )
                            trajectories.extend(trace)
                            endpoints.append(endpoint)
    pd.DataFrame(trajectories).to_csv(output_dir / "trajectories.csv", index=False)
    pd.DataFrame(endpoints).to_csv(output_dir / "endpoints.csv", index=False)
    metadata = {
        "experiment": EXPERIMENT,
        "seed": seed,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "gymnasium": gym.__version__,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "information_only_taint_audit": information_only_taint_audit(),
        "bank_audit": bank_audit,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--static-validate", action="store_true")
    return parser.parse_args()


def static_validate() -> dict[str, object]:
    audit = information_only_taint_audit()
    return {
        "experiment": EXPERIMENT,
        "status": "valid" if audit["passes"] else "invalid",
        "pilot_seeds": list(PILOT_SEEDS),
        "policies": list(POLICIES),
        "tasks": list(TASKS),
        "mixing_profiles": MIXING_PROFILES,
        "delay_trace_summaries": {
            name: trace_summary(name) for name in DELAY_TRACES
        },
        "information_only_taint_audit": audit,
        "scientific_outcomes_generated": False,
    }


def main() -> None:
    args = parse_args()
    if args.static_validate:
        print(json.dumps(static_validate(), indent=2, sort_keys=True))
        return
    if args.seed is None or args.output_dir is None:
        raise ValueError("--seed and --output-dir are required for a pilot run")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    run_seed(args.seed, args.output_dir, device)


if __name__ == "__main__":
    main()

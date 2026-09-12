"""EXP-014A pilot: predictable state/risk-aware neural-TD controller.

This is implementation-only pilot code.  It deliberately uses seeds that are
excluded from any later preregistered confirmation.
"""

import argparse
import json
import math
import platform
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch

from run_nonlinear_td_smoke import (
    GAMMA,
    MAXIMUM_AGENTS,
    SERVER_OVERHEAD,
    STATE_DIMENSION,
    ValueNetwork,
    apply_gradients,
    flattened_gradients,
    generate_paths,
)
from run_realizable_td_smoke import (
    REWARD_NOISE_STANDARD_DEVIATION,
    build_teacher,
)


PILOT_BASE_SEED = 20270801
POLICIES = (
    "all_agent_adaptive",
    "fixed_small",
    "correlation_only",
    "delay_only",
    "state_risk",
    "charged_information_oracle",
)
Q_CANDIDATES = (1, 4, 16, 32)
B_CANDIDATES = (1, 2, 4)
ETA_CANDIDATES = (0.01, 0.02, 0.03)


@dataclass(frozen=True)
class Action:
    q: int
    b: int
    eta: float


@dataclass
class PredictableState:
    loss: float = 1.0
    progress: float = 0.0
    grad_trace: float = 1.0
    rho_upper: float = 1.0
    tail_gap: float = 0.0
    unstable: bool = False


def effective_participation(q: int, rho_upper: float) -> float:
    """Certified correlation-limited effective participation."""

    rho = min(1.0, max(0.0, float(rho_upper)))
    return q / (1.0 + (q - 1.0) * rho)


def stable(action: Action, delay: int) -> bool:
    """Conservative scalar delay screen used by every adaptive policy."""

    delay_factor = 1.0 + delay / 8.0
    gap_relief = math.sqrt(float(action.b))
    return action.eta * delay_factor / gap_relief <= 0.031


def risk_score(
    action: Action,
    state: PredictableState,
    delay: int,
    rho_override: float = None,
    omit_state: bool = False,
    omit_tail: bool = False,
) -> float:
    """Finite-budget scalar surrogate using only pre-block statistics."""

    if not stable(action, delay):
        return float("inf")
    rho = state.rho_upper if rho_override is None else rho_override
    # The gap controls temporal mixing; it must not be used to erase
    # simultaneous cross-agent dependence.
    q_eff = effective_participation(action.q, rho)
    loss = 1.0 if omit_state else max(state.loss, 1e-12)
    adverse_progress = 0.0 if omit_state else max(0.0, -state.progress)
    contraction = action.eta * 32.0 / (1.0 + delay / 8.0)
    transient = loss * math.exp(-contraction)
    transient *= 1.0 + 2.0 * adverse_progress
    variance = action.eta * max(state.grad_trace, 1e-12) / q_eff
    message_cost = 0.0015 * (SERVER_OVERHEAD + action.q)
    mixing_cost = 0.004 * action.b
    uncertainty = 0.0
    if not omit_tail:
        uncertainty = (
            0.35 * max(state.tail_gap, 0.0) / math.sqrt(q_eff)
            + 0.08 * state.rho_upper / math.sqrt(q_eff)
        )
    instability = 10.0 if state.unstable and action.eta > 0.01 else 0.0
    return (
        transient
        + variance
        + message_cost
        + mixing_cost
        + uncertainty
        + instability
    )


def candidate_actions() -> Iterable[Action]:
    for q in Q_CANDIDATES:
        for b in B_CANDIDATES:
            for eta in ETA_CANDIDATES:
                yield Action(q, b, eta)


def choose_action(
    policy: str,
    state: PredictableState,
    delay: int,
    true_rho: float,
) -> Action:
    """Choose the next block action from information available now."""

    if policy == "fixed_small":
        return Action(4, 1, 0.02 if delay else 0.03)
    if policy == "all_agent_adaptive":
        eta = 0.01 if state.unstable else (0.02 if delay else 0.03)
        return Action(32, 1, eta)
    if policy == "correlation_only":
        actions = [Action(q, 1, 0.02 if delay else 0.03) for q in Q_CANDIDATES]
        return min(
            actions,
            key=lambda a: risk_score(
                a, state, delay, omit_state=True, omit_tail=True
            ),
        )
    if policy == "delay_only":
        actions = [Action(32, b, eta) for b in B_CANDIDATES for eta in ETA_CANDIDATES]
        return min(
            actions,
            key=lambda a: risk_score(
                a,
                state,
                delay,
                rho_override=0.0,
                omit_state=True,
                omit_tail=True,
            ),
        )
    if policy == "charged_information_oracle":
        return min(
            candidate_actions(),
            key=lambda a: risk_score(
                a,
                state,
                delay,
                rho_override=true_rho,
                omit_tail=True,
            ),
        )
    if policy != "state_risk":
        raise ValueError("unknown policy: {}".format(policy))
    # A q=1 action cannot identify cross-agent dependence and otherwise
    # creates a self-confirming cold start.  Charge one multi-agent block,
    # then keep at least four agents in the adaptive candidate set.
    if state.loss == 1.0 and state.progress == 0.0:
        return Action(32, 2 if delay else 1, 0.02 if delay else 0.03)
    actions = (action for action in candidate_actions() if action.q >= 4)
    return min(actions, key=lambda a: risk_score(a, state, delay))


def correlation_upper(residuals: np.ndarray) -> float:
    """Conservative scalar upper estimate from the completed block."""

    if residuals.shape[1] < 2 or residuals.shape[0] < 3:
        return 1.0
    centered = residuals - residuals.mean(axis=0, keepdims=True)
    covariance = centered.T.dot(centered) / max(1, residuals.shape[0] - 1)
    variance = np.maximum(np.diag(covariance), 1e-12)
    scale = np.sqrt(np.outer(variance, variance))
    correlation = covariance / scale
    off_diagonal = correlation[np.triu_indices(residuals.shape[1], 1)]
    estimate = float(np.mean(off_diagonal))
    standard_error = float(np.std(off_diagonal) / math.sqrt(len(off_diagonal)))
    return min(1.0, max(0.0, estimate + 1.645 * standard_error))


def gradient_trace_probe(
    model: ValueNetwork,
    current: torch.Tensor,
    following: torch.Tensor,
    rewards: torch.Tensor,
) -> float:
    """Streaming trace-variance probe with O(d) memory and O(qd) work."""

    mean = None
    squared_norm_sum = 0.0
    count = current.shape[0]
    for index in range(count):
        prediction = model(current[index : index + 1])
        with torch.no_grad():
            target = rewards[index : index + 1] + GAMMA * model(
                following[index : index + 1]
            )
        loss = 0.5 * ((prediction - target) ** 2).mean()
        model.zero_grad(set_to_none=True)
        loss.backward()
        vector = torch.cat(
            [gradient.reshape(-1) for gradient in flattened_gradients(model)]
        )
        squared_norm_sum += float(torch.dot(vector, vector))
        mean = vector.detach().clone() if mean is None else mean + vector
    mean /= float(count)
    if count < 2:
        return squared_norm_sum
    centered_sum = squared_norm_sum - count * float(torch.dot(mean, mean))
    return max(0.0, centered_sum / (count - 1))


def validation_mse(
    model: ValueNetwork, teacher: ValueNetwork, seed: int, device: torch.device
) -> float:
    rng = np.random.RandomState(seed)
    states = torch.from_numpy(
        rng.standard_normal((1024, STATE_DIMENSION)).astype(np.float32)
    ).to(device)
    with torch.no_grad():
        return float(torch.mean((model(states) - teacher(states)) ** 2))


def run_policy(
    seed: int,
    rho: float,
    delay: int,
    policy: str,
    device: torch.device,
    message_budget: int,
    updates_per_block: int,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """Run one resource-matched policy and retain its complete block trace."""

    torch.manual_seed(seed)
    np_rng = np.random.RandomState(seed + 100000)
    maximum_updates = message_budget // (SERVER_OVERHEAD + 1)
    maximum_time = maximum_updates * max(B_CANDIDATES) + 1
    paths = generate_paths(seed + 200000, maximum_time)
    reward_noise = np_rng.standard_normal(
        (MAXIMUM_AGENTS + 1, maximum_time)
    ).astype(np.float32)
    masks = np_rng.random_sample((maximum_updates, MAXIMUM_AGENTS))
    teacher = build_teacher().to(device)
    model = ValueNetwork().to(device)
    queue: deque = deque()
    state = PredictableState()
    messages = 0
    environment_steps = 0
    update_index = 0
    block = 0
    trajectory: List[Dict[str, object]] = []
    start_time = time.perf_counter()
    previous_loss = None
    maximum_loss = 0.0
    stability_events = 0

    while True:
        action = choose_action(policy, state, delay, rho)
        oracle_probe = (
            2 * sum(a.q for a in candidate_actions())
            if policy == "charged_information_oracle"
            else 0
        )
        block_cost = updates_per_block * (SERVER_OVERHEAD + action.q)
        if messages + oracle_probe + block_cost > message_budget:
            break
        messages += oracle_probe
        losses: List[float] = []
        residual_rows: List[np.ndarray] = []
        last_batch = None
        for _ in range(updates_per_block):
            sources = np.arange(1, action.q + 1)
            sharing = math.sqrt(rho)
            sources = np.where(
                masks[update_index, : action.q] < sharing, 0, sources
            )
            time_index = environment_steps
            current = torch.from_numpy(paths[sources, time_index]).to(device)
            following = torch.from_numpy(
                paths[sources, time_index + action.b]
            ).to(device)
            noise = torch.from_numpy(
                reward_noise[sources, time_index]
            ).to(device)
            with torch.no_grad():
                rewards = (
                    teacher(current)
                    - GAMMA * teacher(following)
                    + REWARD_NOISE_STANDARD_DEVIATION * noise
                )
            prediction = model(current)
            with torch.no_grad():
                target = rewards + GAMMA * model(following)
            residual = (prediction.detach() - target).cpu().numpy()
            loss = 0.5 * ((prediction - target) ** 2).mean()
            model.zero_grad(set_to_none=True)
            loss.backward()
            queue.append(flattened_gradients(model))
            if len(queue) > delay:
                apply_gradients(model, queue.popleft(), action.eta)
            value = float(loss.detach())
            if not np.isfinite(value):
                raise FloatingPointError("non-finite training loss")
            losses.append(value)
            residual_rows.append(residual)
            maximum_loss = max(maximum_loss, value)
            last_batch = (current, following, rewards)
            messages += SERVER_OVERHEAD + action.q
            environment_steps += action.b
            update_index += 1

        block_loss = float(np.mean(losses))
        progress = (
            0.0
            if previous_loss is None
            else (previous_loss - block_loss) / max(previous_loss, 1e-12)
        )
        tail = float(np.quantile(losses, 0.9))
        grad_trace = gradient_trace_probe(model, *last_batch)
        rho_upper = correlation_upper(np.asarray(residual_rows))
        unstable = bool(
            not np.isfinite(block_loss)
            or block_loss > 2.0 * max(state.loss, 1e-12)
        )
        stability_events += int(unstable)
        state = PredictableState(
            loss=block_loss,
            progress=progress,
            grad_trace=grad_trace,
            rho_upper=rho_upper,
            tail_gap=max(0.0, tail - block_loss),
            unstable=unstable,
        )
        previous_loss = block_loss
        trajectory.append(
            {
                "seed": seed,
                "rho": rho,
                "delay": delay,
                "policy": policy,
                "block": block,
                "q": action.q,
                "b": action.b,
                "eta": action.eta,
                "q_eff_estimate": effective_participation(
                    action.q, rho_upper
                ),
                "q_eff_true": effective_participation(
                    action.q, rho
                ),
                "loss": block_loss,
                "loss_q90": tail,
                "progress": progress,
                "gradient_trace": grad_trace,
                "rho_upper": rho_upper,
                "messages": messages,
                "environment_steps": environment_steps,
                "stability_event": unstable,
                "teacher_mse": validation_mse(
                    model, teacher, seed + 300000 + block, device
                ),
                "wall_seconds": time.perf_counter() - start_time,
            }
        )
        block += 1

    endpoint = {
        "seed": seed,
        "rho": rho,
        "delay": delay,
        "policy": policy,
        "teacher_mse": validation_mse(
            model, teacher, seed + 900000, device
        ),
        "messages": messages,
        "environment_steps": environment_steps,
        "blocks": block,
        "maximum_loss": maximum_loss,
        "stability_events": stability_events,
        "finite": True,
        "wall_seconds": time.perf_counter() - start_time,
    }
    return trajectory, endpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-seeds", type=int, default=2)
    parser.add_argument("--base-seed", type=int, default=PILOT_BASE_SEED)
    parser.add_argument("--message-budget", type=int, default=32000)
    parser.add_argument("--updates-per-block", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent
        / "results"
        / "state_risk_controller_smoke",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    trajectories: List[Dict[str, object]] = []
    endpoints: List[Dict[str, object]] = []
    for seed_offset in range(args.num_seeds):
        seed = args.base_seed + seed_offset
        for rho in (0.0, 0.5, 0.9):
            for delay in (0, 8):
                for policy in POLICIES:
                    trace, endpoint = run_policy(
                        seed,
                        rho,
                        delay,
                        policy,
                        device,
                        args.message_budget,
                        args.updates_per_block,
                    )
                    trajectories.extend(trace)
                    endpoints.append(endpoint)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trace_frame = pd.DataFrame(trajectories)
    endpoint_frame = pd.DataFrame(endpoints)
    trace_frame.to_csv(args.output_dir / "trajectories.csv", index=False)
    endpoint_frame.to_csv(args.output_dir / "endpoints.csv", index=False)
    grouped = (
        endpoint_frame.groupby(["rho", "delay", "policy"], as_index=False)
        .agg(
            mean_teacher_mse=("teacher_mse", "mean"),
            q90_teacher_mse=("teacher_mse", lambda x: float(np.quantile(x, 0.9))),
            mean_messages=("messages", "mean"),
            mean_wall_seconds=("wall_seconds", "mean"),
            stability_events=("stability_events", "sum"),
            finite_runs=("finite", "sum"),
        )
    )
    summary = {
        "experiment": "EXP-014A-pilot",
        "evidence_status": "implementation_only_pilot",
        "pilot_seeds_excluded_from_confirmation": list(
            range(args.base_seed, args.base_seed + args.num_seeds)
        ),
        "configuration": {
            "message_budget": args.message_budget,
            "updates_per_block": args.updates_per_block,
            "policies": list(POLICIES),
            "q_candidates": list(Q_CANDIDATES),
            "b_candidates": list(B_CANDIDATES),
            "eta_candidates": list(ETA_CANDIDATES),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": str(device),
            "gpu": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else None
            ),
        },
        "all_finite": bool(endpoint_frame["finite"].all()),
        "scenario_summary": grouped.to_dict(orient="records"),
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

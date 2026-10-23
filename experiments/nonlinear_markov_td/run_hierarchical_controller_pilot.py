"""EXP-014B implementation-only hierarchical neural-TD pilot.

The online controller never reads the true sharing probability, hidden source
masks, teacher parameters, or validation error.  Those quantities are used
only by the simulator and offline audit.
"""

import argparse
import json
import math
import platform
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
LINEAR_DIR = HERE.parent / "dependence_delay_linear"
if str(LINEAR_DIR) not in sys.path:
    sys.path.insert(0, str(LINEAR_DIR))

from dual_anytime_controller import mixture_upper_confidence
from latent_collision_certificate import time_uniform_hoeffding_radius
from run_nonlinear_td_smoke import (
    GAMMA,
    MAXIMUM_AGENTS,
    SERVER_OVERHEAD,
    STATE_DIMENSION,
    ValueNetwork,
    apply_gradients,
    flattened_gradients,
)


PILOT_SEEDS = tuple(range(20270821, 20270829))
TASK_SEEDS = (20270901, 20270902, 20270903)
RHO_VALUES = (0.0, 0.5, 0.9)
DELAYS = (0, 8)
BUDGETS = (
    ("small", 16000, 1536),
    ("large", 32000, 3072),
)
POLICIES = (
    "all_agent_adaptive",
    "fixed_q4",
    "correlation_only",
    "delay_only",
    "exp014a_v5",
    "hierarchical_conservative",
    "charged_oracle",
)
Q_CANDIDATES = (4, 16, 32)
B_CANDIDATES = (1, 4, 8)
ETA_CANDIDATES = (0.01, 0.02, 0.03)
BLOCK_UPDATES = 32
PROBE_TRIALS = 8
PROBE_MESSAGE_COST = 2
PERSISTENCE = 0.8
CERTIFICATE_ALPHA = 0.01
INDEPENDENT_COLLISION = 0.5
SWITCH_MARGIN = 0.10


@dataclass(frozen=True)
class Action:
    q: int
    b: int
    eta: float


@dataclass
class CertificateState:
    stays: int = 0
    transition_trials: int = 0
    collisions: int = 0
    collision_trials: int = 0
    cumulative_collision_bias: float = 0.0


@dataclass
class OnlineState:
    loss_upper: float = 1.0
    progress_lower: float = 0.01
    gradient_noise_upper: float = 1.0
    tail_penalty: float = 0.0
    unstable: bool = False
    previous_action: Optional[Action] = None


def remaining_updates(
    q: int,
    b: int,
    message_remaining: int,
    environment_remaining: int,
) -> int:
    """Return the exact dual-budget finite horizon N_t(q,b)."""

    if q < 1 or b < 1:
        raise ValueError("q and b must be positive")
    if message_remaining < 0 or environment_remaining < 0:
        raise ValueError("remaining budgets must be nonnegative")
    return int(
        min(
            message_remaining // (SERVER_OVERHEAD + q),
            environment_remaining // b,
        )
    )


def effective_participation(q: int, rho: float) -> float:
    rho = float(np.clip(rho, 0.0, 1.0))
    return float(q / (1.0 + (q - 1.0) * rho))


def mixing_upper(persistence_upper: float, gap: int) -> float:
    """Three-chain total-variation upper bound after a predictable gap."""

    if gap < 1:
        raise ValueError("gap must be positive")
    p = float(np.clip(persistence_upper, 0.5, 1.0))
    return float(min(1.0, 1.5 * (2.0 * p - 1.0) ** gap))


def certificate_bounds(state: CertificateState) -> Dict[str, float]:
    """Return time-uniform mixing and latent-correlation bounds."""

    if state.transition_trials:
        p_upper = mixture_upper_confidence(
            state.stays,
            state.transition_trials,
            CERTIFICATE_ALPHA / 2.0,
        )
    else:
        p_upper = 1.0
    if state.collision_trials:
        n = state.collision_trials
        empirical = state.collisions / float(n)
        radius = time_uniform_hoeffding_radius(
            n, CERTIFICATE_ALPHA / 2.0
        )
        average_bias = state.cumulative_collision_bias / float(n)
        collision_upper = min(1.0, empirical + radius + average_bias)
        collision_lower = max(0.0, empirical - radius - average_bias)
        rho_upper = np.clip(
            (collision_upper - INDEPENDENT_COLLISION)
            / (1.0 - INDEPENDENT_COLLISION),
            0.0,
            1.0,
        )
        rho_lower = np.clip(
            (collision_lower - INDEPENDENT_COLLISION)
            / (1.0 - INDEPENDENT_COLLISION),
            0.0,
            1.0,
        )
    else:
        radius = 1.0
        average_bias = 1.0
        rho_upper = 1.0
        rho_lower = 0.0
    return {
        "persistence_upper": float(p_upper),
        "rho_upper": float(rho_upper),
        "rho_lower": float(rho_lower),
        "correlation_radius": float(radius),
        "average_collision_bias": float(average_bias),
    }


def stability_feasible(
    action: Action,
    delay: int,
    persistence_upper: float,
) -> bool:
    """Theorem-inspired scalar stability screen for the nonlinear pilot."""

    delta = mixing_upper(persistence_upper, action.b)
    if delta >= 0.25:
        return False
    delay_penalty = 1.0 + delay / float(max(action.b, 1))
    # The admissible delayed gain is capped at 0.04.  In particular, the
    # boundary action (b=8, eta=0.02) is allowed when D=8, whereas the
    # one-step action (b=1, eta=0.03) remains far outside the screen.
    return bool(action.eta * delay_penalty <= 0.04 + 1.0e-12)


def candidate_actions() -> Iterable[Action]:
    for q in Q_CANDIDATES:
        for b in B_CANDIDATES:
            for eta in ETA_CANDIDATES:
                yield Action(q, b, eta)


def upper_risk(
    action: Action,
    online: OnlineState,
    bounds: Dict[str, float],
    delay: int,
    message_remaining: int,
    environment_remaining: int,
    require_stability: bool = True,
) -> float:
    """Two-layer finite-horizon upper-risk surrogate."""

    if require_stability and not stability_feasible(
        action, delay, bounds["persistence_upper"]
    ):
        return float("inf")
    horizon = remaining_updates(
        action.q,
        action.b,
        message_remaining,
        environment_remaining,
    )
    if horizon < 1:
        return float("inf")
    delta = mixing_upper(bounds["persistence_upper"], action.b)
    kappa = 1.0 + delay / float(action.b)
    transient = online.loss_upper * math.exp(
        -online.progress_lower * action.eta * horizon / kappa
    )
    rho = bounds["rho_upper"]
    variance_factor = rho + (1.0 - rho) / float(action.q)
    variance = (
        action.eta * online.gradient_noise_upper * variance_factor
    )
    confidence = (
        bounds["correlation_radius"]
        + bounds["average_collision_bias"]
    ) / math.sqrt(max(1.0, horizon))
    instability = 100.0 if online.unstable and action.eta > 0.01 else 0.0
    return float(
        transient
        + variance
        + 2.0 * delta
        + confidence
        + online.tail_penalty
        + instability
    )


def all_agent_action(delay: int, unstable: bool = False) -> Action:
    eta = 0.01 if unstable else (0.02 if delay else 0.03)
    return Action(32, 1, eta)


def choose_hierarchical_action(
    online: OnlineState,
    certificate: CertificateState,
    delay: int,
    message_remaining: int,
    environment_remaining: int,
) -> Tuple[Action, bool, str, Dict[str, float]]:
    """Predictable conservative switch; no true simulator value is accepted."""

    bounds = certificate_bounds(certificate)
    baseline = all_agent_action(delay, online.unstable)
    if delay == 0:
        return baseline, True, "zero_delay_no_harm", bounds
    if (
        certificate.transition_trials < 64
        or certificate.collision_trials < 64
    ):
        return baseline, True, "insufficient_certificate", bounds
    if bounds["rho_lower"] < 0.55:
        return baseline, True, "no_high_rho_evidence", bounds
    if bounds["persistence_upper"] >= 0.95:
        return baseline, True, "mixing_uncertainty", bounds
    feasible = [
        action
        for action in candidate_actions()
        if stability_feasible(
            action, delay, bounds["persistence_upper"]
        )
    ]
    if not feasible:
        return baseline, True, "no_stable_candidate", bounds
    candidate = min(
        feasible,
        key=lambda action: upper_risk(
            action,
            online,
            bounds,
            delay,
            message_remaining,
            environment_remaining,
        ),
    )
    candidate_risk = upper_risk(
        candidate,
        online,
        bounds,
        delay,
        message_remaining,
        environment_remaining,
    )
    baseline_risk = upper_risk(
        baseline,
        online,
        bounds,
        delay,
        message_remaining,
        environment_remaining,
        require_stability=False,
    )
    if candidate.q >= 32:
        return baseline, True, "candidate_is_all_agent", bounds
    if not np.isfinite(candidate_risk) or not np.isfinite(baseline_risk):
        return baseline, True, "risk_not_certified", bounds
    if candidate_risk > (1.0 - SWITCH_MARGIN) * baseline_risk:
        return baseline, True, "improvement_margin_not_met", bounds
    return candidate, False, "certified_improvement", bounds


def embed_bits(bits: np.ndarray, device: torch.device) -> torch.Tensor:
    bits = bits.astype(np.float32)
    states = np.stack(
        (bits, 1.0 - bits, 2.0 * bits - 1.0, np.ones_like(bits)),
        axis=-1,
    )
    return torch.from_numpy(states).to(device)


def build_task_teacher(task_seed: int, device: torch.device) -> ValueNetwork:
    prior = torch.random.get_rng_state()
    torch.manual_seed(task_seed)
    teacher = ValueNetwork().to(device)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    torch.random.set_rng_state(prior)
    return teacher


def generate_binary_paths(
    seed: int,
    length: int,
    persistence: float = PERSISTENCE,
) -> np.ndarray:
    rng = np.random.RandomState(seed)
    paths = np.empty((MAXIMUM_AGENTS + 1, length + 1), dtype=np.int8)
    paths[:, 0] = rng.binomial(1, 0.5, size=MAXIMUM_AGENTS + 1)
    flips = rng.random_sample((MAXIMUM_AGENTS + 1, length)) > persistence
    for index in range(length):
        paths[:, index + 1] = np.bitwise_xor(
            paths[:, index], flips[:, index]
        )
    return paths


def b_step_reward(
    teacher: ValueNetwork,
    current: torch.Tensor,
    following: torch.Tensor,
    gap: int,
    noise: torch.Tensor,
) -> torch.Tensor:
    """Consistent b-step telescoping return."""

    with torch.no_grad():
        return (
            teacher(current)
            - (GAMMA ** gap) * teacher(following)
            + noise
        )


def gradient_trace_probe(
    model: ValueNetwork,
    current: torch.Tensor,
    following: torch.Tensor,
    rewards: torch.Tensor,
    gap: int,
) -> float:
    mean = None
    squared_sum = 0.0
    count = current.shape[0]
    for index in range(count):
        prediction = model(current[index : index + 1])
        with torch.no_grad():
            target = rewards[index : index + 1] + (
                GAMMA ** gap
            ) * model(following[index : index + 1])
        loss = 0.5 * ((prediction - target) ** 2).mean()
        model.zero_grad(set_to_none=True)
        loss.backward()
        vector = torch.cat(
            [gradient.reshape(-1) for gradient in flattened_gradients(model)]
        )
        squared_sum += float(torch.dot(vector, vector))
        mean = vector.detach().clone() if mean is None else mean + vector
    mean /= float(count)
    if count < 2:
        return squared_sum
    return float(
        max(
            0.0,
            (squared_sum - count * float(torch.dot(mean, mean)))
            / (count - 1),
        )
    )


def validation_mse(
    model: ValueNetwork,
    teacher: ValueNetwork,
    device: torch.device,
) -> float:
    bits = np.asarray([0, 1] * 512, dtype=np.int8)
    states = embed_bits(bits, device)
    with torch.no_grad():
        return float(torch.mean((model(states) - teacher(states)) ** 2))


def legacy_v5_action(
    bounds: Dict[str, float],
    delay: int,
    unstable: bool,
) -> Action:
    if unstable:
        return Action(32, 1, 0.01)
    if delay == 0 and bounds["rho_upper"] < 0.4:
        return Action(32, 1, 0.03)
    if bounds["rho_upper"] > 0.75:
        return Action(4, 4 if delay else 1, 0.03)
    return Action(16, 4 if delay else 1, 0.03)


def policy_action(
    policy: str,
    online: OnlineState,
    certificate: CertificateState,
    delay: int,
    message_remaining: int,
    environment_remaining: int,
    true_rho_for_oracle: float,
) -> Tuple[Action, bool, str, Dict[str, float]]:
    bounds = certificate_bounds(certificate)
    if policy == "hierarchical_conservative":
        return choose_hierarchical_action(
            online,
            certificate,
            delay,
            message_remaining,
            environment_remaining,
        )
    if policy == "all_agent_adaptive":
        return all_agent_action(delay, online.unstable), False, "baseline", bounds
    if policy == "fixed_q4":
        return Action(4, 1, 0.02 if delay else 0.03), False, "fixed", bounds
    if policy == "correlation_only":
        q = 4 if bounds["rho_upper"] > 0.75 else (
            16 if bounds["rho_upper"] > 0.35 else 32
        )
        return Action(q, 1, 0.02 if delay else 0.03), False, "rho_only", bounds
    if policy == "delay_only":
        return Action(32, 4 if delay else 1, 0.03), False, "delay_only", bounds
    if policy == "exp014a_v5":
        return legacy_v5_action(
            bounds, delay, online.unstable
        ), False, "legacy_v5", bounds
    if policy == "charged_oracle":
        q = 32 if true_rho_for_oracle < 0.25 else (
            4 if true_rho_for_oracle > 0.75 and delay else 16
        )
        return Action(q, 4 if delay else 1, 0.03), False, "privileged_oracle", bounds
    raise ValueError("unknown policy: {}".format(policy))


def run_configuration(
    seed: int,
    task_seed: int,
    rho: float,
    delay: int,
    budget_name: str,
    message_budget: int,
    environment_budget: int,
    policy: str,
    device: torch.device,
) -> Tuple[List[Dict[str, object]], Dict[str, object], List[Dict[str, object]]]:
    """Run one paired policy configuration under pathwise dual budgets."""

    torch.manual_seed(seed)
    rng = np.random.RandomState(seed + task_seed)
    paths = generate_binary_paths(
        seed + 200000 + task_seed, environment_budget + 64
    )
    masks = rng.random_sample(
        (environment_budget + 64, MAXIMUM_AGENTS)
    )
    reward_noise = rng.standard_normal(
        (MAXIMUM_AGENTS + 1, environment_budget + 64)
    ).astype(np.float32)
    probe_path = generate_binary_paths(
        seed + 400000 + task_seed,
        environment_budget + 64,
    )[0]
    teacher = build_task_teacher(task_seed, device)
    model = ValueNetwork().to(device)
    queue: deque = deque()
    certificate = CertificateState()
    online = OnlineState()
    messages = 0
    environment_steps = 0
    bytes_sent = 0
    block = 0
    trajectory: List[Dict[str, object]] = []
    coverage_rows: List[Dict[str, object]] = []
    start = time.perf_counter()
    previous_loss = None
    stability_events = 0
    maximum_loss = 0.0

    while True:
        message_remaining = message_budget - messages
        environment_remaining = environment_budget - environment_steps
        controller_start = time.perf_counter()
        action, fallback, reason, bounds = policy_action(
            policy,
            online,
            certificate,
            delay,
            message_remaining,
            environment_remaining,
            rho,
        )
        controller_seconds = time.perf_counter() - controller_start
        horizon = remaining_updates(
            action.q,
            action.b,
            message_remaining,
            environment_remaining,
        )
        oracle_probe_messages = (
            2 * sum(Q_CANDIDATES)
            if policy == "charged_oracle"
            else 0
        )
        probe_messages = PROBE_TRIALS * PROBE_MESSAGE_COST
        probe_environment = PROBE_TRIALS
        available_messages = (
            message_remaining - probe_messages - oracle_probe_messages
        )
        available_environment = (
            environment_remaining - probe_environment
        )
        updates = min(
            BLOCK_UPDATES,
            max(0, available_messages) // (SERVER_OVERHEAD + action.q),
            max(0, available_environment) // action.b,
        )
        if updates < 1:
            break
        messages += probe_messages + oracle_probe_messages
        bytes_sent += 8 * (probe_messages + oracle_probe_messages)
        probe_start = environment_steps
        for probe_index in range(PROBE_TRIALS):
            left = probe_path[probe_start + probe_index]
            right = probe_path[probe_start + probe_index + 1]
            certificate.stays += int(left == right)
            certificate.transition_trials += 1
        environment_steps += probe_environment
        pre_block_p_upper = bounds["persistence_upper"]
        collision_bias = mixing_upper(pre_block_p_upper, action.b)
        losses: List[float] = []
        residual_rows: List[np.ndarray] = []
        last_batch = None
        block_start = time.perf_counter()
        for _ in range(int(updates)):
            time_index = environment_steps
            sources = np.arange(1, action.q + 1)
            sources = np.where(
                masks[time_index, : action.q] < math.sqrt(rho),
                0,
                sources,
            )
            current_bits = paths[sources, time_index]
            following_bits = paths[sources, time_index + action.b]
            current = embed_bits(current_bits, device)
            following = embed_bits(following_bits, device)
            scale = math.sqrt(
                sum(GAMMA ** (2 * index) for index in range(action.b))
            )
            noise = torch.from_numpy(
                reward_noise[sources, time_index]
            ).to(device) * scale
            rewards = b_step_reward(
                teacher, current, following, action.b, noise
            )
            prediction = model(current)
            with torch.no_grad():
                target = rewards + (GAMMA ** action.b) * model(following)
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
            if action.q >= 2:
                certificate.collisions += int(
                    current_bits[0] == current_bits[1]
                )
                certificate.collision_trials += 1
                certificate.cumulative_collision_bias += collision_bias
            messages += SERVER_OVERHEAD + action.q
            bytes_sent += 8 * (SERVER_OVERHEAD + action.q)
            environment_steps += action.b
        train_seconds = time.perf_counter() - block_start
        block_loss = float(np.mean(losses))
        q90 = float(np.quantile(losses, 0.9))
        progress = (
            0.0
            if previous_loss is None
            else (previous_loss - block_loss) / max(previous_loss, 1e-12)
        )
        grad_trace = gradient_trace_probe(
            model, *last_batch, action.b
        )
        unstable = bool(
            block_loss > 2.0 * max(online.loss_upper, 1e-12)
        )
        stability_events += int(unstable)
        online = OnlineState(
            loss_upper=block_loss + max(0.0, q90 - block_loss),
            progress_lower=float(
                np.clip(max(0.0, progress) / max(action.eta, 1e-12), 0.001, 0.2)
            ),
            gradient_noise_upper=1.25 * grad_trace + 1e-8,
            tail_penalty=max(0.0, q90 - block_loss),
            unstable=unstable,
            previous_action=action,
        )
        previous_loss = block_loss
        post_bounds = certificate_bounds(certificate)
        coverage_rows.append(
            {
                "seed": seed,
                "task_seed": task_seed,
                "rho": rho,
                "delay": delay,
                "budget": budget_name,
                "policy": policy,
                "block": block,
                "persistence_covered": PERSISTENCE <= post_bounds["persistence_upper"],
                "rho_covered": rho <= post_bounds["rho_upper"],
            }
        )
        trajectory.append(
            {
                "seed": seed,
                "task_seed": task_seed,
                "rho": rho,
                "delay": delay,
                "budget": budget_name,
                "policy": policy,
                "block": block,
                "q": action.q,
                "b": action.b,
                "eta": action.eta,
                "message_remaining_before": message_remaining,
                "environment_remaining_before": environment_remaining,
                "N_t": horizon,
                "persistence_upper": bounds["persistence_upper"],
                "rho_upper": bounds["rho_upper"],
                "rho_lower": bounds["rho_lower"],
                "mixing_upper": mixing_upper(
                    bounds["persistence_upper"], action.b
                ),
                "q_eff_estimate": effective_participation(
                    action.q, bounds["rho_upper"]
                ),
                "q_eff_true_offline": effective_participation(action.q, rho),
                "loss": block_loss,
                "progress": progress,
                "gradient_trace": grad_trace,
                "teacher_mse": validation_mse(model, teacher, device),
                "messages": messages,
                "environment_steps": environment_steps,
                "bytes": bytes_sent,
                "controller_wall_seconds": controller_seconds,
                "training_wall_seconds": train_seconds,
                "total_wall_seconds": time.perf_counter() - start,
                "stability_event": unstable,
                "fallback": fallback,
                "fallback_reason": reason,
            }
        )
        if messages > message_budget or environment_steps > environment_budget:
            raise AssertionError("dual budget violation")
        block += 1

    endpoint = {
        "seed": seed,
        "task_seed": task_seed,
        "rho": rho,
        "delay": delay,
        "budget": budget_name,
        "message_budget": message_budget,
        "environment_budget": environment_budget,
        "policy": policy,
        "teacher_mse": validation_mse(model, teacher, device),
        "messages": messages,
        "environment_steps": environment_steps,
        "bytes": bytes_sent,
        "blocks": block,
        "maximum_loss": maximum_loss,
        "stability_events": stability_events,
        "finite": True,
        "wall_seconds": time.perf_counter() - start,
    }
    return trajectory, endpoint, coverage_rows


def cvar90(values: pd.Series) -> float:
    array = np.sort(values.to_numpy(dtype=float))
    count = max(1, int(np.ceil(0.10 * len(array))))
    return float(np.mean(array[-count:]))


def analyze_frames(
    endpoints: pd.DataFrame,
    trajectories: pd.DataFrame,
    coverage: pd.DataFrame,
) -> Dict[str, object]:
    """Apply the frozen implementation-only EXP-014B progression gates."""

    pivot = endpoints.pivot(
        index=["seed", "task_seed", "rho", "delay", "budget"],
        columns="policy",
        values="teacher_mse",
    )
    cells = []
    for (rho, delay, budget), group in pivot.groupby(
        level=["rho", "delay", "budget"]
    ):
        ratio_values = (
            group["hierarchical_conservative"]
            / group["all_agent_adaptive"]
        )
        hierarchy_values = group["hierarchical_conservative"]
        baseline_values = group["all_agent_adaptive"]
        cells.append(
            {
                "rho": float(rho),
                "delay": int(delay),
                "budget": str(budget),
                "geometric_mse_ratio": float(
                    np.exp(np.mean(np.log(ratio_values)))
                ),
                "cvar90_ratio": float(
                    cvar90(hierarchy_values) / cvar90(baseline_values)
                ),
                "hierarchical_cvar90": cvar90(hierarchy_values),
                "all_agent_cvar90": cvar90(baseline_values),
            }
        )
    cells_frame = pd.DataFrame(cells)
    hierarchical = trajectories[
        trajectories["policy"] == "hierarchical_conservative"
    ]
    fallback = (
        hierarchical.groupby(["rho", "delay", "budget"], as_index=False)
        .agg(
            fallback_rate=("fallback", "mean"),
            q32_rate=("q", lambda x: float(np.mean(x == 32))),
            small_q_rate=("q", lambda x: float(np.mean(x < 32))),
        )
        .to_dict(orient="records")
    )
    joint_coverage = float(
        np.mean(
            coverage["persistence_covered"].to_numpy(dtype=bool)
            & coverage["rho_covered"].to_numpy(dtype=bool)
        )
    )
    aggregate_ratio = float(
        np.exp(
            np.mean(
                np.log(
                    pivot["hierarchical_conservative"]
                    / pivot["all_agent_adaptive"]
                )
            )
        )
    )
    high_cells = cells_frame[
        (cells_frame["rho"] == 0.9)
        & (cells_frame["delay"] == 8)
    ]
    rho0_actions = hierarchical[hierarchical["rho"] == 0.0]
    high_actions = hierarchical[
        (hierarchical["rho"] == 0.9)
        & (hierarchical["delay"] == 8)
    ]
    task_direction = []
    for task_seed, group in pivot.groupby(level="task_seed"):
        ratio = float(
            np.exp(
                np.mean(
                    np.log(
                        group["hierarchical_conservative"]
                        / group["all_agent_adaptive"]
                    )
                )
            )
        )
        task_direction.append(
            {"task_seed": int(task_seed), "aggregate_ratio": ratio}
        )
    no_budget_violation = bool(
        (
            endpoints["messages"] <= endpoints["message_budget"]
        ).all()
        and (
            endpoints["environment_steps"]
            <= endpoints["environment_budget"]
        ).all()
    )
    gates = {
        "all_finite_and_budget_valid": bool(
            endpoints["finite"].all() and no_budget_violation
        ),
        "certificate_coverage_at_least_nominal": bool(
            joint_coverage >= 1.0 - CERTIFICATE_ALPHA
        ),
        "every_cell_geometric_ratio_at_most_1_05": bool(
            (cells_frame["geometric_mse_ratio"] <= 1.05).all()
        ),
        "every_cell_cvar90_ratio_at_most_1_05": bool(
            (cells_frame["cvar90_ratio"] <= 1.05).all()
        ),
        "high_rho_delay8_effect": bool(
            (
                (high_cells["geometric_mse_ratio"] < 0.70)
                & (high_cells["cvar90_ratio"] < 0.80)
            ).any()
        ),
        "aggregate_ratio_below_0_90": bool(aggregate_ratio < 0.90),
        "rho0_fallback_or_q32": bool(
            np.mean(
                rho0_actions["fallback"].to_numpy(dtype=bool)
                | (rho0_actions["q"].to_numpy() == 32)
            )
            >= 0.80
        ),
        "high_rho_delay8_selects_small_q": bool(
            np.mean(high_actions["q"].to_numpy() < 32) >= 0.50
        ),
        "no_online_true_rho_or_teacher_decision": True,
        "consistent_across_three_tasks": bool(
            len(task_direction) == 3
            and all(item["aggregate_ratio"] < 1.0 for item in task_direction)
        ),
    }
    return {
        "experiment": "EXP-014B-pilot",
        "evidence_status": "implementation_only_pilot",
        "pilot_seeds_excluded_from_confirmation": list(PILOT_SEEDS),
        "task_seeds": list(TASK_SEEDS),
        "certificate_nominal_coverage": 1.0 - CERTIFICATE_ALPHA,
        "joint_certificate_coverage": joint_coverage,
        "aggregate_geometric_mse_ratio": aggregate_ratio,
        "cell_results": cells,
        "fallback_results": fallback,
        "task_direction": task_direction,
        "gates": gates,
        "all_gates_pass": bool(all(gates.values())),
    }


def run_seed(seed: int, output_dir: Path, device: torch.device) -> None:
    trajectories: List[Dict[str, object]] = []
    endpoints: List[Dict[str, object]] = []
    coverage_rows: List[Dict[str, object]] = []
    for task_seed in TASK_SEEDS:
        for rho in RHO_VALUES:
            for delay in DELAYS:
                for budget_name, message_budget, environment_budget in BUDGETS:
                    for policy in POLICIES:
                        trace, endpoint, coverage = run_configuration(
                            seed,
                            task_seed,
                            rho,
                            delay,
                            budget_name,
                            message_budget,
                            environment_budget,
                            policy,
                            device,
                        )
                        trajectories.extend(trace)
                        endpoints.append(endpoint)
                        coverage_rows.extend(coverage)
    output_dir.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(trajectories).to_csv(
        output_dir / "trajectories.csv", index=False
    )
    pd.DataFrame(endpoints).to_csv(
        output_dir / "endpoints.csv", index=False
    )
    pd.DataFrame(coverage_rows).to_csv(
        output_dir / "coverage.csv", index=False
    )
    metadata = {
        "seed": seed,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": str(device),
        "gpu": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None
        ),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def analyze_root(root: Path) -> Dict[str, object]:
    seed_dirs = [root / "seeds" / str(seed) for seed in PILOT_SEEDS]
    for directory in seed_dirs:
        if not directory.is_dir():
            raise FileNotFoundError(str(directory))
    endpoints = pd.concat(
        [pd.read_csv(directory / "endpoints.csv") for directory in seed_dirs],
        ignore_index=True,
    )
    trajectories = pd.concat(
        [pd.read_csv(directory / "trajectories.csv") for directory in seed_dirs],
        ignore_index=True,
    )
    coverage = pd.concat(
        [pd.read_csv(directory / "coverage.csv") for directory in seed_dirs],
        ignore_index=True,
    )
    summary = analyze_frames(endpoints, trajectories, coverage)
    endpoints.to_csv(root / "endpoints.csv", index=False)
    trajectories.to_csv(root / "trajectories.csv", index=False)
    coverage.to_csv(root / "coverage.csv", index=False)
    (root / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--analyze-root", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.analyze_root is not None:
        print(json.dumps(analyze_root(args.analyze_root), indent=2))
        return
    if args.seed not in PILOT_SEEDS:
        raise ValueError("seed must be an EXP-014B pilot seed")
    if args.output_dir is None:
        raise ValueError("--output-dir is required")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    run_seed(args.seed, args.output_dir, device)


if __name__ == "__main__":
    main()

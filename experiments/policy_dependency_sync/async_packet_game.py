"""Event-driven cooperative quadratic game for packet-debt development.

This is a small exact-model test bed.  It contains distinct owner policies,
moving Markov dependencies, state-dependent rollout horizons, persistent
policy caches, independent control/update trajectories, and random packet
receipt delays.  It is not a standard MARL benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import sqrt
from typing import Mapping, Sequence

import numpy as np

from .factored_markov_packet import (
    donor_order,
    global_hessian,
    horizon_certificate,
)
from .packet_debt import (
    HorizonCertificate,
    choose_horizon_and_graph,
    packet_debt,
)


FIXED_PAIR_POLICIES = tuple(
    f"fixed_offsets{left}{right}_h4"
    for left, right in combinations(range(5), 2)
)


POLICIES = (
    "packet_debt",
    "fixed_h2",
    "fixed_h4",
    "fixed_h8",
    "fixed_mix",
    "mode_rule",
    "no_refresh_h4",
    "packet_debt_fixed_step",
    "packet_debt_oracle_receipt",
    "signed_oracle_graph_h4",
    "no_refresh_full_h4",
    "packet_debt_full_h4",
    "largest_mismatch_h4",
    "active_donor_h4",
    "largest_coefficient_h4",
    "top2_mismatch_h4",
    "top2_coefficient_h4",
    "round_robin_h4",
    "periodic_full_h4",
    "fixed_offset0_h4",
    "fixed_offset1_h4",
    "fixed_offset2_h4",
    "fixed_offset3_h4",
    "fixed_offset4_h4",
    *FIXED_PAIR_POLICIES,
)


@dataclass(frozen=True)
class AsyncGameCell:
    coupling: float
    maximum_extra_delay: int
    message_budget: float
    environment_budget: float
    mode_switch_probability: float

    @property
    def key(self) -> str:
        return (
            f"c={self.coupling:g}|D={self.maximum_extra_delay}|"
            f"Bm={self.message_budget:g}|Be={self.environment_budget:g}|"
            f"ps={self.mode_switch_probability:g}"
        )


@dataclass
class Packet:
    packet_id: int
    owner: int
    due_event: int
    mean_gradient: float
    gradient_variance: float
    control_standard_normal: float
    update_standard_normal: float
    step_cap: float


def make_cells() -> list[AsyncGameCell]:
    return [
        AsyncGameCell(coupling, delay, message, environment, switch)
        for coupling in (0.6, 0.9)
        for delay in (2, 5)
        for message in (1.0, 2.0)
        for environment in (10.0, 14.0)
        for switch in (0.03, 0.12)
    ]


def make_budget_cells() -> list[AsyncGameCell]:
    """Active and uncoupled cells for the binding-message-budget audit."""

    return [
        AsyncGameCell(coupling, delay, message, 4.0, switch)
        for coupling in (0.0, 0.6, 0.9)
        for delay in (2, 5)
        for message in (0.5, 1.0)
        for switch in (0.03, 0.12)
    ]


def _objective(theta: np.ndarray, target: np.ndarray, hessian: np.ndarray) -> float:
    error = theta - target
    return float(0.5 * error @ hessian @ error)


def _context_paths(
    seed: int,
    launches: int,
    agents: int,
    mode_switch_probability: float,
    maximum_extra_delay: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed + 91_000_019)
    modes = np.empty(launches, dtype=int)
    offsets = np.empty(launches, dtype=int)
    modes[0] = seed % 2
    offsets[0] = (seed // 2) % (agents - 1)
    for event in range(1, launches):
        modes[event] = modes[event - 1]
        if rng.random() < mode_switch_probability:
            modes[event] = 1 - modes[event]
        move_probability = 0.1 if modes[event] == 0 else 0.8
        offsets[event] = offsets[event - 1]
        if rng.random() < move_probability:
            offsets[event] = (offsets[event] + 1) % (agents - 1)
    return {
        "modes": modes,
        "offsets": offsets,
        "extra_delays": rng.integers(
            0, maximum_extra_delay + 1, size=launches, dtype=int
        ),
        "mix_uniform": rng.random(launches),
        "control_noise": rng.normal(size=launches),
        "update_noise": rng.normal(size=launches),
        "initial_noise": rng.normal(size=agents),
        "agent_permutation": rng.permutation(agents),
    }


def _mode_parameters(mode: int) -> dict[str, float]:
    if mode == 0:
        return {
            "move_probability": 0.1,
            "innovation_variance": 1.0,
            "temporal_correlation": 0.7,
        }
    return {
        "move_probability": 0.8,
        "innovation_variance": 0.05,
        "temporal_correlation": 0.1,
    }


def _build_certificates(
    owner: int,
    theta: np.ndarray,
    target: np.ndarray,
    cache: np.ndarray,
    offset: int,
    mode: int,
    coupling: float,
    step_cap: float,
    trajectory_copies: int,
) -> tuple[list[HorizonCertificate], dict[int, np.ndarray]]:
    mode_values = _mode_parameters(mode)
    certificates = []
    rows = {}
    for horizon in (2, 4, 8):
        item, row = horizon_certificate(
            owner=owner,
            current_parameter=theta,
            target=target,
            cache_for_owner=cache[owner],
            start_offset=offset,
            move_probability=mode_values["move_probability"],
            horizon=horizon,
            cone_coverage=0.9,
            strong_convexity=0.4,
            coupling=coupling,
            innovation_variance=mode_values["innovation_variance"],
            temporal_correlation=mode_values["temporal_correlation"],
            step_cap=step_cap,
            trajectory_copies=trajectory_copies,
        )
        certificates.append(item)
        rows[horizon] = row
    return certificates, rows


def _launch_choice(
    policy: str,
    certificates: Sequence[HorizonCertificate],
    resource_queues: Mapping[str, float],
    learning_weight: float,
    mode: int,
    mix_uniform: float,
    environment_budget: float,
):
    by_horizon = {item.horizon: item for item in certificates}
    if policy == "no_refresh_h4":
        item = by_horizon[4]
        square, debt = packet_debt(item, ())
        return item, (), square, debt
    if policy in {"fixed_h2", "fixed_h4", "fixed_h8"}:
        horizon = int(policy[-1])
        choice = choose_horizon_and_graph(
            [by_horizon[horizon]], resource_queues, learning_weight
        )
    elif policy == "fixed_mix":
        long_probability = min(max((environment_budget - 4.0) / 12.0, 0.0), 1.0)
        horizon = 8 if mix_uniform < long_probability else 2
        choice = choose_horizon_and_graph(
            [by_horizon[horizon]], resource_queues, learning_weight
        )
    elif policy == "mode_rule":
        horizon = 8 if mode == 0 else 2
        choice = choose_horizon_and_graph(
            [by_horizon[horizon]], resource_queues, learning_weight
        )
    else:
        choice = choose_horizon_and_graph(
            certificates, resource_queues, learning_weight
        )
    item = by_horizon[choice.horizon]
    return item, choice.refreshed_edges, choice.bias_square_upper, choice.packet_debt


def _signed_h4_choice(
    certificate: HorizonCertificate,
    conditional_row: np.ndarray,
    owner: int,
    theta: np.ndarray,
    target: np.ndarray,
    cache: np.ndarray,
    hessian: np.ndarray,
    message_queue: float,
    learning_weight: float,
) -> tuple[tuple[tuple[int, int], ...], float]:
    """Exact synthetic launch score; no future state or packet noise is used."""

    edges = tuple(certificate.stale_radius_by_edge)
    current_gradient = float(hessian[owner] @ (theta - target))
    best = None
    # The executable graph action is null or one causal-cone refresh.  Repeated
    # launches still create a dynamic graph, while the exact minimization costs
    # O(local interaction degree) rather than enumerating every edge subset.
    for size in range(min(1, len(edges)) + 1):
        for selected in combinations(edges, size):
            candidate_cache = cache[owner].copy()
            for donor, recipient in selected:
                if recipient != owner:
                    raise ValueError("candidate edge has the wrong owner")
                candidate_cache[donor] = theta[donor]
            candidate_cache[owner] = theta[owner]
            mean_gradient = float(conditional_row @ (candidate_cache - target))
            learning_score = (
                -certificate.step_cap * current_gradient * mean_gradient
                + 0.5
                * certificate.smoothness
                * certificate.step_cap
                * certificate.step_cap
                * (mean_gradient * mean_gradient + certificate.gradient_variance)
            )
            index = learning_weight * learning_score + message_queue * len(selected)
            row = (float(index), len(selected), tuple(map(repr, selected)), selected)
            if best is None or row[:3] < best[:3]:
                best = row
    assert best is not None
    return best[3], best[0]


def _receipt_step(
    policy: str,
    packet: Packet,
    packet_index: int,
    pending: Sequence[Packet],
    theta: np.ndarray,
    target: np.ndarray,
    hessian: np.ndarray,
) -> float:
    owner = packet.owner
    current_gradient = float(hessian[owner] @ (theta - target))
    linear = 0.0
    curvature = 0.0
    for index, other in enumerate(pending):
        if index == packet_index:
            continue
        other_gradient = float(hessian[other.owner] @ (theta - target))
        radius = abs(other.mean_gradient - other_gradient)
        sensitivity = abs(float(hessian[other.owner, owner]))
        linear += 2.5 * other.step_cap * radius * sensitivity
        curvature += 2.5 * other.step_cap * sensitivity * sensitivity

    if policy in {
        "packet_debt_fixed_step",
        "signed_oracle_graph_h4",
        "no_refresh_full_h4",
        "packet_debt_full_h4",
        "largest_mismatch_h4",
        "active_donor_h4",
        "largest_coefficient_h4",
        "top2_mismatch_h4",
        "top2_coefficient_h4",
        "round_robin_h4",
        "periodic_full_h4",
        "fixed_offset0_h4",
        "fixed_offset1_h4",
        "fixed_offset2_h4",
        "fixed_offset3_h4",
        "fixed_offset4_h4",
    } or policy.startswith("fixed_offsets"):
        return packet.step_cap
    if policy == "packet_debt_oracle_receipt":
        control_mean = packet.mean_gradient
    else:
        control_mean = packet.mean_gradient + sqrt(packet.gradient_variance) * packet.control_standard_normal
    second_moment = control_mean * control_mean + packet.gradient_variance
    alignment = current_gradient * control_mean - linear * sqrt(second_moment)
    denominator = (float(hessian[owner, owner]) + curvature) * second_moment
    if alignment <= 0.0 or denominator <= 0.0:
        return 0.0
    return min(packet.step_cap, alignment / denominator)


def simulate(
    cell: AsyncGameCell,
    policy: str,
    seed: int,
    launches: int = 400,
) -> dict:
    if policy not in POLICIES:
        raise ValueError(f"unknown policy: {policy}")
    agents = 6
    base_target = np.asarray([0.72, -0.43, 0.86, -0.77, 0.31, -0.58], dtype=float)
    base_theta = np.asarray([-0.84, 0.76, -0.63, 0.93, -0.48, 0.82], dtype=float)
    paths = _context_paths(
        seed,
        launches,
        agents,
        cell.mode_switch_probability,
        cell.maximum_extra_delay,
    )
    permutation = paths["agent_permutation"]
    target = base_target[permutation]
    theta = base_theta[permutation] + 0.025 * paths["initial_noise"]
    cache = np.tile(theta, (agents, 1))
    hessian = global_hessian(agents, 0.4, cell.coupling)
    step_cap = 0.04
    learning_weight = sqrt(float(launches))
    resource_queues = {"message": 0.0, "environment": 0.0}
    pending: list[Packet] = []
    next_packet_id = 0
    messages = 0.0
    environment_steps = 0.0
    horizon_counts = {2: 0, 4: 0, 8: 0}
    mode_horizon_sum = {0: 0.0, 1: 0.0}
    mode_counts = {0: 0, 1: 0}
    graph_supports = set()
    round_robin_pointer = np.zeros(agents, dtype=int)
    baseline_token = 0.0
    accepted_steps = 0
    cumulative_objective = 0.0

    def receive_due(event: int, flush: bool = False) -> None:
        nonlocal pending, theta, accepted_steps
        while True:
            due_indices = [
                index
                for index, packet in enumerate(pending)
                if flush or packet.due_event <= event
            ]
            if not due_indices:
                return
            index = min(due_indices, key=lambda value: pending[value].packet_id)
            packet = pending[index]
            alpha = _receipt_step(
                policy, packet, index, pending, theta, target, hessian
            )
            update_gradient = packet.mean_gradient + sqrt(
                packet.gradient_variance
            ) * packet.update_standard_normal
            theta[packet.owner] -= alpha * update_gradient
            cache[packet.owner, packet.owner] = theta[packet.owner]
            if alpha > 0.0:
                accepted_steps += 1
            pending.pop(index)
            if not flush:
                continue

    for event in range(launches):
        receive_due(event)
        cumulative_objective += _objective(theta, target, hessian)
        owner = (event + seed) % agents
        mode = int(paths["modes"][event])
        offset = int(paths["offsets"][event])
        fixed_receipt_step = policy in {
            "packet_debt_fixed_step",
            "packet_debt_oracle_receipt",
            "signed_oracle_graph_h4",
            "no_refresh_full_h4",
            "packet_debt_full_h4",
            "largest_mismatch_h4",
            "active_donor_h4",
            "largest_coefficient_h4",
            "top2_mismatch_h4",
            "top2_coefficient_h4",
            "round_robin_h4",
            "periodic_full_h4",
            "fixed_offset0_h4",
            "fixed_offset1_h4",
            "fixed_offset2_h4",
            "fixed_offset3_h4",
            "fixed_offset4_h4",
        } or policy.startswith("fixed_offsets")
        certificates, rows = _build_certificates(
            owner,
            theta,
            target,
            cache,
            offset,
            mode,
            cell.coupling,
            step_cap,
            1 if fixed_receipt_step else 2,
        )
        item_h4 = {certificate.horizon: certificate for certificate in certificates}[4]
        row_h4 = rows[4]
        if policy == "signed_oracle_graph_h4":
            item = item_h4
            selected, _ = _signed_h4_choice(
                item,
                row_h4,
                owner,
                theta,
                target,
                cache,
                hessian,
                resource_queues["message"],
                learning_weight,
            )
        elif policy == "no_refresh_full_h4":
            item = item_h4
            selected = ()
        elif policy == "packet_debt_full_h4":
            item = item_h4
            selected = choose_horizon_and_graph(
                [item], resource_queues, learning_weight
            ).refreshed_edges
        elif policy == "largest_mismatch_h4":
            item = item_h4
            candidates = donor_order(owner, agents)
            donor = max(
                candidates,
                key=lambda value: (
                    abs(float(row_h4[value]))
                    * abs(float(theta[value] - cache[owner, value])),
                    -value,
                ),
            )
            selected = ((donor, owner),)
        elif policy == "active_donor_h4":
            item = item_h4
            selected = ((donor_order(owner, agents)[offset], owner),)
        elif policy == "largest_coefficient_h4":
            item = item_h4
            donor = max(
                donor_order(owner, agents),
                key=lambda value: (abs(float(row_h4[value])), -value),
            )
            selected = ((donor, owner),)
        elif policy == "top2_mismatch_h4":
            item = item_h4
            candidates = sorted(
                donor_order(owner, agents),
                key=lambda value: (
                    -abs(float(row_h4[value]))
                    * abs(float(theta[value] - cache[owner, value])),
                    value,
                ),
            )
            selected = tuple((donor, owner) for donor in candidates[:2])
        elif policy == "top2_coefficient_h4":
            item = item_h4
            candidates = sorted(
                donor_order(owner, agents),
                key=lambda value: (-abs(float(row_h4[value])), value),
            )
            selected = tuple((donor, owner) for donor in candidates[:2])
        elif policy == "round_robin_h4":
            item = item_h4
            candidates = donor_order(owner, agents)
            donor = candidates[int(round_robin_pointer[owner] % len(candidates))]
            round_robin_pointer[owner] += 1
            selected = ((donor, owner),)
        elif policy == "periodic_full_h4":
            item = item_h4
            candidates = donor_order(owner, agents)
            selected = tuple((donor, owner) for donor in candidates)
        elif (
            policy.startswith("fixed_offset")
            and not policy.startswith("fixed_offsets")
            and policy.endswith("_h4")
        ):
            item = item_h4
            offset_index = int(policy.removeprefix("fixed_offset").removesuffix("_h4"))
            selected = ((donor_order(owner, agents)[offset_index], owner),)
        elif policy.startswith("fixed_offsets") and policy.endswith("_h4"):
            item = item_h4
            indices = policy.removeprefix("fixed_offsets").removesuffix("_h4")
            if len(indices) != 2:
                raise ValueError("fixed-pair policy requires two offset indices")
            selected = tuple(
                (donor_order(owner, agents)[int(index)], owner) for index in indices
            )
        else:
            item, selected, _, _ = _launch_choice(
                policy,
                certificates,
                resource_queues,
                learning_weight,
                mode,
                float(paths["mix_uniform"][event]),
                cell.environment_budget,
            )
        tokenized_baseline = policy in {
            "largest_mismatch_h4",
            "active_donor_h4",
            "largest_coefficient_h4",
            "top2_mismatch_h4",
            "top2_coefficient_h4",
            "round_robin_h4",
            "periodic_full_h4",
        } or policy.startswith("fixed_offset")
        if tokenized_baseline:
            baseline_token = min(5.0, baseline_token + cell.message_budget)
            if len(selected) <= baseline_token + 1e-12:
                baseline_token -= len(selected)
            else:
                selected = ()
        for donor, recipient in selected:
            cache[recipient, donor] = theta[donor]
        horizon = item.horizon
        horizon_counts[horizon] += 1
        mode_horizon_sum[mode] += horizon
        mode_counts[mode] += 1
        graph_supports.add(tuple(sorted(selected)))
        message_cost = float(len(selected))
        environment_cost = float(item.base_cost_by_resource["environment"])
        messages += message_cost
        environment_steps += environment_cost
        resource_queues["message"] = max(
            0.0,
            resource_queues["message"] + message_cost - cell.message_budget,
        )
        resource_queues["environment"] = max(
            0.0,
            resource_queues["environment"]
            + environment_cost
            - cell.environment_budget,
        )

        row = rows[horizon]
        refreshed_cache = cache[owner].copy()
        refreshed_cache[owner] = theta[owner]
        mean_gradient = float(row @ (refreshed_cache - target))
        due = (
            event
            + 1
            + int(paths["extra_delays"][event])
            + horizon // 2
        )
        pending.append(
            Packet(
                packet_id=next_packet_id,
                owner=owner,
                due_event=due,
                mean_gradient=mean_gradient,
                gradient_variance=item.gradient_variance,
                control_standard_normal=float(paths["control_noise"][event]),
                update_standard_normal=float(paths["update_noise"][event]),
                step_cap=step_cap,
            )
        )
        next_packet_id += 1

    receive_due(launches + 10_000, flush=True)
    terminal = _objective(theta, target, hessian)
    return {
        "cell": cell.key,
        "policy": policy,
        "seed": seed,
        "risk": cumulative_objective / launches,
        "terminal_risk": terminal,
        "messages_per_launch": messages / launches,
        "environment_per_launch": environment_steps / launches,
        "message_queue": resource_queues["message"],
        "environment_queue": resource_queues["environment"],
        "accepted_step_fraction": accepted_steps / launches,
        "horizon_counts": horizon_counts,
        "slow_mean_horizon": mode_horizon_sum[0] / max(1, mode_counts[0]),
        "fast_mean_horizon": mode_horizon_sum[1] / max(1, mode_counts[1]),
        "distinct_graph_supports": len(graph_supports),
        "finite": bool(np.isfinite(theta).all() and np.isfinite(terminal)),
    }


def geometric_mean(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=float)
    if np.any(array <= 0.0) or not np.all(np.isfinite(array)):
        raise ValueError("geometric mean requires positive finite values")
    return float(np.exp(np.mean(np.log(array))))

"""Outcome-free Two Clocks integration smoke on pinned HARL/MPE.

This runner exercises only software contracts.  It deliberately emits no
reward, return, win-rate or method-comparison field and is not scientific
evidence.  The neural common step is an empirical interface instantiation, not
a claim that the unconstrained HARL actor satisfies the finite-policy theorem.
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

from .finite_time_drift import single_flight_pathwise_constant_step
from .harl_packet_overlay import diagonal_gaussian_mean_kl
from .run_harl_layer0_packet_smoke import (
    _add_flat_step,
    _collect_trajectory,
    _distribution_parameters,
    _flat_parameters,
    _policy_gradient,
    _service_duration,
)
from .two_clocks_packet_runtime import PacketTicket, PacketWork, TwoClocksPacketLedger


PROHIBITED_OUTCOME_KEYS = frozenset(
    {"reward", "return", "win_rate", "success_rate", "episode_reward"}
)


@dataclass
class _Payload:
    ticket: PacketTicket
    birth_policies: list[Any]
    reference_observations: list[np.ndarray]
    packet_direction: np.ndarray


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _assert_no_outcome_keys(value: object, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).lower()
            if normalized in PROHIBITED_OUTCOME_KEYS or normalized.endswith("_return"):
                raise RuntimeError(f"outcome field is prohibited in G0 output: {path}.{key}")
            _assert_no_outcome_keys(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_no_outcome_keys(child, f"{path}[{index}]")


def validate_summary(summary: dict[str, Any]) -> None:
    _assert_no_outcome_keys(summary)
    if summary.get("scope") != "outcome-free Two Clocks Layer-0 G0":
        raise RuntimeError("unexpected G0 scope")
    if summary.get("scientific_outcome_generated") is not False:
        raise RuntimeError("G0 must explicitly exclude scientific outcomes")
    invariants = summary.get("invariants")
    if not isinstance(invariants, dict) or not all(invariants.values()):
        raise RuntimeError("not every G0 invariant passed")
    accounting = summary.get("accounting")
    if not isinstance(accounting, dict):
        raise RuntimeError("missing G0 accounting")
    if accounting["completed_packets"] != accounting["applied_updates"]:
        raise RuntimeError("completed and applied packet counts differ")
    if accounting["cancelled_packets"] != 0:
        raise RuntimeError("Layer-0 primary smoke unexpectedly cancelled work")


def _checked_harl_root(path: str, expected_commit: str) -> Path:
    root = Path(path).resolve()
    if not (root / "harl").is_dir():
        raise FileNotFoundError("--harl-root does not contain the HARL package")
    actual = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual != expected_commit:
        raise RuntimeError(f"unexpected HARL commit {actual}; expected {expected_commit}")
    if subprocess.run(
        ["git", "-C", str(root), "status", "--short"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip():
        raise RuntimeError("pinned HARL checkout is not clean")
    return root


def _default_cross_lipschitz(num_agents: int) -> np.ndarray:
    matrix = np.full((num_agents, num_agents), 0.5, dtype=np.float64)
    np.fill_diagonal(matrix, 5.0)
    return matrix


def run(args: argparse.Namespace) -> dict[str, Any]:
    harl_root = _checked_harl_root(args.harl_root, args.harl_commit)
    if str(harl_root) not in sys.path:
        sys.path.insert(0, str(harl_root))
    import pettingzoo
    import supersuit
    import torch
    import yaml
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

    cross_lipschitz = _default_cross_lipschitz(num_agents)
    certificate = single_flight_pathwise_constant_step(
        cross_lipschitz,
        maximum_delay=args.maximum_delay,
        history_inflation=args.history_inflation,
    )
    certified_common_step = float(certificate["step_size"])
    applied_common_step = args.step_fraction * certified_common_step
    if not (0.0 < args.step_fraction <= 1.0):
        raise ValueError("step_fraction must belong to (0,1]")

    parameter_count = sum(parameter.numel() for parameter in policies[0].parameters())
    declared_work = PacketWork(
        environment_steps=args.episode_length,
        actor_transitions=args.episode_length * num_agents,
        optimizer_units=parameter_count,
    )
    ledger = TwoClocksPacketLedger(num_agents)
    queue: list[tuple[float, int, _Payload]] = []
    launched = 0
    logical_time = 0.0
    records: list[dict[str, Any]] = []

    def launch(owner: int) -> None:
        nonlocal launched
        service_time = _service_duration(owner, launched)
        completion_time = logical_time + service_time
        ticket = ledger.launch(
            owner,
            launch_time=logical_time,
            scheduled_completion_time=completion_time,
            declared_work=declared_work,
        )
        birth_policies = [copy.deepcopy(policy) for policy in policies]
        trajectory = _collect_trajectory(
            environment_class=PettingZooMPEEnv,
            policies=birth_policies,
            target_agent=owner,
            seed=args.seed + 10_000 + ticket.ticket_id,
            episode_length=args.episode_length,
        )
        if trajectory.steps != args.episode_length:
            raise RuntimeError("G0 packet did not execute its fixed declared horizon")
        direction = _policy_gradient(
            birth_policies[owner], trajectory, gamma=args.gamma
        )
        norm = float(np.linalg.norm(direction))
        if not math.isfinite(norm):
            raise RuntimeError("non-finite packet direction in G0")
        if norm > args.maximum_packet_norm:
            direction *= args.maximum_packet_norm / norm
        payload = _Payload(
            ticket=ticket,
            birth_policies=birth_policies,
            reference_observations=trajectory.observations,
            packet_direction=direction,
        )
        heapq.heappush(queue, (completion_time, ticket.ticket_id, payload))
        launched += 1

    for owner in range(num_agents):
        launch(owner)

    while queue:
        completion_time, _, payload = heapq.heappop(queue)
        logical_time = completion_time
        ticket = payload.ticket
        completion = ledger.complete(
            ticket.owner,
            ticket_id=ticket.ticket_id,
            completion_time=completion_time,
        )
        owner_parameter_error = float(
            np.max(
                np.abs(
                    _flat_parameters(policies[ticket.owner])
                    - _flat_parameters(payload.birth_policies[ticket.owner])
                )
            )
        )
        if owner_parameter_error > 1e-10:
            raise RuntimeError("owner parameters changed while its packet was in flight")
        teammate_kls: list[float] = []
        for teammate in range(num_agents):
            if teammate == ticket.owner:
                continue
            birth_mean, birth_log_std = _distribution_parameters(
                payload.birth_policies[teammate],
                payload.reference_observations[teammate],
            )
            current_mean, current_log_std = _distribution_parameters(
                policies[teammate],
                payload.reference_observations[teammate],
            )
            teammate_kls.append(
                diagonal_gaussian_mean_kl(
                    birth_mean,
                    birth_log_std,
                    current_mean,
                    current_log_std,
                )
            )
        applied_step = applied_common_step * payload.packet_direction
        _add_flat_step(policies[ticket.owner], applied_step)
        ledger.apply(ticket.owner, ticket_id=ticket.ticket_id)
        if completion.event_delay > args.maximum_delay:
            raise RuntimeError("realized event delay exceeded the declared G0 bound")
        records.append(
            {
                "ticket_id": ticket.ticket_id,
                "owner": ticket.owner,
                "birth_event": completion.birth_event,
                "completion_event": completion.completion_event,
                "event_delay": completion.event_delay,
                "birth_versions": completion.birth_versions,
                "arrival_versions": completion.arrival_versions,
                "teammate_version_increments": completion.teammate_version_increments,
                "owner_parameter_error": owner_parameter_error,
                "teammate_kl_sum": float(sum(teammate_kls)),
                "packet_direction_norm": float(np.linalg.norm(payload.packet_direction)),
                "applied_step_norm": float(np.linalg.norm(applied_step)),
                "completion_time": completion_time,
            }
        )
        if launched < args.packets:
            launch(ticket.owner)

    ledger.assert_quiescent()
    accounting = ledger.accounting()
    expected_work = (
        args.packets * declared_work.environment_steps,
        args.packets * declared_work.actor_transitions,
        args.packets * declared_work.optimizer_units,
    )
    invariants = {
        "pinned_upstream_clean": True,
        "single_writer_versions": sum(ledger.versions) == args.packets,
        "single_flight_teardown": not ledger.active_owners and not ledger.pending_owners,
        "owner_parameter_self_fresh": max(
            record["owner_parameter_error"] for record in records
        )
        <= 1e-10,
        "all_packets_completed_and_applied": accounting["completed_packets"]
        == accounting["applied_updates"]
        == args.packets,
        "exact_completed_work": accounting["total_charged_work"] == expected_work,
        "no_cancelled_work": accounting["cancelled_packets"] == 0,
        "bounded_event_delay": max(record["event_delay"] for record in records)
        <= args.maximum_delay,
        "common_step_condition": float(np.max(certificate["conditions"]))
        <= 1.0 + 1e-12,
        "strategic_staleness_exercised": any(
            sum(record["teammate_version_increments"]) > 0 for record in records
        ),
        "finite_diagnostics": all(
            math.isfinite(float(record[key]))
            for record in records
            for key in (
                "owner_parameter_error",
                "teammate_kl_sum",
                "packet_direction_norm",
                "applied_step_norm",
                "completion_time",
            )
        ),
    }
    summary: dict[str, Any] = {
        "scope": "outcome-free Two Clocks Layer-0 G0",
        "scientific_outcome_generated": False,
        "neural_certificate_status": "empirical_interface_only",
        "seed": args.seed,
        "harl_commit": args.harl_commit,
        "scenario_shape": "simple_spread_v2-continuous-three-distinct-actors",
        "runtime": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "torch": torch.__version__,
            "pettingzoo": getattr(pettingzoo, "__version__", "unknown"),
            "supersuit": getattr(supersuit, "__version__", "unknown"),
        },
        "common_step": {
            "declared_cross_lipschitz": cross_lipschitz.tolist(),
            "maximum_delay": args.maximum_delay,
            "history_inflation": args.history_inflation,
            "certified_finite_policy_step": certified_common_step,
            "empirical_neural_step_fraction": args.step_fraction,
            "applied_common_step": applied_common_step,
            "conditions": np.asarray(certificate["conditions"]).tolist(),
        },
        "packets": args.packets,
        "final_versions": ledger.versions,
        "logical_service_time": logical_time,
        "accounting": accounting,
        "invariants": invariants,
        "records": records,
        "ledger_trace": ledger.trace,
    }
    validate_summary(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--harl-root", required=True)
    parser.add_argument(
        "--harl-commit", default="b1af98b0dbab72a2eee9d160751cd09aedbb8ce2"
    )
    parser.add_argument("--seed", type=int, default=27101)
    parser.add_argument("--packets", type=int, default=12)
    parser.add_argument("--episode-length", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--maximum-delay", type=int, default=8)
    parser.add_argument("--history-inflation", type=float, default=1.0)
    parser.add_argument("--step-fraction", type=float, default=0.25)
    parser.add_argument("--maximum-packet-norm", type=float, default=0.05)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.packets < 3 or args.episode_length <= 0 or args.maximum_delay < 0:
        raise ValueError("packet, horizon or delay argument is invalid")
    if args.maximum_packet_norm <= 0.0 or not math.isfinite(args.maximum_packet_norm):
        raise ValueError("maximum_packet_norm must be finite and positive")
    summary = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key not in {"records", "ledger_trace"}},
            indent=2,
            sort_keys=True,
        )
    )
    print(f"output_sha256={_sha256(args.output)}")


if __name__ == "__main__":
    main()

"""Outcome-free integration gate for the public-MPE Two Clocks bridge.

This gate exercises the four learning schedules on pinned HARL actors but does
not evaluate or emit any learning outcome.  It is allowed to reject an
implementation; it cannot authorize or tune a scientific pilot.
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
from .mpe_bridge_contract import (
    METHODS,
    TASK_AGENTS,
    async_completed_packets,
    barrier_schedule,
    service_bases,
    trajectory_seed,
)
from .run_harl_layer0_development_comparison import (
    _collect_on_reused_environment,
    _policy_gradient_with_frozen_baseline,
)
from .run_harl_layer0_packet_smoke import (
    _add_flat_step,
    _discounted_returns,
    _distribution_parameters,
    _flat_parameters,
)
from .two_clocks_packet_runtime import PacketTicket, PacketWork, TwoClocksPacketLedger


PROHIBITED_OUTPUT_TOKENS = (
    "reward",
    "return",
    "win_rate",
    "success_rate",
    "advantage",
    "gradient",
    "loss",
)


@dataclass
class _Payload:
    ticket: PacketTicket
    owner_index: int
    birth_policies: list[Any] | None
    reference_observations: np.ndarray | None
    direction: np.ndarray | None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_outcome_free(value: object, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).lower()
            if any(token in normalized for token in PROHIBITED_OUTPUT_TOKENS):
                raise RuntimeError(f"outcome-bearing G0 key at {path}.{key}")
            _assert_outcome_free(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_outcome_free(child, f"{path}[{index}]")


def validate_summary(summary: dict[str, Any]) -> None:
    _assert_outcome_free(summary)
    if summary.get("scope") != "outcome-free Two Clocks public-MPE bridge G0":
        raise RuntimeError("unexpected G0 scope")
    if summary.get("scientific_outcome_generated") is not False:
        raise RuntimeError("G0 generated a scientific outcome")
    invariants = summary.get("invariants")
    if not isinstance(invariants, dict) or not all(invariants.values()):
        raise RuntimeError("a mandatory G0 invariant failed")


def _checked_harl_root(path: str, expected_commit: str) -> Path:
    root = Path(path).resolve()
    if not (root / "harl").is_dir():
        raise FileNotFoundError("HARL root does not contain the harl package")
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
        raise RuntimeError("pinned HARL checkout has the wrong revision or is dirty")
    return root


def _policy_digest(policies: list[Any]) -> str:
    digest = hashlib.sha256()
    for policy in policies:
        digest.update(_flat_parameters(policy).tobytes())
    return digest.hexdigest()


def _clip_step(direction: np.ndarray, learning_rate: float, maximum_norm: float) -> np.ndarray:
    step = learning_rate * np.asarray(direction, dtype=np.float64)
    norm = float(np.linalg.norm(step))
    if not math.isfinite(norm):
        raise RuntimeError("non-finite update direction")
    if norm > maximum_norm:
        step *= maximum_norm / norm
    return step


def _declared_cross_lipschitz(agents: int) -> np.ndarray:
    matrix = np.full((agents, agents), 0.5, dtype=np.float64)
    np.fill_diagonal(matrix, 5.0)
    return matrix


def _build_case(
    *,
    specification: dict[str, Any],
    task: str,
    method: str,
    profile: str,
    harl_root: Path,
) -> dict[str, Any]:
    import torch
    import yaml
    from harl.envs.pettingzoo_mpe.pettingzoo_mpe_env import PettingZooMPEEnv
    from harl.models.policy_models.stochastic_policy import StochasticPolicy

    seed = int(specification["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    environment = PettingZooMPEEnv(
        {
            "scenario": task,
            "continuous_actions": True,
            "max_cycles": int(specification["episode_length"]),
        }
    )
    agents = environment.n_agents
    if agents != TASK_AGENTS[task]:
        raise RuntimeError("public task agent count changed")
    with (harl_root / "harl" / "configs" / "algos_cfgs" / "haa2c.yaml").open(
        "r", encoding="utf-8"
    ) as handle:
        upstream = yaml.safe_load(handle)
    policy_args: dict[str, Any] = {}
    for section in ("model", "algo"):
        policy_args.update(upstream[section])
    policy_args["hidden_sizes"] = list(specification["hidden_sizes"])
    policies = [
        StochasticPolicy(
            policy_args, environment.observation_space[index], environment.action_space[index]
        )
        for index in range(agents)
    ]
    initial_digest = _policy_digest(policies)

    baseline_samples = []
    for index in range(int(specification["baseline_episodes"])):
        trajectory = _collect_on_reused_environment(
            environment=environment,
            policies=policies,
            target_agent=0,
            seed=trajectory_seed(
                seed + 700_000,
                task=task,
                owner=0,
                owner_packet_index=index,
                replicate=0,
            ),
            episode_length=int(specification["episode_length"]),
        )
        baseline_samples.append(
            _discounted_returns(trajectory.target_rewards, float(specification["gamma"]))
        )
    frozen_baseline = np.mean(np.asarray(baseline_samples), axis=0)
    frozen_digest = hashlib.sha256(frozen_baseline.tobytes()).hexdigest()
    bases = service_bases(task, profile)
    cross_lipschitz = _declared_cross_lipschitz(agents)
    certificate = single_flight_pathwise_constant_step(
        cross_lipschitz,
        maximum_delay=int(specification["declared_maximum_event_delay"]),
    )
    # Normalize the certified common step by the diagonal-only 1/L step.  This
    # makes the neural adapter inherit the theorem's off-diagonal shrinkage
    # while leaving the absolute neural learning-rate unit explicit.
    diagonal_only_step = 1.0 / float(np.max(np.diag(cross_lipschitz)))
    offdiag_scale = float(certificate["step_size"]) / diagonal_only_step
    if not 0.0 < offdiag_scale <= 1.0:
        raise RuntimeError("invalid off-diagonal Lyapunov scale")
    parameter_count = sum(parameter.numel() for parameter in policies[0].parameters())
    packet_work = PacketWork(
        environment_steps=2 * int(specification["episode_length"]),
        actor_transitions=2 * int(specification["episode_length"]) * agents,
        optimizer_units=parameter_count,
    )
    finite_motion = True
    positive_motion_events = 0
    owner_self_fresh = True

    def packet_payload(
        owner: int, owner_index: int, birth_policies: list[Any]
    ) -> tuple[np.ndarray, np.ndarray]:
        trajectories = [
            _collect_on_reused_environment(
                environment=environment,
                policies=birth_policies,
                target_agent=owner,
                seed=trajectory_seed(
                    seed,
                    task=task,
                    owner=owner,
                    owner_packet_index=owner_index,
                    replicate=replicate,
                ),
                episode_length=int(specification["episode_length"]),
            )
            for replicate in (0, 1)
        ]
        directions = [
            _policy_gradient_with_frozen_baseline(
                birth_policies[owner],
                trajectory,
                gamma=float(specification["gamma"]),
                frozen_baseline=frozen_baseline,
            )
            for trajectory in trajectories
        ]
        return 0.5 * (directions[0] + directions[1]), trajectories[0].target_observations

    completed_packets = 0
    applied_updates = 0
    cancelled_environment_steps = 0
    cancelled_actor_transitions = 0
    logical_time = float(specification["service_horizon"])
    maximum_event_delay = 0

    if method != "frozen_barrier":
        ledger = TwoClocksPacketLedger(agents)
        queue: list[tuple[float, int, _Payload]] = []
        owner_indices = [0 for _ in range(agents)]
        horizon = float(specification["service_horizon"])

        def launch(owner: int, launch_time: float) -> None:
            completion_time = launch_time + bases[owner]
            ticket = ledger.launch(
                owner,
                launch_time=launch_time,
                scheduled_completion_time=completion_time,
                declared_work=packet_work,
            )
            owner_index = owner_indices[owner]
            owner_indices[owner] += 1
            if completion_time <= horizon + 1e-12:
                birth = [copy.deepcopy(policy) for policy in policies]
                direction, reference = packet_payload(owner, owner_index, birth)
            else:
                birth, direction, reference = None, None, None
            heapq.heappush(
                queue,
                (
                    completion_time,
                    ticket.ticket_id,
                    _Payload(ticket, owner_index, birth, reference, direction),
                ),
            )

        for owner in range(agents):
            launch(owner, 0.0)
        while queue and queue[0][0] <= horizon + 1e-12:
            completion_time, _, payload = heapq.heappop(queue)
            ticket = payload.ticket
            completion = ledger.complete(
                ticket.owner,
                ticket_id=ticket.ticket_id,
                completion_time=completion_time,
            )
            maximum_event_delay = max(maximum_event_delay, completion.event_delay)
            assert payload.birth_policies is not None
            assert payload.reference_observations is not None
            assert payload.direction is not None
            error = float(
                np.max(
                    np.abs(
                        _flat_parameters(policies[ticket.owner])
                        - _flat_parameters(payload.birth_policies[ticket.owner])
                    )
                )
            )
            owner_self_fresh = owner_self_fresh and error <= 1e-10
            before_mean, before_log_std = _distribution_parameters(
                policies[ticket.owner], payload.reference_observations
            )
            base_step = _clip_step(
                payload.direction,
                float(specification["learning_rate"]),
                float(specification["maximum_step_norm"]),
            )
            if method == "offdiag_async":
                scale = offdiag_scale
            elif method == "delay_scaled_async":
                scale = 1.0 / (1.0 + completion.event_delay)
            else:
                scale = 1.0
            _add_flat_step(policies[ticket.owner], scale * base_step)
            after_mean, after_log_std = _distribution_parameters(
                policies[ticket.owner], payload.reference_observations
            )
            motion = diagonal_gaussian_mean_kl(
                before_mean, before_log_std, after_mean, after_log_std
            )
            finite_motion = finite_motion and math.isfinite(motion) and motion >= 0.0
            positive_motion_events += int(motion > 0.0)
            ledger.apply(ticket.owner, ticket_id=ticket.ticket_id)
            if completion_time < horizon - 1e-12:
                launch(ticket.owner, completion_time)
        for _, _, payload in queue:
            ticket = payload.ticket
            fraction = max(
                0.0,
                min(
                    1.0,
                    (horizon - ticket.launch_time)
                    / (ticket.scheduled_completion_time - ticket.launch_time),
                ),
            )
            charged_environment = int(math.floor(packet_work.environment_steps * fraction))
            charged_actor = charged_environment * agents
            ledger.cancel(
                ticket.owner,
                ticket_id=ticket.ticket_id,
                charged_work=PacketWork(charged_environment, charged_actor, 0),
                reason="service-horizon tail",
            )
        ledger.assert_quiescent()
        accounting = ledger.accounting()
        completed_packets = int(accounting["completed_packets"])
        applied_updates = int(accounting["applied_updates"])
        cancelled = accounting["cancelled_work"]
        cancelled_environment_steps = int(cancelled[0])
        cancelled_actor_transitions = int(cancelled[1])
        expected = sum(async_completed_packets(bases, horizon))
        schedule_exact = completed_packets == expected
    else:
        schedule = barrier_schedule(
            bases,
            float(specification["service_horizon"]),
            int(specification["episode_length"]),
        )
        total_by_owner = [0 for _ in range(agents)]
        rounds = int(schedule["rounds"])
        round_duration = float(schedule["round_length"])
        for _round in range(rounds):
            birth = [copy.deepcopy(policy) for policy in policies]
            for owner in range(agents):
                count = int(math.floor((round_duration + 1e-12) / bases[owner]))
                directions = []
                references = []
                for _ in range(count):
                    direction, reference = packet_payload(owner, total_by_owner[owner], birth)
                    total_by_owner[owner] += 1
                    directions.append(direction)
                    references.append(reference)
                if not directions:
                    continue
                error = float(
                    np.max(
                        np.abs(_flat_parameters(policies[owner]) - _flat_parameters(birth[owner]))
                    )
                )
                owner_self_fresh = owner_self_fresh and error <= 1e-10
                before_mean, before_log_std = _distribution_parameters(
                    policies[owner], references[0]
                )
                step = _clip_step(
                    np.mean(np.asarray(directions), axis=0),
                    float(specification["learning_rate"]),
                    float(specification["maximum_step_norm"]),
                )
                _add_flat_step(policies[owner], step)
                after_mean, after_log_std = _distribution_parameters(
                    policies[owner], references[0]
                )
                motion = diagonal_gaussian_mean_kl(
                    before_mean, before_log_std, after_mean, after_log_std
                )
                finite_motion = finite_motion and math.isfinite(motion) and motion >= 0.0
                positive_motion_events += int(motion > 0.0)
                applied_updates += 1
        completed_packets = sum(total_by_owner)
        cancelled_steps = tuple(schedule["cancelled_environment_steps_by_owner"])
        cancelled_environment_steps = sum(cancelled_steps)
        cancelled_actor_transitions = cancelled_environment_steps * agents
        schedule_exact = tuple(total_by_owner) == tuple(schedule["completed_by_owner"])

    environment.close()
    completed_environment_steps = (
        completed_packets * 2 * int(specification["episode_length"])
    )
    result = {
        "task": task,
        "method": method,
        "profile": profile,
        "agents": agents,
        "initial_policy_digest": initial_digest,
        "frozen_control_variate_digest": frozen_digest,
        "completed_packets": completed_packets,
        "applied_updates": applied_updates,
        "completed_environment_steps": completed_environment_steps,
        "completed_actor_transitions": completed_environment_steps * agents,
        "cancelled_environment_steps": cancelled_environment_steps,
        "cancelled_actor_transitions": cancelled_actor_transitions,
        "logical_service_time": logical_time,
        "maximum_event_delay": maximum_event_delay,
        "declared_offdiag_lyapunov_scale": offdiag_scale,
        "invariants": {
            "schedule_exact": schedule_exact,
            "owner_self_fresh_or_frozen_round": owner_self_fresh,
            "finite_policy_motion_diagnostic": finite_motion,
            "declared_lyapunov_condition": float(np.max(certificate["conditions"]))
            <= 1.0 + 1e-12,
            "policy_motion_exercised": positive_motion_events > 0,
            "all_complete_packets_applied_or_batched": (
                applied_updates == completed_packets
                if method != "frozen_barrier"
                else 0 < applied_updates <= completed_packets
            ),
        },
    }
    return result


def run(specification: dict[str, Any]) -> dict[str, Any]:
    harl_root = _checked_harl_root(
        str(specification["harl_root"]), str(specification["harl_commit"])
    )
    if str(harl_root) not in sys.path:
        sys.path.insert(0, str(harl_root))
    import numpy
    import pettingzoo
    import supersuit
    import torch

    torch.set_num_threads(1)
    rows = [
        _build_case(
            specification=specification,
            task=task,
            method=method,
            profile=profile,
            harl_root=harl_root,
        )
        for task in TASK_AGENTS
        for profile in ("balanced", "heterogeneous")
        for method in METHODS
    ]
    initial_pairing = all(
        len(
            {
                row["initial_policy_digest"]
                for row in rows
                if row["task"] == task
            }
        )
        == 1
        for task in TASK_AGENTS
    )
    control_variate_pairing = all(
        len(
            {
                row["frozen_control_variate_digest"]
                for row in rows
                if row["task"] == task
            }
        )
        == 1
        for task in TASK_AGENTS
    )
    summary = {
        "scope": "outcome-free Two Clocks public-MPE bridge G0",
        "scientific_outcome_generated": False,
        "neural_certificate_status": "empirical_interface_only",
        "harl_commit": specification["harl_commit"],
        "runtime": {
            "python": sys.version.split()[0],
            "numpy": numpy.__version__,
            "torch": torch.__version__,
            "pettingzoo": getattr(pettingzoo, "__version__", "unknown"),
            "supersuit": getattr(supersuit, "__version__", "unknown"),
        },
        "declared_design": {
            "tasks": list(TASK_AGENTS),
            "methods": list(METHODS),
            "profiles": ["balanced", "heterogeneous"],
            "seed": specification["seed"],
            "episode_length": specification["episode_length"],
            "service_horizon": specification["service_horizon"],
            "two_independent_trajectories_per_complete_packet": True,
            "cancelled_tail_work_is_charged": True,
        },
        "invariants": {
            "pinned_harl_clean": True,
            "all_task_shapes_match": all(
                row["agents"] == TASK_AGENTS[row["task"]] for row in rows
            ),
            "paired_initial_policies": initial_pairing,
            "paired_frozen_control_variate": control_variate_pairing,
            "all_case_invariants": all(
                all(row["invariants"].values()) for row in rows
            ),
            "all_methods_and_tasks_exercised": len(rows)
            == len(TASK_AGENTS) * 2 * len(METHODS),
            "tail_work_visible": any(
                row["cancelled_environment_steps"] > 0 for row in rows
            ),
        },
        "cases": rows,
    }
    validate_summary(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    specification = json.loads(args.config.read_text(encoding="utf-8"))
    summary = run(specification)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["invariants"], indent=2, sort_keys=True))
    print(f"output_sha256={_sha256(args.output)}")


if __name__ == "__main__":
    main()

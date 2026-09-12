"""Frozen-seed CPU pilot for the Two Clocks public-MPE bridge.

The runner uses distinct HARL policy blocks and a past-data, action-independent
control variate.  It records learning curves at common logical service times
and separates completed work, cancelled tail work, optimizer updates, policy
motion, and evaluation work.
"""

from __future__ import annotations

import argparse
import copy
import csv
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
import time
from typing import Any

import numpy as np

from .finite_time_drift import single_flight_pathwise_constant_step
from .harl_packet_overlay import diagonal_gaussian_mean_kl
from .mpe_bridge_contract import METHODS, TASK_AGENTS, barrier_schedule, service_bases, trajectory_seed
from .run_harl_layer0_development_comparison import (
    _collect_on_reused_environment,
    _evaluate_on_reused_environment,
    _policy_gradient_with_frozen_baseline,
)
from .run_harl_layer0_packet_smoke import (
    _add_flat_step,
    _discounted_returns,
    _distribution_parameters,
    _flat_parameters,
)
from .two_clocks_packet_runtime import PacketTicket, PacketWork, TwoClocksPacketLedger


@dataclass
class _Payload:
    ticket: PacketTicket
    owner_index: int
    birth_policies: list[Any] | None
    reference_observations: list[np.ndarray] | None
    direction: np.ndarray | None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checked_harl_root(path: str, expected_commit: str) -> Path:
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
        raise RuntimeError("HARL checkout is not the pinned clean revision")
    return root


def _clip_step(direction: np.ndarray, learning_rate: float, maximum_norm: float) -> np.ndarray:
    step = learning_rate * np.asarray(direction, dtype=np.float64)
    norm = float(np.linalg.norm(step))
    if not math.isfinite(norm):
        raise RuntimeError("non-finite update")
    if norm > maximum_norm:
        step *= maximum_norm / norm
    return step


def _cross_lipschitz(agents: int, diagonal: float, off_diagonal: float) -> np.ndarray:
    matrix = np.full((agents, agents), off_diagonal, dtype=np.float64)
    np.fill_diagonal(matrix, diagonal)
    return matrix


def _run_one(spec: dict[str, Any]) -> dict[str, Any]:
    harl_root = _checked_harl_root(spec["harl_root"], spec["harl_commit"])
    if str(harl_root) not in sys.path:
        sys.path.insert(0, str(harl_root))
    import torch
    import yaml
    from harl.envs.pettingzoo_mpe.pettingzoo_mpe_env import PettingZooMPEEnv
    from harl.models.policy_models.stochastic_policy import StochasticPolicy

    seed = int(spec["seed"])
    task = str(spec["task"])
    profile = str(spec["profile"])
    method = str(spec["method"])
    episode_length = int(spec["episode_length"])
    horizon = float(spec["service_horizon"])
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    environment = PettingZooMPEEnv(
        {"scenario": task, "continuous_actions": True, "max_cycles": episode_length}
    )
    agents = environment.n_agents
    if agents != TASK_AGENTS[task]:
        raise RuntimeError("task shape changed")
    with (harl_root / "harl" / "configs" / "algos_cfgs" / "haa2c.yaml").open(
        "r", encoding="utf-8"
    ) as handle:
        upstream = yaml.safe_load(handle)
    policy_args: dict[str, Any] = {}
    for section in ("model", "algo"):
        policy_args.update(upstream[section])
    policy_args["hidden_sizes"] = list(spec["hidden_sizes"])
    policies = [
        StochasticPolicy(policy_args, environment.observation_space[i], environment.action_space[i])
        for i in range(agents)
    ]
    initial_policy_digest = hashlib.sha256(
        b"".join(_flat_parameters(policy).tobytes() for policy in policies)
    ).hexdigest()
    evaluation_seeds = [
        trajectory_seed(
            seed + 900_000,
            task=task,
            owner=0,
            owner_packet_index=index,
            replicate=0,
        )
        for index in range(int(spec["evaluation_episodes"]))
    ]

    baseline_samples = []
    for index in range(int(spec["baseline_episodes"])):
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
            episode_length=episode_length,
        )
        baseline_samples.append(_discounted_returns(trajectory.target_rewards, float(spec["gamma"])))
    frozen_baseline = np.mean(np.asarray(baseline_samples), axis=0)
    frozen_digest = hashlib.sha256(frozen_baseline.tobytes()).hexdigest()

    bases = service_bases(task, profile)
    cross = _cross_lipschitz(
        agents,
        float(spec["cross_lipschitz_diagonal"]),
        float(spec["cross_lipschitz_off_diagonal"]),
    )
    certificate = single_flight_pathwise_constant_step(
        cross, maximum_delay=int(spec["maximum_event_delay"])
    )
    diagonal_step = 1.0 / float(np.max(np.diag(cross)))
    offdiag_scale = float(certificate["step_size"]) / diagonal_step
    if not 0.0 < offdiag_scale <= 1.0:
        raise RuntimeError("invalid Lyapunov scale")

    parameter_count = sum(parameter.numel() for parameter in policies[0].parameters())
    packet_work = PacketWork(
        2 * episode_length,
        2 * episode_length * agents,
        parameter_count,
    )
    completed_packets = 0
    cancelled_environment_steps = 0
    cancelled_actor_transitions = 0
    optimizer_updates = 0
    cumulative_step_norm = 0.0
    cumulative_policy_kl = 0.0
    cumulative_teammate_birth_arrival_kl = 0.0
    maximum_owner_error = 0.0
    maximum_event_delay = 0
    owner_indices = [0 for _ in range(agents)]
    curve: list[dict[str, Any]] = []
    checkpoints = [horizon * float(value) for value in spec["checkpoint_fractions"]]

    def evaluate(checkpoint: float) -> None:
        score = _evaluate_on_reused_environment(
            environment=environment,
            policies=policies,
            seeds=evaluation_seeds,
            episode_length=episode_length,
        )
        curve.append(
            {
                "seed": seed,
                "task": task,
                "profile": profile,
                "method": method,
                "logical_time": checkpoint,
                "evaluation_return": score,
            }
        )

    def collect_payload(owner: int, index: int, birth: list[Any]) -> tuple[np.ndarray, list[np.ndarray]]:
        trajectories = [
            _collect_on_reused_environment(
                environment=environment,
                policies=birth,
                target_agent=owner,
                seed=trajectory_seed(
                    seed,
                    task=task,
                    owner=owner,
                    owner_packet_index=index,
                    replicate=replicate,
                ),
                episode_length=episode_length,
            )
            for replicate in (0, 1)
        ]
        directions = [
            _policy_gradient_with_frozen_baseline(
                birth[owner], trajectory, gamma=float(spec["gamma"]), frozen_baseline=frozen_baseline
            )
            for trajectory in trajectories
        ]
        return 0.5 * (directions[0] + directions[1]), trajectories[0].observations

    evaluate(checkpoints[0])
    if method != "frozen_barrier":
        ledger = TwoClocksPacketLedger(agents)
        queue: list[tuple[float, int, _Payload]] = []

        def launch(owner: int, launch_time: float) -> None:
            completion_time = launch_time + bases[owner]
            ticket = ledger.launch(
                owner,
                launch_time=launch_time,
                scheduled_completion_time=completion_time,
                declared_work=packet_work,
            )
            index = owner_indices[owner]
            owner_indices[owner] += 1
            if completion_time <= horizon + 1e-12:
                birth = [copy.deepcopy(policy) for policy in policies]
                direction, observations = collect_payload(owner, index, birth)
            else:
                birth, direction, observations = None, None, None
            heapq.heappush(
                queue,
                (completion_time, ticket.ticket_id, _Payload(ticket, index, birth, observations, direction)),
            )

        for owner in range(agents):
            launch(owner, 0.0)

        for checkpoint in checkpoints[1:]:
            while queue and queue[0][0] <= checkpoint + 1e-12:
                completion_time, _, payload = heapq.heappop(queue)
                ticket = payload.ticket
                completion = ledger.complete(
                    ticket.owner, ticket_id=ticket.ticket_id, completion_time=completion_time
                )
                assert payload.birth_policies is not None
                assert payload.reference_observations is not None
                assert payload.direction is not None
                owner_error = float(
                    np.max(
                        np.abs(
                            _flat_parameters(policies[ticket.owner])
                            - _flat_parameters(payload.birth_policies[ticket.owner])
                        )
                    )
                )
                maximum_owner_error = max(maximum_owner_error, owner_error)
                maximum_event_delay = max(maximum_event_delay, completion.event_delay)
                teammate_kl = 0.0
                for teammate in range(agents):
                    if teammate == ticket.owner:
                        continue
                    old_mean, old_log_std = _distribution_parameters(
                        payload.birth_policies[teammate], payload.reference_observations[teammate]
                    )
                    new_mean, new_log_std = _distribution_parameters(
                        policies[teammate], payload.reference_observations[teammate]
                    )
                    teammate_kl += diagonal_gaussian_mean_kl(
                        old_mean, old_log_std, new_mean, new_log_std
                    )
                reference = payload.reference_observations[ticket.owner]
                old_mean, old_log_std = _distribution_parameters(policies[ticket.owner], reference)
                base_step = _clip_step(
                    payload.direction,
                    float(spec["learning_rate"]),
                    float(spec["maximum_step_norm"]),
                )
                if method == "offdiag_async":
                    scale = offdiag_scale
                elif method == "delay_scaled_async":
                    scale = 1.0 / (1.0 + completion.event_delay)
                else:
                    scale = 1.0
                applied_step = scale * base_step
                _add_flat_step(policies[ticket.owner], applied_step)
                new_mean, new_log_std = _distribution_parameters(policies[ticket.owner], reference)
                motion = diagonal_gaussian_mean_kl(old_mean, old_log_std, new_mean, new_log_std)
                if not math.isfinite(motion):
                    raise RuntimeError("non-finite policy motion")
                cumulative_step_norm += float(np.linalg.norm(applied_step))
                cumulative_policy_kl += motion
                cumulative_teammate_birth_arrival_kl += teammate_kl
                ledger.apply(ticket.owner, ticket_id=ticket.ticket_id)
                optimizer_updates += 1
                if completion_time < horizon - 1e-12:
                    launch(ticket.owner, completion_time)
            evaluate(checkpoint)
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
            cancelled_environment = int(math.floor(packet_work.environment_steps * fraction))
            cancelled_actor = cancelled_environment * agents
            ledger.cancel(
                ticket.owner,
                ticket_id=ticket.ticket_id,
                charged_work=PacketWork(cancelled_environment, cancelled_actor, 0),
                reason="service-horizon tail",
            )
        ledger.assert_quiescent()
        accounting = ledger.accounting()
        completed_packets = int(accounting["completed_packets"])
        cancelled_environment_steps = int(accounting["cancelled_work"][0])
        cancelled_actor_transitions = int(accounting["cancelled_work"][1])
    else:
        round_length = max(bases)
        next_checkpoint = 1
        round_start = 0.0
        while round_start < horizon - 1e-12:
            duration = min(round_length, horizon - round_start)
            birth = [copy.deepcopy(policy) for policy in policies]
            pending: list[tuple[int, np.ndarray, list[np.ndarray]]] = []
            for owner in range(agents):
                count = int(math.floor((duration + 1e-12) / bases[owner]))
                directions = []
                observations = []
                for _ in range(count):
                    direction, reference = collect_payload(owner, owner_indices[owner], birth)
                    owner_indices[owner] += 1
                    directions.append(direction)
                    observations.append(reference)
                if directions:
                    pending.append((owner, np.mean(np.asarray(directions), axis=0), observations[0]))
                    completed_packets += count
                remainder = max(0.0, duration - count * bases[owner])
                cancelled_environment_steps += min(
                    2 * episode_length,
                    int(math.floor(2 * episode_length * remainder / bases[owner] + 1e-12)),
                )
            for owner, direction, observations in pending:
                owner_error = float(
                    np.max(np.abs(_flat_parameters(policies[owner]) - _flat_parameters(birth[owner])))
                )
                maximum_owner_error = max(maximum_owner_error, owner_error)
                reference = observations[owner]
                old_mean, old_log_std = _distribution_parameters(policies[owner], reference)
                step = _clip_step(
                    direction,
                    float(spec["learning_rate"]),
                    float(spec["maximum_step_norm"]),
                )
                _add_flat_step(policies[owner], step)
                new_mean, new_log_std = _distribution_parameters(policies[owner], reference)
                motion = diagonal_gaussian_mean_kl(old_mean, old_log_std, new_mean, new_log_std)
                if not math.isfinite(motion):
                    raise RuntimeError("non-finite barrier policy motion")
                cumulative_step_norm += float(np.linalg.norm(step))
                cumulative_policy_kl += motion
                optimizer_updates += 1
            round_start += duration
            while next_checkpoint < len(checkpoints) and round_start >= checkpoints[next_checkpoint] - 1e-12:
                evaluate(checkpoints[next_checkpoint])
                next_checkpoint += 1
        cancelled_actor_transitions = cancelled_environment_steps * agents
        expected = barrier_schedule(bases, horizon, episode_length)
        if completed_packets != sum(expected["completed_by_owner"]):
            raise RuntimeError("barrier schedule mismatch")

    if maximum_owner_error > 1e-10:
        raise RuntimeError("owner self-fresh/frozen-round invariant failed")
    if len(curve) != len(checkpoints):
        raise RuntimeError("learning curve has the wrong checkpoint count")
    completed_environment_steps = completed_packets * 2 * episode_length
    baseline_environment_steps = int(spec["baseline_episodes"]) * episode_length
    evaluation_environment_steps = len(checkpoints) * len(evaluation_seeds) * episode_length
    times = np.asarray([row["logical_time"] for row in curve], dtype=float)
    values = np.asarray([row["evaluation_return"] for row in curve], dtype=float)
    auc = float(np.trapz(values, times) / horizon)
    environment.close()
    return {
        "endpoint": {
            "seed": seed,
            "task": task,
            "profile": profile,
            "method": method,
            "initial_policy_digest": initial_policy_digest,
            "frozen_control_variate_digest": frozen_digest,
            "initial_return": float(values[0]),
            "final_return": float(values[-1]),
            "return_change": float(values[-1] - values[0]),
            "return_auc": auc,
            "completed_packets": completed_packets,
            "optimizer_updates": optimizer_updates,
            "completed_environment_steps": completed_environment_steps,
            "completed_actor_transitions": completed_environment_steps * agents,
            "cancelled_environment_steps": cancelled_environment_steps,
            "cancelled_actor_transitions": cancelled_actor_transitions,
            "baseline_environment_steps": baseline_environment_steps,
            "baseline_actor_transitions": baseline_environment_steps * agents,
            "evaluation_environment_steps": evaluation_environment_steps,
            "evaluation_actor_transitions": evaluation_environment_steps * agents,
            "cumulative_step_norm": cumulative_step_norm,
            "cumulative_policy_kl": cumulative_policy_kl,
            "cumulative_teammate_birth_arrival_kl": cumulative_teammate_birth_arrival_kl,
            "maximum_owner_error": maximum_owner_error,
            "maximum_event_delay": maximum_event_delay,
            "offdiag_lyapunov_scale": offdiag_scale,
            "lyapunov_condition_max": float(np.max(certificate["conditions"])),
            "logical_service_time": horizon,
        },
        "curve": curve,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write empty CSV")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(config: dict[str, Any], output: Path) -> dict[str, Any]:
    harl_root = _checked_harl_root(config["harl_root"], config["harl_commit"])
    tasks = []
    for seed in config["pilot_seeds"]:
        for task, task_config in config["tasks"].items():
            for profile in ("balanced", "heterogeneous"):
                for method in METHODS:
                    tasks.append(
                        {
                            **config,
                            **task_config,
                            "harl_root": str(harl_root),
                            "seed": seed,
                            "task": task,
                            "profile": profile,
                            "method": method,
                        }
                    )
    started = time.perf_counter()
    results = []
    with ProcessPoolExecutor(max_workers=int(config["workers"])) as executor:
        futures = [executor.submit(_run_one, task) for task in tasks]
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            endpoint = result["endpoint"]
            print(
                f"[{index}/{len(futures)}] {endpoint['task']} {endpoint['profile']} "
                f"{endpoint['method']} seed={endpoint['seed']}"
            )
    endpoints = [result["endpoint"] for result in results]
    curves = [row for result in results for row in result["curve"]]
    endpoints.sort(key=lambda row: (row["seed"], row["task"], row["profile"], row["method"]))
    curves.sort(
        key=lambda row: (
            row["seed"], row["task"], row["profile"], row["method"], row["logical_time"]
        )
    )
    output.mkdir(parents=True, exist_ok=False)
    endpoints_path = output / "endpoints.csv"
    curves_path = output / "curves.csv"
    _write_csv(endpoints_path, endpoints)
    _write_csv(curves_path, curves)
    manifest = {
        "experiment_id": config["experiment_id"],
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip(),
        "rows": len(endpoints),
        "curve_rows": len(curves),
        "endpoints_sha256": _sha256(endpoints_path),
        "curves_sha256": _sha256(curves_path),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    runtime = {"wall_seconds": time.perf_counter() - started, "workers": config["workers"]}
    (output / "runtime.json").write_text(
        json.dumps(runtime, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    print(json.dumps(run(config, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

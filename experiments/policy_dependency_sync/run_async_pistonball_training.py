"""Development runner for asynchronous policy-cache CTDE on Pistonball.

This is an implementation-development runner, not a preregistered scientific
experiment.  Its purpose is to exercise the complete launch--rollout--receipt
state machine before freezing a GPU pilot.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .async_pistonball_ctde import (
    CompressedReplay,
    OwnerGradientQueue,
    PistonActor,
    PistonCentralCritic,
    PolicyCacheBank,
    apply_refresh_action,
    collect_parallel_segment,
    compress_parallel_snapshot,
    critic_td_loss,
    hard_budget_remaining,
    joint_policy_actions,
    owner_critic_gradient,
    polyak_update,
    select_age_refresh,
    select_mismatch_refresh,
    signed_refresh_for_batch,
)
from .pistonball_tail import predictive_tube_support


SCHEDULERS = (
    "signed_lyapunov",
    "signed_only",
    "no_refresh",
    "age",
    "mismatch",
    "round_robin",
    "static_chain",
    "complete_burst",
)


@dataclass(frozen=True)
class TrainingConfig:
    n_agents: int = 20
    launches: int = 160
    rollout_horizon: int = 4
    max_cycles: int = 125
    replay_capacity: int = 4096
    batch_size: int = 32
    discount: float = 0.99
    exploration_std: float = 0.2
    critic_step: float = 3e-4
    actor_receipt_step: float = 1e-3
    target_polyak: float = 0.01
    gradient_clip: float = 5.0
    budget_rate: float = 0.5
    cone_shell: int = 6
    maximum_delay: int = 4
    lyapunov_weight: float = 100000.0
    score_smoothness: float = 1.0
    score_second_moment: float = 1.0
    score_motion_bound: float = 0.0
    score_taylor_coefficient: float = 0.0
    cache_debt_weight: float = 100000.0
    evaluations: int = 5
    evaluation_episodes: int = 2
    random_drop: bool = False
    random_rotate: bool = False

    def __post_init__(self) -> None:
        positive_integer_fields = (
            self.n_agents,
            self.launches,
            self.rollout_horizon,
            self.max_cycles,
            self.replay_capacity,
            self.batch_size,
            self.evaluations,
            self.evaluation_episodes,
        )
        if min(positive_integer_fields) <= 0 or self.n_agents < 2:
            raise ValueError("counts must be positive and n_agents at least two")
        if self.replay_capacity < self.batch_size or self.evaluations < 2:
            raise ValueError("replay must hold a batch and evaluations include endpoints")
        if not 0.0 <= self.discount <= 1.0:
            raise ValueError("discount must lie in [0,1]")
        nonnegative = (
            self.exploration_std,
            self.critic_step,
            self.actor_receipt_step,
            self.gradient_clip,
            self.budget_rate,
            self.cone_shell,
            self.maximum_delay,
            self.lyapunov_weight,
            self.score_smoothness,
            self.score_second_moment,
            self.score_motion_bound,
            self.score_taylor_coefficient,
            self.cache_debt_weight,
        )
        if min(nonnegative) < 0.0 or not 0.0 <= self.target_polyak <= 1.0:
            raise ValueError("rates, bounds, and weights must be nonnegative")
        strictly_positive = (
            self.critic_step,
            self.actor_receipt_step,
            self.gradient_clip,
            self.lyapunov_weight,
            self.score_smoothness,
        )
        if min(strictly_positive) <= 0.0:
            raise ValueError("optimization steps, clip, and Lyapunov terms must be positive")


@dataclass
class Worker:
    environment: Any
    observations: Mapping[str, np.ndarray]
    resets: int


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected = torch.device(name)
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return selected


def _make_environment(config: TrainingConfig):
    from pettingzoo.butterfly import pistonball_v6

    return pistonball_v6.parallel_env(
        n_pistons=config.n_agents,
        continuous=True,
        random_drop=config.random_drop,
        random_rotate=config.random_rotate,
        max_cycles=config.max_cycles,
        render_mode=None,
    )


def _make_workers(config: TrainingConfig, seed: int) -> list[Worker]:
    workers: list[Worker] = []
    for owner in range(config.n_agents):
        environment = _make_environment(config)
        observations, _ = environment.reset(seed=int(seed + 1000 * owner))
        workers.append(Worker(environment, observations, 1))
    return workers


def _clip_gradients(
    gradients: Sequence[torch.Tensor], maximum_norm: float
) -> tuple[torch.Tensor, ...]:
    norm = math.sqrt(
        sum(float(gradient.detach().square().sum().cpu()) for gradient in gradients)
    )
    scale = 1.0 if norm <= maximum_norm else float(maximum_norm) / max(norm, 1e-12)
    return tuple(gradient.detach() * scale for gradient in gradients)


def _eligible_donors(worker: Worker, owner: int, config: TrainingConfig) -> tuple[int, ...]:
    raw = worker.environment.unwrapped
    support = predictive_tube_support(
        n_agents=config.n_agents,
        launch_x=float(raw.ball.position.x),
        launch_velocity_x=float(raw.ball.velocity.x),
        horizon=config.rollout_horizon,
        radius=config.cone_shell,
    )
    return tuple(support[int(owner)])


def _round_robin_donor(
    eligible: Sequence[int], launch: int, maximum_edges: int
) -> tuple[int, ...]:
    if not eligible or maximum_edges <= 0:
        return ()
    ordered = tuple(sorted(int(donor) for donor in eligible))
    start = int(launch) % len(ordered)
    rotated = ordered[start:] + ordered[:start]
    return tuple(rotated[: int(maximum_edges)])


def _scheduler_action(
    *,
    scheduler: str,
    owner: int,
    eligible: tuple[int, ...],
    launch: int,
    actors: Sequence[PistonActor],
    caches: PolicyCacheBank,
    critic: PistonCentralCritic,
    replay: CompressedReplay,
    score_observations: torch.Tensor,
    score_states: torch.Tensor,
    device: torch.device,
    queue_value: float,
    remaining_units: int,
    config: TrainingConfig,
) -> tuple[int, ...]:
    maximum_edges = min(int(remaining_units), len(eligible))
    if maximum_edges <= 0 or scheduler == "no_refresh":
        return ()
    if scheduler == "age":
        return select_age_refresh(
            recipient=owner,
            eligible_donors=eligible,
            caches=caches,
            maximum_edges=1,
        )
    if scheduler == "mismatch":
        return select_mismatch_refresh(
            recipient=owner,
            eligible_donors=eligible,
            actors=actors,
            caches=caches,
            maximum_edges=1,
        )
    if scheduler == "round_robin":
        stale = tuple(
            donor for donor in eligible if caches.age(owner, donor) > 0
        )
        return _round_robin_donor(stale, launch, 1)
    if scheduler == "static_chain":
        static = tuple(
            donor for donor in (owner - 1, owner + 1) if donor in set(eligible)
        )
        return select_age_refresh(
            recipient=owner,
            eligible_donors=static,
            caches=caches,
            maximum_edges=min(1, maximum_edges),
        )
    if scheduler == "complete_burst":
        return select_age_refresh(
            recipient=owner,
            eligible_donors=eligible,
            caches=caches,
            maximum_edges=maximum_edges,
        )
    if scheduler not in ("signed_lyapunov", "signed_only"):
        raise ValueError(f"unknown scheduler {scheduler}")
    if len(replay) < config.batch_size:
        return ()
    choice = signed_refresh_for_batch(
        owner=owner,
        eligible_donors=eligible,
        actors=actors,
        caches=caches,
        critic=critic,
        observations=score_observations,
        states=score_states,
        step=config.actor_receipt_step,
        communication_queue=queue_value,
        learning_weight=config.lyapunov_weight,
        smoothness=config.score_smoothness,
        packet_second_moment_upper=config.score_second_moment,
        receipt_motion_upper=config.score_motion_bound,
        taylor_coefficient=config.score_taylor_coefficient,
        cache_debt_weight=(
            config.cache_debt_weight if scheduler == "signed_lyapunov" else 0.0
        ),
        can_refresh=remaining_units > 0,
    )
    return () if choice.donor is None else (int(choice.donor),)


def _evaluate(
    *,
    actors: Sequence[PistonActor],
    config: TrainingConfig,
    seed: int,
    device: torch.device,
) -> float:
    returns: list[float] = []
    for episode in range(config.evaluation_episodes):
        environment = _make_environment(config)
        observations, _ = environment.reset(seed=int(seed + episode))
        total = 0.0
        try:
            while environment.agents:
                compressed, _ = compress_parallel_snapshot(
                    observations=observations,
                    possible_agents=environment.possible_agents,
                    state=environment.state(),
                )
                with torch.no_grad():
                    actions = joint_policy_actions(
                        actors=actors,
                        observations=torch.as_tensor(compressed[None], device=device),
                    )[0].cpu().numpy()
                action_dict = {
                    agent: np.asarray([actions[index]], dtype=np.float32)
                    for index, agent in enumerate(environment.possible_agents)
                }
                observations, rewards, _, _, _ = environment.step(action_dict)
                total += float(rewards[environment.possible_agents[0]])
        finally:
            environment.close()
        returns.append(total)
    return float(np.mean(returns))


def run_training(
    *,
    scheduler: str,
    seed: int,
    config: TrainingConfig,
    device_name: str,
) -> dict[str, Any]:
    if scheduler not in SCHEDULERS:
        raise ValueError(f"scheduler must be one of {SCHEDULERS}")
    device = _device(device_name)
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    delay_rng = np.random.default_rng(int(seed) + 11)
    action_rngs = [
        np.random.default_rng(int(seed) + 101 + owner) for owner in range(config.n_agents)
    ]
    replay_rng = np.random.default_rng(int(seed) + 211)

    base_actor = PistonActor().to(device)
    actors = tuple(copy.deepcopy(base_actor).to(device) for _ in range(config.n_agents))
    target_actors = tuple(copy.deepcopy(actor).to(device) for actor in actors)
    critic = PistonCentralCritic(config.n_agents).to(device)
    target_critic = copy.deepcopy(critic).to(device)
    caches = PolicyCacheBank(actors)
    replay = CompressedReplay(config.replay_capacity)
    packet_queue = OwnerGradientQueue()
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=config.critic_step)
    workers = _make_workers(config, int(seed) + 10000)

    spent_units = 0
    spent_bytes = 0
    queue_value = 0.0
    launched_transitions = 0
    received_packets = 0
    selected_edges = 0
    cumulative_train_reward = 0.0
    evaluation_rows: list[dict[str, float | int]] = []
    launch_rows: list[dict[str, Any]] = []
    evaluation_launches = set(
        int(value)
        for value in np.linspace(0, config.launches, config.evaluations, dtype=int)
    )
    started = time.perf_counter()
    try:
        if 0 in evaluation_launches:
            evaluation_rows.append(
                {
                    "launch": 0,
                    "actor_transitions": 0,
                    "optional_policy_bytes": 0,
                    "return": _evaluate(
                        actors=actors,
                        config=config,
                        seed=int(seed) + 700000,
                        device=device,
                    ),
                }
            )
        for launch in range(config.launches):
            received = packet_queue.apply_due(
                event=launch,
                actors=actors,
                caches=caches,
                step=config.actor_receipt_step,
            )
            received_packets += len(received)
            if received:
                polyak_update(actors, target_actors, config.target_polyak)

            owner = launch % config.n_agents
            worker = workers[owner]
            eligible = _eligible_donors(worker, owner, config)
            launch_observations, launch_state = compress_parallel_snapshot(
                observations=worker.observations,
                possible_agents=worker.environment.possible_agents,
                state=worker.environment.state(),
            )
            remaining = hard_budget_remaining(
                launches_after_action=launch + 1,
                budget_rate=config.budget_rate,
                spent_units=spent_units,
            )
            selected = _scheduler_action(
                scheduler=scheduler,
                owner=owner,
                eligible=eligible,
                launch=launch,
                actors=actors,
                caches=caches,
                critic=critic,
                replay=replay,
                score_observations=torch.as_tensor(
                    launch_observations[None], device=device
                ),
                score_states=torch.as_tensor(launch_state[None], device=device),
                device=device,
                queue_value=queue_value,
                remaining_units=remaining,
                config=config,
            )
            refresh = apply_refresh_action(
                recipient=owner,
                donors=selected,
                actors=actors,
                caches=caches,
            )
            spent_units += refresh.refresh_units
            spent_bytes += refresh.optional_policy_bytes
            selected_edges += refresh.refresh_units
            queue_value = max(
                queue_value + refresh.refresh_units - config.budget_rate, 0.0
            )

            transitions, observations, terminated = collect_parallel_segment(
                environment=worker.environment,
                owner=owner,
                actors=actors,
                caches=caches,
                observations=worker.observations,
                horizon=config.rollout_horizon,
                exploration_std=config.exploration_std,
                rng=action_rngs[owner],
                device=device,
            )
            for transition in transitions:
                replay.add(transition)
                cumulative_train_reward += transition.reward
            launched_transitions += len(transitions) * config.n_agents
            launch_rows.append(
                {
                    "launch": launch,
                    "owner": owner,
                    "eligible_donors": list(eligible),
                    "selected_donors": list(refresh.donors),
                    "refresh_units": refresh.refresh_units,
                    "optional_policy_bytes": refresh.optional_policy_bytes,
                    "queue_after_action": queue_value,
                    "segment_cycles": len(transitions),
                    "segment_team_reward": float(
                        sum(transition.reward for transition in transitions)
                    ),
                }
            )
            if terminated:
                worker.resets += 1
                worker.observations, _ = worker.environment.reset(
                    seed=int(seed + 1000 * owner + worker.resets)
                )
            else:
                worker.observations = observations

            if len(replay) >= config.batch_size:
                batch = replay.sample(config.batch_size, replay_rng, device)
                critic_optimizer.zero_grad(set_to_none=True)
                td_loss = critic_td_loss(
                    critic=critic,
                    target_critic=target_critic,
                    target_actors=target_actors,
                    states=batch.states,
                    actions=batch.actions,
                    rewards=batch.rewards,
                    next_states=batch.next_states,
                    next_observations=batch.next_observations,
                    done=batch.done,
                    discount=config.discount,
                )
                td_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), config.gradient_clip)
                critic_optimizer.step()

                actor_batch = replay.sample(config.batch_size, replay_rng, device)
                gradients = owner_critic_gradient(
                    owner=owner,
                    actors=actors,
                    caches=caches,
                    critic=critic,
                    observations=actor_batch.observations,
                    states=actor_batch.states,
                )
                gradients = _clip_gradients(gradients, config.gradient_clip)
                delay = int(delay_rng.integers(0, config.maximum_delay + 1))
                packet_queue.launch(
                    owner=owner,
                    launch_event=launch,
                    delay=delay,
                    gradients=gradients,
                )
                immediately_received = packet_queue.apply_due(
                    event=launch,
                    actors=actors,
                    caches=caches,
                    step=config.actor_receipt_step,
                )
                received_packets += len(immediately_received)
                polyak_update((critic,), (target_critic,), config.target_polyak)
                if immediately_received:
                    polyak_update(actors, target_actors, config.target_polyak)

            completed_launches = launch + 1
            if completed_launches in evaluation_launches:
                evaluation_rows.append(
                    {
                        "launch": completed_launches,
                        "actor_transitions": launched_transitions,
                        "optional_policy_bytes": spent_bytes,
                        "return": _evaluate(
                            actors=actors,
                            config=config,
                            seed=int(seed) + 700000,
                            device=device,
                        ),
                    }
                )

        drain_event = packet_queue.drain_event()
        if drain_event is not None:
            drained = packet_queue.apply_due(
                event=drain_event,
                actors=actors,
                caches=caches,
                step=config.actor_receipt_step,
            )
            received_packets += len(drained)
    finally:
        for worker in workers:
            worker.environment.close()

    runtime = time.perf_counter() - started
    allowance = int(math.floor(config.launches * config.budget_rate + 1e-12))
    finite = all(
        math.isfinite(float(row["return"])) for row in evaluation_rows
    ) and all(
        torch.isfinite(parameter).all().item()
        for module in (*actors, critic)
        for parameter in module.parameters()
    )
    action_trace_sha256 = hashlib.sha256(
        json.dumps(launch_rows, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest().upper()
    return {
        "scheduler": scheduler,
        "seed": int(seed),
        "config": asdict(config),
        "device": str(device),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "evaluations": evaluation_rows,
        "launch_trace": launch_rows,
        "action_trace_sha256": action_trace_sha256,
        "spent_refresh_units": spent_units,
        "allowed_refresh_units": allowance,
        "optional_policy_bytes": spent_bytes,
        "selected_edges": selected_edges,
        "launched_actor_transitions": launched_transitions,
        "received_packets": received_packets,
        "remaining_packets_after_drain": len(packet_queue),
        "cumulative_training_team_reward": cumulative_train_reward,
        "finite": finite,
        "budget_feasible": spent_units <= allowance,
        "runtime_seconds": runtime,
        "development_only": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheduler", choices=SCHEDULERS, required=True)
    parser.add_argument("--seed", type=int, default=74001)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--launches", type=int, default=160)
    parser.add_argument("--n-agents", type=int, default=20)
    parser.add_argument("--rollout-horizon", type=int, default=4)
    parser.add_argument("--max-cycles", type=int, default=125)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--evaluations", type=int, default=5)
    parser.add_argument("--evaluation-episodes", type=int, default=2)
    parser.add_argument("--budget-rate", type=float, default=0.5)
    parser.add_argument("--cone-shell", type=int, default=6)
    parser.add_argument("--maximum-delay", type=int, default=4)
    parser.add_argument("--lyapunov-weight", type=float, default=100000.0)
    parser.add_argument("--cache-debt-weight", type=float, default=100000.0)
    parser.add_argument("--random-drop", action="store_true")
    parser.add_argument("--random-rotate", action="store_true")
    args = parser.parse_args()
    config = TrainingConfig(
        n_agents=args.n_agents,
        launches=args.launches,
        rollout_horizon=args.rollout_horizon,
        max_cycles=args.max_cycles,
        batch_size=args.batch_size,
        evaluations=args.evaluations,
        evaluation_episodes=args.evaluation_episodes,
        budget_rate=args.budget_rate,
        cone_shell=args.cone_shell,
        maximum_delay=args.maximum_delay,
        lyapunov_weight=args.lyapunov_weight,
        cache_debt_weight=args.cache_debt_weight,
        random_drop=args.random_drop,
        random_rotate=args.random_rotate,
    )
    result = run_training(
        scheduler=args.scheduler,
        seed=args.seed,
        config=config,
        device_name=args.device,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not result["finite"] or not result["budget_feasible"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

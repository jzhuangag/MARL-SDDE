"""Run the TSP-V2 online Lyapunov participation controller on HARL MAPPO.

The controller observes only a registered, disjoint validation stream.  Each
decision is charged for one MAPPO update and for both validation rollouts.
The final return stream has distinct seeds and is never exposed to the
controller.  Fixed-q comparators use the same two physical budgets and do not
pay for observations that they do not consume.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from online_drift_queue_controller import (
    OnlineActionCost,
    SlidingWindowDriftQueueController,
)


def _old_experiment_path() -> Path:
    path = Path(__file__).resolve().parents[2] / "TSP" / "experiments"
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    return path


def _to_numpy(value):
    return value.detach().cpu().numpy()


def bounded_validation_risk(mean_step_return: float, scale: float) -> float:
    """Map a return (larger is better) monotonically into a bounded risk."""

    if not math.isfinite(mean_step_return) or scale <= 0.0:
        raise ValueError("return must be finite and scale positive")
    return float(0.5 - math.atan(mean_step_return / scale) / math.pi)


def action_costs(
    candidate_q: list[int],
    *,
    rollout_length: int,
    train_updates_per_decision: int,
    validation_q: int,
    validation_horizon: int,
    server_overhead: int,
) -> dict[int, OnlineActionCost]:
    """Fully charge one training update and the before/after validation pair."""

    if min(
        rollout_length,
        train_updates_per_decision,
        validation_q,
        validation_horizon,
    ) <= 0:
        raise ValueError("rollout and validation dimensions must be positive")
    if server_overhead < 0:
        raise ValueError("server overhead cannot be negative")
    validation_messages = 2 * (
        server_overhead + validation_q * validation_horizon
    )
    validation_ticks = 2 * validation_horizon
    return {
        q: OnlineActionCost(
            messages=train_updates_per_decision
            * (server_overhead + q * rollout_length)
            + validation_messages,
            environment_ticks=train_updates_per_decision * rollout_length
            + validation_ticks,
        )
        for q in sorted(set(candidate_q))
    }


def fixed_q_updates(
    q: int,
    *,
    rollout_length: int,
    server_overhead: int,
    message_budget: int,
    environment_budget: int,
) -> int:
    """Maximum fully charged fixed-q updates under both physical budgets."""

    if min(q, rollout_length, message_budget, environment_budget) <= 0:
        raise ValueError("q, rollout length, and budgets must be positive")
    per_update_messages = server_overhead + q * rollout_length
    return min(
        message_budget // per_update_messages,
        environment_budget // rollout_length,
    )


def parameter_hash(runner) -> str:
    digest = hashlib.sha256()
    modules = [actor.actor for actor in runner.actor] + [runner.critic.critic]
    for module_index, module in enumerate(modules):
        for name, value in sorted(module.state_dict().items()):
            digest.update(f"{module_index}:{name}".encode("utf-8"))
            digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _training_state(runner) -> dict:
    return {
        "actors": [copy.deepcopy(actor.actor.state_dict()) for actor in runner.actor],
        "actor_optimizers": [
            copy.deepcopy(actor.actor_optimizer.state_dict()) for actor in runner.actor
        ],
        "critic": copy.deepcopy(runner.critic.critic.state_dict()),
        "critic_optimizer": copy.deepcopy(
            runner.critic.critic_optimizer.state_dict()
        ),
        "value_normalizer": (
            copy.deepcopy(runner.value_normalizer.state_dict())
            if runner.value_normalizer is not None
            else None
        ),
    }


def synchronize_training_state(source, targets: list) -> None:
    """Synchronize learned state while preserving each q stream's Markov state."""

    state = _training_state(source)
    for target in targets:
        if len(target.actor) != len(state["actors"]):
            raise ValueError("runner actor counts differ")
        for actor, actor_state, optimizer_state in zip(
            target.actor,
            state["actors"],
            state["actor_optimizers"],
        ):
            actor.actor.load_state_dict(actor_state)
            actor.actor_optimizer.load_state_dict(optimizer_state)
        target.critic.critic.load_state_dict(state["critic"])
        target.critic.critic_optimizer.load_state_dict(state["critic_optimizer"])
        if state["value_normalizer"] is not None:
            if target.value_normalizer is None:
                raise ValueError("value-normalizer configuration differs")
            target.value_normalizer.load_state_dict(state["value_normalizer"])


def train_one_update(runner) -> float:
    """Execute one MAPPO update and return its observed mean step reward."""

    runner.prep_rollout()
    rewards = []
    for step in range(runner.algo_args["train"]["episode_length"]):
        values, actions, action_log_probs, rnn_states, rnn_states_critic = (
            runner.collect(step)
        )
        obs, share_obs, reward, dones, infos, available_actions = runner.envs.step(
            actions
        )
        rewards.append(np.asarray(reward, dtype=float))
        runner.insert(
            (
                obs,
                share_obs,
                reward,
                dones,
                infos,
                available_actions,
                values,
                actions,
                action_log_probs,
                rnn_states,
                rnn_states_critic,
            )
        )
    runner.compute()
    runner.prep_training()
    runner.train()
    runner.after_update()
    return float(np.mean(rewards))


@torch.no_grad()
def deterministic_horizon_return(runner, *, seed: int, horizon: int) -> float:
    """Evaluate the current decentralized policies on a fresh seeded stream."""

    if horizon <= 0:
        raise ValueError("evaluation horizon must be positive")
    from harl.utils.envs_tools import make_eval_env

    eval_env = make_eval_env(runner.env_name, seed, 1, runner.env_args)
    try:
        obs, _, available_actions = eval_env.reset()
        rnn_states = np.zeros(
            (1, runner.num_agents, runner.recurrent_n, runner.rnn_hidden_size),
            dtype=np.float32,
        )
        masks = np.ones((1, runner.num_agents, 1), dtype=np.float32)
        rewards = []
        for _ in range(horizon):
            action_columns = []
            for agent_id in range(runner.num_agents):
                actions, next_state = runner.actor[agent_id].act(
                    obs[:, agent_id],
                    rnn_states[:, agent_id],
                    masks[:, agent_id],
                    available_actions[:, agent_id]
                    if available_actions[0] is not None
                    else None,
                    deterministic=True,
                )
                rnn_states[:, agent_id] = _to_numpy(next_state)
                action_columns.append(_to_numpy(actions))
            joint_actions = np.asarray(action_columns).transpose(1, 0, 2)
            obs, _, reward, dones, _, available_actions = eval_env.step(
                joint_actions
            )
            rewards.append(float(np.mean(reward)))
            done_env = np.all(dones, axis=1)
            rnn_states[done_env] = 0.0
            masks[:] = 1.0
            masks[done_env] = 0.0
        return float(np.mean(rewards))
    finally:
        eval_env.close()


def _make_runner(args: argparse.Namespace, q: int, root: Path):
    _old_experiment_path()
    from run_mappo_probe_commit import make_harl_runner

    return make_harl_runner(
        args,
        q=q,
        seed=args.seed,
        num_env_steps=max(1, q * args.rollout_length * args.max_updates_hint),
        results_root=root / f"q{q}",
        exp_name=f"v2_{args.method}_{args.coupling}_q{q}",
        use_eval=False,
    )


def _task_name(args: argparse.Namespace) -> str:
    return (
        f"{args.scenario}/{args.agent_conf}"
        if args.env_name == "mamujoco"
        else args.scenario
    )


def run(args: argparse.Namespace) -> Path:
    root = args.results_root.resolve()
    if root.exists():
        raise FileExistsError(f"refusing to overwrite {root}")
    root.mkdir(parents=True)
    start = time.perf_counter()
    rows = []
    runners: dict[int, object] = {}
    controller = None
    try:
        if args.method == "controller":
            qs = sorted(set(args.candidate_q))
            costs = action_costs(
                qs,
                rollout_length=args.rollout_length,
                train_updates_per_decision=args.train_updates_per_decision,
                validation_q=args.validation_q,
                validation_horizon=args.validation_horizon,
                server_overhead=args.server_overhead,
            )
            controller = SlidingWindowDriftQueueController(
                costs=costs,
                total_messages=args.message_budget,
                total_environment_ticks=args.environment_budget,
                decisions=args.controller_decisions,
                window=args.window,
                bonus_scale=args.bonus_scale,
                probe_interval=args.probe_interval,
                progress_scale=args.progress_scale,
                queue_weight=args.queue_weight,
                confidence_delta=args.confidence_delta,
                markov_bias=args.markov_bias,
            )
            for q in qs:
                runners[q] = _make_runner(args, q, root / "training")
                runners[q].warmup()
            hashes = {q: parameter_hash(runner) for q, runner in runners.items()}
            if len(set(hashes.values())) != 1:
                raise RuntimeError("candidate q runners do not share initialization")
            synchronize_training_state(runners[qs[0]], list(runners.values()))
            cumulative_messages = 0
            cumulative_ticks = 0
            rows.append(
                {
                    "block": 0,
                    "budget_fraction": 0.0,
                    "q": None,
                    "train_mean_step_reward": None,
                    "validation_return_before": None,
                    "validation_return_after": None,
                    "validation_progress": None,
                    "evaluation_mean_step_return": deterministic_horizon_return(
                        runners[qs[0]],
                        seed=args.evaluation_seed,
                        horizon=args.evaluation_horizon,
                    ),
                    "cumulative_messages": 0,
                    "cumulative_environment_ticks": 0,
                }
            )
            for block in range(args.controller_decisions):
                decision = controller.decide()
                q = decision.action
                validation_seed = args.validation_seed_base + block
                before_return = deterministic_horizon_return(
                    runners[q], seed=validation_seed, horizon=args.validation_horizon
                )
                block_training_rewards = [
                    train_one_update(runners[q])
                    for _ in range(args.train_updates_per_decision)
                ]
                train_reward = float(np.mean(block_training_rewards))
                synchronize_training_state(runners[q], list(runners.values()))
                after_return = deterministic_horizon_return(
                    runners[q], seed=validation_seed, horizon=args.validation_horizon
                )
                observation = controller.observe(
                    risk_before=bounded_validation_risk(
                        before_return, args.validation_return_scale
                    ),
                    risk_after=bounded_validation_risk(
                        after_return, args.validation_return_scale
                    ),
                )
                cost = costs[q]
                cumulative_messages += cost.messages
                cumulative_ticks += cost.environment_ticks
                if block % args.evaluation_interval == 0 or block + 1 == args.controller_decisions:
                    final_return = deterministic_horizon_return(
                        runners[q],
                        seed=args.evaluation_seed,
                        horizon=args.evaluation_horizon,
                    )
                    rows.append(
                        {
                            "block": block + 1,
                            "budget_fraction": max(
                                cumulative_messages / args.message_budget,
                                cumulative_ticks / args.environment_budget,
                            ),
                            "q": q,
                            "train_mean_step_reward": train_reward,
                            "validation_return_before": before_return,
                            "validation_return_after": after_return,
                            "validation_progress": observation.progress,
                            "evaluation_mean_step_return": final_return,
                            "cumulative_messages": cumulative_messages,
                            "cumulative_environment_ticks": cumulative_ticks,
                        }
                    )
            selected_counts = {str(q): len(controller.histories[q]) for q in qs}
            accounting = {
                "charged_messages": cumulative_messages,
                "charged_environment_ticks": cumulative_ticks,
                "message_remaining": controller.message_remaining,
                "environment_remaining": controller.environment_remaining,
                "selected_counts": selected_counts,
            }
            initial_hashes = hashes
        else:
            q = int(args.method.removeprefix("fixed_q"))
            updates = fixed_q_updates(
                q,
                rollout_length=args.rollout_length,
                server_overhead=args.server_overhead,
                message_budget=args.message_budget,
                environment_budget=args.environment_budget,
            )
            runners[q] = _make_runner(args, q, root / "training")
            runners[q].warmup()
            initial_hashes = {q: parameter_hash(runners[q])}
            cumulative_messages = 0
            cumulative_ticks = 0
            rows.append(
                {
                    "block": 0,
                    "budget_fraction": 0.0,
                    "q": q,
                    "train_mean_step_reward": None,
                    "validation_return_before": None,
                    "validation_return_after": None,
                    "validation_progress": None,
                    "evaluation_mean_step_return": deterministic_horizon_return(
                        runners[q],
                        seed=args.evaluation_seed,
                        horizon=args.evaluation_horizon,
                    ),
                    "cumulative_messages": 0,
                    "cumulative_environment_ticks": 0,
                }
            )
            for block in range(updates):
                train_reward = train_one_update(runners[q])
                cumulative_messages += args.server_overhead + q * args.rollout_length
                cumulative_ticks += args.rollout_length
                if block % args.evaluation_interval == 0 or block + 1 == updates:
                    final_return = deterministic_horizon_return(
                        runners[q],
                        seed=args.evaluation_seed,
                        horizon=args.evaluation_horizon,
                    )
                    rows.append(
                        {
                            "block": block + 1,
                            "budget_fraction": max(
                                cumulative_messages / args.message_budget,
                                cumulative_ticks / args.environment_budget,
                            ),
                            "q": q,
                            "train_mean_step_reward": train_reward,
                            "validation_return_before": None,
                            "validation_return_after": None,
                            "validation_progress": None,
                            "evaluation_mean_step_return": final_return,
                            "cumulative_messages": cumulative_messages,
                            "cumulative_environment_ticks": cumulative_ticks,
                        }
                    )
            accounting = {
                "charged_messages": cumulative_messages,
                "charged_environment_ticks": cumulative_ticks,
                "message_remaining": args.message_budget - cumulative_messages,
                "environment_remaining": args.environment_budget - cumulative_ticks,
                "selected_counts": {str(q): updates},
            }
    finally:
        for runner in runners.values():
            runner.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not rows:
        raise RuntimeError("no evaluation rows were produced")
    progress_path = root / "progress.jsonl"
    progress_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    metadata = {
        "experiment_id": args.experiment_id,
        "role": "development" if args.development else "confirmation",
        "environment": args.env_name,
        "task": _task_name(args),
        "coupling": args.coupling,
        "method": args.method,
        "seed": args.seed,
        "validation_seed_base": args.validation_seed_base,
        "evaluation_seed": args.evaluation_seed,
        "candidate_q": sorted(set(args.candidate_q)),
        "initial_parameter_sha256": initial_hashes,
        "message_budget": args.message_budget,
        "environment_budget": args.environment_budget,
        "rollout_length": args.rollout_length,
        "validation_horizon": args.validation_horizon,
        "train_updates_per_decision": args.train_updates_per_decision,
        "evaluation_horizon": args.evaluation_horizon,
        "server_overhead": args.server_overhead,
        "controller_parameters": {
            "decisions": args.controller_decisions,
            "window": args.window,
            "bonus_scale": args.bonus_scale,
            "probe_interval": args.probe_interval,
            "progress_scale": args.progress_scale,
            "queue_weight": args.queue_weight,
            "validation_return_scale": args.validation_return_scale,
            "confidence_delta": args.confidence_delta,
            "markov_bias": args.markov_bias,
        },
        "accounting": accounting,
        "final_evaluation_return": rows[-1]["evaluation_mean_step_return"],
        "evaluation_used_by_controller": False,
        "wall_seconds": time.perf_counter() - start,
        "harl_commit": subprocess.check_output(
            ["git", "-C", str(args.harl_root.resolve()), "rev-parse", "HEAD"],
            text=True,
        ).strip(),
    }
    path = root / "metadata.json"
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--harl-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--experiment-id", default="TSP-V2-MARL-ONLINE-DEV-001")
    parser.add_argument("--development", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--env-name", choices=("pettingzoo_mpe", "mamujoco"), required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--agent-conf", default="2x3")
    parser.add_argument("--agent-obsk", type=int, default=0)
    parser.add_argument("--episode-limit", type=int, default=1000)
    parser.add_argument("--coupling", choices=("independent", "shared"), required=True)
    parser.add_argument("--method", choices=("controller", "fixed_q1", "fixed_q2", "fixed_q4", "fixed_q8"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--validation-seed-base", type=int, required=True)
    parser.add_argument("--evaluation-seed", type=int, required=True)
    parser.add_argument("--seed-registry-base", type=int, default=150000)
    parser.add_argument("--seed-registry-size", type=int, default=4096)
    parser.add_argument("--candidate-q", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--controller-decisions", type=int, required=True)
    parser.add_argument("--max-updates-hint", type=int, default=4096)
    parser.add_argument("--rollout-length", type=int, required=True)
    parser.add_argument("--train-updates-per-decision", type=int, default=1)
    parser.add_argument("--validation-q", type=int, default=1)
    parser.add_argument("--validation-horizon", type=int, required=True)
    parser.add_argument("--evaluation-horizon", type=int, required=True)
    parser.add_argument("--evaluation-interval", type=int, default=25)
    parser.add_argument("--validation-return-scale", type=float, required=True)
    parser.add_argument("--message-budget", type=int, required=True)
    parser.add_argument("--environment-budget", type=int, required=True)
    parser.add_argument("--server-overhead", type=int, required=True)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--bonus-scale", type=float, default=0.08)
    parser.add_argument("--probe-interval", type=int, default=40)
    parser.add_argument("--progress-scale", type=float, default=0.02)
    parser.add_argument("--queue-weight", type=float, default=80.0)
    parser.add_argument("--confidence-delta", type=float, default=0.05)
    parser.add_argument("--markov-bias", type=float, default=0.0)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--activation", default="tanh")
    parser.add_argument("--actor-lr", type=float, default=5e-4)
    parser.add_argument("--critic-lr", type=float, default=5e-4)
    parser.add_argument("--ppo-epoch", type=int, default=5)
    parser.add_argument("--critic-epoch", type=int, default=5)
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--eval-threads", type=int, default=1)
    parser.add_argument("--eval-episodes", type=int, default=1)
    parser.add_argument("--eval-interval", type=int, default=10_000_000)
    parser.add_argument("--log-interval", type=int, default=10_000_000)
    parser.add_argument("--continuous-actions", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--share-param", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = run(args)
    print(json.dumps({"metadata": str(path)}, sort_keys=True))


if __name__ == "__main__":
    main()

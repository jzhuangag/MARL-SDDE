"""Outcome-isolated interface test for a MARL learning-progress sensor.

The script performs a small, fully charged micro-training branch for every
candidate participation level.  It records only rewards observed by those
training branches and optimizer diagnostics; it does not run the registered
evaluation-return protocol and cannot be used as manuscript performance
evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch


def parameter_hash(runner) -> str:
    digest = hashlib.sha256()
    modules = [runner.actor[0].actor, runner.critic.critic]
    for module in modules:
        for name, value in sorted(module.state_dict().items()):
            digest.update(name.encode("utf-8"))
            digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def run_updates(runner, updates: int) -> np.ndarray:
    if updates < 4:
        raise ValueError("at least four micro-training updates are required")
    runner.warmup()
    reward_means = []
    for _ in range(updates):
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
        reward_means.append(float(np.mean(rewards)))
    return np.asarray(reward_means, dtype=float)


def progress_statistics(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size < 4 or not np.isfinite(values).all():
        raise ValueError("progress values must contain at least four finite entries")
    midpoint = values.size // 2
    early = values[:midpoint]
    late = values[midpoint:]
    x = np.arange(values.size, dtype=float)
    slope = float(np.polyfit(x, values, 1)[0])
    return {
        "updates": int(values.size),
        "mean": float(values.mean()),
        "early_mean": float(early.mean()),
        "late_mean": float(late.mean()),
        "late_minus_early": float(late.mean() - early.mean()),
        "slope": slope,
        "standard_error": float(values.std(ddof=1) / np.sqrt(values.size)),
    }


def make_runner(args: argparse.Namespace, q: int):
    repo_root = Path(__file__).resolve().parents[2]
    old_experiments = repo_root / "TSP" / "experiments"
    if str(old_experiments) not in sys.path:
        sys.path.insert(0, str(old_experiments))
    from run_mappo_probe_commit import make_harl_runner

    runner_args = argparse.Namespace(**vars(args))
    runner_args.q = q
    runner_args.results_root = args.results_root / f"q{q}"
    runner_args.exp_name = f"progress_g0_q{q}_{args.coupling}"
    return make_harl_runner(
        runner_args,
        q=q,
        seed=args.seed,
        num_env_steps=q * args.rollout_length * args.micro_updates,
        results_root=runner_args.results_root,
        exp_name=runner_args.exp_name,
        use_eval=False,
    )


def run(args: argparse.Namespace) -> dict:
    args.results_root = args.results_root.resolve()
    if args.results_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.results_root}")
    args.results_root.mkdir(parents=True)
    results = {}
    initial_hash = None
    for q in sorted(set(args.candidate_q)):
        runner = make_runner(args, q)
        try:
            current_hash = parameter_hash(runner)
            if initial_hash is None:
                initial_hash = current_hash
            if current_hash != initial_hash:
                raise RuntimeError("candidate branches do not share initialization")
            values = run_updates(runner, args.micro_updates)
            results[str(q)] = {
                "initial_parameter_sha256": current_hash,
                "progress": progress_statistics(values),
                "reward_trace": values.tolist(),
                "charged_messages": args.micro_updates
                * (args.server_overhead + q * args.rollout_length),
                "charged_environment_ticks": args.micro_updates
                * args.rollout_length,
            }
        finally:
            runner.close()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    selected_q = max(
        (int(q) for q in results),
        key=lambda q: (results[str(q)]["progress"]["late_mean"], -q),
    )
    output = {
        "experiment_id": args.experiment_id,
        "role": "outcome-isolated learning-progress sensor qualification",
        "environment": args.env_name,
        "task": (
            f"{args.scenario}/{args.agent_conf}"
            if args.env_name == "mamujoco"
            else args.scenario
        ),
        "coupling": args.coupling,
        "seed": args.seed,
        "candidate_q": sorted(set(args.candidate_q)),
        "micro_updates": args.micro_updates,
        "selected_q_diagnostic": selected_q,
        "results": results,
        "evaluation_return_computed": False,
    }
    path = args.results_root / "progress_sensor_g0.json"
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--harl-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--experiment-id", default="TSP-V2-MARL-PROGRESS-G0")
    parser.add_argument(
        "--env-name", choices=("pettingzoo_mpe", "mamujoco"), required=True
    )
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--agent-conf", default="2x3")
    parser.add_argument("--agent-obsk", type=int, default=0)
    parser.add_argument("--episode-limit", type=int, default=1000)
    parser.add_argument("--coupling", choices=("independent", "shared"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--candidate-q", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--micro-updates", type=int, default=24)
    parser.add_argument("--rollout-length", type=int, default=200)
    parser.add_argument("--server-overhead", type=int, default=800)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--activation", default="tanh")
    parser.add_argument("--actor-lr", type=float, default=5e-4)
    parser.add_argument("--critic-lr", type=float, default=5e-4)
    parser.add_argument("--ppo-epoch", type=int, default=5)
    parser.add_argument("--critic-epoch", type=int, default=5)
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--eval-threads", type=int, default=1)
    parser.add_argument("--eval-episodes", type=int, default=1)
    parser.add_argument("--eval-interval", type=int, default=10_000)
    parser.add_argument("--log-interval", type=int, default=10_000)
    parser.add_argument("--seed-registry-base", type=int, default=140000)
    parser.add_argument("--seed-registry-size", type=int, default=4096)
    parser.add_argument("--continuous-actions", action="store_true")
    parser.add_argument("--share-param", action="store_true")
    parser.add_argument("--cuda", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(args)
    print(
        json.dumps(
            {
                "selected_q_diagnostic": result["selected_q_diagnostic"],
                "evaluation_return_computed": result["evaluation_return_computed"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()


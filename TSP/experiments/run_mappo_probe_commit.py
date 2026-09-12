"""Run the fully charged MAPPO Lyapunov probe-then-commit controller."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import run_mappo_return_bridge as bridge
from lyapunov_probe_commit import choose_participation, controller_accounting


def make_harl_runner(
    args: argparse.Namespace,
    *,
    q: int,
    seed: int,
    num_env_steps: int,
    results_root: Path,
    exp_name: str,
    use_eval: bool,
):
    """Construct a pinned HARL MAPPO runner without editing upstream source."""

    harl_root = args.harl_root.resolve()
    if not (harl_root / "harl" / "runners").is_dir():
        raise FileNotFoundError(f"invalid HARL root: {harl_root}")
    if str(harl_root) not in sys.path:
        sys.path.insert(0, str(harl_root))

    from harl.utils import envs_tools
    from harl.utils.configs_tools import get_defaults_yaml_args

    envs_tools.make_train_env = bridge.coupled_train_env_factory(
        args.coupling, args.seed_registry_base, args.seed_registry_size
    )
    if args.coupling == "shared":
        bridge.install_shared_action_coupling()
    from harl.runners import RUNNER_REGISTRY
    from harl.runners import on_policy_base_runner

    # OnPolicyBaseRunner imports this symbol at module load.  Assign it
    # explicitly so repeated probe/training construction uses the registered
    # coupling factory without altering the HARL checkout.
    on_policy_base_runner.make_train_env = envs_tools.make_train_env

    env_name = getattr(args, "env_name", "pettingzoo_mpe")
    algo_args, env_args = get_defaults_yaml_args("mappo", env_name)
    algo_args["seed"].update({"seed_specify": True, "seed": seed})
    algo_args["device"].update(
        {"cuda": args.cuda, "torch_threads": args.torch_threads}
    )
    algo_args["train"].update(
        {
            "n_rollout_threads": q,
            "num_env_steps": num_env_steps,
            "episode_length": args.rollout_length,
            "log_interval": args.log_interval,
            "eval_interval": args.eval_interval,
            "use_valuenorm": True,
            "use_linear_lr_decay": False,
        }
    )
    algo_args["eval"].update(
        {
            "use_eval": use_eval,
            "n_eval_rollout_threads": args.eval_threads,
            "eval_episodes": args.eval_episodes,
        }
    )
    algo_args["model"].update(
        {
            "hidden_sizes": [args.hidden_size, args.hidden_size],
            "activation_func": args.activation,
            "lr": args.actor_lr,
            "critic_lr": args.critic_lr,
        }
    )
    algo_args["algo"].update(
        {
            "ppo_epoch": args.ppo_epoch,
            "critic_epoch": args.critic_epoch,
            "share_param": True,
            "fixed_order": True,
        }
    )
    algo_args["logger"]["log_dir"] = str(results_root.resolve())
    if env_name == "pettingzoo_mpe":
        env_args.update(
            {
                "scenario": args.scenario,
                "continuous_actions": args.continuous_actions,
            }
        )
    else:
        env_args["map_name"] = args.map_name
    main_args = {"algo": "mappo", "env": env_name, "exp_name": exp_name}
    return RUNNER_REGISTRY["mappo"](main_args, algo_args, env_args)


def residual_fingerprint(residuals: np.ndarray) -> np.ndarray:
    """Reduce a HARL critic-residual block to one scalar per rollout worker."""

    residuals = np.asarray(residuals, dtype=float)
    if residuals.ndim < 2:
        raise ValueError("residuals must contain time and rollout-worker axes")
    reduction_axes = (0,) + tuple(range(2, residuals.ndim))
    result = residuals.mean(axis=reduction_axes)
    if result.ndim != 1:
        raise RuntimeError(f"unexpected fingerprint shape {result.shape}")
    return result


def collect_probe_fingerprints(runner, blocks: int) -> np.ndarray:
    """Collect one unnormalized critic-residual fingerprint per block/worker."""

    if blocks <= 0:
        raise ValueError("probe blocks must be positive")
    runner.warmup()
    fingerprints = []
    for _ in range(blocks):
        runner.prep_rollout()
        for step in range(runner.algo_args["train"]["episode_length"]):
            values, actions, action_log_probs, rnn_states, rnn_states_critic = (
                runner.collect(step)
            )
            obs, share_obs, rewards, dones, infos, available_actions = (
                runner.envs.step(actions)
            )
            runner.insert(
                (
                    obs,
                    share_obs,
                    rewards,
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
        value_predictions = runner.critic_buffer.value_preds[:-1]
        if runner.value_normalizer is not None:
            value_predictions = runner.value_normalizer.denormalize(value_predictions)
        # HARL stores [time, rollout worker, agent, value component].
        fingerprints.append(
            residual_fingerprint(runner.critic_buffer.returns[:-1] - value_predictions)
        )
        runner.after_update()
    result = np.asarray(fingerprints, dtype=float)
    if result.shape != (blocks, runner.algo_args["train"]["n_rollout_threads"]):
        raise RuntimeError(f"unexpected fingerprint shape {result.shape}")
    return result


def charged_progress(
    progress_path: Path,
    *,
    selected_q: int,
    rollout_length: int,
    server_overhead: int,
    message_budget: int,
    environment_budget: int,
    probe_messages: int,
    probe_environment_ticks: int,
) -> list[dict]:
    """Map HARL's training axis to cumulative fully charged budget fractions."""

    rows = []
    for line in progress_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        fields = line.split(",")
        if len(fields) not in {2, 3}:
            raise ValueError("expected return or return-and-win-rate progress row")
        actor_transitions_text, team_return_text = fields[:2]
        actor_transitions = int(actor_transitions_text)
        denominator = selected_q * rollout_length
        if actor_transitions % denominator:
            raise ValueError("evaluation transition count is not an integer update count")
        updates = actor_transitions // denominator
        messages = probe_messages + updates * (
            server_overhead + selected_q * rollout_length
        )
        environment_ticks = probe_environment_ticks + updates * rollout_length
        if messages > message_budget or environment_ticks > environment_budget:
            raise ValueError("progress row exceeds a physical budget")
        row = {
                "budget_fraction": max(
                    messages / message_budget,
                    environment_ticks / environment_budget,
                ),
                "cumulative_messages": messages,
                "cumulative_environment_ticks": environment_ticks,
                "training_actor_transitions": actor_transitions,
                "team_return": float(team_return_text),
            }
        if len(fields) == 3:
            row["win_rate"] = float(fields[2])
        rows.append(row)
    if not rows:
        raise ValueError("training run produced no evaluation rows")
    return rows


def write_charged_progress(path: Path, rows: list[dict]) -> None:
    columns = list(rows[0])
    lines = [",".join(columns)]
    lines.extend(",".join(str(row[column]) for column in columns) for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def dry_run_spec(args: argparse.Namespace) -> dict:
    possible = {}
    for q in sorted(set(args.candidate_q)):
        possible[str(q)] = controller_accounting(
            selected_q=q,
            probe_q=args.probe_q,
            probe_blocks=args.probe_blocks,
            rollout_length=args.rollout_length,
            message_budget=args.message_budget,
            environment_budget=args.environment_budget,
            server_overhead=args.server_overhead,
        ).to_dict()
    env_name = getattr(args, "env_name", "pettingzoo_mpe")
    return {
        "experiment_id": getattr(args, "experiment_id", "TSP-MARL-DEV-002"),
        "environment": env_name,
        "map_name": getattr(args, "map_name", None) if env_name == "smacv2" else None,
        "scenario": args.scenario if env_name == "pettingzoo_mpe" else None,
        "share_param": True,
        "coupling": args.coupling,
        "training_seed": args.seed,
        "probe_seed": args.probe_seed,
        "probe_q": args.probe_q,
        "probe_blocks": args.probe_blocks,
        "candidate_q": sorted(set(args.candidate_q)),
        "possible_accounting": possible,
    }


def run(args: argparse.Namespace) -> Path:
    results_root = args.results_root.resolve()
    if results_root.exists():
        raise FileExistsError(f"refusing to overwrite {results_root}")
    results_root.mkdir(parents=True)

    probe_steps = args.probe_blocks * args.probe_q * args.rollout_length
    probe_start = time.perf_counter()
    probe_runner = make_harl_runner(
        args,
        q=args.probe_q,
        seed=args.probe_seed,
        num_env_steps=probe_steps,
        results_root=results_root / "probe",
        exp_name=f"probe_{args.coupling}",
        use_eval=False,
    )
    probe_run_dir = Path(probe_runner.run_dir)
    try:
        fingerprints = collect_probe_fingerprints(probe_runner, args.probe_blocks)
    finally:
        probe_runner.close()
    probe_seconds = time.perf_counter() - probe_start
    np.save(results_root / "probe_fingerprints.npy", fingerprints)

    selection_start = time.perf_counter()
    decision = choose_participation(
        fingerprints,
        args.candidate_q,
        args.server_overhead,
        args.rollout_length,
        args.correlation_delta,
    )
    selection_seconds = time.perf_counter() - selection_start
    accounting = controller_accounting(
        selected_q=decision.selected_q,
        probe_q=args.probe_q,
        probe_blocks=args.probe_blocks,
        rollout_length=args.rollout_length,
        message_budget=args.message_budget,
        environment_budget=args.environment_budget,
        server_overhead=args.server_overhead,
    )

    train_args = argparse.Namespace(**vars(args))
    train_args.q = decision.selected_q
    train_args.results_root = results_root / "training"
    train_args.exp_name = f"controller_{args.coupling}_q{decision.selected_q}"
    train_args.message_budget = args.message_budget - accounting.probe_messages
    train_args.environment_budget = (
        args.environment_budget - accounting.probe_environment_ticks
    )
    train_args.seed = args.seed
    training_spec = bridge.specification(train_args)
    if training_spec["usable_updates"] != accounting.training_updates:
        raise RuntimeError("controller and fixed-runner accounting disagree")

    training_start = time.perf_counter()
    training_run_dir = bridge.run(train_args, training_spec)
    training_seconds = time.perf_counter() - training_start
    progress_rows = charged_progress(
        training_run_dir / "progress.txt",
        selected_q=decision.selected_q,
        rollout_length=args.rollout_length,
        server_overhead=args.server_overhead,
        message_budget=args.message_budget,
        environment_budget=args.environment_budget,
        probe_messages=accounting.probe_messages,
        probe_environment_ticks=accounting.probe_environment_ticks,
    )
    charged_path = results_root / "charged_progress.csv"
    write_charged_progress(charged_path, progress_rows)

    env_name = getattr(args, "env_name", "pettingzoo_mpe")
    metadata = {
        "experiment_id": getattr(args, "experiment_id", "TSP-MARL-DEV-002"),
        "experiment_role": "development Lyapunov probe-then-commit controller",
        "environment": env_name,
        "map_name": getattr(args, "map_name", None) if env_name == "smacv2" else None,
        "scenario": args.scenario if env_name == "pettingzoo_mpe" else None,
        "share_param": True,
        "method": "lyapunov_probe_commit",
        "coupling": args.coupling,
        "training_seed": args.seed,
        "probe_seed": args.probe_seed,
        "probe_q": args.probe_q,
        "probe_blocks": args.probe_blocks,
        "probe_updates_model": False,
        "critic_lr": args.critic_lr,
        "actor_lr": args.actor_lr,
        "rollout_length": args.rollout_length,
        "message_budget": args.message_budget,
        "environment_budget": args.environment_budget,
        "server_overhead": args.server_overhead,
        "decision": decision.to_dict(),
        "accounting": accounting.to_dict(),
        "probe_seconds": probe_seconds,
        "selection_seconds": selection_seconds,
        "training_seconds": training_seconds,
        "selection_overhead_fraction": selection_seconds
        / max(training_seconds, np.finfo(float).eps),
        "probe_run_dir": str(probe_run_dir.resolve()),
        "training_run_dir": str(training_run_dir.resolve()),
        "charged_progress": str(charged_path.resolve()),
        "harl_commit": bridge.git_head(args.harl_root.resolve()),
        "runner_sha256": bridge.sha256(Path(__file__).resolve()),
        "controller_sha256": bridge.sha256(
            Path(__file__).with_name("lyapunov_probe_commit.py").resolve()
        ),
        "upstream_modified": bool(
            subprocess.check_output(
                ["git", "-C", str(args.harl_root.resolve()), "status", "--porcelain"],
                text=True,
            ).strip()
        ),
    }
    metadata_path = results_root / "tsp_probe_commit_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata_path


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--harl-root", type=Path, default=repo_root / "tmp" / "HARL")
    parser.add_argument("--experiment-id", default="TSP-MARL-DEV-002")
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument(
        "--env-name", choices=("pettingzoo_mpe", "smacv2"), default="pettingzoo_mpe"
    )
    parser.add_argument("--scenario", default="simple_spread_v2")
    parser.add_argument("--map-name", default="terran_10_vs_10")
    parser.add_argument("--coupling", choices=("independent", "shared"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--probe-seed", type=int, required=True)
    parser.add_argument("--seed-registry-base", type=int, default=90000)
    parser.add_argument("--seed-registry-size", type=int, default=10000)
    parser.add_argument("--probe-q", type=int, default=8)
    parser.add_argument("--probe-blocks", type=int, default=128)
    parser.add_argument("--candidate-q", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--correlation-delta", type=float, default=0.05)
    parser.add_argument("--rollout-length", type=int, default=25)
    parser.add_argument("--message-budget", type=int, default=5_000_000)
    parser.add_argument("--environment-budget", type=int, default=1_000_000)
    parser.add_argument("--server-overhead", type=int, default=100)
    parser.add_argument("--critic-lr", type=float, default=5e-4)
    parser.add_argument("--actor-lr", type=float, default=5e-4)
    parser.add_argument("--continuous-actions", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--eval-threads", type=int, default=4)
    parser.add_argument("--eval-episodes", type=int, default=32)
    parser.add_argument("--eval-interval", type=int, default=1000)
    parser.add_argument("--log-interval", type=int, default=1000)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--activation", choices=("relu", "tanh"), default="tanh")
    parser.add_argument("--ppo-epoch", type=int, default=2)
    parser.add_argument("--critic-epoch", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run:
        print(json.dumps(dry_run_spec(args), indent=2, sort_keys=True))
        return
    metadata = run(args)
    print(json.dumps({"metadata": str(metadata.resolve())}, sort_keys=True))


if __name__ == "__main__":
    main()

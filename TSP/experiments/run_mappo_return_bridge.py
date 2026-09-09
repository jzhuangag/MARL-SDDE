"""Run a fully charged fixed-action MAPPO bridge for TSP development.

This is an action evaluator, not the adaptive controller.  It leaves the
upstream HARL checkout untouched and changes only the cross-rollout seed
coupling used by the training environments.  Evaluation environments always
use HARL's independent deterministic evaluation streams.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def usable_updates(
    q: int,
    rollout_length: int,
    message_budget: int,
    environment_budget: int,
    server_overhead: int,
) -> int:
    if min(q, rollout_length, message_budget, environment_budget) <= 0:
        raise ValueError("q, rollout length, and budgets must be positive")
    if server_overhead < 0:
        raise ValueError("server overhead must be nonnegative")
    message_cost = server_overhead + q * rollout_length
    environment_cost = rollout_length
    return min(message_budget // message_cost, environment_budget // environment_cost)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_head(root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()


def specification(args: argparse.Namespace) -> dict:
    updates = usable_updates(
        args.q,
        args.rollout_length,
        args.message_budget,
        args.environment_budget,
        args.server_overhead,
    )
    if updates < 2:
        raise ValueError("budgets admit fewer than two learner updates")
    return {
        "experiment_role": "development fixed-action evaluator",
        "scenario": args.scenario,
        "continuous_actions": args.continuous_actions,
        "coupling": args.coupling,
        "q_rollout_workers": args.q,
        "rollout_length": args.rollout_length,
        "critic_lr": args.critic_lr,
        "actor_lr": args.actor_lr,
        "message_budget": args.message_budget,
        "environment_budget": args.environment_budget,
        "server_overhead": args.server_overhead,
        "message_cost_per_update": args.server_overhead + args.q * args.rollout_length,
        "environment_cost_per_update": args.rollout_length,
        "usable_updates": updates,
        "charged_training_actor_transitions": updates * args.q * args.rollout_length,
        "charged_training_environment_ticks": updates * args.rollout_length,
        "charged_training_messages": updates
        * (args.server_overhead + args.q * args.rollout_length),
        "harl_num_env_steps": updates * args.q * args.rollout_length,
        "seed": args.seed,
        "seed_registry_base": args.seed_registry_base,
        "seed_registry_size": args.seed_registry_size,
    }


def worker_seed(
    coupling: str, run_seed: int, rank: int, registry_base: int, registry_size: int
) -> int:
    if registry_size <= 0 or not registry_base <= run_seed < registry_base + registry_size:
        raise ValueError("run seed must belong to the declared contiguous registry")
    if coupling == "shared":
        return run_seed
    index = run_seed - registry_base
    return registry_base + (index + rank) % registry_size


def common_categorical_sample(probabilities, uniform=None):
    """Inverse-CDF categorical sample with one public uniform across workers."""
    import torch

    if probabilities.ndim < 2:
        raise ValueError("expected worker-by-action probabilities")
    if uniform is None:
        shape = (1,) + tuple(probabilities.shape[1:-1]) + (1,)
        uniform = torch.rand(shape, device=probabilities.device)
    cumulative = probabilities.cumsum(dim=-1)
    return (uniform > cumulative).sum(dim=-1, keepdim=True)


def install_shared_action_coupling() -> None:
    from harl.models.base.distributions import FixedCategorical

    def sample(self):
        return common_categorical_sample(self.probs)

    FixedCategorical.sample = sample


def coupled_train_env_factory(coupling: str, registry_base: int, registry_size: int):
    from harl.envs.env_wrappers import ShareDummyVecEnv, ShareSubprocVecEnv

    def make_train_env(env_name, seed, n_threads, env_args):
        if env_name != "pettingzoo_mpe":
            raise ValueError("the TSP bridge currently supports pettingzoo_mpe only")

        def get_env_fn(rank):
            def init_env():
                from harl.envs.pettingzoo_mpe.pettingzoo_mpe_env import (
                    PettingZooMPEEnv,
                )

                env = PettingZooMPEEnv(env_args)
                env_seed = worker_seed(
                    coupling, seed, rank, registry_base, registry_size
                )
                env.seed(env_seed)
                return env

            return init_env

        constructors = [get_env_fn(rank) for rank in range(n_threads)]
        return ShareDummyVecEnv(constructors) if n_threads == 1 else ShareSubprocVecEnv(constructors)

    return make_train_env


def run(args: argparse.Namespace, spec: dict) -> Path:
    harl_root = args.harl_root.resolve()
    if not (harl_root / "harl" / "runners").is_dir():
        raise FileNotFoundError(f"invalid HARL root: {harl_root}")
    sys.path.insert(0, str(harl_root))

    from harl.utils import envs_tools
    from harl.utils.configs_tools import get_defaults_yaml_args

    envs_tools.make_train_env = coupled_train_env_factory(
        args.coupling, args.seed_registry_base, args.seed_registry_size
    )
    if args.coupling == "shared":
        install_shared_action_coupling()
    # Import after patching: OnPolicyBaseRunner copies this symbol at import.
    from harl.runners import RUNNER_REGISTRY

    algo_args, env_args = get_defaults_yaml_args("mappo", "pettingzoo_mpe")
    algo_args["seed"].update({"seed_specify": True, "seed": args.seed})
    algo_args["device"].update({"cuda": args.cuda, "torch_threads": args.torch_threads})
    algo_args["train"].update(
        {
            "n_rollout_threads": args.q,
            "num_env_steps": spec["harl_num_env_steps"],
            "episode_length": args.rollout_length,
            "log_interval": args.log_interval,
            "eval_interval": args.eval_interval,
            "use_valuenorm": True,
            "use_linear_lr_decay": False,
        }
    )
    algo_args["eval"].update(
        {
            "use_eval": True,
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
    algo_args["logger"]["log_dir"] = str(args.results_root.resolve())
    env_args.update(
        {
            "scenario": args.scenario,
            "continuous_actions": args.continuous_actions,
        }
    )

    main_args = {
        "algo": "mappo",
        "env": "pettingzoo_mpe",
        "exp_name": args.exp_name,
    }
    runner = RUNNER_REGISTRY["mappo"](main_args, algo_args, env_args)
    run_dir = Path(runner.run_dir)
    try:
        runner.run()
    finally:
        runner.close()

    metadata = {
        **spec,
        "harl_commit": git_head(harl_root),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "run_dir": str(run_dir.resolve()),
        "upstream_modified": bool(
            subprocess.check_output(
                ["git", "-C", str(harl_root), "status", "--porcelain"], text=True
            ).strip()
        ),
    }
    (run_dir / "tsp_bridge_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return run_dir


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    tsp_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--harl-root", type=Path, default=repo_root / "tmp" / "HARL")
    # Keep the default short enough for tensorboardX's nested tags on Windows.
    parser.add_argument("--results-root", type=Path, default=tsp_root / "tmp" / "mr")
    parser.add_argument("--exp-name", default="bridge")
    parser.add_argument("--scenario", default="simple_spread_v2")
    parser.add_argument("--coupling", choices=("independent", "shared"), required=True)
    parser.add_argument("--q", type=int, required=True)
    parser.add_argument("--rollout-length", type=int, default=25)
    parser.add_argument("--message-budget", type=int, default=320_000)
    parser.add_argument("--environment-budget", type=int, default=100_000)
    parser.add_argument("--server-overhead", type=int, default=100)
    parser.add_argument("--critic-lr", type=float, default=5e-4)
    parser.add_argument("--actor-lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=92001)
    parser.add_argument("--seed-registry-base", type=int, default=92001)
    parser.add_argument("--seed-registry-size", type=int, default=128)
    parser.add_argument("--continuous-actions", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--eval-threads", type=int, default=2)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--activation", choices=("relu", "tanh"), default="tanh")
    parser.add_argument("--ppo-epoch", type=int, default=2)
    parser.add_argument("--critic-epoch", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = specification(args)
    if args.dry_run:
        print(json.dumps(spec, indent=2, sort_keys=True))
        return
    output = run(args, spec)
    print(json.dumps({"run_dir": str(output.resolve()), **spec}, sort_keys=True))


if __name__ == "__main__":
    main()

"""Outcome-free MaMuJoCo controller-interface smoke test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lyapunov_probe_commit import average_pairwise_correlation_certificate
from run_mappo_probe_commit import collect_probe_fingerprints, make_harl_runner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--harl-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--coupling", choices=("independent", "shared"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args0 = parser.parse_args()
    args = argparse.Namespace(
        harl_root=args0.harl_root,
        coupling=args0.coupling,
        seed_registry_base=127001,
        seed_registry_size=128,
        cuda=True,
        torch_threads=8,
        rollout_length=200,
        log_interval=200,
        eval_interval=200,
        eval_threads=2,
        eval_episodes=2,
        hidden_size=128,
        activation="tanh",
        actor_lr=5e-4,
        critic_lr=5e-4,
        ppo_epoch=5,
        critic_epoch=5,
        share_param=True,
        env_name="mamujoco",
        scenario="HalfCheetah-v2",
        map_name="",
        agent_conf="2x3",
        agent_obsk=0,
        episode_limit=1000,
        continuous_actions=True,
    )
    runner = make_harl_runner(
        args,
        q=8,
        seed=args0.seed,
        num_env_steps=8 * 8 * 200,
        results_root=args0.results_root,
        exp_name=f"mamujoco_g0_{args0.coupling}",
        use_eval=False,
    )
    try:
        fingerprints = collect_probe_fingerprints(runner, 8)
    finally:
        runner.close()
    certificate = average_pairwise_correlation_certificate(fingerprints, 0.05)
    result = {
        "task": "HalfCheetah-v2/2x3",
        "coupling": args0.coupling,
        "shape": list(fingerprints.shape),
        "finite": bool(np.isfinite(fingerprints).all()),
        "worker_mean_max_abs": float(np.max(np.abs(fingerprints.mean(axis=0)))),
        "worker_variance_min": float(fingerprints.var(axis=0, ddof=1).min()),
        "correlation_estimate": certificate.estimate,
        "correlation_upper": certificate.upper,
        "contains_return": False,
    }
    args0.output.parent.mkdir(parents=True, exist_ok=True)
    args0.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

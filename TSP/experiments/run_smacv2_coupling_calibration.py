"""Probe-only qualification of the SMACv2 rollout coupling.

This integration experiment never updates a policy and never observes return
or win-rate curves.  It estimates only the cross-worker dependence of the
predeclared critic-residual fingerprints used by the controller.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

import run_mappo_probe_commit as controller
from lyapunov_probe_commit import choose_participation


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dependence_summary(fingerprints: np.ndarray) -> dict:
    x = np.asarray(fingerprints, dtype=float)
    if x.ndim != 2 or min(x.shape) < 2 or not np.isfinite(x).all():
        raise ValueError("fingerprints must be a finite blocks-by-workers matrix")
    correlation = np.corrcoef(x, rowvar=False)
    upper = correlation[np.triu_indices(x.shape[1], k=1)]
    if not np.isfinite(upper).all():
        raise ValueError("worker fingerprints must have nonzero variation")
    return {
        "blocks": int(x.shape[0]),
        "workers": int(x.shape[1]),
        "median_pairwise_correlation": float(np.median(upper)),
        "minimum_pairwise_correlation": float(np.min(upper)),
        "maximum_pairwise_correlation": float(np.max(upper)),
        "maximum_within_block_range": float(np.max(np.ptp(x, axis=1))),
        "pooled_mean": float(np.mean(x)),
        "pooled_standard_deviation": float(np.std(x, ddof=1)),
    }


def run(args: argparse.Namespace) -> Path:
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    probe_steps = args.blocks * args.workers * args.rollout_length
    start = time.perf_counter()
    runner = controller.make_harl_runner(
        args,
        q=args.workers,
        seed=args.seed,
        num_env_steps=probe_steps,
        results_root=output / "probe",
        exp_name=f"coupling_calibration_{args.coupling}",
        use_eval=False,
    )
    try:
        fingerprints = controller.collect_probe_fingerprints(runner, args.blocks)
    finally:
        runner.close()
    elapsed = time.perf_counter() - start
    fingerprint_path = output / "probe_fingerprints.npy"
    np.save(fingerprint_path, fingerprints)
    decision = choose_participation(
        fingerprints,
        args.candidate_q,
        args.server_overhead,
        args.rollout_length,
        args.correlation_delta,
    )
    metadata = {
        "experiment_id": "TSP-SMACV2-COUPLING-CAL-001",
        "role": "outcome-free probe-only integration qualification",
        "coupling": args.coupling,
        "seed": args.seed,
        "environment": "smacv2",
        "map_name": args.map_name,
        "policy_updates": 0,
        "return_or_win_rate_observed": False,
        "probe_actor_transitions": probe_steps,
        "elapsed_seconds": elapsed,
        "dependence": dependence_summary(fingerprints),
        "decision": decision.to_dict(),
        "fingerprint_sha256": sha256(fingerprint_path),
        "runner_sha256": sha256(Path(__file__).resolve()),
    }
    metadata_path = output / "calibration.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, sort_keys=True))
    return metadata_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--harl-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-name", choices=("smacv2",), default="smacv2")
    parser.add_argument("--coupling", choices=("independent", "shared"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--seed-registry-base", type=int, required=True)
    parser.add_argument("--seed-registry-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--blocks", type=int, default=64)
    parser.add_argument("--rollout-length", type=int, default=200)
    parser.add_argument("--map-name", default="terran_10_vs_10")
    parser.add_argument("--candidate-q", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--correlation-delta", type=float, default=0.05)
    parser.add_argument("--server-overhead", type=int, default=800)
    parser.add_argument("--critic-lr", type=float, default=5e-4)
    parser.add_argument("--actor-lr", type=float, default=5e-4)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--activation", default="relu")
    parser.add_argument("--ppo-epoch", type=int, default=5)
    parser.add_argument("--critic-epoch", type=int, default=5)
    parser.add_argument("--eval-threads", type=int, default=1)
    parser.add_argument("--eval-episodes", type=int, default=1)
    parser.add_argument("--eval-interval", type=int, default=64)
    parser.add_argument("--log-interval", type=int, default=64)
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--continuous-actions", action="store_true")
    parser.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

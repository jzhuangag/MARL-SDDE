"""Audit exact per-worker reset marginals for the MAPPO coupling bridge."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


def load_bridge(path: Path):
    spec = importlib.util.spec_from_file_location("tsp_mappo_bridge", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def digest_observation(observation) -> str:
    array = np.asarray(observation, dtype=np.float32)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def audit(harl_root: Path, seed_base: int, seed_count: int, workers: int) -> dict:
    sys.path.insert(0, str(harl_root.resolve()))
    from harl.envs.pettingzoo_mpe.pettingzoo_mpe_env import PettingZooMPEEnv

    bridge = load_bridge(Path(__file__).with_name("run_mappo_return_bridge.py"))
    regimes = {}
    args = {
        "scenario": "simple_spread_v2",
        "continuous_actions": False,
        "max_cycles": 25,
    }
    for coupling in ("independent", "shared"):
        worker_hashes = [[] for _ in range(workers)]
        same_run_unique = []
        for run_seed in range(seed_base, seed_base + seed_count):
            hashes = []
            for rank in range(workers):
                env = PettingZooMPEEnv(args)
                try:
                    env.seed(
                        bridge.worker_seed(
                            coupling, run_seed, rank, seed_base, seed_count
                        )
                    )
                    _, shared_observation, _ = env.reset()
                    digest = digest_observation(shared_observation[0])
                    worker_hashes[rank].append(digest)
                    hashes.append(digest)
                finally:
                    env.close()
            same_run_unique.append(len(set(hashes)))
        regimes[coupling] = {
            "worker_hashes": worker_hashes,
            "same_run_unique_min": min(same_run_unique),
            "same_run_unique_max": max(same_run_unique),
        }

    marginal_match = all(
        sorted(regimes["independent"]["worker_hashes"][rank])
        == sorted(regimes["shared"]["worker_hashes"][rank])
        for rank in range(workers)
    )
    shared_synchrony = regimes["shared"]["same_run_unique_max"] == 1
    independent_diversity = regimes["independent"]["same_run_unique_min"] > 1
    return {
        "audit": "marginal-preserving reset coupling",
        "seed_base": seed_base,
        "seed_count": seed_count,
        "workers": workers,
        "marginal_hash_multisets_match": marginal_match,
        "shared_same_run_synchrony": shared_synchrony,
        "independent_same_run_diversity": independent_diversity,
        "pass": marginal_match and shared_synchrony and independent_diversity,
        "regime_diagnostics": {
            key: {
                "same_run_unique_min": value["same_run_unique_min"],
                "same_run_unique_max": value["same_run_unique_max"],
            }
            for key, value in regimes.items()
        },
    }


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--harl-root", type=Path, default=repo / "tmp" / "HARL")
    parser.add_argument("--seed-base", type=int, default=93001)
    parser.add_argument("--seed-count", type=int, default=17)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=repo / "TSP" / "internal" / "marl_coupling_audit.json",
    )
    args = parser.parse_args()
    result = audit(args.harl_root, args.seed_base, args.seed_count, args.workers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

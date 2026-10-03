"""Audit the outcome-free coupling implementation for the SMACv2 study."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch


def load_bridge(path: Path):
    spec = importlib.util.spec_from_file_location("tsp_mappo_bridge", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def audit(config_path: Path, smoke_root: Path | None = None) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    bridge = load_bridge(Path(__file__).with_name("run_mappo_return_bridge.py"))
    workers = int(config["controller"]["probe_q"])
    registry_base = min(config["seed_registry"]["development_training"])
    registry_size = 128
    registry = list(range(registry_base, registry_base + registry_size))

    independent = [
        [
            bridge.worker_seed(
                "independent", run_seed, rank, registry_base, registry_size
            )
            for run_seed in registry
        ]
        for rank in range(workers)
    ]
    shared = [
        [
            bridge.worker_seed("shared", run_seed, rank, registry_base, registry_size)
            for run_seed in registry
        ]
        for rank in range(workers)
    ]
    marginal_seed_multisets_match = all(
        sorted(independent[rank]) == sorted(shared[rank]) == registry
        for rank in range(workers)
    )
    independent_within_run_unique = all(
        len(
            {
                bridge.worker_seed(
                    "independent", run_seed, rank, registry_base, registry_size
                )
                for rank in range(workers)
            }
        )
        == workers
        for run_seed in registry
    )
    shared_within_run_synchronous = all(
        len(
            {
                bridge.worker_seed(
                    "shared", run_seed, rank, registry_base, registry_size
                )
                for rank in range(workers)
            }
        )
        == 1
        for run_seed in registry
    )

    probabilities = torch.tensor(
        [[0.05, 0.15, 0.80], [0.60, 0.25, 0.15]], dtype=torch.float64
    )
    uniforms = torch.linspace(0.0005, 0.9995, 1000, dtype=torch.float64).reshape(
        1000, 1, 1
    )
    expanded = probabilities.unsqueeze(0).expand(1000, -1, -1)
    samples = bridge.common_categorical_sample(expanded, uniforms).squeeze(-1)
    frequencies = torch.stack(
        [(samples == action).double().mean(dim=0) for action in range(3)], dim=1
    )
    categorical_marginal_max_error = float(
        torch.max(torch.abs(frequencies - probabilities)).item()
    )
    categorical_marginal_match = categorical_marginal_max_error <= 0.0011

    smoke = {
        "requested": smoke_root is not None,
        "complete_runs": 0,
        "metadata_environment_and_map_match": False,
    }
    if smoke_root is not None and smoke_root.exists():
        metadata_paths = list(smoke_root.rglob("tsp_bridge_metadata.json"))
        valid = []
        for path in metadata_paths:
            metadata = json.loads(path.read_text(encoding="utf-8"))
            valid.append(
                metadata.get("environment") == "smacv2"
                and metadata.get("map_name") == config["task"]["map_name"]
                and metadata.get("harl_commit")
                == config["upstream"]["harl_commit"]
                and not metadata.get("upstream_modified")
            )
        smoke = {
            "requested": True,
            "complete_runs": len(metadata_paths),
            "metadata_environment_and_map_match": bool(valid) and all(valid),
        }

    passed = all(
        (
            marginal_seed_multisets_match,
            independent_within_run_unique,
            shared_within_run_synchronous,
            categorical_marginal_match,
            not smoke["requested"]
            or (
                smoke["complete_runs"] >= 2
                and smoke["metadata_environment_and_map_match"]
            ),
        )
    )
    return {
        "audit": "SMACv2 marginal-preserving rollout coupling",
        "registry_base": registry_base,
        "registry_size": registry_size,
        "workers": workers,
        "marginal_seed_multisets_match": marginal_seed_multisets_match,
        "independent_within_run_unique": independent_within_run_unique,
        "shared_within_run_synchronous": shared_within_run_synchronous,
        "categorical_marginal_max_error": categorical_marginal_max_error,
        "categorical_marginal_match": categorical_marginal_match,
        "smoke": smoke,
        "pass": passed,
    }


def main() -> None:
    tsp_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=tsp_root / "experiments" / "marl_smacv2_development.json",
    )
    parser.add_argument("--smoke-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.config, args.smoke_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

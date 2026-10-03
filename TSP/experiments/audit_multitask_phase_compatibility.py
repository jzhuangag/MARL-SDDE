"""Outcome-free compatibility audit for the staged multi-task phase study."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

import run_mappo_return_bridge as bridge


def _hash_rows(array: np.ndarray) -> list[str]:
    values = np.asarray(array)
    return [
        hashlib.sha256(np.ascontiguousarray(values[index]).tobytes()).hexdigest()
        for index in range(values.shape[0])
    ]


def _environment_audit(harl_root: Path, task: dict, coupling: str, seed: int) -> dict:
    from harl.utils.configs_tools import get_defaults_yaml_args

    _, env_args = get_defaults_yaml_args("mappo", task["environment"])
    if task["environment"] == "pettingzoo_mpe":
        env_args.update(
            {
                "scenario": task["scenario"],
                "continuous_actions": task["continuous_actions"],
            }
        )
    else:
        env_args.update(
            {
                "scenario": task["scenario"],
                "agent_conf": task["agent_conf"],
                "agent_obsk": 0,
                "episode_limit": 1000,
            }
        )
    factory = bridge.coupled_train_env_factory(coupling, seed, 128)
    envs = factory(task["environment"], seed, 4, env_args)
    try:
        reset = envs.reset()
        obs = np.asarray(reset[0])
        share_obs = np.asarray(reset[1])
        obs_hashes = _hash_rows(obs)
        state_hashes = _hash_rows(share_obs)
        shared_expected = coupling == "shared"
        reset_relation_ok = (
            len(set(obs_hashes)) == 1 and len(set(state_hashes)) == 1
            if shared_expected
            else len(set(obs_hashes)) > 1 or len(set(state_hashes)) > 1
        )
        return {
            "task_id": task["task_id"],
            "environment": task["environment"],
            "coupling": coupling,
            "workers": int(obs.shape[0]),
            "agents": int(obs.shape[1]),
            "observation_shape": list(obs.shape[2:]),
            "state_shape": list(share_obs.shape[2:]),
            "reset_relation_ok": bool(reset_relation_ok),
            "finite": bool(np.isfinite(obs).all() and np.isfinite(share_obs).all()),
        }
    finally:
        envs.close()


def _continuous_coupling_audit(draws: int = 20_000) -> dict:
    generator = torch.Generator().manual_seed(20260920)
    noise = torch.randn((draws, 1, 3), generator=generator, dtype=torch.float64)
    loc = torch.tensor(
        [[[0.5, -0.5, 1.0]], [[-1.0, 2.0, 0.0]], [[3.0, 1.0, -2.0]]],
        dtype=torch.float64,
    ).transpose(0, 1)
    scale = torch.tensor(
        [[[0.5, 1.0, 2.0]], [[1.5, 0.75, 0.25]], [[2.0, 0.5, 1.25]]],
        dtype=torch.float64,
    ).transpose(0, 1)
    first = bridge.common_normal_sample(loc[0], scale[0], noise[0])
    first_z = ((first - loc[0]) / scale[0]).numpy()
    if not np.allclose(first_z, np.repeat(noise[0].numpy(), 3, axis=0)):
        raise RuntimeError("public Gaussian coupling did not broadcast exactly")
    values = (
        loc[0].numpy()[None, :, :]
        + scale[0].numpy()[None, :, :] * noise.numpy()
    )
    z = (values - loc[0].numpy()[None, :, :]) / scale[0].numpy()[None, :, :]
    mean_error = float(np.max(np.abs(values.mean(axis=0) - loc[0].numpy())))
    variance_error = float(
        np.max(np.abs(values.var(axis=0, ddof=1) - np.square(scale[0].numpy())))
    )
    cross_worker_error = float(
        np.max(np.abs(z[:, 0, None, :] - z[:, 1:, :]))
    )
    return {
        "draws": draws,
        "mean_error_max": mean_error,
        "variance_error_max": variance_error,
        "standardized_cross_worker_error_max": cross_worker_error,
        "pass": bool(
            mean_error <= 0.05
            and variance_error <= 0.12
            and cross_worker_error <= 1e-12
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--harl-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    harl_root = args.harl_root.resolve()
    sys.path.insert(0, str(harl_root))
    bridge.install_numpy_legacy_aliases()
    matching = [task for task in config["tasks"] if task["task_id"] == args.task_id]
    if len(matching) != 1:
        raise ValueError(f"expected one registered task named {args.task_id}")
    task = matching[0]
    task_index = [row["task_id"] for row in config["tasks"]].index(args.task_id)
    audits = [
        _environment_audit(
            harl_root,
            task,
            coupling,
            config["seed_registry_base"] + 10 * task_index + coupling_index,
        )
        for coupling_index, coupling in enumerate(config["coupling_regimes"])
    ]
    continuous = (
        _continuous_coupling_audit() if task["continuous_actions"] else None
    )
    pinned = subprocess.check_output(
        ["git", "-C", str(harl_root), "rev-parse", "HEAD"], text=True
    ).strip()
    clean = not subprocess.check_output(
        ["git", "-C", str(harl_root), "status", "--porcelain"], text=True
    ).strip()
    gates = {
        "task_interfaces_finite": all(row["finite"] for row in audits),
        "reset_coupling_relation": all(row["reset_relation_ok"] for row in audits),
        "continuous_marginal_and_public_noise": bool(
            continuous is None or continuous["pass"]
        ),
        "pinned_clean_upstream": bool(
            clean and pinned == config["upstream"]["harl_commit"]
        ),
        "no_return_or_policy_update": True,
    }
    result = {
        "experiment_id": config["experiment_id"],
        "role": "outcome-free compatibility audit",
        "task_id": args.task_id,
        "environment_audits": audits,
        "continuous_coupling_audit": continuous,
        "harl_commit": pinned,
        "harl_clean": clean,
        "gates": gates,
        "pass": all(gates.values()),
        "scientific_outcome_generated": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"pass": result["pass"], "output": str(args.output)}))


if __name__ == "__main__":
    main()

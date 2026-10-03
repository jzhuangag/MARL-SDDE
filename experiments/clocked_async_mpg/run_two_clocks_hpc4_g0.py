"""Outcome-free GPU/environment qualification for the Two Clocks runtime.

This is an installation and interface gate, not a learning experiment.  It
starts pinned HARL environments, performs one environment transition, runs
distinct actor modules on CUDA, and checks process teardown.  The emitted JSON
is structurally forbidden from containing task-performance outcomes.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable

import numpy as np


EXPECTED_SCOPE = "outcome-free Two Clocks HPC4 G0"
PROHIBITED_KEYS = frozenset(
    {
        "episode_reward",
        "reward",
        "return",
        "success_rate",
        "win_rate",
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _assert_outcome_free(value: object, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).strip().lower()
            if normalized in PROHIBITED_KEYS or normalized.endswith("_return"):
                raise RuntimeError(f"prohibited G0 outcome key: {path}.{key}")
            _assert_outcome_free(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_outcome_free(child, f"{path}[{index}]")


def validate_summary(summary: dict[str, Any]) -> None:
    _assert_outcome_free(summary)
    if summary.get("scope") != EXPECTED_SCOPE:
        raise RuntimeError("unexpected G0 scope")
    if summary.get("scientific_outcome_generated") is not False:
        raise RuntimeError("G0 must explicitly exclude scientific outcomes")
    tasks = summary.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 2:
        raise RuntimeError("G0 must cover exactly the registered two task shapes")
    if {task.get("task") for task in tasks} != {"mamujoco_ant_4x2", "smacv2_terran_5v5"}:
        raise RuntimeError("unexpected G0 task set")
    invariants = summary.get("invariants")
    if not isinstance(invariants, dict) or not invariants or not all(invariants.values()):
        raise RuntimeError("not every G0 invariant passed")
    if not all(task.get("status") == "pass" for task in tasks):
        raise RuntimeError("a G0 task did not pass")


def _clean_checkout(path: Path, expected_commit: str, label: str) -> dict[str, str]:
    path = path.resolve()
    actual = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(path), "status", "--short"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual != expected_commit:
        raise RuntimeError(f"{label} commit {actual} != {expected_commit}")
    if dirty:
        raise RuntimeError(f"{label} checkout is not clean")
    return {"commit": actual, "path": str(path)}


def _policy_args(harl_root: Path) -> dict[str, Any]:
    import yaml

    config_path = harl_root / "harl" / "configs" / "algos_cfgs" / "haa2c.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        configuration = yaml.safe_load(handle)
    args: dict[str, Any] = {}
    for section in ("model", "algo"):
        args.update(configuration[section])
    args["hidden_sizes"] = [64, 64]
    args["use_naive_recurrent_policy"] = False
    args["use_recurrent_policy"] = False
    return args


def _children(process: Any) -> set[int]:
    return {child.pid for child in process.children(recursive=True) if child.is_running()}


def _wait_for_children_to_exit(process: Any, original: Iterable[int]) -> set[int]:
    import psutil

    original = set(original)
    deadline = time.monotonic() + 15.0
    remaining: set[int] = set()
    while time.monotonic() < deadline:
        remaining = _children(process) - original
        if not remaining:
            return set()
        time.sleep(0.25)
    for pid in remaining:
        try:
            psutil.Process(pid).terminate()
        except psutil.Error:
            pass
    processes = []
    for pid in remaining:
        try:
            processes.append(psutil.Process(pid))
        except psutil.Error:
            pass
    _, alive = psutil.wait_procs(processes, timeout=5.0)
    return {child.pid for child in alive if child.is_running()}


def _forward_distinct_policies(
    *,
    observations: list[np.ndarray],
    observation_spaces: list[Any] | tuple[Any, ...],
    action_spaces: list[Any] | tuple[Any, ...],
    available_actions: list[np.ndarray] | None,
    harl_root: Path,
    device: Any,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    import torch
    from harl.models.policy_models.stochastic_policy import StochasticPolicy

    args = _policy_args(harl_root)
    policies = [
        StochasticPolicy(args, observation_spaces[index], action_spaces[index], device)
        for index in range(len(observations))
    ]
    if len({id(policy) for policy in policies}) != len(policies):
        raise RuntimeError("G0 requires distinct actor objects")
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    actions: list[np.ndarray] = []
    action_shapes: list[list[int]] = []
    log_probability_shapes: list[list[int]] = []
    with torch.inference_mode():
        for index, policy in enumerate(policies):
            obs = np.asarray(observations[index], dtype=np.float32)[None, :]
            rnn = np.zeros((1, args["recurrent_n"], args["hidden_sizes"][-1]), dtype=np.float32)
            masks = np.ones((1, 1), dtype=np.float32)
            available = None
            if available_actions is not None:
                available = np.asarray(available_actions[index], dtype=np.float32)[None, :]
            action, log_probability, _ = policy(
                obs, rnn, masks, available_actions=available, deterministic=False
            )
            action_array = action.detach().cpu().numpy()[0]
            actions.append(action_array)
            action_shapes.append(list(action.shape))
            log_probability_shapes.append(list(log_probability.shape))
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    parameter_counts = [
        sum(parameter.numel() for parameter in policy.parameters())
        for policy in policies
    ]
    details = {
        "actors": len(policies),
        "distinct_actor_objects": True,
        "parameter_counts": parameter_counts,
        "action_shapes": action_shapes,
        "log_probability_shapes": log_probability_shapes,
        "cuda_forward_seconds": elapsed,
        "cuda_peak_bytes": int(torch.cuda.max_memory_allocated(device)),
    }
    del policies
    gc.collect()
    torch.cuda.empty_cache()
    return actions, details


def _run_mamujoco(harl_root: Path, device: Any, seed: int) -> dict[str, Any]:
    from harl.envs.mamujoco.multiagent_mujoco.mujoco_multi import MujocoMulti

    environment = MujocoMulti(
        env_args={
            "scenario": "Ant-v2",
            "agent_conf": "4x2",
            "agent_obsk": 0,
            "episode_limit": 25,
        }
    )
    try:
        environment.seed(seed)
        observations, states, available = environment.reset()
        actions, policy = _forward_distinct_policies(
            observations=observations,
            observation_spaces=environment.observation_space,
            action_spaces=environment.action_space,
            available_actions=available,
            harl_root=harl_root,
            device=device,
        )
        environment.step(actions)
        return {
            "task": "mamujoco_ant_4x2",
            "status": "pass",
            "agents": environment.n_agents,
            "observation_shapes": [list(np.asarray(value).shape) for value in observations],
            "state_shapes": [list(np.asarray(value).shape) for value in states],
            "action_space_types": [space.__class__.__name__ for space in environment.action_space],
            "environment_transitions": 1,
            "actor_transitions": environment.n_agents,
            "policy": policy,
        }
    finally:
        environment.close()


def _run_smacv2(harl_root: Path, device: Any, seed: int) -> dict[str, Any]:
    from harl.envs.smacv2.smacv2_env import SMACv2Env

    environment = SMACv2Env({"map_name": "terran_5_vs_5"})
    try:
        environment.seed(seed)
        observations, states, available = environment.reset()
        actions, policy = _forward_distinct_policies(
            observations=observations,
            observation_spaces=environment.observation_space,
            action_spaces=environment.action_space,
            available_actions=available,
            harl_root=harl_root,
            device=device,
        )
        discrete_actions = np.asarray(
            [[int(np.asarray(action).reshape(-1)[0])] for action in actions], dtype=np.int64
        )
        environment.step(discrete_actions)
        return {
            "task": "smacv2_terran_5v5",
            "status": "pass",
            "agents": environment.n_agents,
            "observation_shapes": [list(np.asarray(value).shape) for value in observations],
            "state_shapes": [list(np.asarray(value).shape) for value in states],
            "available_action_shapes": [list(np.asarray(value).shape) for value in available],
            "action_space_types": [space.__class__.__name__ for space in environment.action_space],
            "environment_transitions": 1,
            "actor_transitions": environment.n_agents,
            "policy": policy,
        }
    finally:
        environment.close()


def run(args: argparse.Namespace) -> dict[str, Any]:
    harl_root = Path(args.harl_root).resolve()
    smacv2_root = Path(args.smacv2_root).resolve()
    sources = {
        "harl": _clean_checkout(harl_root, args.harl_commit, "HARL"),
        "smacv2": _clean_checkout(smacv2_root, args.smacv2_commit, "SMACv2"),
    }
    for root in (str(harl_root), str(smacv2_root)):
        if root not in sys.path:
            sys.path.insert(0, root)

    import psutil
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable inside the allocated G0 job")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.torch_threads)
    device = torch.device("cuda:0")
    process = psutil.Process()
    inherited_children = _children(process)
    tasks = [
        _run_mamujoco(harl_root, device, args.seed),
        _run_smacv2(harl_root, device, args.seed + 1),
    ]
    leaked_children = _wait_for_children_to_exit(process, inherited_children)
    summary: dict[str, Any] = {
        "scope": EXPECTED_SCOPE,
        "scientific_outcome_generated": False,
        "seed": args.seed,
        "sources": sources,
        "runtime": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(device),
            "gpu_capability": list(torch.cuda.get_device_capability(device)),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", "missing"),
        },
        "tasks": tasks,
        "invariants": {
            "cuda_available": True,
            "exact_task_set": len(tasks) == 2,
            "distinct_actor_blocks": all(
                task["policy"]["distinct_actor_objects"] for task in tasks
            ),
            "one_environment_transition_per_task": all(
                task["environment_transitions"] == 1 for task in tasks
            ),
            "fully_charged_actor_transitions": all(
                task["actor_transitions"] == task["agents"] for task in tasks
            ),
            "no_descendant_process_leak": not leaked_children,
            "pinned_sources_clean": True,
            "finite_timing": all(
                np.isfinite(task["policy"]["cuda_forward_seconds"]) for task in tasks
            ),
        },
        "remaining_descendant_pids": sorted(leaked_children),
    }
    validate_summary(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--harl-root", required=True)
    parser.add_argument("--smacv2-root", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=29021)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument(
        "--harl-commit", default="b1af98b0dbab72a2eee9d160751cd09aedbb8ce2"
    )
    parser.add_argument(
        "--smacv2-commit", default="577ab5a2cff2391f8df582da5731ea9cd6adf3c6"
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.torch_threads <= 0:
        raise ValueError("torch_threads must be positive")
    summary = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"output_sha256={_sha256(args.output)}")


if __name__ == "__main__":
    main()

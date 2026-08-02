"""EXP-018A frozen-parameter nonlinear TD gradient CPU pilot runner."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from exp018a_direct_gradient_config import (
    BLOCK_LENGTH,
    CHECKPOINTS,
    EXPERIMENT,
    HIDDEN_WIDTH,
    MAXIMUM_AGENTS,
    MIXING_PROFILES,
    PILOT_SEEDS,
    PROJECTION_COUNT,
    PROJECTION_SEED,
    Q_LEVELS,
    RHO_LEVELS,
    STATIC_MANIFEST_HASH,
    TASKS,
    build_static_manifest,
    expected_rows,
    expected_source_gradient_evaluations,
    sha256_json,
    variance_factor,
)


PROJECTION_COLUMNS = tuple(f"projection_{index:02d}" for index in range(PROJECTION_COUNT))
OUTPUT_COLUMNS = (
    "experiment",
    "manifest_hash",
    "seed",
    "task",
    "mixing",
    "checkpoint",
    "rho",
    "q",
    "theoretical_variance_factor",
    "shared_agents",
    "pairwise_trials",
    "pairwise_shared_fraction",
    "parameter_hash_before",
    "parameter_hash_after",
    *PROJECTION_COLUMNS,
)


@dataclass(frozen=True)
class TransitionBank:
    states: np.ndarray
    following: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray


class ValueNetwork(nn.Module):
    def __init__(self, input_dimension: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dimension, HIDDEN_WIDTH),
            nn.ReLU(),
            nn.Linear(HIDDEN_WIDTH, HIDDEN_WIDTH),
            nn.ReLU(),
            nn.Linear(HIDDEN_WIDTH, 1),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.layers(states).squeeze(-1)


def normalize_states(task_name: str, states: np.ndarray) -> np.ndarray:
    if task_name == "cartpole":
        scale = np.asarray([2.4, 3.0, 0.2095, 3.5], dtype=np.float32)
    elif task_name == "acrobot":
        scale = np.asarray(
            [1.0, 1.0, 1.0, 1.0, 4.0 * np.pi, 9.0 * np.pi], dtype=np.float32
        )
    else:
        raise ValueError(task_name)
    return np.clip(states.astype(np.float32) / scale, -5.0, 5.0)


def behavior_action(task_name: str, state: np.ndarray, uniform: float) -> int:
    epsilon = 0.20
    if task_name == "cartpole":
        preferred = int(1.8 * float(state[2]) + 0.35 * float(state[3]) >= 0.0)
        return int((uniform / epsilon) * 2) % 2 if uniform < epsilon else preferred
    if task_name == "acrobot":
        signal = (
            float(state[1])
            + 0.5 * (float(state[1]) * float(state[2]) + float(state[0]) * float(state[3]))
            + 0.08 * float(state[4])
            + 0.04 * float(state[5])
        )
        preferred = 2 if signal >= 0.0 else 0
        return int((uniform / epsilon) * 3) % 3 if uniform < epsilon else preferred
    raise ValueError(task_name)


def _offset(*parts: object) -> int:
    return int(sha256_json(list(parts))[:8], 16)


def generate_independent_transition_bank(
    task_name: str,
    mixing_name: str,
    seed: int,
    length: int = BLOCK_LENGTH,
    source_count: int = MAXIMUM_AGENTS + 1,
) -> TransitionBank:
    """Generate iid complete streams with independent regeneration clocks."""

    task = TASKS[task_name]
    regeneration = float(MIXING_PROFILES[mixing_name]["regeneration_probability"])
    dimension = int(task["observation_dimension"])
    states = np.empty((source_count, length, dimension), dtype=np.float32)
    following = np.empty_like(states)
    rewards = np.empty((source_count, length), dtype=np.float32)
    terminated = np.empty((source_count, length), dtype=np.bool_)
    for source in range(source_count):
        source_seed = seed + _offset(task_name, mixing_name, source) % 1_000_000_000
        rng = np.random.RandomState(source_seed)
        environment = gym.make(str(task["gym_id"]))
        reset_count = 0
        observation, _ = environment.reset(seed=source_seed)
        observation = np.asarray(observation, dtype=np.float32)
        for tick in range(length):
            if tick > 0 and rng.random_sample() < regeneration:
                reset_count += 1
                observation, _ = environment.reset(
                    seed=source_seed + 1_000_003 * reset_count
                )
                observation = np.asarray(observation, dtype=np.float32)
            action = behavior_action(task_name, observation, rng.random_sample())
            next_state, reward, ended, truncated, _ = environment.step(action)
            done = bool(ended or truncated)
            states[source, tick] = normalize_states(task_name, observation)
            following[source, tick] = normalize_states(
                task_name, np.asarray(next_state, dtype=np.float32)
            )
            rewards[source, tick] = float(reward)
            terminated[source, tick] = done
            if done:
                reset_count += 1
                next_state, _ = environment.reset(
                    seed=source_seed + 1_000_003 * reset_count
                )
            observation = np.asarray(next_state, dtype=np.float32)
        environment.close()
    return TransitionBank(states, following, rewards, terminated)


def parameter_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for parameter in model.parameters():
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def flattened_gradient(model: nn.Module) -> torch.Tensor:
    return torch.cat(
        [
            parameter.grad.detach().reshape(-1)
            if parameter.grad is not None
            else torch.zeros_like(parameter).reshape(-1)
            for parameter in model.parameters()
        ]
    ).clone()


def projection_matrix(parameter_count: int, checkpoint_seed: int) -> torch.Tensor:
    rng = np.random.RandomState(PROJECTION_SEED + checkpoint_seed + parameter_count)
    signs = rng.randint(0, 2, size=(PROJECTION_COUNT, parameter_count)).astype(np.float32)
    signs = 2.0 * signs - 1.0
    return torch.from_numpy(signs / math.sqrt(float(parameter_count)))


def source_gradient_projections(
    bank: TransitionBank,
    task_name: str,
    checkpoint_seed: int,
) -> tuple[np.ndarray, str, str]:
    torch.manual_seed(checkpoint_seed)
    model = ValueNetwork(int(TASKS[task_name]["observation_dimension"]))
    model.eval()
    before = parameter_hash(model)
    count = sum(parameter.numel() for parameter in model.parameters())
    projections = projection_matrix(count, checkpoint_seed)
    output = np.empty((bank.states.shape[0], PROJECTION_COUNT), dtype=np.float64)
    discount = float(TASKS[task_name]["discount"])
    for source in range(bank.states.shape[0]):
        states = torch.from_numpy(bank.states[source])
        following = torch.from_numpy(bank.following[source])
        rewards = torch.from_numpy(bank.rewards[source])
        continuation = torch.from_numpy((~bank.terminated[source]).astype(np.float32))
        predictions = model(states)
        with torch.no_grad():
            target = rewards + discount * continuation * model(following)
        loss = 0.5 * torch.mean((predictions - target) ** 2)
        model.zero_grad(set_to_none=True)
        loss.backward()
        gradient = flattened_gradient(model)
        output[source] = torch.mv(projections, gradient).detach().numpy()
    after = parameter_hash(model)
    return output, before, after


def source_assignment(seed: int, task_name: str, mixing_name: str, rho: float) -> np.ndarray:
    rng = np.random.RandomState(
        seed + _offset("assignment", task_name, mixing_name, rho) % 1_000_000_000
    )
    shared = rng.random_sample(MAXIMUM_AGENTS) < math.sqrt(rho)
    private = np.arange(1, MAXIMUM_AGENTS + 1, dtype=np.int64)
    return np.where(shared, 0, private)


def pairwise_share(shared_agents: int, q: int) -> tuple[int, float]:
    trials = q * (q - 1) // 2
    if trials == 0:
        return 0, 0.0
    shared_pairs = shared_agents * (shared_agents - 1) // 2
    return trials, shared_pairs / float(trials)


def rows_for_seed(seed: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for task_name in TASKS:
        for mixing_name in MIXING_PROFILES:
            bank = generate_independent_transition_bank(task_name, mixing_name, seed)
            for checkpoint, checkpoint_seed in CHECKPOINTS.items():
                source_projections, before, after = source_gradient_projections(
                    bank, task_name, checkpoint_seed
                )
                for rho in RHO_LEVELS:
                    assignment = source_assignment(seed, task_name, mixing_name, rho)
                    for q in Q_LEVELS:
                        selected = assignment[:q]
                        averaged = np.mean(source_projections[selected], axis=0)
                        shared_agents = int(np.sum(selected == 0))
                        trials, share_fraction = pairwise_share(shared_agents, q)
                        row: dict[str, object] = {
                            "experiment": EXPERIMENT,
                            "manifest_hash": STATIC_MANIFEST_HASH,
                            "seed": seed,
                            "task": task_name,
                            "mixing": mixing_name,
                            "checkpoint": checkpoint,
                            "rho": rho,
                            "q": q,
                            "theoretical_variance_factor": variance_factor(q, rho),
                            "shared_agents": shared_agents,
                            "pairwise_trials": trials,
                            "pairwise_shared_fraction": share_fraction,
                            "parameter_hash_before": before,
                            "parameter_hash_after": after,
                        }
                        row.update(
                            {
                                column: float(value)
                                for column, value in zip(PROJECTION_COLUMNS, averaged)
                            }
                        )
                        rows.append(row)
    return rows


def run_pilot(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = output_dir / "static_manifest.json"
    manifest_path.write_text(
        json.dumps(build_static_manifest(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_path = output_dir / "projections.csv"
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for index, seed in enumerate(PILOT_SEEDS, start=1):
            writer.writerows(rows_for_seed(seed))
            handle.flush()
            print(f"completed seed {index}/{len(PILOT_SEEDS)}: {seed}", flush=True)
    return output_path


def static_validate() -> dict[str, object]:
    return {
        "experiment": EXPERIMENT,
        "manifest_hash": STATIC_MANIFEST_HASH,
        "pilot_seed_count": len(PILOT_SEEDS),
        "pilot_seed_unique": len(set(PILOT_SEEDS)) == len(PILOT_SEEDS),
        "expected_rows": expected_rows(),
        "expected_source_gradient_evaluations": expected_source_gradient_evaluations(),
        "scientific_trajectories_generated": 0,
        "gpu_required": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--validate", action="store_true")
    group.add_argument("--estimate", action="store_true")
    group.add_argument("--pilot", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.validate:
        print(json.dumps(static_validate(), indent=2, sort_keys=True))
        return
    if args.estimate:
        estimate = static_validate()
        estimate.update(
            {
                "base_environment_transitions": len(PILOT_SEEDS)
                * len(TASKS)
                * len(MIXING_PROFILES)
                * (MAXIMUM_AGENTS + 1)
                * BLOCK_LENGTH,
                "projected_csv_megabytes": 8.0,
                "projected_peak_memory_megabytes": 512,
                "execution_recommendation": "local_CPU",
            }
        )
        print(json.dumps(estimate, indent=2, sort_keys=True))
        return
    if args.output_dir is None:
        parser.error("--output-dir is required with --pilot")
    path = run_pilot(args.output_dir.resolve())
    print(json.dumps({"status": "completed", "output": str(path)}, indent=2))


if __name__ == "__main__":
    main()

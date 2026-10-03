"""CPU smoke test with a realizable neural value and stochastic rewards."""

import argparse
import json
from collections import deque
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

from run_nonlinear_td_smoke import (
    AGENT_COUNTS,
    CORRELATIONS,
    DELAYS,
    GAMMA,
    LEARNING_RATE,
    MAXIMUM_AGENTS,
    MESSAGE_BUDGET,
    SERVER_OVERHEAD,
    STATE_DIMENSION,
    VALIDATION_SIZE,
    ValueNetwork,
    apply_gradients,
    flattened_gradients,
    generate_paths,
)


REWARD_NOISE_STANDARD_DEVIATION = 1.0
TEACHER_SEED = 20270522


def build_teacher() -> ValueNetwork:
    """Construct the fixed, exactly realizable teacher network."""

    prior_state = torch.random.get_rng_state()
    torch.manual_seed(TEACHER_SEED)
    teacher = ValueNetwork()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    torch.random.set_rng_state(prior_state)
    return teacher


def run_configuration(
    seed: int, rho: float, delay: int, num_agents: int
) -> Dict[str, object]:
    """Train one delayed configuration with correlated reward noise."""

    torch.manual_seed(seed)
    torch.set_num_threads(1)
    rng = np.random.RandomState(seed + 100000)
    updates = MESSAGE_BUDGET // (SERVER_OVERHEAD + num_agents)
    paths = generate_paths(seed + 200000, updates + 1)
    reward_noise = rng.standard_normal(
        (MAXIMUM_AGENTS + 1, updates)
    ).astype(np.float32)
    masks = rng.random_sample((updates, num_agents))
    teacher = build_teacher()
    model = ValueNetwork()
    queue: deque = deque()
    maximum_loss = 0.0
    finite = True
    share_probability = np.sqrt(rho)
    for update in range(updates):
        sources = np.arange(1, num_agents + 1)
        sources = np.where(
            masks[update] < share_probability, 0, sources
        )
        current = torch.from_numpy(paths[sources, update])
        following = torch.from_numpy(paths[sources, update + 1])
        with torch.no_grad():
            rewards = (
                teacher(current)
                - GAMMA * teacher(following)
                + REWARD_NOISE_STANDARD_DEVIATION
                * torch.from_numpy(reward_noise[sources, update])
            )
        prediction = model(current)
        with torch.no_grad():
            target = rewards + GAMMA * model(following)
        loss = 0.5 * ((prediction - target) ** 2).mean()
        model.zero_grad(set_to_none=True)
        loss.backward()
        queue.append(flattened_gradients(model))
        if len(queue) > delay:
            apply_gradients(
                model, queue.popleft(), LEARNING_RATE
            )
        loss_value = float(loss.detach())
        maximum_loss = max(maximum_loss, loss_value)
        if not np.isfinite(loss_value):
            finite = False
            break
    validation_rng = np.random.RandomState(seed + 300000)
    validation = torch.from_numpy(
        validation_rng.standard_normal(
            (VALIDATION_SIZE, STATE_DIMENSION)
        ).astype(np.float32)
    )
    with torch.no_grad():
        error = torch.mean(
            (model(validation) - teacher(validation)) ** 2
        )
    return {
        "seed": seed,
        "rho": rho,
        "delay": delay,
        "num_agents": num_agents,
        "updates": int(updates),
        "teacher_mse": float(error),
        "maximum_loss": maximum_loss,
        "finite": bool(finite and np.isfinite(float(error))),
    }


def save_figure(metrics: pd.DataFrame, output_dir: Path) -> None:
    """Plot the resource-matched teacher error."""

    summary = (
        metrics.groupby(
            ["rho", "delay", "num_agents"], as_index=False
        )
        .agg(mse=("teacher_mse", "mean"))
    )
    figure, axes = plt.subplots(
        1, 2, figsize=(9.5, 3.9), sharey=True
    )
    for axis, rho in zip(axes, CORRELATIONS):
        for delay, marker in zip(DELAYS, ("o", "s")):
            group = summary[
                (summary["rho"] == rho)
                & (summary["delay"] == delay)
            ]
            axis.plot(
                group["num_agents"],
                group["mse"],
                marker=marker,
                label=f"$D={delay}$",
            )
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_xlabel("participating agents $q$")
        axis.set_title(f"$\\rho={rho:g}$")
        axis.grid(alpha=0.3)
        axis.legend()
    axes[0].set_ylabel("realizable teacher MSE")
    figure.tight_layout()
    figure.savefig(
        output_dir / "realizable_td_smoke.png", dpi=180
    )
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-seeds", type=int, default=2)
    parser.add_argument("--base-seed", type=int, default=20270523)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent
        / "results"
        / "realizable_td_smoke",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: List[Dict[str, object]] = []
    for seed_index in range(args.num_seeds):
        seed = args.base_seed + seed_index
        for rho in CORRELATIONS:
            for delay in DELAYS:
                for num_agents in AGENT_COUNTS:
                    rows.append(
                        run_configuration(
                            seed, rho, delay, num_agents
                        )
                    )
    metrics = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    save_figure(metrics, args.output_dir)
    summary = {
        "experiment": "EXP-013A-realizable-TD-smoke",
        "evidence_status": "implementation_only",
        "num_seeds": args.num_seeds,
        "reward_noise_standard_deviation": (
            REWARD_NOISE_STANDARD_DEVIATION
        ),
        "message_budget": MESSAGE_BUDGET,
        "server_overhead": SERVER_OVERHEAD,
        "all_finite": bool(metrics["finite"].all()),
        "scenario_means": (
            metrics.groupby(
                ["rho", "delay", "num_agents"], as_index=False
            )
            .agg(
                mse=("teacher_mse", "mean"),
                updates=("updates", "first"),
                maximum_loss=("maximum_loss", "max"),
            )
            .to_dict(orient="records")
        ),
    }
    with (args.output_dir / "summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

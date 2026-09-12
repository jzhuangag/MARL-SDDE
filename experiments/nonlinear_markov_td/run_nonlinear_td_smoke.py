"""CPU smoke test for delayed nonlinear multi-agent Markov TD."""

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


STATE_DIMENSION = 4
MAXIMUM_AGENTS = 32
AGENT_COUNTS = (1, 4, 16, 32)
CORRELATIONS = (0.0, 0.9)
DELAYS = (0, 8)
MIXING = 0.8
GAMMA = 0.9
# A substantial fixed server cost prevents the resource-matched comparison
# from degenerating into "q=1 receives many more optimization steps."
SERVER_OVERHEAD = 64
MESSAGE_BUDGET = 64000
LEARNING_RATE = 0.03
VALIDATION_SIZE = 4096


class ValueNetwork(nn.Module):
    """Small nonlinear value approximator."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(STATE_DIMENSION, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.layers(states).squeeze(-1)


def teacher_value(states: torch.Tensor) -> torch.Tensor:
    """Known nonlinear value function used to construct exact rewards."""

    linear = (
        0.8 * states[..., 0]
        - 0.5 * states[..., 1]
        + 0.3 * states[..., 2]
        + 0.2 * states[..., 3]
    )
    quadratic = 0.08 * (states * states).sum(dim=-1)
    interaction = 0.25 * torch.sin(
        states[..., 0] * states[..., 1]
    )
    return torch.sin(linear) + quadratic + interaction


def generate_paths(seed: int, length: int) -> np.ndarray:
    """Generate common plus private stationary Gaussian Markov paths."""

    rng = np.random.RandomState(seed)
    paths = np.empty(
        (MAXIMUM_AGENTS + 1, length + 1, STATE_DIMENSION),
        dtype=np.float32,
    )
    paths[:, 0] = rng.standard_normal(
        (MAXIMUM_AGENTS + 1, STATE_DIMENSION)
    )
    scale = np.sqrt(1.0 - MIXING ** 2)
    for time in range(length):
        paths[:, time + 1] = (
            MIXING * paths[:, time]
            + scale
            * rng.standard_normal(
                (MAXIMUM_AGENTS + 1, STATE_DIMENSION)
            )
        )
    return paths


def flattened_gradients(model: nn.Module) -> List[torch.Tensor]:
    """Copy the current parameter gradients."""

    return [
        parameter.grad.detach().clone()
        for parameter in model.parameters()
    ]


def apply_gradients(
    model: nn.Module, gradients: List[torch.Tensor], step_size: float
) -> None:
    """Apply one queued stochastic-gradient update."""

    with torch.no_grad():
        for parameter, gradient in zip(model.parameters(), gradients):
            parameter.add_(gradient, alpha=-step_size)


def run_configuration(
    seed: int, rho: float, delay: int, num_agents: int
) -> Dict[str, object]:
    """Train one delayed neural TD configuration."""

    torch.manual_seed(seed)
    rng = np.random.RandomState(seed + 100000)
    torch.set_num_threads(1)
    updates = MESSAGE_BUDGET // (SERVER_OVERHEAD + num_agents)
    paths = generate_paths(seed + 200000, updates + 1)
    masks = rng.random_sample((updates, num_agents))
    model = ValueNetwork()
    queue = deque()
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
        rewards = (
            teacher_value(current)
            - GAMMA * teacher_value(following)
        )
        prediction = model(current)
        with torch.no_grad():
            target = rewards + GAMMA * model(following)
        loss = 0.5 * ((prediction - target) ** 2).mean()
        model.zero_grad(set_to_none=True)
        loss.backward()
        queue.append(flattened_gradients(model))
        if len(queue) > delay:
            apply_gradients(model, queue.popleft(), LEARNING_RATE)
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
            (model(validation) - teacher_value(validation)) ** 2
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
        "message_budget": MESSAGE_BUDGET,
        "server_overhead": SERVER_OVERHEAD,
        "learning_rate": LEARNING_RATE,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent
        / "results"
        / "nonlinear_td_overhead_smoke",
    )
    parser.add_argument("--num-seeds", type=int, default=2)
    parser.add_argument("--base-seed", type=int, default=20270501)
    return parser.parse_args()


def save_figure(metrics: pd.DataFrame, output_dir: Path) -> None:
    summary = (
        metrics.groupby(["rho", "delay", "num_agents"], as_index=False)
        .agg(mse=("teacher_mse", "mean"))
    )
    figure, axes = plt.subplots(1, 2, figsize=(9.5, 3.9), sharey=True)
    for axis, rho in zip(axes, CORRELATIONS):
        for delay, marker in ((0, "o"), (8, "s")):
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
    axes[0].set_ylabel("teacher value MSE")
    figure.tight_layout()
    figure.savefig(output_dir / "nonlinear_td_smoke.png", dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
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
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    save_figure(metrics, args.output_dir)
    summary = {
        "experiment": "EXP-013A-smoke",
        "evidence_status": "implementation_only",
        "num_seeds": args.num_seeds,
        "message_budget": MESSAGE_BUDGET,
        "server_overhead": SERVER_OVERHEAD,
        "rows": int(len(metrics)),
        "finite_runs": int(metrics["finite"].sum()),
        "all_finite": bool(metrics["finite"].all()),
        "scenario_means": (
            metrics.groupby(
                ["rho", "delay", "num_agents"], as_index=False
            )
            .agg(
                mse=("teacher_mse", "mean"),
                maximum_loss=("maximum_loss", "max"),
                updates=("updates", "first"),
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

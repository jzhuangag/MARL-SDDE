"""Run the TSP finite-budget convergence-curve experiment on CPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / "experiments" / "dependence_delay_linear"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from affine_markov_certificate import (  # noqa: E402
    affine_candidate_actions,
    select_affine_action,
)
from linear_model import make_agent_delays  # noqa: E402
from multistate_certificate import (  # noqa: E402
    build_transfer_mrp,
    generate_unit_paths,
)

try:
    from numba import njit
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("TSP-CURVE-001 requires numba") from exc


CONFIG_PATH = Path(__file__).with_name("convergence_config.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config() -> Dict[str, object]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def seed_sequence(config: Mapping[str, object], phase: str, limit: int | None) -> Tuple[int, ...]:
    key = "development_seeds" if phase == "development" else "confirmation_seeds"
    registry = config[key]
    start = int(registry["start"])
    count = int(registry["count"])
    if limit is not None:
        count = min(count, int(limit))
    return tuple(range(start, start + count))


@njit(cache=False, nogil=True)
def _trace_kernel(
    paths: np.ndarray,
    masks: np.ndarray,
    features: np.ndarray,
    reward: np.ndarray,
    theta_star: np.ndarray,
    delays: np.ndarray,
    rho: float,
    gap: int,
    eta: float,
    update_targets: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    dimension = features.shape[1]
    num_agents = delays.shape[0]
    maximum_delay = int(np.max(delays))
    maximum_updates = int(update_targets[-1])
    history = np.zeros((maximum_delay + maximum_updates + 1, dimension), dtype=np.float64)
    parameter_errors = np.empty(update_targets.shape[0], dtype=np.float64)
    return_errors = np.empty(update_targets.shape[0], dtype=np.float64)
    initial_parameter = 0.0
    initial_return = 0.0
    for coordinate in range(dimension):
        initial_parameter += theta_star[coordinate] * theta_star[coordinate]
        value = features[0, coordinate] * theta_star[coordinate]
        initial_return += value
    target_index = 0
    while target_index < update_targets.shape[0] and update_targets[target_index] == 0:
        parameter_errors[target_index] = initial_parameter
        return_errors[target_index] = initial_return * initial_return
        target_index += 1
    gradient = np.zeros(dimension, dtype=np.float64)
    share_probability = math.sqrt(rho)
    for update in range(maximum_updates):
        for coordinate in range(dimension):
            gradient[coordinate] = 0.0
        physical_time = update * gap
        for agent in range(num_agents):
            source = agent + 1
            if masks[agent, physical_time] < share_probability:
                source = 0
            state = int(paths[source, physical_time])
            following = int(paths[source, physical_time + 1])
            stale_index = maximum_delay + update - int(delays[agent])
            current_value = 0.0
            next_value = 0.0
            for coordinate in range(dimension):
                current_value += features[state, coordinate] * history[stale_index, coordinate]
                next_value += features[following, coordinate] * history[stale_index, coordinate]
            td_error = reward[state] + 0.9 * next_value - current_value
            for coordinate in range(dimension):
                gradient[coordinate] += features[state, coordinate] * td_error
        new_index = maximum_delay + update + 1
        old_index = maximum_delay + update
        for coordinate in range(dimension):
            history[new_index, coordinate] = (
                history[old_index, coordinate]
                + eta * gradient[coordinate] / num_agents
            )
        completed = update + 1
        while target_index < update_targets.shape[0] and update_targets[target_index] == completed:
            parameter_error = 0.0
            return_difference = 0.0
            for coordinate in range(dimension):
                difference = history[new_index, coordinate] - theta_star[coordinate]
                parameter_error += difference * difference
                return_difference += features[0, coordinate] * difference
            parameter_errors[target_index] = parameter_error
            return_errors[target_index] = return_difference * return_difference
            target_index += 1
    return parameter_errors, return_errors


def action_table(config: Mapping[str, object]) -> Tuple[pd.DataFrame, Dict[Tuple[float, float, int, int], Dict[str, float]]]:
    rows: List[Dict[str, float]] = []
    lookup: Dict[Tuple[float, float, int, int], Dict[str, float]] = {}
    budget = int(config["resource_budget"])
    counts = tuple(int(value) for value in config["candidate_agent_counts"])
    for persistence in config["persistences"]:
        model = build_transfer_mrp(float(persistence))
        for rho in config["correlations"]:
            for maximum_delay in config["maximum_delays"]:
                candidates = affine_candidate_actions(
                    model,
                    float(rho),
                    int(maximum_delay),
                    resource_budget=budget,
                    agent_counts=counts,
                )
                joint = select_affine_action(candidates)
                for q in counts:
                    selected = select_affine_action(candidates, restricted_q=q)
                    key = (float(persistence), float(rho), int(maximum_delay), int(q))
                    lookup[key] = selected
                    rows.append({"policy": f"q{q}", "persistence": persistence, **selected})
                rows.append({"policy": "joint", "persistence": persistence, **joint})
    return pd.DataFrame(rows), lookup


def normalized_auc(frame: pd.DataFrame, metric: str) -> float:
    ordered = frame.sort_values("resource_fraction")
    x = ordered["resource_fraction"].to_numpy(dtype=float)
    y = ordered[metric].to_numpy(dtype=float)
    return float(np.trapezoid(y, x) / max(x[-1] - x[0], np.finfo(float).eps))


def geometric_mean(values: Iterable[float]) -> float:
    array = np.asarray(tuple(values), dtype=float)
    return float(np.exp(np.mean(np.log(np.maximum(array, np.finfo(float).tiny)))))


def summarize_development(metrics: pd.DataFrame, counts: Sequence[int]) -> Dict[str, object]:
    auc_rows = []
    group_columns = ["seed", "persistence", "rho", "maximum_delay", "policy"]
    for keys, frame in metrics.groupby(group_columns, sort=True):
        auc_rows.append(
            {
                **dict(zip(group_columns, keys)),
                "parameter_auc": normalized_auc(frame, "normalized_parameter_error"),
                "return_auc": normalized_auc(frame, "normalized_return_error"),
            }
        )
    auc = pd.DataFrame(auc_rows)
    scores = {}
    for q in counts:
        subset = auc[auc["policy"] == f"q{q}"]
        scores[str(q)] = {
            "parameter_auc_geometric_mean": geometric_mean(subset["parameter_auc"]),
            "return_auc_geometric_mean": geometric_mean(subset["return_auc"]),
        }
    selected_q = min(
        counts,
        key=lambda q: (scores[str(q)]["parameter_auc_geometric_mean"], q),
    )
    return {
        "phase": "development",
        "selected_strong_fixed_q": int(selected_q),
        "fixed_q_scores": scores,
        "selection_metric": "geometric mean normalized parameter-error AUC",
    }


def paired_log_ratio_interval(
    numerator: np.ndarray,
    denominator: np.ndarray,
    seed: int,
    replicates: int,
) -> Dict[str, float]:
    log_ratio = np.log(np.maximum(numerator, np.finfo(float).tiny)) - np.log(
        np.maximum(denominator, np.finfo(float).tiny)
    )
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=float)
    for index in range(replicates):
        sample = rng.integers(0, len(log_ratio), size=len(log_ratio))
        draws[index] = math.exp(float(np.mean(log_ratio[sample])))
    return {
        "ratio": math.exp(float(np.mean(log_ratio))),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
    }


def summarize_confirmation(
    metrics: pd.DataFrame,
    selected_q: int,
    config: Mapping[str, object],
) -> Dict[str, object]:
    auc_rows = []
    group_columns = ["seed", "persistence", "rho", "maximum_delay", "policy"]
    for keys, frame in metrics.groupby(group_columns, sort=True):
        auc_rows.append(
            {
                **dict(zip(group_columns, keys)),
                "parameter_auc": normalized_auc(frame, "normalized_parameter_error"),
                "return_auc": normalized_auc(frame, "normalized_return_error"),
                "terminal_parameter": float(frame.sort_values("resource_fraction").iloc[-1]["normalized_parameter_error"]),
                "terminal_return": float(frame.sort_values("resource_fraction").iloc[-1]["normalized_return_error"]),
            }
        )
    auc = pd.DataFrame(auc_rows)
    joint = auc[auc["policy"] == "joint"].sort_values(group_columns[:-1]).reset_index(drop=True)
    fixed = auc[auc["policy"] == f"q{selected_q}"].sort_values(group_columns[:-1]).reset_index(drop=True)
    if not joint[group_columns[:-1]].equals(fixed[group_columns[:-1]]):
        raise RuntimeError("paired confirmation rows are misaligned")
    seed = int(config["bootstrap_seed"])
    replicates = int(config["bootstrap_replicates"])
    comparisons = {
        metric: paired_log_ratio_interval(
            joint[metric].to_numpy(dtype=float),
            fixed[metric].to_numpy(dtype=float),
            seed + index,
            replicates,
        )
        for index, metric in enumerate(
            ("parameter_auc", "return_auc", "terminal_parameter", "terminal_return")
        )
    }
    cell_rows = []
    for keys, joint_cell in joint.groupby(["persistence", "rho", "maximum_delay"], sort=True):
        fixed_cell = fixed[
            (fixed["persistence"] == keys[0])
            & (fixed["rho"] == keys[1])
            & (fixed["maximum_delay"] == keys[2])
        ]
        cell_rows.append(
            {
                "persistence": keys[0],
                "rho": keys[1],
                "maximum_delay": keys[2],
                "parameter_auc_ratio": geometric_mean(joint_cell["parameter_auc"])
                / geometric_mean(fixed_cell["parameter_auc"]),
                "return_auc_ratio": geometric_mean(joint_cell["return_auc"])
                / geometric_mean(fixed_cell["return_auc"]),
            }
        )
    return {
        "phase": "confirmation",
        "selected_strong_fixed_q": int(selected_q),
        "paired_comparisons": comparisons,
        "cell_comparisons": cell_rows,
        "finite": bool(np.isfinite(metrics.select_dtypes(include=[np.number]).to_numpy()).all()),
        "within_budget": bool(metrics["within_budget"].all()),
    }


def run(phase: str, output_dir: Path, limit: int | None, selection_path: Path | None) -> None:
    config = load_config()
    seeds = seed_sequence(config, phase, limit)
    table, lookup = action_table(config)
    checkpoints = np.linspace(0.0, 1.0, int(config["checkpoints"]))
    budget = int(config["resource_budget"])
    counts = tuple(int(value) for value in config["candidate_agent_counts"])
    rows: List[Dict[str, object]] = []
    start_time = time.perf_counter()
    for seed in seeds:
        for persistence in config["persistences"]:
            model = build_transfer_mrp(float(persistence))
            streams = generate_unit_paths(
                int(seed), model, length=budget, num_agents=max(counts)
            )
            initial_parameter = float(model["theta_star"].dot(model["theta_star"]))
            initial_return = float(model["features"][0].dot(model["theta_star"])) ** 2
            for rho in config["correlations"]:
                for maximum_delay in config["maximum_delays"]:
                    scenario_actions = {
                        f"q{q}": lookup[(float(persistence), float(rho), int(maximum_delay), q)]
                        for q in counts
                    }
                    joint_action = min(
                        scenario_actions.values(),
                        key=lambda action: (
                            action["finite_time_bound"],
                            action["num_agents"],
                            action["gap"],
                            action["eta"],
                        ),
                    )
                    scenario_actions["joint"] = joint_action
                    for policy, action in scenario_actions.items():
                        q = int(action["num_agents"])
                        delays = np.ascontiguousarray(
                            make_agent_delays(32, int(maximum_delay))[:q], dtype=np.int64
                        )
                        update_targets = np.asarray(
                            np.floor(checkpoints * int(action["updates"])), dtype=np.int64
                        )
                        parameter_errors, return_errors = _trace_kernel(
                            streams["paths"],
                            streams["masks"],
                            model["features"],
                            model["reward"],
                            model["theta_star"],
                            delays,
                            float(rho),
                            int(action["gap"]),
                            float(action["eta"]),
                            update_targets,
                        )
                        for index, target in enumerate(update_targets):
                            charged = int(target) * int(action["update_cost"])
                            rows.append(
                                {
                                    "phase": phase,
                                    "seed": int(seed),
                                    "persistence": float(persistence),
                                    "rho": float(rho),
                                    "maximum_delay": int(maximum_delay),
                                    "policy": policy,
                                    "q": q,
                                    "gap": int(action["gap"]),
                                    "eta": float(action["eta"]),
                                    "checkpoint": int(index),
                                    "updates": int(target),
                                    "charged_resource": charged,
                                    "resource_fraction": charged / float(budget),
                                    "normalized_parameter_error": float(parameter_errors[index] / initial_parameter),
                                    "normalized_return_error": float(return_errors[index] / initial_return),
                                    "within_budget": bool(charged <= budget),
                                }
                            )
    elapsed = time.perf_counter() - start_time
    metrics = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=False)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    table.to_csv(output_dir / "action_table.csv", index=False)
    if phase == "development":
        summary = summarize_development(metrics, counts)
    else:
        if selection_path is None:
            raise ValueError("confirmation requires --selection")
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        summary = summarize_confirmation(
            metrics, int(selection["selected_strong_fixed_q"]), config
        )
    summary.update(
        {
            "experiment_id": config["experiment_id"],
            "num_seeds": len(seeds),
            "seeds": list(seeds),
            "rows": len(metrics),
            "elapsed_seconds": elapsed,
            "config_sha256": sha256(CONFIG_PATH),
            "source_sha256": sha256(Path(__file__)),
        }
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    metadata = {
        "command_phase": phase,
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "python": sys.version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("development", "confirmation"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-seeds", type=int)
    parser.add_argument("--selection", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments.phase, arguments.output_dir, arguments.num_seeds, arguments.selection)

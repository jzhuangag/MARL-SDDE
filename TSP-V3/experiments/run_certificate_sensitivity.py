"""Run the preregistered certificate-sensitivity and controller-ablation study."""

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
TSP_EXPERIMENTS = ROOT / "TSP" / "experiments"
for module_path in (CORE, TSP_EXPERIMENTS):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

from linear_model import make_agent_delays  # noqa: E402
from multistate_certificate import build_transfer_mrp, generate_unit_paths  # noqa: E402
from run_convergence_curves import _trace_kernel  # noqa: E402

from certificate_sensitivity import build_policy_actions  # noqa: E402


CONFIG_PATH = Path(__file__).with_name("certificate_sensitivity_preregistration.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config() -> Dict[str, object]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def seed_sequence(config: Mapping[str, object], phase: str) -> Tuple[int, ...]:
    registry = config[f"{phase}_seeds"]
    return tuple(
        range(int(registry["start"]), int(registry["start"]) + int(registry["count"]))
    )


def geometric_mean(values: Iterable[float]) -> float:
    array = np.asarray(tuple(values), dtype=float)
    return float(np.exp(np.mean(np.log(np.maximum(array, np.finfo(float).tiny)))))


def normalized_auc(frame: pd.DataFrame, metric: str) -> float:
    ordered = frame.sort_values("resource_fraction")
    x = ordered["resource_fraction"].to_numpy(dtype=float)
    y = ordered[metric].to_numpy(dtype=float)
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is None:
        trapezoid = np.trapz
    return float(trapezoid(y, x) / max(x[-1] - x[0], np.finfo(float).eps))


def build_actions(config: Mapping[str, object]) -> Tuple[pd.DataFrame, Dict[Tuple[float, float, int, str], Dict[str, float]]]:
    rows: List[Dict[str, object]] = []
    lookup: Dict[Tuple[float, float, int, str], Dict[str, float]] = {}
    for persistence in config["persistences"]:
        model = build_transfer_mrp(float(persistence))
        for rho in config["correlations"]:
            for maximum_delay in config["maximum_delays"]:
                policies, _ = build_policy_actions(
                    model,
                    float(rho),
                    int(maximum_delay),
                    resource_budget=int(config["resource_budget"]),
                    agent_counts=tuple(int(q) for q in config["candidate_agent_counts"]),
                    sensitivity_levels=config["sensitivity_levels"],
                )
                for policy, action in policies.items():
                    key = (float(persistence), float(rho), int(maximum_delay), policy)
                    lookup[key] = action
                    rows.append(
                        {
                            "persistence": float(persistence),
                            "rho": float(rho),
                            "maximum_delay": int(maximum_delay),
                            "policy": policy,
                            **action,
                        }
                    )
    return pd.DataFrame(rows), lookup


def simulate(
    config: Mapping[str, object],
    seeds: Sequence[int],
    actions: Mapping[Tuple[float, float, int, str], Mapping[str, float]],
) -> pd.DataFrame:
    checkpoints = np.linspace(0.0, 1.0, int(config["checkpoints"]))
    counts = tuple(int(q) for q in config["candidate_agent_counts"])
    budget = int(config["resource_budget"])
    rows: List[Dict[str, object]] = []
    for seed in seeds:
        for persistence in config["persistences"]:
            model = build_transfer_mrp(float(persistence))
            streams = generate_unit_paths(int(seed), model, length=budget, num_agents=max(counts))
            initial_parameter = float(model["theta_star"].dot(model["theta_star"]))
            initial_return = float(model["features"][0].dot(model["theta_star"])) ** 2
            for rho in config["correlations"]:
                for maximum_delay in config["maximum_delays"]:
                    for policy in config["policies"]:
                        action = actions[(float(persistence), float(rho), int(maximum_delay), str(policy))]
                        q = int(action["num_agents"])
                        delays = np.ascontiguousarray(
                            make_agent_delays(32, int(maximum_delay))[:q], dtype=np.int64
                        )
                        targets = np.asarray(
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
                            targets,
                        )
                        for checkpoint, target in enumerate(targets):
                            charged = int(target) * int(action["update_cost"])
                            rows.append(
                                {
                                    "seed": int(seed),
                                    "persistence": float(persistence),
                                    "rho": float(rho),
                                    "maximum_delay": int(maximum_delay),
                                    "policy": str(policy),
                                    "q": q,
                                    "gap": int(action["gap"]),
                                    "eta": float(action["eta"]),
                                    "checkpoint": int(checkpoint),
                                    "updates": int(target),
                                    "charged_resource": charged,
                                    "resource_fraction": charged / float(budget),
                                    "normalized_parameter_error": float(parameter_errors[checkpoint] / initial_parameter),
                                    "normalized_return_error": float(return_errors[checkpoint] / initial_return),
                                    "certificate_bound": float(action["finite_time_bound"]),
                                    "within_budget": bool(charged <= budget),
                                }
                            )
    return pd.DataFrame(rows)


def seed_cell_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    group = ["seed", "persistence", "rho", "maximum_delay", "policy"]
    rows = []
    for keys, frame in metrics.groupby(group, sort=True):
        ordered = frame.sort_values("resource_fraction")
        rows.append(
            {
                **dict(zip(group, keys)),
                "parameter_auc": normalized_auc(ordered, "normalized_parameter_error"),
                "return_auc": normalized_auc(ordered, "normalized_return_error"),
                "terminal_parameter": float(ordered.iloc[-1]["normalized_parameter_error"]),
                "terminal_return": float(ordered.iloc[-1]["normalized_return_error"]),
                "certificate_bound": float(ordered.iloc[-1]["certificate_bound"]),
            }
        )
    return pd.DataFrame(rows)


def paired_ratio_interval(
    numerator: np.ndarray,
    denominator: np.ndarray,
    *,
    seed: int,
    replications: int,
) -> Dict[str, float]:
    log_ratio = np.log(np.maximum(numerator, np.finfo(float).tiny)) - np.log(
        np.maximum(denominator, np.finfo(float).tiny)
    )
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(replications), dtype=float)
    for index in range(int(replications)):
        sample = rng.integers(0, len(log_ratio), size=len(log_ratio))
        draws[index] = math.exp(float(np.mean(log_ratio[sample])))
    return {
        "ratio": math.exp(float(np.mean(log_ratio))),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
    }


def one_sided_upper(values: np.ndarray, *, seed: int, replications: int) -> float:
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(replications), dtype=float)
    for index in range(int(replications)):
        sample = rng.integers(0, len(values), size=len(values))
        draws[index] = float(np.mean(values[sample]))
    return float(np.quantile(draws, 0.99))


def summarize(
    metrics: pd.DataFrame,
    action_table: pd.DataFrame,
    config: Mapping[str, object],
    phase: str,
) -> Tuple[Dict[str, object], pd.DataFrame, pd.DataFrame]:
    cell = seed_cell_metrics(metrics)
    policies = tuple(str(policy) for policy in config["policies"])
    aggregate: Dict[str, Dict[str, float]] = {}
    for policy in policies:
        part = cell[cell["policy"] == policy]
        aggregate[policy] = {
            metric: geometric_mean(part[metric])
            for metric in ("parameter_auc", "return_auc", "terminal_parameter", "terminal_return")
        }
    reference = aggregate["finite_horizon"]
    ratios = {
        policy: {
            metric: aggregate[policy][metric] / reference[metric]
            for metric in reference
        }
        for policy in policies
        if policy != "finite_horizon"
    }

    paired_rows = []
    base_seed = int(config["bootstrap_seed"])
    replications = int(config["bootstrap_replicates"])
    sort_keys = ["seed", "persistence", "rho", "maximum_delay"]
    reference_rows = cell[cell["policy"] == "finite_horizon"].sort_values(sort_keys)
    for policy_index, policy in enumerate(policies):
        if policy == "finite_horizon":
            continue
        comparator = cell[cell["policy"] == policy].sort_values(sort_keys)
        if not reference_rows[sort_keys].reset_index(drop=True).equals(
            comparator[sort_keys].reset_index(drop=True)
        ):
            raise RuntimeError("paired rows are misaligned")
        for metric_index, metric in enumerate(("parameter_auc", "return_auc", "terminal_parameter", "terminal_return")):
            interval = paired_ratio_interval(
                reference_rows[metric].to_numpy(float),
                comparator[metric].to_numpy(float),
                seed=base_seed + 100 * policy_index + metric_index,
                replications=replications,
            )
            paired_rows.append(
                {
                    "numerator": "finite_horizon",
                    "denominator": policy,
                    "metric": metric,
                    **interval,
                }
            )
    paired = pd.DataFrame(paired_rows)

    coverage_rows = []
    for policy_index, policy in enumerate(("conservative_moderate", "conservative_strong")):
        part = cell[cell["policy"] == policy]
        for cell_index, (keys, group) in enumerate(
            part.groupby(["persistence", "rho", "maximum_delay"], sort=True)
        ):
            upper = one_sided_upper(
                group["terminal_parameter"].to_numpy(float),
                seed=base_seed + 1000 + 100 * policy_index + cell_index,
                replications=replications,
            )
            bound = float(group["certificate_bound"].iloc[0])
            coverage_rows.append(
                {
                    "policy": policy,
                    "persistence": float(keys[0]),
                    "rho": float(keys[1]),
                    "maximum_delay": int(keys[2]),
                    "bootstrap_99_upper_mean": upper,
                    "certificate_bound": bound,
                    "covered": bool(upper <= bound),
                }
            )
    coverage = pd.DataFrame(coverage_rows)

    finite_stable = bool(
        np.isfinite(action_table[["contraction", "forcing", "finite_time_bound"]].to_numpy()).all()
        and (action_table["contraction"] > 0.0).all()
        and (action_table["contraction"] < 1.0).all()
    )
    payload = bool(
        np.isfinite(metrics.select_dtypes(include=[np.number]).to_numpy()).all()
        and metrics["within_budget"].all()
        and len(metrics)
        == len(seed_sequence(config, phase))
        * 12
        * len(policies)
        * int(config["checkpoints"])
    )
    gate_spec = config[f"mandatory_{phase}_gates"]
    gates: Dict[str, object] = {
        "payload_validity": payload,
        "all_conservative_bounds_finite_and_stable": finite_stable,
        "moderate_terminal_parameter_ratio": ratios["conservative_moderate"]["terminal_parameter"],
        "strong_terminal_parameter_ratio": ratios["conservative_strong"]["terminal_parameter"],
        "finite_horizon_vs_myopic_terminal_parameter_ratio": 1.0 / ratios["one_step_myopic"]["terminal_parameter"],
        "finite_horizon_vs_correlation_only_terminal_parameter_ratio": 1.0 / ratios["correlation_only"]["terminal_parameter"],
    }
    passes = [
        payload,
        finite_stable,
        gates["moderate_terminal_parameter_ratio"] <= float(gate_spec["moderate_terminal_parameter_ratio_at_most"]),
        gates["strong_terminal_parameter_ratio"] <= float(gate_spec["strong_terminal_parameter_ratio_at_most"]),
        gates["finite_horizon_vs_myopic_terminal_parameter_ratio"]
        <= float(gate_spec["finite_horizon_vs_myopic_terminal_parameter_ratio_at_most"]),
        gates["finite_horizon_vs_correlation_only_terminal_parameter_ratio"]
        <= float(gate_spec["finite_horizon_vs_correlation_only_terminal_parameter_ratio_at_most"]),
    ]
    if phase == "confirmation":
        coverage_pass = bool(len(coverage) == 24 and coverage["covered"].all())
        paired_ablation = paired[
            paired["denominator"].isin(["one_step_myopic", "correlation_only"])
            & paired["metric"].eq("terminal_parameter")
        ]
        paired_pass = bool(len(paired_ablation) == 2 and (paired_ablation["ci95_high"] < 1.0).all())
        gates["conservative_99pct_upper_mean_coverage"] = {
            "covered": int(coverage["covered"].sum()),
            "total": int(len(coverage)),
            "pass": coverage_pass,
        }
        gates["paired_bootstrap_upper_ratio_below_one_for_both_ablations"] = paired_pass
        passes.extend([coverage_pass, paired_pass])
    gates["overall_pass"] = bool(all(passes))
    return (
        {
            "experiment_id": config["experiment_id"],
            "phase": phase,
            "aggregate": aggregate,
            "ratios_to_finite_horizon": ratios,
            "gates": gates,
        },
        paired,
        coverage,
    )


def run(phase: str, output_dir: Path) -> None:
    config = load_config()
    seeds = seed_sequence(config, phase)
    start = time.perf_counter()
    action_table, actions = build_actions(config)
    metrics = simulate(config, seeds, actions)
    summary, paired, coverage = summarize(metrics, action_table, config, phase)
    summary.update(
        {
            "num_seeds": len(seeds),
            "seeds": list(seeds),
            "rows": len(metrics),
            "elapsed_seconds": time.perf_counter() - start,
            "config_sha256": sha256(CONFIG_PATH),
            "source_sha256": sha256(Path(__file__)),
            "implementation_sha256": sha256(Path(__file__).with_name("certificate_sensitivity.py")),
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        }
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    action_table.to_csv(output_dir / "action_table.csv", index=False)
    seed_cell_metrics(metrics).to_csv(output_dir / "seed_cell_metrics.csv", index=False)
    paired.to_csv(output_dir / "paired_comparisons.csv", index=False)
    coverage.to_csv(output_dir / "coverage.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("development", "confirmation"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.phase, args.output_dir)

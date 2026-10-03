"""Run the preregistered robust-feasibility certificate study."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(__file__).with_name(
    "certificate_robust_shield_preregistration.json"
)

import run_certificate_sensitivity as base
from certificate_robust_shield import build_shield_policy_actions
from multistate_certificate import build_transfer_mrp


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config() -> Dict[str, object]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def build_actions(
    config: Mapping[str, object],
) -> Tuple[pd.DataFrame, Dict[Tuple[float, float, int, str], Dict[str, float]]]:
    rows: List[Dict[str, object]] = []
    lookup: Dict[Tuple[float, float, int, str], Dict[str, float]] = {}
    for persistence in config["persistences"]:
        model = build_transfer_mrp(float(persistence))
        for rho in config["correlations"]:
            for maximum_delay in config["maximum_delays"]:
                policies, _ = build_shield_policy_actions(
                    model,
                    float(rho),
                    int(maximum_delay),
                    resource_budget=int(config["resource_budget"]),
                    agent_counts=tuple(
                        int(q) for q in config["candidate_agent_counts"]
                    ),
                    sensitivity_levels=config["sensitivity_levels"],
                )
                for policy, raw_action in policies.items():
                    action = dict(raw_action)
                    if policy.startswith("shield_"):
                        action["finite_time_bound"] = float(
                            action["certificate_bound"]
                        )
                    key = (
                        float(persistence),
                        float(rho),
                        int(maximum_delay),
                        str(policy),
                    )
                    lookup[key] = action
                    rows.append(
                        {
                            "persistence": float(persistence),
                            "rho": float(rho),
                            "maximum_delay": int(maximum_delay),
                            "policy": str(policy),
                            **action,
                        }
                    )
    return pd.DataFrame(rows), lookup


def summarize(
    metrics: pd.DataFrame,
    action_table: pd.DataFrame,
    config: Mapping[str, object],
    phase: str,
) -> Tuple[Dict[str, object], pd.DataFrame, pd.DataFrame]:
    cell = base.seed_cell_metrics(metrics)
    policies = tuple(str(policy) for policy in config["policies"])
    aggregate = {}
    for policy in policies:
        part = cell[cell["policy"] == policy]
        aggregate[policy] = {
            metric: base.geometric_mean(part[metric])
            for metric in (
                "parameter_auc",
                "return_auc",
                "terminal_parameter",
                "terminal_return",
            )
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

    sort_keys = ["seed", "persistence", "rho", "maximum_delay"]
    reference_rows = cell[cell["policy"] == "finite_horizon"].sort_values(
        sort_keys
    )
    paired_rows = []
    for policy_index, policy in enumerate(policies):
        if policy == "finite_horizon":
            continue
        comparator = cell[cell["policy"] == policy].sort_values(sort_keys)
        if not reference_rows[sort_keys].reset_index(drop=True).equals(
            comparator[sort_keys].reset_index(drop=True)
        ):
            raise RuntimeError("paired rows are misaligned")
        for metric_index, metric in enumerate(
            ("parameter_auc", "return_auc", "terminal_parameter", "terminal_return")
        ):
            interval = base.paired_ratio_interval(
                reference_rows[metric].to_numpy(float),
                comparator[metric].to_numpy(float),
                seed=int(config["bootstrap_seed"])
                + 100 * policy_index
                + metric_index,
                replications=int(config["bootstrap_replicates"]),
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
    for policy_index, policy in enumerate(("shield_moderate", "shield_strong")):
        part = cell[cell["policy"] == policy]
        for cell_index, (keys, group) in enumerate(
            part.groupby(["persistence", "rho", "maximum_delay"], sort=True)
        ):
            upper = base.one_sided_upper(
                group["terminal_parameter"].to_numpy(float),
                seed=int(config["bootstrap_seed"])
                + 1000
                + 100 * policy_index
                + cell_index,
                replications=int(config["bootstrap_replicates"]),
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

    shield_rows = action_table[action_table["policy"].str.startswith("shield_")]
    upper_stable = bool(
        np.isfinite(
            shield_rows[
                [
                    "certificate_contraction",
                    "certificate_forcing",
                    "certificate_bound",
                ]
            ].to_numpy()
        ).all()
        and (shield_rows["certificate_contraction"] > 0.0).all()
        and (shield_rows["certificate_contraction"] < 1.0).all()
    )
    expected_rows = (
        len(base.seed_sequence(config, phase))
        * 12
        * len(policies)
        * int(config["checkpoints"])
    )
    payload = bool(
        np.isfinite(metrics.select_dtypes(include=[np.number]).to_numpy()).all()
        and metrics["within_budget"].all()
        and len(metrics) == expected_rows
    )
    gate_spec = config[f"mandatory_{phase}_gates"]
    gates: Dict[str, object] = {
        "payload_validity": payload,
        "upper_certificate_stability": upper_stable,
        "moderate_terminal_parameter_ratio": ratios["shield_moderate"][
            "terminal_parameter"
        ],
        "strong_terminal_parameter_ratio": ratios["shield_strong"][
            "terminal_parameter"
        ],
        "finite_horizon_vs_myopic_terminal_parameter_ratio": 1.0
        / ratios["one_step_myopic"]["terminal_parameter"],
        "finite_horizon_vs_correlation_only_terminal_parameter_ratio": 1.0
        / ratios["correlation_only"]["terminal_parameter"],
    }
    passes = [
        payload,
        upper_stable,
        gates["moderate_terminal_parameter_ratio"]
        <= float(gate_spec["moderate_terminal_parameter_ratio_at_most"]),
        gates["strong_terminal_parameter_ratio"]
        <= float(gate_spec["strong_terminal_parameter_ratio_at_most"]),
        gates["finite_horizon_vs_myopic_terminal_parameter_ratio"]
        <= float(
            gate_spec[
                "finite_horizon_vs_myopic_terminal_parameter_ratio_at_most"
            ]
        ),
        gates["finite_horizon_vs_correlation_only_terminal_parameter_ratio"]
        <= float(
            gate_spec[
                "finite_horizon_vs_correlation_only_terminal_parameter_ratio_at_most"
            ]
        ),
    ]
    if phase == "confirmation":
        coverage_pass = bool(len(coverage) == 24 and coverage["covered"].all())
        ablation = paired[
            paired["denominator"].isin(["one_step_myopic", "correlation_only"])
            & paired["metric"].eq("terminal_parameter")
        ]
        paired_pass = bool(
            len(ablation) == 2 and (ablation["ci95_high"] < 1.0).all()
        )
        gates["conservative_99pct_upper_mean_coverage"] = {
            "covered": int(coverage["covered"].sum()),
            "total": int(len(coverage)),
            "pass": coverage_pass,
        }
        gates[
            "paired_bootstrap_upper_ratio_below_one_for_both_ablations"
        ] = paired_pass
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
    seeds = base.seed_sequence(config, phase)
    start = time.perf_counter()
    action_table, actions = build_actions(config)
    metrics = base.simulate(config, seeds, actions)
    summary, paired, coverage = summarize(
        metrics, action_table, config, phase
    )
    summary.update(
        {
            "num_seeds": len(seeds),
            "seeds": list(seeds),
            "rows": len(metrics),
            "elapsed_seconds": time.perf_counter() - start,
            "config_sha256": sha256(CONFIG_PATH),
            "source_sha256": sha256(Path(__file__)),
            "implementation_sha256": sha256(
                Path(__file__).with_name("certificate_robust_shield.py")
            ),
            "git_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
        }
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    action_table.to_csv(output_dir / "action_table.csv", index=False)
    base.seed_cell_metrics(metrics).to_csv(
        output_dir / "seed_cell_metrics.csv", index=False
    )
    paired.to_csv(output_dir / "paired_comparisons.csv", index=False)
    coverage.to_csv(output_dir / "coverage.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase", choices=("development", "confirmation"), required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.phase, args.output_dir)


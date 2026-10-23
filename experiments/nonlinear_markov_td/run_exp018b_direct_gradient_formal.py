"""Frozen runner for the EXP-018B local-CPU formal study."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from exp018a_direct_gradient_config import (
    CHECKPOINTS,
    MIXING_PROFILES,
    Q_LEVELS,
    RHO_LEVELS,
    TASKS,
    variance_factor,
)
from exp018b_direct_gradient_config import (
    EXPERIMENT,
    FORMAL_SEEDS,
    STATIC_MANIFEST_HASH,
    build_static_manifest,
    expected_rows,
    expected_source_gradient_evaluations,
)
from run_exp018a_direct_gradient import (
    PROJECTION_COLUMNS,
    generate_independent_transition_bank,
    pairwise_share,
    source_assignment,
    source_gradient_projections,
)


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
                        if q == 1:
                            selected = np.asarray([1], dtype=np.int64)
                        else:
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


def run_formal(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "static_manifest.json").write_text(
        json.dumps(build_static_manifest(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_path = output_dir / "projections.csv"
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for index, seed in enumerate(FORMAL_SEEDS, start=1):
            writer.writerows(rows_for_seed(seed))
            handle.flush()
            print(f"completed seed {index}/{len(FORMAL_SEEDS)}: {seed}", flush=True)
    return output_path


def static_validate() -> dict[str, object]:
    return {
        "experiment": EXPERIMENT,
        "manifest_hash": STATIC_MANIFEST_HASH,
        "formal_seed_count": len(FORMAL_SEEDS),
        "formal_seed_unique": len(set(FORMAL_SEEDS)) == len(FORMAL_SEEDS),
        "expected_rows": expected_rows(),
        "expected_source_gradient_evaluations": expected_source_gradient_evaluations(),
        "q1_crn_exact_by_construction": True,
        "scientific_trajectories_generated": 0,
        "gpu_required": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--validate", action="store_true")
    group.add_argument("--estimate", action="store_true")
    group.add_argument("--formal", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.validate:
        print(json.dumps(static_validate(), indent=2, sort_keys=True))
        return
    if args.estimate:
        result = static_validate()
        result.update(
            {
                "base_environment_transitions": 1_622_016,
                "projected_csv_megabytes": 24.0,
                "projected_peak_memory_megabytes": 768,
                "projected_runtime_minutes_from_exp018a": 8.1,
                "execution_recommendation": "local_CPU",
            }
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.output_dir is None:
        parser.error("--output-dir is required with --formal")
    path = run_formal(args.output_dir.resolve())
    print(json.dumps({"status": "completed", "output": str(path)}, indent=2))


if __name__ == "__main__":
    main()

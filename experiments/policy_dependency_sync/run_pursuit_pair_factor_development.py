"""Run the privileged Pursuit pair-factor representation kill test."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .pursuit_delayed_alignment_interface import collect_alignment_launches
from .pursuit_pair_factor_development import (
    build_counterfactual_pair_design,
    fit_pair_factor_ridge,
    heldout_pair_metrics,
    predict_pair_factor_ridge,
)


DEFAULT_TRAIN_SEEDS = tuple(range(93600, 93604))
DEFAULT_VALIDATION_SEEDS = tuple(range(93604, 93606))
DEFAULT_TEST_SEEDS = tuple(range(93606, 93608))
RIDGE_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)


def _source_hash() -> str:
    digest = hashlib.sha256()
    for filename in (
        "pursuit_delayed_alignment_interface.py",
        "pursuit_pair_factor_development.py",
        "run_pursuit_pair_factor_development.py",
    ):
        digest.update((Path(__file__).parent / filename).read_bytes())
    return digest.hexdigest()


def _collect(seeds: tuple[int, ...]):
    return collect_alignment_launches(
        seeds=seeds,
        cycles=32,
        n_evaders=24,
        maximum_delay=3,
        drift_scale=0.20,
        rollout_horizon=8,
        counterfactual_replicates=4,
        audit_counterfactuals=True,
        actor_mode="heuristic_path",
    )


def _collect_one(seed: int):
    return _collect((int(seed),))


def _collect_parallel(seeds: tuple[int, ...], *, workers: int = 4):
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        pieces = list(executor.map(_collect_one, seeds))
    rows = [row for piece, _ in pieces for row in piece]
    diagnostics_by_seed = [diagnostics for _, diagnostics in pieces]
    additive = (
        "launches",
        "delivered_packets",
        "nonnull_launches",
        "charged_policy_bytes",
        "counterfactual_steps",
        "elapsed_seconds",
    )
    diagnostics: dict[str, float | int | str] = {
        key: sum(float(item[key]) for item in diagnostics_by_seed)
        for key in additive
    }
    for key in ("launches", "delivered_packets", "nonnull_launches", "charged_policy_bytes", "counterfactual_steps"):
        diagnostics[key] = int(diagnostics[key])
    diagnostics["maximum_candidates"] = max(
        int(item["maximum_candidates"]) for item in diagnostics_by_seed
    )
    diagnostics["maximum_selected_reward_error"] = max(
        float(item["maximum_selected_reward_error"])
        for item in diagnostics_by_seed
    )
    for key in (
        "feature_dimension",
        "feedback_scale",
        "drift_scale",
        "rollout_horizon",
        "discount",
        "actor_mode",
        "counterfactual_replicates",
    ):
        values = {item[key] for item in diagnostics_by_seed}
        if len(values) != 1:
            raise AssertionError(f"per-seed diagnostic {key} changed")
        diagnostics[key] = values.pop()
    diagnostics["workers"] = int(workers)
    return rows, diagnostics


def run() -> dict[str, object]:
    started = time.perf_counter()
    train_rows, train_diagnostics = _collect_parallel(DEFAULT_TRAIN_SEEDS)
    print("completed train collection", flush=True)
    validation_rows, validation_diagnostics = _collect_parallel(
        DEFAULT_VALIDATION_SEEDS
    )
    print("completed validation collection", flush=True)
    test_rows, test_diagnostics = _collect_parallel(DEFAULT_TEST_SEEDS)
    print("completed test collection", flush=True)
    train = build_counterfactual_pair_design(train_rows)
    validation = build_counterfactual_pair_design(validation_rows)
    test = build_counterfactual_pair_design(test_rows)

    candidates: list[dict[str, float | int]] = []
    models = []
    for ridge in RIDGE_GRID:
        model = fit_pair_factor_ridge(
            train.matrix,
            train.response,
            ridge=ridge,
        )
        prediction = predict_pair_factor_ridge(model, validation.matrix)
        metrics = heldout_pair_metrics(
            truth=validation.response,
            prediction=prediction,
            packet_id=validation.packet_id,
        )
        candidates.append({"ridge": ridge, **metrics})
        models.append(model)
    best_index = min(
        range(len(candidates)),
        key=lambda index: (
            -float(candidates[index]["r_squared"]),
            float(candidates[index]["ridge"]),
        ),
    )
    chosen = models[best_index]
    prediction = predict_pair_factor_ridge(chosen, test.matrix)
    test_metrics = heldout_pair_metrics(
        truth=test.response,
        prediction=prediction,
        packet_id=test.packet_id,
    )
    null_scale = float(np.mean(np.abs(test.response)))
    return {
        "status": "development_only_privileged_counterfactual_audit",
        "authorization": "none",
        "source_sha256": _source_hash(),
        "train_seeds": DEFAULT_TRAIN_SEEDS,
        "validation_seeds": DEFAULT_VALIDATION_SEEDS,
        "test_seeds": DEFAULT_TEST_SEEDS,
        "configuration": {
            "cycles": 32,
            "n_evaders": 24,
            "maximum_delay": 3,
            "drift_scale": 0.20,
            "rollout_horizon": 8,
            "counterfactual_replicates": 4,
            "ridge_grid": RIDGE_GRID,
            "workers": 4,
        },
        "train_diagnostics": train_diagnostics,
        "validation_diagnostics": validation_diagnostics,
        "test_diagnostics": test_diagnostics,
        "design_rows": {
            "train": int(train.response.size),
            "validation": int(validation.response.size),
            "test": int(test.response.size),
            "dimension": int(train.matrix.shape[1]),
        },
        "ridge_validation": candidates,
        "selected_ridge": float(candidates[best_index]["ridge"]),
        "test_metrics": test_metrics,
        "mean_absolute_edge_effect": null_scale,
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = run()
    payload = json.dumps(result, indent=2, sort_keys=True)
    if arguments.output is None:
        print(payload)
        return
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(payload + "\n", encoding="utf-8")
    print(arguments.output)


if __name__ == "__main__":
    main()

"""Develop the selected-packet Pursuit profile-factor critic on local CPU."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .pursuit_delayed_alignment_interface import collect_alignment_launches
from .pursuit_profile_factor_critic import (
    counterfactual_alignment_metrics,
    fit_factor_normalization,
    normalize_selected_batch,
    predict_selected_cost,
    selected_packet_batch,
    train_profile_factor_critic,
)


TRAIN_SEEDS = tuple(range(94200, 94216))
VALIDATION_SEEDS = tuple(range(94216, 94220))
TEST_SEEDS = tuple(range(94220, 94222))


def _collect_selected_one(seed: int):
    return collect_alignment_launches(
        seeds=(int(seed),),
        cycles=64,
        n_evaders=24,
        maximum_delay=3,
        drift_scale=0.20,
        rollout_horizon=8,
        audit_counterfactuals=False,
        actor_mode="heuristic_path",
    )


def _collect_audit_one(seed: int):
    return collect_alignment_launches(
        seeds=(int(seed),),
        cycles=32,
        n_evaders=24,
        maximum_delay=3,
        drift_scale=0.20,
        rollout_horizon=8,
        counterfactual_replicates=8,
        audit_counterfactuals=True,
        actor_mode="heuristic_path",
    )


def _parallel(function, seeds: tuple[int, ...], *, workers: int = 4):
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        pieces = list(executor.map(function, seeds))
    return [row for rows, _ in pieces for row in rows], [
        diagnostics for _, diagnostics in pieces
    ]


def _prediction_metrics(target: np.ndarray, prediction: np.ndarray):
    centered = target - float(np.mean(target))
    denominator = float(centered @ centered)
    return {
        "mse": float(np.mean((prediction - target) ** 2)),
        "mae": float(np.mean(np.abs(prediction - target))),
        "target_std": float(np.std(target)),
        "r_squared": float(
            "nan"
            if denominator <= 1e-15
            else 1.0 - float(np.sum((prediction - target) ** 2)) / denominator
        ),
    }


def _source_hash() -> str:
    digest = hashlib.sha256()
    for filename in (
        "pursuit_delayed_alignment_interface.py",
        "pursuit_pair_factor_development.py",
        "pursuit_profile_factor_critic.py",
        "run_pursuit_profile_factor_development.py",
    ):
        digest.update((Path(__file__).parent / filename).read_bytes())
    return digest.hexdigest()


def run() -> dict[str, object]:
    started = time.perf_counter()
    train_rows, train_collection = _parallel(_collect_selected_one, TRAIN_SEEDS)
    print("completed selected-packet train collection", flush=True)
    validation_rows, validation_collection = _parallel(
        _collect_selected_one, VALIDATION_SEEDS
    )
    print("completed selected-packet validation collection", flush=True)
    train_raw = selected_packet_batch(train_rows)
    validation_raw = selected_packet_batch(validation_rows)
    normalization = fit_factor_normalization(train_raw)
    train = normalize_selected_batch(train_raw, normalization)
    validation = normalize_selected_batch(validation_raw, normalization)
    model, training = train_profile_factor_critic(
        train,
        validation,
        seed=94290,
        hidden_dimension=48,
        epochs=400,
        learning_rate=0.003,
        weight_decay=1e-4,
        patience=50,
    )
    validation_prediction = predict_selected_cost(model, validation)
    validation_metrics = _prediction_metrics(
        validation.cost_return.astype(float), validation_prediction
    )
    print("completed selected-packet critic fit", flush=True)
    test_rows, test_collection = _parallel(_collect_audit_one, TEST_SEEDS)
    print("completed untouched counterfactual test collection", flush=True)
    alignment = counterfactual_alignment_metrics(
        model=model,
        normalization=normalization,
        rows=test_rows,
    )
    passed = bool(
        float(alignment["r_squared"]) > 0.0
        and float(alignment["sign_accuracy_nonzero"]) >= 0.60
        and float(alignment["best_action_accuracy"]) >= 0.55
        and float(alignment["edge_effect_to_null_scale"]) >= 0.05
    )
    return {
        "status": "development_only_selected_packet_critic",
        "source_sha256": _source_hash(),
        "authorization": (
            "independent_cpu_calibration_design_only" if passed else "none"
        ),
        "passed_development_gate": passed,
        "train_seeds": TRAIN_SEEDS,
        "validation_seeds": VALIDATION_SEEDS,
        "test_seeds": TEST_SEEDS,
        "configuration": {
            "selected_cycles": 64,
            "audit_cycles": 32,
            "rollout_horizon": 8,
            "counterfactual_test_replicates": 8,
            "drift_scale": 0.20,
            "n_evaders": 24,
            "workers": 4,
            "hidden_dimension": 48,
        },
        "sample_counts": {
            "train_selected_packets": len(train_rows),
            "validation_selected_packets": len(validation_rows),
            "test_audit_packets": len(test_rows),
        },
        "training": training,
        "validation_selected_packet_metrics": validation_metrics,
        "test_counterfactual_alignment_metrics": alignment,
        "collection": {
            "train": train_collection,
            "validation": validation_collection,
            "test": test_collection,
        },
        "gate": {
            "test_edge_r_squared_strictly_positive": float(
                alignment["r_squared"]
            ) > 0.0,
            "test_nonzero_sign_accuracy_at_least_0_60": float(
                alignment["sign_accuracy_nonzero"]
            ) >= 0.60,
            "test_best_action_accuracy_at_least_0_55": float(
                alignment["best_action_accuracy"]
            ) >= 0.55,
            "test_edge_to_null_scale_at_least_0_05": float(
                alignment["edge_effect_to_null_scale"]
            ) >= 0.05,
        },
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

"""Final Pursuit estimator-class development gate using raw observations."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .pursuit_delayed_alignment_interface import collect_alignment_launches
from .pursuit_raw_factor_critic import (
    predict_raw_selected_cost,
    raw_counterfactual_alignment_metrics,
    raw_selected_packet_batch,
    train_raw_factor_critic,
)


TRAIN_SEEDS = tuple(range(94500, 94516))
VALIDATION_SEEDS = tuple(range(94516, 94520))
TEST_SEEDS = tuple(range(94520, 94522))


def _collect_selected(seed: int):
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


def _collect_test(seed: int):
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


def _return_metrics(target: np.ndarray, prediction: np.ndarray):
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
        "pursuit_raw_factor_critic.py",
        "run_pursuit_raw_factor_development.py",
    ):
        digest.update((Path(__file__).parent / filename).read_bytes())
    return digest.hexdigest()


def run() -> dict[str, object]:
    started = time.perf_counter()
    train_rows, train_collection = _parallel(_collect_selected, TRAIN_SEEDS)
    print("completed raw-critic selected train collection", flush=True)
    validation_rows, validation_collection = _parallel(
        _collect_selected, VALIDATION_SEEDS
    )
    print("completed raw-critic selected validation collection", flush=True)
    train = raw_selected_packet_batch(train_rows)
    validation = raw_selected_packet_batch(validation_rows)
    model, scale, training = train_raw_factor_critic(
        train,
        validation,
        seed=94590,
        epochs=250,
        batch_size=128,
        learning_rate=0.001,
        weight_decay=1e-4,
        patience=35,
    )
    validation_prediction = predict_raw_selected_cost(model, scale, validation)
    validation_metrics = _return_metrics(
        validation.cost_return.astype(float), validation_prediction
    )
    print("completed frozen raw-observation critic fit", flush=True)
    test_rows, test_collection = _parallel(_collect_test, TEST_SEEDS)
    print("completed raw-critic untouched counterfactual test", flush=True)
    alignment = raw_counterfactual_alignment_metrics(
        model=model,
        scale=scale,
        rows=test_rows,
    )
    gate = {
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
    }
    passed = all(gate.values())
    return {
        "status": "final_pursuit_estimator_class_development_gate",
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
            "encoder": "shared_2x12_channel_3x3_CNN",
            "embedding_dimension": 32,
            "hidden_dimension": 64,
        },
        "sample_counts": {
            "train_selected_packets": len(train_rows),
            "validation_selected_packets": len(validation_rows),
            "test_audit_packets": len(test_rows),
        },
        "training": training,
        "target_scale": {
            "location": scale.target_location,
            "scale": scale.target_scale,
        },
        "validation_selected_packet_metrics": validation_metrics,
        "test_counterfactual_alignment_metrics": alignment,
        "gate": gate,
        "collection": {
            "train": train_collection,
            "validation": validation_collection,
            "test": test_collection,
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
